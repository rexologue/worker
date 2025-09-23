# model_manager.py
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout

# Optional third-party: huggingface_hub
try:
    from huggingface_hub import (
        HfApi,
        snapshot_download,
        configure_http_backend,
    )
    _HF_AVAILABLE = True
except Exception:  # pragma: no cover
    _HF_AVAILABLE = False


###############################################################################
# Logging
###############################################################################

logger = logging.getLogger("ModelManager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


###############################################################################
# Small cross-platform file lock (best-effort, no extra deps)
###############################################################################

class FileLock:
    """Lightweight file lock using fcntl/msvcrt. Best-effort."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self._fh = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt  # type: ignore
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl  # type: ignore
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX)
        except Exception as e:
            logger.debug(f"Lock not fully enforced: {e}")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fh:
                if os.name == "nt":
                    import msvcrt  # type: ignore
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl  # type: ignore
                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            if self._fh:
                self._fh.close()
                self._fh = None


###############################################################################
# Data models
###############################################################################

ModelKind = Literal["auto", "repo", "gguf"]
SourceName = Literal["hf"]  # extend later (e.g., "local", "s3", "gcs", ...)

@dataclass
class ModelSpec:
    """Declarative description of a model to install."""
    source: SourceName = "hf"
    kind: ModelKind = "auto"          # "auto" tries to detect by files; "repo" or "gguf"
    repo_id: Optional[str] = None     # For HF
    revision: Optional[str] = None    # Commit hash, tag, or branch
    repo_type: Literal["model", "dataset", "space"] = "model"  # HF repo type
    gguf_quant: Optional[str] = None  # e.g. "Q4_K_M" (if kind="gguf")
    allow_patterns: Optional[List[str]] = None  # Optional filters
    # UX sugar: arbitrary notes/tags
    tags: List[str] = field(default_factory=list)

@dataclass
class InstallInfo:
    installed: bool = False
    installed_at: Optional[float] = None
    path: Optional[str] = None
    files: List[str] = field(default_factory=list)
    selected_file: Optional[str] = None  # for gguf
    revision_resolved: Optional[str] = None
    size_bytes: Optional[int] = None

@dataclass
class RegistryRecord:
    name: str
    spec: ModelSpec
    install: InstallInfo = field(default_factory=InstallInfo)
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


###############################################################################
# Source interface (pluggable)
###############################################################################

class BaseSource(ABC):
    @abstractmethod
    def install(self, dest_dir: Path, spec: ModelSpec) -> InstallInfo:
        ...

    @abstractmethod
    def detect_kind(self, spec: ModelSpec) -> ModelKind:
        ...


###############################################################################
# Hugging Face source implementation
###############################################################################

class HuggingFaceSource(BaseSource):
    def __init__(self, token: Optional[str] = None, http_timeout: int = 15):
        if not _HF_AVAILABLE:
            raise RuntimeError("huggingface_hub is required for HuggingFaceSource")
        print(token)
        self.api = HfApi(token=token)
        self.token = token
        self.http_timeout = http_timeout
        # Глобальный таймаут HTTP-запросов HF Hub
        try:
            configure_http_backend(timeout=http_timeout)
        except Exception as e:
            logger.debug(f"configure_http_backend failed: {e}")

    # ---------- helpers ----------

    def _run_with_timeout(self, fn, timeout: int):
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(fn)
            return fut.result(timeout=timeout)

    def _list_files_fast(self, spec: ModelSpec) -> List[str]:
        """
        Многоступенчатое перечисление файлов:
          1) model_info.siblings (верхний уровень) — быстро
          2) list_repo_tree(recursive=False) — немного тяжелее
          3) list_repo_files(...) — самый тяжелый; под таймаутом и с ретраями
        Возвращает пути относительно корня репозитория.
        """
        repo_id = spec.repo_id
        revision = spec.revision
        repo_type = spec.repo_type

        # Step 1: model_info (top-level only)
        try:
            def _info():
                return self.api.model_info(repo_id=repo_id, revision=revision) if repo_type == "model" \
                    else self.api.repo_info(repo_id=repo_id, revision=revision, repo_type=repo_type)
            info = self._run_with_timeout(_info, self.http_timeout)
            siblings = getattr(info, "siblings", None) or getattr(info, "files", None)
            if siblings:
                names = []
                for s in siblings:
                    rfn = getattr(s, "rfilename", None) or getattr(s, "path", None) or getattr(s, "filename", None)
                    if rfn:
                        names.append(rfn)
                if names:
                    logger.debug(f"[HF] list(top-level via info): {len(names)} files")
                    return names
        except FutTimeout:
            logger.warning(f"[HF] model_info timeout after {self.http_timeout}s")
        except Exception as e:
            logger.debug(f"[HF] model_info failed: {e}")

        # Step 2: top-level tree (non-recursive)
        try:
            def _tree():
                return self.api.list_repo_tree(
                    repo_id=repo_id,
                    revision=revision,
                    repo_type=repo_type,
                    recursive=False,
                )
            items = self._run_with_timeout(_tree, self.http_timeout)
            top_names = [it.path for it in items if getattr(it, "type", None) in (None, "file")]
            if top_names:
                logger.debug(f"[HF] list(top-level via tree): {len(top_names)} files")
                return top_names
        except FutTimeout:
            logger.warning(f"[HF] list_repo_tree timeout after {self.http_timeout}s")
        except Exception as e:
            logger.debug(f"[HF] list_repo_tree failed: {e}")

        # Step 3: full listing with retries/backoff
        delays = [self.http_timeout, int(self.http_timeout * 1.5), int(self.http_timeout * 2)]
        last_err = None
        for i, t in enumerate(delays, 1):
            try:
                logger.info(f"[HF] full list_repo_files attempt {i}/{len(delays)} (timeout {t}s)...")
                def _files():
                    return self.api.list_repo_files(
                        repo_id=repo_id,
                        repo_type=repo_type,
                        revision=revision
                    )
                files = self._run_with_timeout(_files, t)
                logger.info(f"[HF] list_repo_files ok: {len(files)} files")
                return files
            except FutTimeout as e:
                last_err = e
                logger.warning(f"[HF] list_repo_files timeout after {t}s (attempt {i})")
            except Exception as e:
                last_err = e
                logger.warning(f"[HF] list_repo_files error (attempt {i}): {e}")
            time.sleep(0.3 * i)  # мягкий backoff
        # Всё сломалось
        raise RuntimeError(f"[HF] Could not list files for {repo_type}:{repo_id}@{revision} — {last_err}")

    def _list_files(self, spec: ModelSpec) -> List[str]:
        # Просто обёртка, чтобы заменить вызовы в остальных методах
        return self._list_files_fast(spec)

    # ---------- detect kind ----------

    def detect_kind(self, spec: ModelSpec) -> ModelKind:
        if spec.kind != "auto":
            return spec.kind
        files = self._list_files(spec)
        if any(f.lower().endswith(".gguf") for f in files):
            return "gguf"
        return "repo"

    # ---------- GGUF helpers (как было в твоём коде) ----------

    _QUANT_TOKEN_RE = re.compile(r"(?i)\b(I?Q[0-9]+[A-Z0-9_+-]*)\b")

    @staticmethod
    def _extract_quant_tokens(filename: str) -> List[str]:
        return HuggingFaceSource._QUANT_TOKEN_RE.findall(filename)

    @staticmethod
    def _quant_score(token: str) -> float:
        t = token.upper()
        base = 0.0
        iq = t.startswith("IQ")
        m = re.search(r"(I?Q)(\d+)", t)
        if m:
            num = int(m.group(2))
            base = float(num) * 10.0
            if iq:
                base += 1.0
        if "_K" in t:
            base += 0.6
        if "_M" in t or "_S" in t or "_XL" in t:
            base += 0.2
        if "_0" in t or "_1" in t:
            base += 0.05
        if "NL" in t:
            base += 0.3
        return base

    def _pick_gguf(self, files: List[str], preferred_quant: Optional[str] = None) -> str:
        ggufs = [f for f in files if f.lower().endswith(".gguf")]
        if not ggufs:
            raise FileNotFoundError("No .gguf files in the repo")
        if preferred_quant:
            cand = [f for f in ggufs if preferred_quant.lower() in f.lower()]
            if not cand:
                found = {}
                for f in ggufs:
                    toks = self._extract_quant_tokens(f)
                    for t in toks:
                        found.setdefault(t.upper(), 0)
                        found[t.upper()] += 1
                raise ValueError(
                    f"Requested quant '{preferred_quant}' not found. "
                    f"Available (approx): {', '.join(sorted(found))}"
                )
            if len(cand) > 1:
                cand.sort(key=lambda x: (-max([self._quant_score(t) for t in self._extract_quant_tokens(x)] or [0]), len(x)))
            return cand[0]
        def best_score(path: str) -> float:
            toks = self._extract_quant_tokens(path)
            return max([self._quant_score(t) for t in toks] or [0.0])
        ggufs.sort(key=lambda p: (best_score(p), -len(Path(p).name)), reverse=True)
        return ggufs[0]

    # ---------- install ----------

    def install(self, dest_dir: Path, spec: ModelSpec) -> InstallInfo:
        dest_dir.mkdir(parents=True, exist_ok=True)
        resolved_kind = self.detect_kind(spec)
        info = InstallInfo(installed=False)

        if resolved_kind == "gguf":
            files = self._list_files(spec)
            chosen_rel = self._pick_gguf(files, spec.gguf_quant)
            logger.info(f"Downloading GGUF: {spec.repo_id}@{spec.revision or 'latest'} :: {chosen_rel}")
            snapshot_download(
                repo_id=spec.repo_id,  # type: ignore[arg-type]
                repo_type=spec.repo_type,
                revision=spec.revision,
                local_dir=str(dest_dir),
                allow_patterns=[chosen_rel, "README*", "LICENSE*", "tokenizer.*", "*.json", "*.txt"],
                token=self.token
            )
            chosen_abs = dest_dir / chosen_rel
            files_on_disk = [str(p.relative_to(dest_dir)) for p in dest_dir.rglob("*") if p.is_file()]
            info.installed = True
            info.installed_at = time.time()
            info.path = str(dest_dir)
            info.files = files_on_disk
            info.selected_file = str(chosen_abs)
            info.revision_resolved = spec.revision or "latest"
            info.size_bytes = sum((dest_dir / f).stat().st_size for f in files_on_disk if (dest_dir / f).exists())
            return info

        # Generic repo
        logger.info(f"Downloading repo: {spec.repo_id}@{spec.revision or 'latest'}")
        snapshot_download(
            repo_id=spec.repo_id,  # type: ignore[arg-type]
            repo_type=spec.repo_type,
            revision=spec.revision,
            local_dir=str(dest_dir),
            allow_patterns=spec.allow_patterns,
            token=self.token
        )

        files_on_disk = [str(p.relative_to(dest_dir)) for p in dest_dir.rglob("*") if p.is_file()]
        info.installed = True
        info.installed_at = time.time()
        info.path = str(dest_dir)
        info.files = files_on_disk
        info.revision_resolved = spec.revision or "latest"
        info.size_bytes = sum((dest_dir / f).stat().st_size for f in files_on_disk if (dest_dir / f).exists())
        
        return info


###############################################################################
# ModelManager
###############################################################################

def _sanitize_name(name: str) -> str:
    name = name.strip().lower()
    return re.sub(r"[^a-z0-9_.:-]+", "-", name)

class ModelManager:
    """
    Unified model manager:
      - base_dir: ~/.models_mgr
      - registry.json (atomic, locked)
      - pluggable sources (currently HF)
    """

    REGISTRY_VERSION = 1

    def __init__(self, base_dir: Union[str, Path] = "~/.models_mgr", token_hf: Optional[str] = None):
        self.base_dir = Path(os.path.expanduser(base_dir))
        self.models_dir = self.base_dir / "models"
        self.registry_path = self.base_dir / "registry.json"
        self.lock_path = self.base_dir / ".lock"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.sources: Dict[str, BaseSource] = {
            "hf": HuggingFaceSource(token=token_hf) if _HF_AVAILABLE else None  # type: ignore[dict-item]
        }
        if self.sources["hf"] is None and token_hf is not None:
            # Token provided but package missing
            logger.warning("huggingface_hub not available; HF source disabled")

        self._registry: Dict[str, RegistryRecord] = {}
        self._load_registry()

    # ---------------- Registry I/O ----------------

    def _load_registry(self) -> None:
        if not self.registry_path.exists():
            self._save_registry()
            return
        try:
            with FileLock(self.lock_path):
                data = json.loads(self.registry_path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        # migrate / parse
        models = data.get("models", {})
        out: Dict[str, RegistryRecord] = {}
        for name, rec in models.items():
            spec = ModelSpec(**rec["spec"])
            install = InstallInfo(**rec.get("install", {}))
            rr = RegistryRecord(name=name, spec=spec, install=install,
                                created_at=rec.get("created_at", time.time()),
                                updated_at=rec.get("updated_at", time.time()))
            out[name] = rr
        self._registry = out

    def _save_registry(self) -> None:
        to_dump = {
            "version": self.REGISTRY_VERSION,
            "models": {
                name: {
                    "name": rec.name,
                    "spec": asdict(rec.spec),
                    "install": asdict(rec.install),
                    "created_at": rec.created_at,
                    "updated_at": rec.updated_at,
                }
                for name, rec in sorted(self._registry.items())
            },
        }
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(self.base_dir))
        try:
            json.dump(to_dump, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            with FileLock(self.lock_path):
                Path(tmp.name).replace(self.registry_path)
        finally:
            try:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
            except Exception:
                pass

    # ---------------- Public API ----------------

    def add(self, name: str, spec: ModelSpec, overwrite: bool = False) -> None:
        """Register model spec (without installing)."""
        key = _sanitize_name(name)
        if key in self._registry and not overwrite:
            raise ValueError(f"Model '{key}' already exists. Use overwrite=True to replace.")
        self._registry[key] = RegistryRecord(name=key, spec=spec)
        self._registry[key].updated_at = time.time()
        self._save_registry()
        logger.info(f"Registered model '{key}'")

    def remove(self, name: str, delete_files: bool = False) -> None:
        key = _sanitize_name(name)
        rec = self._registry.get(key)
        if not rec:
            raise KeyError(f"No such model: {key}")
        # delete files
        if delete_files and rec.install.installed and rec.install.path:
            p = Path(rec.install.path)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        del self._registry[key]
        self._save_registry()
        logger.info(f"Removed model '{key}'")

    def install(self, name: str, force: bool = False) -> Path:
        """Install/download model into manager storage. Returns local path."""
        key = _sanitize_name(name)
        rec = self._registry.get(key)
        if not rec:
            raise KeyError(f"No such model: {key}")

        # Source
        src = self.sources.get(rec.spec.source)
        if src is None:
            raise RuntimeError(f"Source '{rec.spec.source}' is not available for this manager.")

        dest_dir = self._model_dir_for(key)
        if dest_dir.exists() and not force and rec.install.installed:
            logger.info(f"Already installed: {key} -> {dest_dir}")
            return dest_dir

        if dest_dir.exists() and force:
            logger.info(f"Force re-install: wiping {dest_dir}")
            shutil.rmtree(dest_dir, ignore_errors=True)

        info = src.install(dest_dir, rec.spec)
        rec.install = info
        rec.updated_at = time.time()
        self._save_registry()
        return Path(info.path) if info.path else dest_dir

    def get_model(
        self,
        name: str,
        prefer_quant: Optional[str] = None,
        install_if_missing: bool = True
    ) -> Path:
        """
        Returns a path suitable for inference:
          - for 'gguf': returns absolute path to selected *.gguf
          - for 'repo': returns directory path with snapshot
        """
        key = _sanitize_name(name)
        rec = self._registry.get(key)
        if not rec:
            raise KeyError(f"No such model: {key}")

        if not rec.install.installed or not rec.install.path or not Path(rec.install.path).exists():
            if not install_if_missing:
                raise FileNotFoundError(f"Model '{key}' is not installed")
            self.install(key)

        # If GGUF and prefer_quant given, optionally reselect (and download if needed)
        src = self.sources.get(rec.spec.source)
        if isinstance(src, HuggingFaceSource):
            kind = src.detect_kind(rec.spec)
        else:
            kind = rec.spec.kind if rec.spec.kind != "auto" else "repo"

        if kind == "gguf":
            # if prefer_quant differs, ensure we have that file; if not, download it
            if prefer_quant and (not rec.install.selected_file or prefer_quant.lower() not in rec.install.selected_file.lower()):
                # list repo, pick quant, download file only
                files = src._list_files(rec.spec)  # type: ignore[attr-defined]
                chosen_rel = src._pick_gguf(files, preferred_quant=prefer_quant)  # type: ignore[attr-defined]
                dest_dir = Path(rec.install.path)  # type: ignore[arg-type]
                snapshot_download(
                    repo_id=rec.spec.repo_id,  # type: ignore[arg-type]
                    repo_type=rec.spec.repo_type,
                    revision=rec.spec.revision,
                    local_dir=str(dest_dir),
                    local_dir_use_symlinks=True,
                    allow_patterns=[chosen_rel]
                )
                rec.install.selected_file = str(dest_dir / chosen_rel)
                rec.install.files = [str(p.relative_to(dest_dir)) for p in dest_dir.rglob("*") if p.is_file()]
                rec.updated_at = time.time()
                self._save_registry()
            # return gguf file
            gguf_path = Path(rec.install.selected_file) if rec.install.selected_file else None
            if not gguf_path or not gguf_path.exists():
                # fall back: try to find any gguf
                candidates = list(Path(rec.install.path).rglob("*.gguf"))  # type: ignore[arg-type]
                if not candidates:
                    raise FileNotFoundError(f"No GGUF file found for '{key}'")
                # choose best by heuristic
                chosen = max(
                    candidates,
                    key=lambda p: max([HuggingFaceSource._quant_score(t) for t in HuggingFaceSource._extract_quant_tokens(p.name)] or [0.0])
                )
                rec.install.selected_file = str(chosen)
                self._save_registry()
                gguf_path = chosen
            return gguf_path.resolve()

        # generic repo -> return directory
        return Path(rec.install.path).resolve()  # type: ignore[arg-type]

    def list_models(self, installed_only: bool = False) -> List[Dict]:
        out = []
        for name, rec in sorted(self._registry.items()):
            if installed_only and not rec.install.installed:
                continue
            out.append({
                "name": rec.name,
                "source": rec.spec.source,
                "kind": rec.spec.kind,
                "repo_id": rec.spec.repo_id,
                "revision": rec.spec.revision,
                "installed": rec.install.installed,
                "path": rec.install.path,
                "selected_file": rec.install.selected_file,
                "size_bytes": rec.install.size_bytes,
                "tags": rec.spec.tags,
            })
        return out

    def info(self, name: str) -> Dict:
        key = _sanitize_name(name)
        rec = self._registry.get(key)
        if not rec:
            raise KeyError(f"No such model: {key}")
        return {
            "name": rec.name,
            "spec": asdict(rec.spec),
            "install": asdict(rec.install),
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
        }

    def update(self, name: str) -> Path:
        """Reinstall (refresh) a model (same revision or latest)."""
        return self.install(name, force=True)

    def find_files(self, name: str, pattern: str) -> List[Path]:
        key = _sanitize_name(name)
        rec = self._registry.get(key)
        if not rec or not rec.install.path:
            raise KeyError(f"No such model or not installed: {key}")
        root = Path(rec.install.path)
        return [p for p in root.rglob("*") if p.is_file() and p.match(pattern)]

    def uninstall(self, name: str) -> None:
        self.remove(name, delete_files=True)

    # ---------------- Paths & Helpers ----------------

    def _model_dir_for(self, name: str) -> Path:
        return self.models_dir / _sanitize_name(name)


###############################################################################
# Example usage (comment out in production)
###############################################################################
if __name__ == "__main__":
    """
    Quick demo:

    from model_manager import ModelManager, ModelSpec
    mm = ModelManager()
    mm.add("llama3-8b-q4", ModelSpec(
        source="hf",
        repo_id="TheBloke/Llama-3-8B-Instruct-GGUF",
        kind="gguf",
        gguf_quant="Q4_K_M"
    ))
    path = mm.get_model("llama3-8b-q4")  # -> /home/user/.models_mgr/models/llama3-8b-q4/.../model.Q4_K_M.gguf
    print("Use this in llama.cpp:", path)

    # Generic repo:
    mm.add("phi-mini", ModelSpec(source="hf", repo_id="microsoft/phi-4-mini", kind="repo"))
    repo_path = mm.get_model("phi-mini")  # returns directory path
    print("Repo snapshot at:", repo_path)

    print(mm.list_models())
    """
    pass
