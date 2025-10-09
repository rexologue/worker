from __future__ import annotations
import os
import sys
import time
import json
import socket
import atexit
import platform
import subprocess
from typing import Any, Dict, List, Optional

import requests

from .base import EngineWrapper, GenerationResult

class LlamaCppEngine(EngineWrapper):
    """
    Обёртка над llama_cpp.server (OpenAI-совместимый API).
    Цели:
      - CPU-режим по умолчанию (n_gpu_layers=0)
      - Автозапуск/остановка сервера
      - Полный проксинг параметров генерации в /v1/chat/completions
        (и /v1/responses при необходимости для 'reasoning')
    """

    def __init__(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 8000,
        served_name: str = "local-model",
        attach_if_running: bool = True,
        http_timeout: float = 60.0,
        extra_server_args: Optional[List[str]] = None,
    ) -> None:
        super().__init__(model_path=model_path, host=host, port=port)

        self.served_name = served_name
        self.attach_if_running = attach_if_running
        self.http_timeout = http_timeout
        self.extra_server_args = extra_server_args or []
        self._proc: Optional[subprocess.Popen] = None

        atexit.register(self._atexit_stop)

    # ---------- lifecycle ----------
    def start(self) -> None:
        # если уже поднят и отвечает - просто используем
        if self.attach_if_running and self._http_ready(timeout=2.0):
            return

        cmd = [
            sys.executable, "-m", "llama_cpp.server",
            "--model", self.model_path,
            "--host", self.host,
            "--port", str(self.port),
            "--model_alias", self.served_name,
            "--n_gpu_layers", "0",          # CPU-only как просили
            "--n_ctx", "8192",              # разумный дефолт; можно править из extra_server_args
        ] + self.extra_server_args

        creationflags = 0
        start_new_session = False
        if platform.system() == "Windows":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            start_new_session = True

        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=start_new_session,
            creationflags=creationflags,
        )

        # ждём готовности по /v1/models
        self._wait_http_ready()

    def stop(self) -> None:
        if not self._proc:
            return
        try:
            if platform.system() == "Windows":
                try:
                    self._proc.terminate()
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            else:
                try:
                    os.killpg(self._proc.pid, 15)  # SIGTERM
                except Exception:
                    self._proc.terminate()
                try:
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self._proc.pid, 9)  # SIGKILL
                    except Exception:
                        self._proc.kill()
        finally:
            self._proc = None

    def _atexit_stop(self) -> None:
        try:
            self.stop()
        except Exception:
            pass

    def ready(self) -> bool:
        return self._http_ready(timeout=1.0)

    # ---------- generation ----------
    def generate(self, prompt: str, **gen_params: Any) -> GenerationResult:
        """
        Делает один запрос в /v1/chat/completions (или /v1/responses если указан reasoning/use_responses).
        Возвращает текст и usage.
        """
        if not self.ready():
            raise RuntimeError("llama.cpp server is not ready. Did you call start()?")

        # 1) определим, куда слать
        use_responses = bool(gen_params.pop("use_responses", False) or gen_params.get("reasoning"))

        # лёгкий проброс в /v1/responses если задан reasoning, но сервер может не уметь — тогда fallback
        if use_responses and self._has_responses():
            return self._generate_via_responses(prompt, **gen_params)
        else:
            return self._generate_via_chat(prompt, **gen_params)

    # ---------- internals ----------
    @property
    def _base(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _generate_via_chat(self, prompt: str, **params: Any) -> GenerationResult:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.served_name,
            "messages": messages,
        }

        # поддерживаем популярные параметры + произвольные поля на проброс
        # стандартные
        for key in (
            "temperature", "top_p", "top_k", "max_tokens", "stop",
            "frequency_penalty", "presence_penalty",
            "n", "logprobs", "logit_bias",
        ):
            if key in params and params[key] is not None:
                payload[key] = params[key]

        # параметры llama.cpp, которые часто нужны
        for key in (
            "repeat_penalty", "mirostat", "mirostat_tau", "mirostat_eta",
            "typical_p", "min_p", "tfs", "seed", "grammar",
        ):
            if key in params and params[key] is not None:
                payload[key] = params[key]

        # если пользователь положил что-то ещё — тоже отправим
        for k, v in params.items():
            if k not in payload and v is not None:
                payload[k] = v

        r = requests.post(f"{self._base}/v1/chat/completions", json=payload, timeout=self.http_timeout)
        if r.status_code != 200:
            raise RuntimeError(f"chat completion failed: {r.status_code} {r.text}")

        out = r.json()
        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        return GenerationResult(text=text, usage=usage)

    def _generate_via_responses(self, prompt: str, **params: Any) -> GenerationResult:
        """
        /v1/responses — если сервер поддерживает (некоторые сборки llama.cpp добавляют этот роут).
        Формируем input из system+user; reasoning можно передать как dict: {"effort": "medium"}.
        """
        input_parts = []
        if self.system_prompt:
            input_parts.append(self.system_prompt.strip())
        input_parts.append(prompt)

        payload: Dict[str, Any] = {
            "model": self.served_name,
            "input": "\n\n".join(input_parts),
        }

        if "reasoning" in params and params["reasoning"] is not None:
            payload["reasoning"] = params.pop("reasoning")

        # те же параметры вероятностной выборки
        for key in (
            "temperature", "top_p", "top_k", "max_tokens", "stop",
            "frequency_penalty", "presence_penalty",
            "repeat_penalty", "mirostat", "mirostat_tau", "mirostat_eta",
            "typical_p", "min_p", "tfs", "seed", "grammar",
        ):
            if key in params and params[key] is not None:
                payload[key] = params[key]
        for k, v in params.items():
            if k not in payload and v is not None:
                payload[k] = v

        r = requests.post(f"{self._base}/v1/responses", json=payload, timeout=self.http_timeout)
        if r.status_code == 404:
            # сервер не умеет — откат на чат
            return self._generate_via_chat(prompt, **params)
        if r.status_code != 200:
            raise RuntimeError(f"responses failed: {r.status_code} {r.text}")

        out = r.json()

        # некоторые реализации кладут ответ в choices/message, некоторые — в output_text
        text = out.get("output_text")
        if text is None and "choices" in out:
            text = out["choices"][0].get("message", {}).get("content")
        if text is None:
            text = json.dumps(out)  # крайний случай, чтобы не потерять ответ
            
        usage = out.get("usage", {})
        return GenerationResult(text=text, usage=usage)

    def _http_ready(self, timeout: float = 1.0) -> bool:
        t0 = time.time()
        url = f"{self._base}/v1/models"
        while time.time() - t0 < timeout:
            try:
                r = requests.get(url, timeout=0.5)
                if r.ok:
                    return True
            except requests.RequestException:
                pass
            time.sleep(0.1)
        return False

    def _wait_http_ready(self, timeout: float = 90.0, interval: float = 0.5) -> None:
        t0 = time.time()
        url = f"{self._base}/v1/models"
        last_err = None

        while time.time() - t0 < timeout:
            try:
                r = requests.get(url, timeout=5)
                if r.ok:
                    js = r.json()
                    ids = [m.get("id") for m in js.get("data", [])]
                    if self.served_name in ids or not ids:
                        return
                    
            except Exception as e:
                last_err = e

            time.sleep(interval)

        # финальная проверка, что порт вообще открыт
        self._wait_port(timeout=5.0)
        raise TimeoutError(f"llama_cpp.server is not ready. Last error: {last_err}")

    def _wait_port(self, timeout: float = 60.0, interval: float = 0.5) -> None:
        t0 = time.time()

        while time.time() - t0 < timeout:
            with socket.socket() as s:
                s.settimeout(interval)
                if s.connect_ex((self.host, self.port)) == 0:
                    return
                
            time.sleep(interval)

        raise TimeoutError(f"Port {self.host}:{self.port} did not open within {timeout}s.")

    def _has_responses(self) -> bool:
        try:
            r = requests.options(f"{self._base}/v1/responses", timeout=2)
            return r.status_code < 500
        except requests.RequestException:
            return False
