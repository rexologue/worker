from __future__ import annotations
from typing import Any, Dict, Optional

from engines.base import EngineWrapper, GenerationResult
from engines.llama_cpp_engine import LlamaCppEngine

from utils.enums import EngineType

class ModelHost:
    """
    Тонкая обёртка над конкретным EngineWrapper.
    Сейчас — только llama.cpp; позднее добавишь фабрику для vLLM и др.
    """

    def __init__(
        self,
        engine: str,
        model_path: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8000,
        served_name: str = "local-model",
        http_timeout: float = 60.0,
        engine_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.model_path = model_path

        engine_kwargs = engine_kwargs or {}

        if engine == EngineType.llama_cpp:
            self.wrapper: EngineWrapper = LlamaCppEngine(
                model_path=model_path,
                host=host,
                port=port,
                served_name=served_name,
                http_timeout=http_timeout,
                **engine_kwargs,
            )
        else:
            raise NotImplementedError(f"Engine '{engine}' is not supported yet.")

    # ---- lifecycle ----
    def start(self) -> None:
        self.wrapper.start()

    def stop(self) -> None:
        self.wrapper.stop()

    def ready(self) -> bool:
        return self.wrapper.ready()

    def set_system_prompt(self, prompt: Optional[str]) -> None:
        self.wrapper.set_system_prompt(prompt)

    # ---- generation ----
    def generate(self, prompt: str, **gen_params: Any) -> GenerationResult:
        return self.wrapper.generate(prompt, **gen_params)
