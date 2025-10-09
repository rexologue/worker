from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

@dataclass
class GenerationResult:
    text: str
    usage: Dict[str, Any]  # {"prompt_tokens":..., "completion_tokens":..., "total_tokens":...}

class EngineWrapper(ABC):
    """
    Унифицированный интерфейс для движков инференса (llama.cpp, vLLM и др.).
    """

    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 8000):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.system_prompt: Optional[str] = None

    # -------- lifecycle ----------
    @abstractmethod
    def start(self) -> None:
        """Запуск сервера/движка (если нужно). Должен блокироваться до готовности."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Корректная остановка сервера/движка."""
        ...

    @abstractmethod
    def ready(self) -> bool:
        """Готов ли бэкенд принимать запросы."""
        ...

    # -------- generation ----------
    @abstractmethod
    def generate(self, prompt: str, **gen_params: Any) -> GenerationResult:
        """
        Синхронная генерация (запрос-ответ). gen_params — все параметры, специфичные движку:
        temperature, top_p, top_k, max_tokens, stop, frequency_penalty, presence_penalty,
        repeat_penalty, mirostat, typical_p, min_p, tfs, grammar, reasoning и т.п.
        """
        ...

    # -------- optional helpers ----------
    def set_system_prompt(self, prompt: Optional[str]) -> None:
        self.system_prompt = prompt
