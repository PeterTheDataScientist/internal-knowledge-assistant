"""Language model provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    name: str = "base"
    model: str = "unknown"

    @abstractmethod
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Return a single completion."""

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        """Yield the completion in pieces. Providers that cannot stream yield once."""
        response = self.complete(
            system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        yield response.text

    def health(self) -> Dict[str, str]:
        return {"status": "ok", "provider": self.name, "model": self.model}

    def classify_injection(self, text: str) -> Optional[float]:
        """Return an injection probability if the provider offers a guard model."""
        return None
