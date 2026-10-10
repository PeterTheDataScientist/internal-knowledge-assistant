"""Chooses the language model provider for the running process."""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.mock_provider import MockProvider

logger = logging.getLogger(__name__)


def build_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "mock":
        logger.info("using offline mock provider by configuration")
        return MockProvider()

    if not settings.groq_api_keys:
        if settings.llm_provider == "groq":
            raise RuntimeError("LLM_PROVIDER=groq but GROQ_API_KEYS is empty")
        logger.warning("no Groq key found, starting with the offline mock provider")
        return MockProvider()

    from app.llm.groq_provider import GroqProvider

    try:
        provider = GroqProvider(
            api_keys=settings.groq_api_keys,
            model=settings.groq_model,
            fast_model=settings.groq_fast_model,
            guard_model=settings.guard_model if settings.use_model_guard else None,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        logger.info("using Groq provider", extra={"model": settings.groq_model})
        return provider
    except Exception as exc:  # noqa: BLE001
        if settings.llm_provider == "groq":
            raise
        logger.error(
            "Groq provider could not start, falling back to mock",
            extra={"reason": str(exc)[:200]},
        )
        return MockProvider()
