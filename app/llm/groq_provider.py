"""Groq backed provider.

Groq is used because it serves open weight models at low latency and its free tier
supplies keys. Nothing above this file knows which vendor is in use, so replacing it
with Azure OpenAI for a real bank deployment is a single class.

Two operational details are handled here rather than left to chance:

* several keys can be supplied and are rotated on rate limit responses, which is what
  actually happens under load on a shared tenancy;
* retries are bounded and only applied to transient status codes, so a bad request is
  surfaced immediately instead of being retried three times.
"""

from __future__ import annotations

import itertools
import logging
import random
import re
import time
from typing import Dict, Iterator, List, Optional

from app.errors import LLMTimeoutError, LLMUnavailableError
from app.llm.base import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}

# Groq reports the wait in the error body, for example "try again in 12.5s". Honouring it
# is the difference between a run that completes and a run that burns its retries in two
# seconds and gives up.
RETRY_AFTER_RE = re.compile(r"try again in\s+([0-9.]+)\s*(ms|s)\b", re.IGNORECASE)
MAX_BACKOFF_SECONDS = 65.0
DAILY_LIMIT_RE = re.compile(r"tokens per day|requests per day|\bTPD\b|\bRPD\b", re.IGNORECASE)
DAILY_COOLDOWN_SECONDS = 3600.0


def _suggested_wait(message: str) -> Optional[float]:
    match = RETRY_AFTER_RE.search(message or "")
    if not match:
        return None
    value = float(match.group(1))
    seconds = value / 1000 if match.group(2).lower() == "ms" else value
    return min(seconds + 0.5, MAX_BACKOFF_SECONDS)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(
        self,
        api_keys: List[str],
        model: str,
        *,
        fast_model: Optional[str] = None,
        guard_model: Optional[str] = None,
        timeout: float = 25.0,
        max_retries: int = 2,
        temperature: float = 0.0,
        max_tokens: int = 700,
    ) -> None:
        if not api_keys:
            raise LLMUnavailableError("no Groq API key configured")

        from groq import Groq  # imported lazily so the mock provider needs no SDK

        self.model = model
        self.fast_model = fast_model or model
        self.guard_model = guard_model
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_temperature = temperature
        self.default_max_tokens = max_tokens

        self._clients = [Groq(api_key=key, timeout=timeout, max_retries=0) for key in api_keys]
        self._cycle = itertools.cycle(range(len(self._clients)))
        self._current = next(self._cycle)
        # A key that has burned its daily allowance is useless for the rest of the day, so
        # it is parked rather than rotated back into the pool a second later. Without this
        # the pool spends its retries on keys that are certain to fail.
        self._cooldown_until: Dict[int, float] = {}

    def _client(self):
        return self._clients[self._current]

    def _available(self, index: int) -> bool:
        return time.time() >= self._cooldown_until.get(index, 0.0)

    def _rotate_key(self, cooldown: float = 0.0) -> None:
        if cooldown:
            self._cooldown_until[self._current] = time.time() + cooldown
        if len(self._clients) == 1:
            return
        for _ in range(len(self._clients)):
            candidate = next(self._cycle)
            if self._available(candidate):
                self._current = candidate
                logger.info("rotated Groq API key", extra={"key_index": candidate})
                return
        logger.warning("every Groq key is in cooldown, continuing with the current one")

    @staticmethod
    def _status_of(exc: Exception) -> Optional[int]:
        return getattr(exc, "status_code", None) or getattr(
            getattr(exc, "response", None), "status_code", None
        )

    def _all_keys_cooling(self) -> bool:
        return all(not self._available(index) for index in range(len(self._clients)))

    def _call(self, model: str, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        last_error: Optional[Exception] = None
        started = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            try:
                completion = self._client().chat.completions.create(
                    model=model, messages=messages, **kwargs
                )
                usage = getattr(completion, "usage", None)
                return LLMResponse(
                    text=(completion.choices[0].message.content or "").strip(),
                    model=model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:  # noqa: BLE001 - normalised into domain errors below
                last_error = exc
                status = self._status_of(exc)
                name = type(exc).__name__.lower()

                if "timeout" in name and attempt >= self.max_retries:
                    raise LLMTimeoutError() from exc

                if status is not None and status not in RETRYABLE_STATUS:
                    logger.error(
                        "non retryable model error",
                        extra={"status": status, "model": model, "reason": str(exc)[:200]},
                    )
                    raise LLMUnavailableError(f"model request rejected with status {status}") from exc

                if status == 429:
                    daily = DAILY_LIMIT_RE.search(str(exc)) is not None
                    self._rotate_key(DAILY_COOLDOWN_SECONDS if daily else 0.0)

                if attempt < self.max_retries:
                    backoff = _suggested_wait(str(exc)) or (
                        (0.4 * (2**attempt)) + random.uniform(0, 0.2)
                    )
                    logger.warning(
                        "retrying model call",
                        extra={"attempt": attempt + 1, "status": status, "backoff": round(backoff, 2)},
                    )
                    time.sleep(backoff)

        raise LLMUnavailableError(f"model call failed after retries: {str(last_error)[:200]}")

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        kwargs: Dict[str, object] = {
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_completion_tokens": max_tokens or self.default_max_tokens,
            "top_p": 1,
            "stream": False,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            return self._call(self.model, messages, **kwargs)
        except LLMUnavailableError:
            # Graceful degradation. When the primary model has no quota left on any key,
            # a smaller model on the same account is a better answer than a 503, and the
            # response reports which model actually served the request.
            if self.fast_model == self.model or not self._all_keys_cooling():
                raise
            logger.warning(
                "primary model exhausted, degrading to the fast model",
                extra={"primary": self.model, "fallback": self.fast_model},
            )
            self._cooldown_until.clear()
            return self._call(self.fast_model, messages, **kwargs)

    def stream(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        completion = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.default_temperature if temperature is None else temperature,
            max_completion_tokens=max_tokens or self.default_max_tokens,
            stream=True,
        )
        for piece in completion:
            delta = piece.choices[0].delta.content
            if delta:
                yield delta

    def classify_injection(self, text: str) -> Optional[float]:
        """Score text with Llama Prompt Guard 2, which returns a probability as a string.

        A failure here must never block a request, so any error returns None and the
        deterministic pattern rules remain in force.
        """
        if not self.guard_model or not text.strip():
            return None
        try:
            completion = self._client().chat.completions.create(
                model=self.guard_model,
                messages=[{"role": "user", "content": text[:4000]}],
            )
            return float((completion.choices[0].message.content or "0").strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("guard model unavailable", extra={"reason": str(exc)[:200]})
            return None

    def health(self) -> Dict[str, str]:
        try:
            self._call(
                self.fast_model,
                [{"role": "user", "content": "ping"}],
                temperature=0,
                max_completion_tokens=1,
            )
            return {"status": "ok", "provider": self.name, "model": self.model}
        except Exception as exc:  # noqa: BLE001
            return {"status": "degraded", "provider": self.name, "detail": str(exc)[:200]}
