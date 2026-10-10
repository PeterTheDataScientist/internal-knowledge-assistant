"""Three guard stages: input, retrieved context, and generated output.

The threat model for an internal banking assistant is not only a malicious employee. It
is also an ordinary employee whose question happens to retrieve a document that somebody
else edited. Both are covered here.

Stage 1 screens the question. Stage 2 treats every retrieved passage as data rather than
instruction and strips imperative lines from documents that were not written by a policy
owner. Stage 3 checks what the model produced before it leaves the process.
"""

from __future__ import annotations

import dataclasses
import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from app.ingestion.loader import TrustTier
from app.retrieval.retriever import RetrievedChunk
from app.safety.patterns import (
    CONFIDENTIAL_REQUEST_RULES,
    CONTEXT_INSTRUCTION_RE,
    INJECTION_RULES,
    OUTPUT_LEAK_RULES,
    looks_like_policy_question,
    match_rules,
)

logger = logging.getLogger(__name__)

CONFIDENTIALITY_REFUSAL = (
    "I cannot provide customer account information, credentials, or other confidential "
    "data. Customer account data is classified as Confidential under the Data "
    "Classification Policy and may only be accessed in approved systems where you have a "
    "verified legitimate business need."
)


@dataclass
class InputVerdict:
    allowed: bool
    reason: Optional[str] = None
    refusal_text: Optional[str] = None
    triggered_rules: List[str] = field(default_factory=list)
    injection_score: Optional[float] = None
    sanitised_question: str = ""


@dataclass
class ContextVerdict:
    chunks: List[RetrievedChunk]
    sanitised: bool = False
    removed_lines: int = 0
    untrusted_sources: List[str] = field(default_factory=list)
    triggered_rules: List[str] = field(default_factory=list)


@dataclass
class OutputVerdict:
    allowed: bool
    text: str
    triggered_rules: List[str] = field(default_factory=list)


class InputGuard:
    """Screens the incoming question before any retrieval happens."""

    def __init__(
        self,
        *,
        max_length: int = 1000,
        classifier=None,
        classifier_threshold: float = 0.8,
    ) -> None:
        self.max_length = max_length
        self.classifier = classifier
        self.classifier_threshold = classifier_threshold

    def check(self, question: str) -> InputVerdict:
        cleaned = self._normalise(question)
        triggered: List[str] = []

        policy_shaped = looks_like_policy_question(cleaned)

        injection_hits = match_rules(cleaned, INJECTION_RULES)
        if policy_shaped:
            # "Who approves an admin override" and "what should a teller do if asked to
            # bypass dual control" are questions about controls, not attempts to defeat
            # them. Only the two rules a real question can plausibly trip are softened;
            # developer mode, fake authority and prompt extraction are never softened.
            softenable = {"ignore_prior_instructions", "role_override"}
            dropped = [hit for hit in injection_hits if hit in softenable]
            if dropped and len(dropped) == len(injection_hits):
                logger.info("injection pattern treated as a policy question")
                injection_hits = []
        triggered.extend(injection_hits)

        confidential_hits = match_rules(cleaned, CONFIDENTIAL_REQUEST_RULES)
        if confidential_hits and policy_shaped and not injection_hits:
            # A question about the confidentiality rules is not a request for the data.
            logger.info("confidential pattern treated as a policy question")
            confidential_hits = []
        triggered.extend(confidential_hits)

        score = self._classify(cleaned)
        if score is not None and score >= self.classifier_threshold:
            triggered.append("prompt_guard_model")

        if confidential_hits:
            return InputVerdict(
                allowed=False,
                reason="confidential_request",
                refusal_text=CONFIDENTIALITY_REFUSAL,
                triggered_rules=sorted(set(triggered)),
                injection_score=score,
                sanitised_question=cleaned,
            )

        if injection_hits or "prompt_guard_model" in triggered:
            return InputVerdict(
                allowed=False,
                reason="prompt_injection",
                refusal_text=(
                    "That request tries to change how this assistant operates, so it was not "
                    "processed. Ask a question about Contoso Financial internal documentation and it will be "
                    "answered from the approved documents."
                ),
                triggered_rules=sorted(set(triggered)),
                injection_score=score,
                sanitised_question=cleaned,
            )

        return InputVerdict(
            allowed=True,
            triggered_rules=[],
            injection_score=score,
            sanitised_question=cleaned,
        )

    def _classify(self, text: str) -> Optional[float]:
        if self.classifier is None:
            return None
        try:
            return self.classifier(text)
        except Exception as exc:  # noqa: BLE001 - the guard must never break a request
            logger.warning("input classifier failed", extra={"reason": str(exc)[:200]})
            return None

    def _normalise(self, question: str) -> str:
        """Strip control characters and collapse whitespace used to hide instructions."""
        cleaned = re.sub(r"[​-‏‪-‮﻿]", "", question)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
        return cleaned[: self.max_length]


class ContextGuard:
    """Neutralises instructions hidden inside retrieved documents."""

    def sanitise(self, chunks: Sequence[RetrievedChunk]) -> ContextVerdict:
        cleaned: List[RetrievedChunk] = []
        removed = 0
        untrusted_sources: List[str] = []
        triggered: List[str] = []

        for item in chunks:
            text = item.chunk.text
            if item.chunk.trust is not TrustTier.AUTHORITATIVE:
                untrusted_sources.append(item.chunk.source)
                stripped, count = CONTEXT_INSTRUCTION_RE.subn(
                    "[instruction removed by content filter]", text
                )
                if count:
                    removed += count
                    triggered.append("context_instruction_stripped")
                    text = stripped

            if text != item.chunk.text:
                patched = replace_chunk_text(item, text)
                cleaned.append(patched)
            else:
                cleaned.append(item)

        return ContextVerdict(
            chunks=cleaned,
            sanitised=removed > 0,
            removed_lines=removed,
            untrusted_sources=sorted(set(untrusted_sources)),
            triggered_rules=sorted(set(triggered)),
        )


def replace_chunk_text(item: RetrievedChunk, text: str) -> RetrievedChunk:
    """Copy a retrieved chunk with different text so the indexed original is untouched."""
    return dataclasses.replace(item, chunk=dataclasses.replace(item.chunk, text=text))


class OutputGuard:
    """Final check on the generated answer."""

    def __init__(self, refusal_message: str) -> None:
        self.refusal_message = refusal_message

    def check(self, answer: str) -> OutputVerdict:
        if not answer.strip():
            return OutputVerdict(False, self.refusal_message, ["empty_answer"])

        hits = match_rules(answer, OUTPUT_LEAK_RULES)
        if hits:
            logger.warning("output blocked by leak rules", extra={"rules": hits})
            return OutputVerdict(False, CONFIDENTIALITY_REFUSAL, hits)

        return OutputVerdict(True, answer.strip(), [])
