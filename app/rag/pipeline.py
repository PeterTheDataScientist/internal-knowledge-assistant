"""The RAG pipeline.

Order of operations, and why:

    question
      -> input guard          reject injection and confidential data requests early,
                              before any retrieval cost is spent
      -> question rewrite     resolve follow-ups against conversation history
      -> hybrid retrieval     dense plus BM25, fused, then MMR selection
      -> relevance gate       refuse before calling the model when nothing is relevant,
                              which removes the most common hallucination trigger
      -> context guard        strip imperative lines from untrusted documents
      -> generation           JSON only, temperature 0
      -> parse and repair     one bounded retry for malformed JSON
      -> grounding checks     citation validity, numeric grounding, lexical support,
                              with one bounded regeneration when a figure is ungrounded
      -> output guard         leak patterns
      -> response

Every refusal produces the same sentence. The reason is reported in a separate field so
that operators can see which control fired while a caller probing the system cannot.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from app.config import Settings
from app.llm.base import LLMProvider
from app.observability.logging_config import hash_text
from app.rag import grounding
from app.rag.knowledge_base import KnowledgeBase
from app.rag.memory import ConversationMemory
from app.rag.prompts import (
    STANDALONE_QUESTION_PROMPT,
    SYSTEM_PROMPT,
    build_answer_prompt,
)
from app.retrieval.retriever import RetrievedChunk
from app.safety.guards import ContextGuard, InputGuard, OutputGuard
from app.schemas import AskResponse, Citation, Confidence, SafetyReport

logger = logging.getLogger(__name__)

JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
MIN_SUPPORT_SCORE = 0.30
EXCERPT_LENGTH = 320


@dataclass
class ModelAnswer:
    sufficient: bool
    answer: str
    citations: List[int]


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        knowledge_base: KnowledgeBase,
        llm: LLMProvider,
        memory: Optional[ConversationMemory] = None,
    ) -> None:
        self.settings = settings
        self.kb = knowledge_base
        self.llm = llm
        self.memory = memory or ConversationMemory(
            settings.memory_max_turns, settings.memory_ttl_seconds, settings.memory_max_sessions
        )
        self.input_guard = InputGuard(
            max_length=settings.max_question_length,
            classifier=llm.classify_injection if settings.use_model_guard else None,
            classifier_threshold=settings.guard_threshold,
        )
        self.context_guard = ContextGuard()
        self.output_guard = OutputGuard(settings.refusal_message)

    # ------------------------------------------------------------------ public API

    def answer(
        self, question: str, session_id: Optional[str] = None, top_k: Optional[int] = None
    ) -> AskResponse:
        started = time.perf_counter()
        request_id = uuid.uuid4().hex[:16]
        top_k = top_k or self.settings.retrieval_top_k
        log_context = {"request_id": request_id, "question_hash": hash_text(question)}

        verdict = self.input_guard.check(question)
        if not verdict.allowed:
            logger.warning(
                "request refused by input guard",
                extra={**log_context, "reason": verdict.reason, "rules": verdict.triggered_rules},
            )
            return self._refusal(
                request_id,
                started,
                reason=verdict.reason or "blocked",
                safety=SafetyReport(
                    input_flagged=True,
                    triggered_rules=verdict.triggered_rules,
                    injection_score=verdict.injection_score,
                ),
                session_id=session_id,
            )

        cleaned_question = verdict.sanitised_question
        history = self.memory.history(session_id) if session_id else []
        search_query = self._standalone_question(cleaned_question, history)

        retrieved = self.kb.retriever.retrieve(search_query, top_k)
        best_score = max((item.score for item in retrieved), default=0.0)

        if not retrieved or best_score < self.settings.min_relevance_score:
            logger.info(
                "no sufficiently relevant context",
                extra={**log_context, "best_score": round(best_score, 3)},
            )
            return self._refusal(
                request_id,
                started,
                reason="no_relevant_context",
                safety=SafetyReport(injection_score=verdict.injection_score),
                session_id=session_id,
            )

        context = self.context_guard.sanitise(retrieved)
        if context.sanitised:
            logger.warning(
                "injection stripped from retrieved content",
                extra={**log_context, "sources": context.untrusted_sources},
            )

        parsed, raw_text = self._generate(search_query, context.chunks, history)

        if parsed is None:
            logger.error("model output could not be parsed as JSON", extra=log_context)
            return self._refusal(
                request_id,
                started,
                reason="invalid_model_output",
                safety=SafetyReport(
                    context_sanitised=context.sanitised,
                    triggered_rules=context.triggered_rules,
                    injection_score=verdict.injection_score,
                ),
                session_id=session_id,
            )

        if not parsed.sufficient or not parsed.answer.strip():
            return self._refusal(
                request_id,
                started,
                reason="model_reported_insufficient_context",
                safety=SafetyReport(
                    context_sanitised=context.sanitised,
                    triggered_rules=context.triggered_rules,
                    injection_score=verdict.injection_score,
                ),
                session_id=session_id,
            )

        report = grounding.assess(parsed.answer, parsed.citations, context.chunks, search_query)

        if report.ungrounded_numbers:
            logger.warning(
                "regenerating after ungrounded figure",
                extra={**log_context, "numbers": report.ungrounded_numbers[:5]},
            )
            repaired, _ = self._generate(
                search_query,
                context.chunks,
                history,
                correction=(
                    "Your previous answer contained figures that do not appear in the CONTEXT: "
                    f"{', '.join(report.ungrounded_numbers[:5])}. Every number must be copied "
                    "from the CONTEXT. If the CONTEXT does not state the figure, set "
                    '"sufficient" to false.'
                ),
            )
            if repaired is not None and repaired.sufficient and repaired.answer.strip():
                candidate = grounding.assess(
                    repaired.answer, repaired.citations, context.chunks, search_query
                )
                if not candidate.ungrounded_numbers:
                    parsed, report = repaired, candidate

        rejection = self._grounding_rejection(report)
        if rejection:
            logger.warning(
                "answer rejected by grounding checks",
                extra={
                    **log_context,
                    "reason": rejection,
                    "support": report.support_score,
                    "numeric_support": report.numeric_support,
                },
            )
            return self._refusal(
                request_id,
                started,
                reason=rejection,
                safety=SafetyReport(
                    context_sanitised=context.sanitised,
                    output_flagged=True,
                    triggered_rules=context.triggered_rules + [rejection],
                    injection_score=verdict.injection_score,
                ),
                session_id=session_id,
            )

        output = self.output_guard.check(parsed.answer)
        if not output.allowed:
            return self._refusal(
                request_id,
                started,
                reason="output_blocked",
                safety=SafetyReport(
                    context_sanitised=context.sanitised,
                    output_flagged=True,
                    triggered_rules=context.triggered_rules + output.triggered_rules,
                    injection_score=verdict.injection_score,
                ),
                session_id=session_id,
            )

        citations = self._build_citations(report.valid_citations, context.chunks)
        confidence = grounding.score_confidence(report, context.chunks, bool(report.valid_citations))

        if session_id:
            self.memory.append(session_id, cleaned_question, output.text)

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "question answered",
            extra={
                **log_context,
                "sources": report.sources,
                "confidence": confidence.value,
                "support": report.support_score,
                "latency_ms": latency_ms,
                "raw_length": len(raw_text),
            },
        )

        return AskResponse(
            answer=output.text,
            sources=report.sources,
            confidence=confidence,
            answered_from_context=True,
            refusal_reason=None,
            citations=citations,
            safety=SafetyReport(
                context_sanitised=context.sanitised,
                triggered_rules=context.triggered_rules,
                injection_score=verdict.injection_score,
            ),
            request_id=request_id,
            latency_ms=latency_ms,
            model=self.llm.model,
            session_id=session_id,
        )

    def stream_answer(self, question: str, session_id: Optional[str] = None, top_k: Optional[int] = None):
        """Answer, then stream. Kept for callers that want one call."""
        yield from self.stream_prepared(
            self.answer(question, session_id=session_id, top_k=top_k)
        )

    def stream_prepared(self, response: AskResponse):
        """Stream a response that has already passed every check.

        Streaming raw model tokens would mean publishing text before the grounding and
        leak checks have run, which is not acceptable for this use case. The answer is
        validated first and then released in pieces, so the caller gets progressive
        rendering without the service ever emitting unverified text. Generating before
        the stream opens also keeps the failure path on a real status code.
        """
        words = response.answer.split(" ")
        for position in range(0, len(words), 6):
            piece = " ".join(words[position : position + 6])
            yield piece + (" " if position + 6 < len(words) else "")
        yield None, response  # sentinel carrying the full response object

    # ------------------------------------------------------------------ internals

    def _standalone_question(self, question: str, history: Sequence[Tuple[str, str]]) -> str:
        if not history:
            return question
        try:
            rendered = "\n".join(
                f"Employee: {user}\nAssistant: {assistant}" for user, assistant in history[-3:]
            )
            response = self.llm.complete(
                "You rewrite follow-up questions into standalone questions. Reply with the "
                "question only.",
                STANDALONE_QUESTION_PROMPT.format(history=rendered, question=question),
                temperature=0.0,
                max_tokens=120,
            )
            rewritten = response.text.strip().strip('"')
            if not self._usable_rewrite(rewritten):
                return question
            return rewritten
        except Exception as exc:  # noqa: BLE001 - a rewrite failure must not fail the request
            logger.warning("question rewrite failed", extra={"reason": str(exc)[:200]})
            return question

    @staticmethod
    def _usable_rewrite(text: str) -> bool:
        """Reject a rewrite that is not a question.

        A provider that cannot rewrite returns something else, and the offline provider
        returns its JSON refusal object. Feeding that to the retriever produced a query
        made of punctuation and a guaranteed refusal on the second turn of every
        conversation.
        """
        if not 3 < len(text) <= 400:
            return False
        if text.lstrip().startswith(("{", "[")):
            return False
        return sum(character.isalpha() for character in text) >= 0.5 * len(text)

    def _generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Sequence[Tuple[str, str]],
        correction: Optional[str] = None,
    ) -> Tuple[Optional[ModelAnswer], str]:
        prompt = build_answer_prompt(question, chunks, history)
        if correction:
            prompt = f"{prompt}\n\nCORRECTION: {correction}"

        response = self.llm.complete(SYSTEM_PROMPT, prompt, json_mode=True)
        parsed = self._parse(response.text)
        if parsed is not None:
            return parsed, response.text

        # One bounded repair attempt. Beyond that the request is refused rather than
        # retried indefinitely, because latency is a user-visible cost.
        retry = self.llm.complete(
            SYSTEM_PROMPT,
            f"{prompt}\n\nYour previous reply was not valid JSON. Reply with the JSON object "
            'only, in the form {"sufficient": true, "answer": "...", "citations": [1]}.',
            json_mode=True,
        )
        return self._parse(retry.text), retry.text

    @staticmethod
    def _parse(text: str) -> Optional[ModelAnswer]:
        if not text or not text.strip():
            return None

        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", candidate).strip()

        payload: Optional[dict] = None
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            match = JSON_OBJECT_RE.search(candidate)
            if match:
                try:
                    payload = json.loads(match.group(0))
                except json.JSONDecodeError:
                    payload = None

        if not isinstance(payload, dict):
            return None

        raw_citations = payload.get("citations") or []
        if not isinstance(raw_citations, list):
            raw_citations = []

        answer = payload.get("answer")
        raw_sufficient = payload.get("sufficient")
        # Some models return the flag as the string "true" rather than a boolean.
        sufficient = (
            raw_sufficient
            if isinstance(raw_sufficient, bool)
            else str(raw_sufficient).strip().lower() == "true"
        )

        return ModelAnswer(
            sufficient=sufficient,
            answer=answer.strip() if isinstance(answer, str) else "",
            citations=[c for c in raw_citations if isinstance(c, (int, str))],
        )

    @staticmethod
    def _grounding_rejection(report: grounding.GroundingReport) -> Optional[str]:
        if report.ungrounded_numbers:
            return "ungrounded_figures"
        if report.untrusted_only:
            return "untrusted_sources_only"
        if not report.valid_citations:
            return "missing_citations"
        if report.support_score < MIN_SUPPORT_SCORE:
            return "low_groundedness"
        return None

    @staticmethod
    def _build_citations(
        indices: Sequence[int], chunks: Sequence[RetrievedChunk]
    ) -> List[Citation]:
        citations: List[Citation] = []
        for index in indices:
            if not 1 <= index <= len(chunks):
                continue
            item = chunks[index - 1]
            body = item.chunk.text.split("\n", 1)[-1].strip()
            excerpt = body[:EXCERPT_LENGTH] + ("..." if len(body) > EXCERPT_LENGTH else "")
            citations.append(
                Citation(
                    source=item.chunk.source,
                    chunk_id=item.chunk.chunk_id,
                    section=item.chunk.section,
                    score=item.score,
                    excerpt=excerpt,
                )
            )
        return citations

    def _refusal(
        self,
        request_id: str,
        started: float,
        *,
        reason: str,
        safety: SafetyReport,
        session_id: Optional[str],
    ) -> AskResponse:
        return AskResponse(
            answer=self.settings.refusal_message,
            sources=[],
            confidence=Confidence.NONE,
            answered_from_context=False,
            refusal_reason=reason,
            citations=[],
            safety=safety,
            request_id=request_id,
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=self.llm.model,
            session_id=session_id,
        )

    def stats(self) -> Dict[str, object]:
        return {
            "documents": self.kb.document_count,
            "chunks": self.kb.chunk_count,
            "embedder": self.kb.embedder.name,
            "vector_backend": self.kb.vector_index.backend,
            "sessions": self.memory.session_count,
        }
