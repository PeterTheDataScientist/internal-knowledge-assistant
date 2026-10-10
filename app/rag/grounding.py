"""Citation validation, groundedness scoring and confidence.

Asking a model for citations is not the same as having citations. This module checks
them, which is the difference between a demo and something a compliance officer will
sign off.

Three checks run on every answer:

1. Citation validity. Every number the model returned must point at a block that was
   actually retrieved. Invalid numbers are dropped rather than passed through.
2. Numeric grounding. Every figure in the answer must appear in the cited text. This is
   the check that catches the most damaging failure mode in a policy assistant, which is
   a right-sounding answer with the wrong number.
3. Lexical support. The share of the answer's content words that appear in the cited
   text. Low overlap means the model wrote from memory rather than from the passage.

Confidence is derived from those signals plus retrieval strength, so it reflects the
evidence rather than the model's tone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Sequence, Set, Tuple

from app.ingestion.loader import TrustTier
from app.retrieval.retriever import RetrievedChunk
from app.schemas import Confidence

WORD_RE = re.compile(r"[a-z0-9%$][a-z0-9%$/\.,-]*")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

FILLER_WORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is", "are", "be",
    "must", "may", "can", "will", "shall", "with", "that", "this", "it", "as", "at",
    "by", "from", "not", "no", "you", "your", "employee", "employees", "per", "any",
    "all", "each", "which", "their", "they", "there", "than", "then", "into", "under",
    "when", "where", "who", "has", "have", "had", "does", "do", "if", "up", "out",
}


@dataclass
class GroundingReport:
    valid_citations: List[int]
    dropped_citations: List[int]
    sources: List[str]
    support_score: float
    numeric_support: float
    ungrounded_numbers: List[str] = field(default_factory=list)
    untrusted_only: bool = False


def _content_words(text: str) -> Set[str]:
    return {
        token.strip(".,")
        for token in WORD_RE.findall(text.lower())
        if token.strip(".,") and token.strip(".,") not in FILLER_WORDS and len(token) > 1
    }


def _numbers(text: str) -> List[str]:
    return [match.replace(",", "") for match in NUMBER_RE.findall(text)]


def validate_citations(
    raw_citations: Sequence[int], retrieved: Sequence[RetrievedChunk]
) -> Tuple[List[int], List[int]]:
    valid: List[int] = []
    dropped: List[int] = []
    for number in raw_citations:
        try:
            index = int(number)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(retrieved):
            if index not in valid:
                valid.append(index)
        else:
            dropped.append(index)
    return valid, dropped


def assess(
    answer: str,
    citations: Sequence[int],
    retrieved: Sequence[RetrievedChunk],
    question: str = "",
) -> GroundingReport:
    """Score an answer against the passages it cites.

    ``question`` is passed in so that figures the employee supplied themselves are not
    counted as ungrounded. An answer to "can I approve a US$30,000 loan" has to be able
    to repeat US$30,000 without that being treated as an invented figure.
    """
    valid, dropped = validate_citations(citations, retrieved)

    # A model that answers without citing anything still has to be checked, so fall back
    # to the full retrieved set when the citation list is empty.
    evaluation_indices = valid or list(range(1, len(retrieved) + 1))
    cited_chunks = [retrieved[index - 1] for index in evaluation_indices]
    cited_text = " ".join(item.chunk.text for item in cited_chunks).lower()

    answer_words = _content_words(answer)
    support = (
        len(answer_words & _content_words(cited_text)) / len(answer_words) if answer_words else 0.0
    )

    answer_numbers = _numbers(answer)
    context_numbers = set(_numbers(cited_text)) | set(_numbers(question))
    ungrounded = [number for number in answer_numbers if number not in context_numbers]
    numeric_support = (
        (len(answer_numbers) - len(ungrounded)) / len(answer_numbers) if answer_numbers else 1.0
    )

    sources: List[str] = []
    for item in cited_chunks:
        if item.chunk.source not in sources:
            sources.append(item.chunk.source)

    untrusted_only = bool(cited_chunks) and all(
        item.chunk.trust is not TrustTier.AUTHORITATIVE for item in cited_chunks
    )

    return GroundingReport(
        valid_citations=valid,
        dropped_citations=dropped,
        sources=sources,
        support_score=round(support, 3),
        numeric_support=round(numeric_support, 3),
        ungrounded_numbers=ungrounded,
        untrusted_only=untrusted_only,
    )


def score_confidence(
    report: GroundingReport, retrieved: Sequence[RetrievedChunk], has_citations: bool
) -> Confidence:
    if not retrieved:
        return Confidence.NONE

    top_score = max(item.score for item in retrieved)
    cited_scores = [
        retrieved[index - 1].score for index in report.valid_citations if index <= len(retrieved)
    ]
    evidence = max(cited_scores) if cited_scores else top_score

    if report.numeric_support < 1.0 or report.untrusted_only:
        return Confidence.LOW

    if has_citations and evidence >= 0.55 and report.support_score >= 0.55:
        return Confidence.HIGH

    if has_citations and evidence >= 0.35 and report.support_score >= 0.35:
        return Confidence.MEDIUM

    return Confidence.LOW
