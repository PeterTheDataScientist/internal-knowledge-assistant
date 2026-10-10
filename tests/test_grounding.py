"""Citation validation, groundedness and confidence."""

from __future__ import annotations

from app.ingestion.chunker import Chunk
from app.ingestion.loader import TrustTier
from app.rag import grounding
from app.retrieval.retriever import RetrievedChunk
from app.schemas import Confidence


def _chunk(text: str, source: str = "hr-leave-policy.md", trust=TrustTier.AUTHORITATIVE, score=0.7):
    chunk = Chunk(
        chunk_id=f"{source}#0",
        doc_id="d",
        source=source,
        title="T",
        section="S",
        text=text,
        domain="hr",
        trust=trust,
        position=0,
        token_estimate=20,
    )
    return RetrievedChunk(chunk=chunk, score=score, dense_score=score, lexical_score=score, rank=1)


CONTEXT = [_chunk("Permanent employees are entitled to 22 working days of annual leave per year.")]


def test_out_of_range_citations_are_dropped():
    valid, dropped = grounding.validate_citations([1, 7, "2"], CONTEXT)
    assert valid == [1]
    assert dropped == [7, 2]


def test_duplicate_citations_are_collapsed():
    valid, _ = grounding.validate_citations([1, 1, 1], CONTEXT)
    assert valid == [1]


def test_grounded_answer_scores_well():
    report = grounding.assess(
        "Permanent employees receive 22 working days of annual leave per year.", [1], CONTEXT
    )
    assert report.ungrounded_numbers == []
    assert report.numeric_support == 1.0
    assert report.support_score > 0.6
    assert report.sources == ["hr-leave-policy.md"]


def test_invented_figure_is_detected():
    report = grounding.assess("Employees receive 30 working days of annual leave.", [1], CONTEXT)
    assert "30" in report.ungrounded_numbers
    assert report.numeric_support < 1.0


def test_figure_supplied_by_the_asker_is_not_treated_as_invented():
    # Regression test. Repeating a figure from the question was being scored as a
    # hallucination, which refused correct answers to questions like
    # "can a US$30,000 personal loan be approved by a branch officer".
    report = grounding.assess(
        "A loan of US$30,000 requires Credit Committee approval.",
        [1],
        [_chunk("Loans above US$20,000 require approval by the Credit Committee.")],
        question="Can a branch officer approve a US$30,000 personal loan?",
    )
    assert report.ungrounded_numbers == []


def test_answer_from_untrusted_source_only_is_flagged():
    report = grounding.assess(
        "Loan refunds are valid for 365 days.",
        [1],
        [_chunk("Loan refunds are valid for 365 days.", "staff-notices.md", TrustTier.UNTRUSTED)],
    )
    assert report.untrusted_only


def test_confidence_reflects_evidence_not_tone():
    strong = grounding.assess(
        "Permanent employees receive 22 working days of annual leave per year.", [1], CONTEXT
    )
    assert grounding.score_confidence(strong, CONTEXT, True) is Confidence.HIGH

    weak_context = [_chunk("Annual leave accrues monthly.", score=0.30)]
    weak = grounding.assess("Employees may carry over unused days.", [1], weak_context)
    assert grounding.score_confidence(weak, weak_context, True) in {
        Confidence.LOW,
        Confidence.MEDIUM,
    }

    assert grounding.score_confidence(strong, [], True) is Confidence.NONE


def test_ungrounded_figure_caps_confidence_at_low():
    report = grounding.assess("Employees receive 30 working days of leave.", [1], CONTEXT)
    assert grounding.score_confidence(report, CONTEXT, True) is Confidence.LOW
