"""Retrieval behaviour."""

from __future__ import annotations

import pytest

from app.indexing.lexical_index import BM25Index, tokenize


def test_tokenizer_keeps_currency_and_indexes_digit_variants():
    tokens = tokenize("Loans above US$20,000 require approval")
    assert "us$20,000" in tokens
    assert "20000" in tokens


def test_bm25_ranks_the_document_that_contains_the_term():
    index = BM25Index()
    index.build(
        [
            "Annual leave is 22 working days per calendar year",
            "Sanctions screening records are retained for five years",
            "Mortgage loan to value ratio is 80 percent",
        ]
    )
    top = index.search("how many years are sanctions records kept", 3)
    assert top
    assert top[0][0] == 1


def test_bm25_returns_nothing_for_unknown_terms():
    index = BM25Index()
    index.build(["Annual leave is 22 working days"])
    assert index.search("quantum chromodynamics", 3) == []


@pytest.mark.parametrize(
    "question,expected_source",
    [
        ("How many days of annual leave do permanent employees get?", "hr-leave-policy.md"),
        ("What is the maximum loan to value ratio on a mortgage?", "credit-mortgage-policy.md"),
        ("Who files a suspicious transaction report?", "aml-kyc-policy.md"),
        ("How long are sanctions screening records retained?", "aml-sanctions-screening.md"),
        ("What documents are needed to open an account?", "sop-account-opening.md"),
        ("What is the waiting period on the funeral plan?", "product-funeral-plan.md"),
    ],
)
def test_expected_document_is_retrieved(knowledge_base, question, expected_source):
    hits = knowledge_base.retriever.retrieve(question, 5)
    assert expected_source in {hit.chunk.source for hit in hits}


def test_lexical_channel_survives_selection(knowledge_base):
    """Regression test for the MMR bug.

    Selection once scored candidates on dense similarity alone, which discarded the
    lexical half of the hybrid ranking. This question has one overwhelming lexical match
    and a mediocre dense score for that same chunk, so it fails if that returns.
    """
    hits = knowledge_base.retriever.retrieve(
        "A politically exposed person wants to open a personal account. "
        "What documents are required and what additional due diligence applies?",
        6,
    )
    sources = {hit.chunk.source for hit in hits}
    assert "aml-kyc-policy.md" in sources
    assert "sop-account-opening.md" in sources


def test_scores_are_ordered_and_bounded(knowledge_base):
    hits = knowledge_base.retriever.retrieve("annual leave entitlement", 5)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_empty_query_returns_nothing(knowledge_base):
    assert knowledge_base.retriever.retrieve("   ", 5) == []
