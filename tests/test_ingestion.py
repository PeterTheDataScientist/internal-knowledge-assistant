"""Ingestion and chunking."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunker import chunk_document, chunk_documents
from app.ingestion.loader import TrustTier, load_documents

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_every_corpus_file_is_loaded():
    documents = load_documents(DATA_DIR)
    sources = {document.source for document in documents}
    assert "hr-leave-policy.md" in sources
    assert "aml-kyc-policy.md" in sources
    assert len(documents) >= 17


def test_notice_board_is_marked_untrusted():
    documents = {document.source: document for document in load_documents(DATA_DIR)}
    assert documents["staff-notices.md"].trust is TrustTier.UNTRUSTED
    assert documents["hr-leave-policy.md"].trust is TrustTier.AUTHORITATIVE


def test_corpus_notice_is_marked_meta_not_untrusted():
    # The notice describes the corpus and mentions the phrase "user-generated content",
    # which an ordering mistake in the classifier would misread as a notice board.
    documents = {document.source: document for document in load_documents(DATA_DIR)}
    assert documents["00-README-CORPUS.md"].trust is TrustTier.META


def test_missing_directory_raises():
    with pytest.raises(FileNotFoundError):
        load_documents(DATA_DIR / "does-not-exist")


def test_chunks_carry_provenance():
    documents = load_documents(DATA_DIR)
    chunks = chunk_documents(documents, 450, 80)
    assert chunks
    for chunk in chunks:
        assert chunk.source
        assert chunk.chunk_id.startswith(chunk.source)
        assert chunk.text.strip()


def test_chunk_ids_are_unique():
    chunks = chunk_documents(load_documents(DATA_DIR), 450, 80)
    ids = [chunk.chunk_id for chunk in chunks]
    assert len(ids) == len(set(ids))


def test_long_section_is_split_with_overlap():
    document = next(d for d in load_documents(DATA_DIR) if d.source == "credit-policy.md")
    chunks = chunk_document(document, 200, 60)
    assert len(chunks) > 1
    assert all(len(chunk.text) < 700 for chunk in chunks)


def test_context_block_labels_trust():
    documents = {d.source: d for d in load_documents(DATA_DIR)}
    trusted = chunk_document(documents["hr-leave-policy.md"], 450, 80)[0]
    untrusted = chunk_document(documents["staff-notices.md"], 450, 80)[0]
    assert "AUTHORITATIVE POLICY" in trusted.as_context_block(1)
    assert "UNTRUSTED USER CONTENT" in untrusted.as_context_block(1)
