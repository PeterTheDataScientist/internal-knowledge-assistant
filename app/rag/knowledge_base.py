"""Builds the searchable knowledge base from the corpus on disk."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.config import Settings
from app.embeddings.base import Embedder
from app.embeddings.providers import build_embedder
from app.indexing.lexical_index import BM25Index
from app.indexing.vector_store import VectorIndex
from app.ingestion.chunker import Chunk, chunk_documents
from app.ingestion.loader import Document, TrustTier, load_documents
from app.retrieval.retriever import CrossEncoderReranker, HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class KnowledgeBase:
    documents: List[Document]
    chunks: List[Chunk]
    embedder: Embedder
    vector_index: VectorIndex
    lexical_index: BM25Index
    retriever: HybridRetriever
    build_seconds: float

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def sources(self) -> List[str]:
        return sorted({document.source for document in self.documents})


def build_knowledge_base(settings: Settings, data_dir: Optional[Path] = None) -> KnowledgeBase:
    started = time.perf_counter()
    directory = data_dir or settings.data_dir

    all_documents = load_documents(directory)
    # The corpus notice describes the corpus itself and holds no policy content, so it is
    # loaded for provenance but kept out of the index where it would only add noise.
    documents = [d for d in all_documents if d.trust is not TrustTier.META]
    skipped = [d.source for d in all_documents if d.trust is TrustTier.META]
    if skipped:
        logger.info("documents excluded from index", extra={"sources": skipped})

    chunks = chunk_documents(documents, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise RuntimeError("chunking produced no chunks, check the corpus directory")

    embedder = build_embedder(
        settings.embedding_backend, settings.embedding_model, settings.embedding_dim_fallback
    )
    texts = [chunk.text for chunk in chunks]
    embedder.fit(texts)
    vectors = embedder.encode(texts)

    vector_index = VectorIndex(embedder.dimension)
    vector_index.add(vectors)

    lexical_index = BM25Index()
    lexical_index.build(texts)

    reranker = None
    if settings.use_reranker:
        try:
            reranker = CrossEncoderReranker(settings.reranker_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker unavailable, continuing without it", extra={"reason": str(exc)[:200]})

    retriever = HybridRetriever(
        chunks=chunks,
        embedder=embedder,
        vector_index=vector_index,
        lexical_index=lexical_index,
        candidate_pool=settings.candidate_pool,
        hybrid_alpha=settings.hybrid_alpha,
        rrf_k=settings.rrf_k,
        mmr_lambda=settings.mmr_lambda,
        reranker=reranker,
    )
    retriever.attach_vectors(np.asarray(vectors, dtype="float32"))

    elapsed = time.perf_counter() - started
    logger.info(
        "knowledge base ready",
        extra={
            "documents": len(documents),
            "chunks": len(chunks),
            "embedder": embedder.name,
            "vector_backend": vector_index.backend,
            "build_seconds": round(elapsed, 3),
        },
    )

    return KnowledgeBase(
        documents=documents,
        chunks=chunks,
        embedder=embedder,
        vector_index=vector_index,
        lexical_index=lexical_index,
        retriever=retriever,
        build_seconds=elapsed,
    )
