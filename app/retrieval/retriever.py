"""Hybrid retrieval.

Dense search finds paraphrases, BM25 finds exact policy terms and figures. Neither is
enough alone on this corpus, so both run and their rankings are fused with reciprocal
rank fusion. RRF is used for ordering because it combines rankings without needing the
two score distributions to be comparable, which they are not.

A separate calibrated score is computed for thresholding and confidence, because a rank
position tells you nothing about whether the top result is actually relevant. That score
is a weighted blend of the normalised dense and lexical scores.

Selection then applies maximal marginal relevance so that the context window is not
filled with three near duplicate chunks from the same section.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.embeddings.base import Embedder
from app.indexing.lexical_index import BM25Index
from app.indexing.vector_store import VectorIndex
from app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    dense_score: float
    lexical_score: float
    rank: int

    @property
    def source(self) -> str:
        return self.chunk.source


# BM25 scores are unbounded, so they are squashed with a saturating transform rather than
# min-max normalised. Min-max would make the best candidate score 1.0 in every query,
# including queries where nothing in the corpus is relevant, which is exactly the case the
# relevance gate has to catch.
BM25_SATURATION = 6.0


def _saturate(score: float, half_point: float = BM25_SATURATION) -> float:
    return score / (score + half_point) if score > 0 else 0.0


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        vector_index: VectorIndex,
        lexical_index: BM25Index,
        *,
        candidate_pool: int = 20,
        hybrid_alpha: float = 0.5,
        rrf_k: int = 60,
        mmr_lambda: float = 0.7,
        reranker: Optional["CrossEncoderReranker"] = None,
    ) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        self.vector_index = vector_index
        self.lexical_index = lexical_index
        self.candidate_pool = candidate_pool
        self.hybrid_alpha = hybrid_alpha
        self.rrf_k = rrf_k
        self.mmr_lambda = mmr_lambda
        self.reranker = reranker
        self._vectors: np.ndarray | None = None

    def attach_vectors(self, vectors: np.ndarray) -> None:
        """Keep the chunk matrix so MMR can measure chunk to chunk similarity."""
        self._vectors = vectors

    def retrieve(self, query: str, top_k: int) -> List[RetrievedChunk]:
        if not query.strip() or not self.chunks:
            return []

        pool = max(self.candidate_pool, top_k)
        query_vector = self.embedder.encode_one(query)

        dense_hits = self.vector_index.search(query_vector, pool)
        lexical_hits = self.lexical_index.search(query, pool)

        dense_raw = dict(dense_hits)
        lexical_raw = dict(lexical_hits)
        lexical_norm = {index: _saturate(score) for index, score in lexical_raw.items()}

        dense_rank = {index: rank for rank, (index, _) in enumerate(dense_hits, start=1)}
        lexical_rank = {index: rank for rank, (index, _) in enumerate(lexical_hits, start=1)}

        candidates = set(dense_raw) | set(lexical_raw)
        fused: Dict[int, float] = {}
        blended: Dict[int, float] = {}

        for index in candidates:
            rrf = 0.0
            if index in dense_rank:
                rrf += 1.0 / (self.rrf_k + dense_rank[index])
            if index in lexical_rank:
                rrf += 1.0 / (self.rrf_k + lexical_rank[index])
            fused[index] = rrf

            dense_component = min(1.0, max(0.0, dense_raw.get(index, 0.0)))
            blended[index] = (
                self.hybrid_alpha * dense_component
                + (1 - self.hybrid_alpha) * lexical_norm.get(index, 0.0)
            )

        ordered = sorted(candidates, key=lambda i: (fused[i], blended[i]), reverse=True)
        shortlist = ordered[: max(pool, top_k)]

        if self.reranker is not None and shortlist:
            shortlist = self.reranker.rerank(query, shortlist, self.chunks)

        selected = self._select_with_mmr(shortlist, blended, top_k)
        # MMR chooses a diverse set; the set is then presented strongest first so the
        # rank a caller sees always matches the score next to it.
        selected = sorted(selected, key=lambda i: blended.get(i, 0.0), reverse=True)

        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=round(float(blended.get(index, 0.0)), 4),
                dense_score=round(float(dense_raw.get(index, 0.0)), 4),
                lexical_score=round(float(lexical_norm.get(index, 0.0)), 4),
                rank=position,
            )
            for position, index in enumerate(selected, start=1)
        ]

    def _select_with_mmr(
        self, shortlist: Sequence[int], scores: Dict[int, float], top_k: int
    ) -> List[int]:
        """Pick a diverse subset, scoring relevance with the fused score.

        Using the dense score alone here was a real bug during development. A question
        about a politically exposed person opening an account has one overwhelming
        lexical match, the AML policy, and a mediocre dense score for that same chunk.
        Selecting on dense similarity threw the fused ranking away and dropped the one
        document that answered half the question.
        """
        if self._vectors is None or len(shortlist) <= top_k:
            return list(shortlist)[:top_k]

        remaining = list(shortlist)
        chosen: List[int] = []

        while remaining and len(chosen) < top_k:
            best_index = None
            best_value = -1e9
            for index in remaining:
                relevance = scores.get(index, 0.0)
                redundancy = (
                    max(float(self._vectors[index] @ self._vectors[other]) for other in chosen)
                    if chosen
                    else 0.0
                )
                value = self.mmr_lambda * relevance - (1 - self.mmr_lambda) * redundancy
                if value > best_value:
                    best_value = value
                    best_index = index
            chosen.append(best_index)
            remaining.remove(best_index)

        return chosen


class CrossEncoderReranker:
    """Optional cross encoder stage, disabled by default.

    On a corpus this small it adds latency for no measurable gain, but the hook is here
    because at 500,000 documents re-ranking the top 50 is the single highest value
    change to retrieval precision.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # imported lazily

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, shortlist: Sequence[int], chunks: Sequence[Chunk]) -> List[int]:
        pairs = [(query, chunks[index].text) for index in shortlist]
        scores = self._model.predict(pairs)
        ordered = sorted(zip(shortlist, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [index for index, _ in ordered]
