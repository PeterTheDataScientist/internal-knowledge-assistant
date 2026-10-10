"""Dense vector index.

FAISS is used when it is installed because that is what a production deployment would
run. With 500,000 documents the same interface is backed by a managed vector database,
so the rest of the pipeline is written against ``VectorIndex.search`` and nothing else.
For this corpus a flat inner-product index is exact and takes microseconds, so there is
no reason to approximate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised by whichever backend is installed
    import faiss

    FAISS_AVAILABLE = True
except Exception:  # noqa: BLE001
    faiss = None
    FAISS_AVAILABLE = False


class VectorIndex:
    """Inner product search over L2 normalised vectors, which equals cosine similarity."""

    def __init__(self, dimension: int, prefer_faiss: bool = True) -> None:
        self.dimension = dimension
        self.backend = "faiss" if (FAISS_AVAILABLE and prefer_faiss) else "numpy"
        self._matrix: np.ndarray | None = None
        self._index = faiss.IndexFlatIP(dimension) if self.backend == "faiss" else None
        self.size = 0

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected vectors of dimension {self.dimension}, got {vectors.shape[1]}"
            )
        if self.backend == "faiss":
            self._index.add(vectors)
        self._matrix = vectors if self._matrix is None else np.vstack([self._matrix, vectors])
        self.size = int(self._matrix.shape[0])

    def search(self, query: np.ndarray, top_k: int) -> List[Tuple[int, float]]:
        if self.size == 0:
            return []
        top_k = min(top_k, self.size)
        query = np.ascontiguousarray(query, dtype="float32").reshape(1, -1)

        if self.backend == "faiss":
            scores, ids = self._index.search(query, top_k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

        scores = (self._matrix @ query.T).ravel()
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._matrix is not None:
            np.save(directory / "vectors.npy", self._matrix)
        (directory / "vector_meta.json").write_text(
            json.dumps({"dimension": self.dimension, "size": self.size, "backend": self.backend}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path, prefer_faiss: bool = True) -> "VectorIndex":
        meta = json.loads((directory / "vector_meta.json").read_text(encoding="utf-8"))
        index = cls(meta["dimension"], prefer_faiss=prefer_faiss)
        vectors = np.load(directory / "vectors.npy")
        index.add(vectors)
        return index
