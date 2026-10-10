"""Embedding provider interface.

Two implementations exist behind this interface so the service can start with no network
access. Swapping in Azure OpenAI or Bedrock embeddings means adding one more subclass and
one branch in the factory, nothing else in the pipeline changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np


class Embedder(ABC):
    name: str = "base"
    dimension: int = 0

    @abstractmethod
    def fit(self, corpus: Sequence[str]) -> None:
        """Prepare the embedder. Neural models ignore this, statistical ones need it."""

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2 normalised float32 matrix of shape (len(texts), dimension)."""

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float32")
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def batched(items: Sequence[str], size: int) -> List[Sequence[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
