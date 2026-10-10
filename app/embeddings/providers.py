"""Concrete embedding backends and the factory that picks one."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from app.embeddings.base import Embedder, batched, l2_normalise

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(Embedder):
    """Local sentence-transformer model. Default choice when the weights are available."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # imported lazily

        self.name = f"sentence_transformer:{model_name}"
        self._model = SentenceTransformer(model_name)
        self.dimension = int(self._model.get_sentence_embedding_dimension())

    def fit(self, corpus: Sequence[str]) -> None:
        return None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for batch in batched(list(texts), 32):
            vectors.append(
                self._model.encode(list(batch), show_progress_bar=False, convert_to_numpy=True)
            )
        return l2_normalise(np.vstack(vectors))


class TfidfSvdEmbedder(Embedder):
    """Character and word TF-IDF reduced with truncated SVD.

    This is the fallback when the neural model cannot be downloaded, for example on an
    air gapped build agent. It is weaker at paraphrase matching than a neural encoder,
    which is one reason retrieval is hybrid: the lexical channel carries most of the
    load in that configuration.
    """

    def __init__(self, dimension: int = 256) -> None:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        self.name = f"tfidf_svd:{dimension}"
        self.dimension = dimension
        self._features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        min_df=1,
                        stop_words="english",
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        sublinear_tf=True,
                        min_df=1,
                    ),
                ),
            ]
        )
        self._svd = TruncatedSVD(n_components=dimension, random_state=42)
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> None:
        matrix = self._features.fit_transform(list(corpus))
        max_components = max(2, min(self.dimension, min(matrix.shape) - 1))
        if max_components != self._svd.n_components:
            self._svd.set_params(n_components=max_components)
            self.dimension = max_components
            self.name = f"tfidf_svd:{max_components}"
        self._svd.fit(matrix)
        self._fitted = True

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("TfidfSvdEmbedder.fit must be called before encode")
        matrix = self._features.transform(list(texts))
        return l2_normalise(self._svd.transform(matrix))


def build_embedder(backend: str, model_name: str, fallback_dim: int) -> Embedder:
    """Return an embedder, degrading to the statistical backend if the model is missing."""
    if backend == "tfidf":
        return TfidfSvdEmbedder(fallback_dim)

    try:
        return SentenceTransformerEmbedder(model_name)
    except Exception as exc:  # noqa: BLE001 - any load failure must degrade, not crash
        if backend == "sentence_transformer":
            raise
        logger.warning(
            "sentence-transformer model unavailable, falling back to tfidf embedder",
            extra={"model": model_name, "reason": str(exc)[:200]},
        )
        return TfidfSvdEmbedder(fallback_dim)
