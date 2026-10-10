"""BM25 lexical index.

Written out rather than pulled from a library because it is thirty lines, it removes a
dependency, and the behaviour matters: this corpus is full of exact tokens that a dense
encoder blurs. "US$20,000", "300 months", "MLRO" and "OFAC" are precisely the terms an
employee types, and lexical matching is what retrieves them reliably.

Numbers with separators are indexed both as written and stripped, so "20,000", "20000"
and "us$20,000" all reach the same chunk.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

TOKEN_RE = re.compile(r"[a-z0-9$%][a-z0-9$%,\.\-/]*")

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "for", "on", "and", "or", "is", "are", "be",
    "was", "were", "it", "its", "as", "at", "by", "that", "this", "with", "from",
    "how", "what", "when", "which", "who", "whom", "do", "does", "did", "can", "may",
    "i", "we", "you", "my", "our", "your", "me", "us", "if", "there", "their", "they",
    "much", "many", "long", "please", "tell",
}


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = raw.strip(".,-/")
        if not token or token in STOPWORDS:
            continue
        tokens.append(token)
        # Index a digits-only variant so thousands separators and currency prefixes
        # do not stop a numeric question from matching.
        digits = re.sub(r"[^0-9]", "", token)
        if digits and digits != token and len(digits) >= 2:
            tokens.append(digits)
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: List[Counter] = []
        self._lengths: List[int] = []
        self._avg_length = 0.0
        self._idf: Dict[str, float] = {}
        self.size = 0

    def build(self, texts: Sequence[str]) -> None:
        self._docs = [Counter(tokenize(text)) for text in texts]
        self._lengths = [sum(counter.values()) for counter in self._docs]
        self.size = len(self._docs)
        self._avg_length = (sum(self._lengths) / self.size) if self.size else 0.0

        document_frequency: Counter = Counter()
        for counter in self._docs:
            document_frequency.update(counter.keys())

        self._idf = {
            term: math.log(1 + (self.size - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.size == 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores = [0.0] * self.size
        for term in set(query_terms):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for doc_id, counter in enumerate(self._docs):
                frequency = counter.get(term, 0)
                if not frequency:
                    continue
                length_norm = 1 - self.b + self.b * (self._lengths[doc_id] / (self._avg_length or 1))
                scores[doc_id] += idf * (frequency * (self.k1 + 1)) / (
                    frequency + self.k1 * length_norm
                )

        ranked = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
