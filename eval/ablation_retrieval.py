"""Retrieval ablation.

Answer quality is bounded by retrieval. If the passage holding the answer never reaches
the prompt, no amount of prompt engineering recovers it, so retrieval is measured on its
own rather than only through end to end answers. This runs without a model, so it costs
nothing and is fast enough to sit in a pull request check.

Four configurations are compared over every answerable case in the golden set:

  dense only          sentence-transformer embeddings
  lexical only        BM25
  hybrid, dense MMR   RRF fusion, then selection scored on dense similarity alone.
                      This is the configuration that shipped first and it contains a real
                      bug: selecting on dense similarity throws away the lexical half of
                      the fused ranking.
  hybrid, fused MMR   RRF fusion, then selection scored on the fused score. Current.

Metrics, all at the document level because that is what a citation names:

  recall@k    share of expected source documents that appear in the top k
  hit rate    share of questions where at least one expected document appears
  MRR         mean reciprocal rank of the first expected document

Run:  python eval/ablation_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.knowledge_base import build_knowledge_base  # noqa: E402
from app.retrieval.retriever import HybridRetriever  # noqa: E402

TOP_K = 6


def load_answerable(path: Path) -> List[dict]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [case for case in cases if case.get("expected_sources")]


def dense_only(retriever: HybridRetriever, query: str, top_k: int) -> List[str]:
    vector = retriever.embedder.encode_one(query)
    hits = retriever.vector_index.search(vector, top_k)
    return [retriever.chunks[index].source for index, _ in hits]


def lexical_only(retriever: HybridRetriever, query: str, top_k: int) -> List[str]:
    hits = retriever.lexical_index.search(query, top_k)
    return [retriever.chunks[index].source for index, _ in hits]


def hybrid_dense_mmr(retriever: HybridRetriever, query: str, top_k: int) -> List[str]:
    """Reproduces the shipped-then-fixed selection bug for comparison."""
    original = retriever._select_with_mmr  # noqa: SLF001 - deliberate, this is the ablation

    def buggy(shortlist: Sequence[int], scores: Dict[int, float], k: int) -> List[int]:
        vectors = retriever._vectors  # noqa: SLF001
        if vectors is None or len(shortlist) <= k:
            return list(shortlist)[:k]
        query_vector = np.asarray(retriever.embedder.encode_one(query), dtype="float32").reshape(-1)
        remaining, chosen = list(shortlist), []
        while remaining and len(chosen) < k:
            best, best_value = None, -1e9
            for index in remaining:
                relevance = float(vectors[index] @ query_vector)
                redundancy = (
                    max(float(vectors[index] @ vectors[other]) for other in chosen) if chosen else 0.0
                )
                value = retriever.mmr_lambda * relevance - (1 - retriever.mmr_lambda) * redundancy
                if value > best_value:
                    best, best_value = index, value
            chosen.append(best)
            remaining.remove(best)
        return chosen

    retriever._select_with_mmr = buggy  # noqa: SLF001
    try:
        return [hit.chunk.source for hit in retriever.retrieve(query, top_k)]
    finally:
        retriever._select_with_mmr = original  # noqa: SLF001


def hybrid_fused_mmr(retriever: HybridRetriever, query: str, top_k: int) -> List[str]:
    return [hit.chunk.source for hit in retriever.retrieve(query, top_k)]


def score(cases: List[dict], retrieved_per_case: List[List[str]]) -> dict:
    recalls, hits, reciprocal_ranks = [], [], []

    for case, retrieved in zip(cases, retrieved_per_case):
        expected = set(case["expected_sources"])
        ordered_unique: List[str] = []
        for source in retrieved:
            if source not in ordered_unique:
                ordered_unique.append(source)

        found = expected & set(ordered_unique)
        recalls.append(len(found) / len(expected))
        hits.append(1.0 if found else 0.0)

        rank = next(
            (position for position, source in enumerate(ordered_unique, start=1) if source in expected),
            None,
        )
        reciprocal_ranks.append(1 / rank if rank else 0.0)

    return {
        "recall_at_k": round(sum(recalls) / len(recalls), 3),
        "hit_rate": round(sum(hits) / len(hits), 3),
        "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 3),
    }


def main() -> int:
    cases = load_answerable(Path(__file__).parent / "questions.jsonl")
    knowledge_base = build_knowledge_base(get_settings())
    retriever = knowledge_base.retriever

    configurations = {
        "dense only": dense_only,
        "lexical only, BM25": lexical_only,
        "hybrid, selection on dense score": hybrid_dense_mmr,
        "hybrid, selection on fused score": hybrid_fused_mmr,
    }

    results = {}
    for name, function in configurations.items():
        retrieved = [function(retriever, case["question"], TOP_K) for case in cases]
        results[name] = score(cases, retrieved)

    multi = [case for case in cases if len(case["expected_sources"]) > 1]
    multi_results = {}
    for name, function in configurations.items():
        retrieved = [function(retriever, case["question"], TOP_K) for case in multi]
        multi_results[name] = score(multi, retrieved)

    print(f"\nAnswerable cases: {len(cases)}, of which multi document: {len(multi)}, k = {TOP_K}\n")
    header = f"{'configuration':<36}{'recall@k':>10}{'hit rate':>10}{'MRR':>8}"
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(f"{name:<36}{metrics['recall_at_k']:>10.3f}{metrics['hit_rate']:>10.3f}{metrics['mrr']:>8.3f}")

    print(f"\nMulti document cases only ({len(multi)})\n")
    print(header)
    print("-" * len(header))
    for name, metrics in multi_results.items():
        print(f"{name:<36}{metrics['recall_at_k']:>10.3f}{metrics['hit_rate']:>10.3f}{metrics['mrr']:>8.3f}")

    out = Path(__file__).parent / "results" / "retrieval_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"top_k": TOP_K, "cases": len(cases), "multi_document_cases": len(multi),
             "all": results, "multi_document": multi_results},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
