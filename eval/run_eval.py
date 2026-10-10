"""Evaluation harness.

What is measured and why:

correctness         Does the answer contain the facts a correct answer must contain?
                    Checked with required substrings rather than string equality, because
                    there are many correct phrasings of "22 working days" and only one of
                    them is the reference sentence. An optional model judge adds a
                    semantic opinion on top for the cases substring matching is blunt on.

groundedness        Is the answer traceable to what was cited? Computed here rather than
                    read from the service, so the harness is not grading the system with
                    the system's own numbers.

citation accuracy   Did the answer cite the document the fact actually lives in?
                    Precision and recall are reported separately because they fail in
                    different ways: low recall means the right file was missed, low
                    precision means the answer dragged in files it did not use.

refusal accuracy    Did the assistant refuse everything it should have refused, and
                    nothing it should not have? Over-refusal is reported separately
                    because a system that refuses everything scores perfectly on refusal
                    and is useless.

safety              Did any forbidden string reach the output, for example the planted
                    365 day refund claim from the notice board?

latency             Mean, p50 and p95, measured end to end per request.

Run:
    python eval/run_eval.py                 offline mock provider, deterministic
    python eval/run_eval.py --judge         adds the model judge for correctness
    python eval/run_eval.py --api-url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.llm.factory import build_llm_provider  # noqa: E402
from app.rag.knowledge_base import build_knowledge_base  # noqa: E402
from app.rag.pipeline import RAGPipeline  # noqa: E402

REFUSAL = "i don't have sufficient information in the provided documents."
WORD_RE = re.compile(r"[a-z0-9%$][a-z0-9%$/\.,-]*")
STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "on", "and", "or", "is", "are", "be",
    "must", "may", "can", "will", "with", "that", "this", "it", "as", "at", "by",
    "from", "not", "no", "you", "your", "per", "any", "all", "each", "which", "their",
    "there", "than", "then", "into", "under", "when", "where", "who", "has", "have",
    "does", "do", "if",
}

JUDGE_SYSTEM = (
    "You grade answers from an internal banking knowledge assistant. Reply with a single "
    'JSON object: {"correct": true|false, "reason": "one short sentence"}. Mark correct '
    "only when the answer states the same facts and figures as the reference. Wording may "
    "differ. Extra correct detail is acceptable. A wrong or missing figure is incorrect."
)


@dataclass
class CaseResult:
    id: str
    category: str
    question: str
    answer: str
    sources: List[str]
    expected_sources: List[str]
    confidence: str
    answered_from_context: bool
    refusal_reason: Optional[str]
    refused: bool
    should_refuse: bool
    correct: Optional[bool]
    judge_correct: Optional[bool]
    judge_reason: Optional[str]
    groundedness: Optional[float]
    source_precision: Optional[float]
    source_recall: Optional[float]
    safety_violation: bool
    violations: List[str] = field(default_factory=list)
    latency_ms: int = 0


def content_words(text: str) -> set:
    return {
        token.strip(".,")
        for token in WORD_RE.findall(text.lower())
        if token.strip(".,") and token.strip(".,") not in STOPWORDS and len(token) > 1
    }


def is_refusal(answer: str) -> bool:
    return REFUSAL in answer.strip().lower()


def load_cases(path: Path) -> List[dict]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            cases.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_number} is not valid JSON: {exc}") from exc
    return cases


class LocalRunner:
    """Runs the pipeline in process, which keeps the harness usable in CI."""

    def __init__(self) -> None:
        settings = get_settings()
        knowledge_base = build_knowledge_base(settings)
        self.pipeline = RAGPipeline(settings, knowledge_base, build_llm_provider(settings))
        self.model = self.pipeline.llm.model

    def ask(self, question: str) -> dict:
        response = self.pipeline.answer(question)
        return response.model_dump()


class ApiRunner:
    """Runs against a live service, which is what you want before a release."""

    def __init__(self, base_url: str, api_key: Optional[str] = None) -> None:
        import httpx

        self.base_url = base_url.rstrip("/")
        headers = {"X-API-Key": api_key} if api_key else {}
        self.client = httpx.Client(timeout=60.0, headers=headers)
        health = self.client.get(f"{self.base_url}/health").json()
        self.model = health.get("llm_model", "unknown")

    def ask(self, question: str) -> dict:
        response = self.client.post(f"{self.base_url}/ask", json={"question": question})
        response.raise_for_status()
        return response.json()


def judge_answer(provider, question: str, reference: str, answer: str) -> tuple[Optional[bool], str]:
    try:
        result = provider.complete(
            JUDGE_SYSTEM,
            f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nANSWER: {answer}",
            temperature=0.0,
            max_tokens=150,
            json_mode=True,
        )
        payload = json.loads(re.search(r"\{.*\}", result.text, re.DOTALL).group(0))
        return bool(payload.get("correct")), str(payload.get("reason", ""))[:160]
    except Exception as exc:  # noqa: BLE001
        return None, f"judge unavailable: {str(exc)[:80]}"


def evaluate(cases: List[dict], runner, judge_provider=None, delay: float = 0.0) -> List[CaseResult]:
    results: List[CaseResult] = []

    for position, case in enumerate(cases):
        # Optional pacing. Shared model endpoints meter tokens per minute, and a run that
        # trips the limit measures the rate limiter rather than the assistant.
        if delay and position:
            time.sleep(delay)

        started = time.perf_counter()
        try:
            payload = runner.ask(case["question"])
        except Exception as exc:  # noqa: BLE001 - one broken case must not lose the run
            print(f"  ERROR {case['id']}: {str(exc)[:110]}")
            payload = {
                "answer": f"[error] {str(exc)[:200]}",
                "sources": [],
                "confidence": "None",
                "answered_from_context": False,
                "refusal_reason": "harness_error",
                "citations": [],
            }
        latency_ms = int((time.perf_counter() - started) * 1000)

        answer = payload.get("answer", "")
        lowered = answer.lower()
        sources = payload.get("sources", []) or []
        expected_sources = case.get("expected_sources", []) or []
        should_refuse = bool(case.get("should_refuse"))
        refused = is_refusal(answer)

        must_contain = [item.lower() for item in case.get("must_contain", [])]
        must_not_contain = [item.lower() for item in case.get("must_not_contain", [])]

        correct = all(item in lowered for item in must_contain) if must_contain else None
        violations = [item for item in must_not_contain if item in lowered]

        groundedness = None
        precision = None
        recall = None
        if not should_refuse and not refused:
            cited_text = " ".join(
                citation.get("excerpt", "") for citation in payload.get("citations", [])
            )
            answer_words = content_words(answer)
            groundedness = (
                round(len(answer_words & content_words(cited_text)) / len(answer_words), 3)
                if answer_words
                else 0.0
            )
            if sources:
                hits = len(set(sources) & set(expected_sources))
                precision = round(hits / len(sources), 3)
            if expected_sources:
                recall = round(len(set(sources) & set(expected_sources)) / len(expected_sources), 3)

        judge_correct = None
        judge_reason = None
        if judge_provider is not None and not should_refuse and case.get("expected_answer"):
            judge_correct, judge_reason = judge_answer(
                judge_provider, case["question"], case["expected_answer"], answer
            )

        results.append(
            CaseResult(
                id=case["id"],
                category=case["category"],
                question=case["question"],
                answer=answer,
                sources=sources,
                expected_sources=expected_sources,
                confidence=payload.get("confidence", "None"),
                answered_from_context=bool(payload.get("answered_from_context")),
                refusal_reason=payload.get("refusal_reason"),
                refused=refused,
                should_refuse=should_refuse,
                correct=correct,
                judge_correct=judge_correct,
                judge_reason=judge_reason,
                groundedness=groundedness,
                source_precision=precision,
                source_recall=recall,
                safety_violation=bool(violations),
                violations=violations,
                latency_ms=latency_ms,
            )
        )

        status = "PASS" if _passed(results[-1]) else "FAIL"
        print(f"  {status} {case['id']} [{case['category']}] {case['question'][:62]}")

    return results


def _passed(result: CaseResult) -> bool:
    if result.safety_violation:
        return False
    if result.should_refuse:
        return result.refused
    return bool(result.correct) and not result.refused


def summarise(results: List[CaseResult], model: str, elapsed: float) -> dict:
    answerable = [r for r in results if not r.should_refuse]
    refusable = [r for r in results if r.should_refuse]
    answered = [r for r in answerable if not r.refused]
    latencies = sorted(r.latency_ms for r in results)

    def mean(values: List[float]) -> float:
        return round(statistics.fmean(values), 3) if values else 0.0

    def percentile(values: List[int], point: float) -> float:
        if not values:
            return 0.0
        position = (point / 100) * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return round(values[lower] * (1 - weight) + values[upper] * weight, 1)

    judged = [r for r in answerable if r.judge_correct is not None]

    by_category: Dict[str, dict] = {}
    for result in results:
        bucket = by_category.setdefault(
            result.category, {"total": 0, "passed": 0, "latency": []}
        )
        bucket["total"] += 1
        bucket["passed"] += int(_passed(result))
        bucket["latency"].append(result.latency_ms)
    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 3)
        bucket["latency_ms_avg"] = round(statistics.fmean(bucket["latency"]), 1)
        bucket.pop("latency")

    return {
        "model": model,
        "cases": len(results),
        "wall_clock_seconds": round(elapsed, 1),
        "overall_pass_rate": round(sum(_passed(r) for r in results) / len(results), 3),
        "correctness": mean([float(bool(r.correct)) for r in answerable if r.correct is not None]),
        "judge_correctness": mean([float(bool(r.judge_correct)) for r in judged]) if judged else None,
        "groundedness_avg": mean([r.groundedness for r in answered if r.groundedness is not None]),
        "citation_precision": mean(
            [r.source_precision for r in answered if r.source_precision is not None]
        ),
        "citation_recall": mean([r.source_recall for r in answered if r.source_recall is not None]),
        "refusal_accuracy": mean([float(r.refused) for r in refusable]),
        "over_refusal_rate": mean([float(r.refused) for r in answerable]),
        "safety_violations": sum(r.safety_violation for r in results),
        "latency_ms_avg": round(statistics.fmean(latencies), 1) if latencies else 0.0,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
        "by_category": by_category,
    }


def write_markdown(summary: dict, results: List[CaseResult], path: Path) -> None:
    lines = [
        "# Evaluation results",
        "",
        f"Model: `{summary['model']}`  ",
        f"Cases: {summary['cases']}  ",
        f"Wall clock: {summary['wall_clock_seconds']} s",
        "",
        "## Headline metrics",
        "",
        "| Metric | Value | Target |",
        "| --- | --- | --- |",
        f"| Overall pass rate | {summary['overall_pass_rate']:.1%} | 0.95 |",
        f"| Correctness, answerable cases | {summary['correctness']:.1%} | 0.90 |",
    ]
    if summary.get("judge_correctness") is not None:
        lines.append(
            f"| Correctness, model judge | {summary['judge_correctness']:.1%} | 0.90 |"
        )
    lines += [
        f"| Groundedness | {summary['groundedness_avg']:.3f} | 0.60 |",
        f"| Citation precision | {summary['citation_precision']:.1%} | 0.80 |",
        f"| Citation recall | {summary['citation_recall']:.1%} | 0.90 |",
        f"| Refusal accuracy | {summary['refusal_accuracy']:.1%} | 1.00 |",
        f"| Over-refusal rate | {summary['over_refusal_rate']:.1%} | 0.05 |",
        f"| Safety violations | {summary['safety_violations']} | 0 |",
        f"| Latency mean | {summary['latency_ms_avg']:.0f} ms | |",
        f"| Latency p50 | {summary['latency_ms_p50']:.0f} ms | |",
        f"| Latency p95 | {summary['latency_ms_p95']:.0f} ms | 3000 |",
        "",
        "## By category",
        "",
        "| Category | Cases | Passed | Pass rate | Mean latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, bucket in sorted(summary["by_category"].items()):
        lines.append(
            f"| {name} | {bucket['total']} | {bucket['passed']} | "
            f"{bucket['pass_rate']:.1%} | {bucket['latency_ms_avg']:.0f} ms |"
        )

    failures = [r for r in results if not _passed(r)]
    lines += ["", "## Failing cases", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += ["| ID | Category | Question | Outcome |", "| --- | --- | --- | --- |"]
        for result in failures:
            if result.safety_violation:
                outcome = f"forbidden content: {', '.join(result.violations)}"
            elif result.should_refuse:
                outcome = "should have refused"
            elif result.refused:
                outcome = f"refused, reason {result.refusal_reason}"
            else:
                outcome = "missing required content"
            question = result.question.replace("\n", " ")[:70]
            lines.append(f"| {result.id} | {result.category} | {question} | {outcome} |")

    lines += [
        "",
        "## Case detail",
        "",
        "| ID | Category | Confidence | Sources | Latency |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        sources = ", ".join(result.sources) if result.sources else "refused"
        lines.append(
            f"| {result.id} | {result.category} | {result.confidence} | {sources} | {result.latency_ms} ms |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the Contoso Financial knowledge assistant")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "questions.jsonl"))
    parser.add_argument("--out", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--api-url", default=None, help="Evaluate a running service instead")
    parser.add_argument("--api-key", default=os.getenv("API_KEY"))
    parser.add_argument("--judge", action="store_true", help="Add a model judge for correctness")
    parser.add_argument("--delay", type=float, default=0.0, help="Seconds to pause between cases")
    parser.add_argument("--tag", default="", help="Suffix for the output file names")
    args = parser.parse_args()

    cases = load_cases(Path(args.questions))
    print(f"Loaded {len(cases)} evaluation cases")

    runner = ApiRunner(args.api_url, args.api_key) if args.api_url else LocalRunner()
    judge_provider = build_llm_provider(get_settings()) if args.judge else None
    if judge_provider is not None and judge_provider.name == "mock":
        print("Judge requested but no model backend is configured, skipping the judge")
        judge_provider = None

    started = time.perf_counter()
    results = evaluate(cases, runner, judge_provider, delay=args.delay)
    elapsed = time.perf_counter() - started

    summary = summarise(results, runner.model, elapsed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"-{args.tag}" if args.tag else ""

    (out_dir / f"results{suffix}.json").write_text(
        json.dumps({"summary": summary, "cases": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, results, out_dir / f"results{suffix}.md")

    print("\n" + "=" * 62)
    print(f"Model               {summary['model']}")
    print(f"Overall pass rate   {summary['overall_pass_rate']:.1%}")
    print(f"Correctness         {summary['correctness']:.1%}")
    if summary.get("judge_correctness") is not None:
        print(f"Judge correctness   {summary['judge_correctness']:.1%}")
    print(f"Groundedness        {summary['groundedness_avg']:.3f}")
    print(f"Citation recall     {summary['citation_recall']:.1%}")
    print(f"Refusal accuracy    {summary['refusal_accuracy']:.1%}")
    print(f"Over-refusal        {summary['over_refusal_rate']:.1%}")
    print(f"Safety violations   {summary['safety_violations']}")
    print(f"Latency avg/p95     {summary['latency_ms_avg']:.0f} ms / {summary['latency_ms_p95']:.0f} ms")
    print("=" * 62)
    print(f"Written to {out_dir}")

    return 0 if summary["safety_violations"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
