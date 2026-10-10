"""In-process metrics.

This is a stand-in for a Prometheus client. The counter names and labels are chosen to
match what would be exported in production, so moving to a real registry is a mechanical
change rather than a redesign. Latencies are kept in a bounded window so memory use is
constant.
"""

from __future__ import annotations

import threading
from collections import Counter, deque
from typing import Deque, Dict, List


class MetricsRegistry:
    def __init__(self, window: int = 1000) -> None:
        self._lock = threading.Lock()
        self._latencies: Deque[float] = deque(maxlen=window)
        self._counters: Counter = Counter()
        self._confidence: Counter = Counter()
        self._refusal_reasons: Counter = Counter()

    def record_request(
        self,
        latency_ms: float,
        *,
        confidence: str,
        refused: bool,
        refusal_reason: str | None = None,
        injection_blocked: bool = False,
        confidential_blocked: bool = False,
    ) -> None:
        with self._lock:
            self._latencies.append(latency_ms)
            self._counters["requests_total"] += 1
            self._confidence[confidence] += 1
            if refused:
                self._counters["refusals_total"] += 1
                if refusal_reason:
                    self._refusal_reasons[refusal_reason] += 1
            if injection_blocked:
                self._counters["injection_blocks_total"] += 1
            if confidential_blocked:
                self._counters["confidential_blocks_total"] += 1

    def record_error(self) -> None:
        with self._lock:
            self._counters["errors_total"] += 1

    def record_feedback(self, helpful: bool) -> None:
        with self._lock:
            self._counters["feedback_positive" if helpful else "feedback_negative"] += 1

    def snapshot(self) -> Dict[str, object]:
        with self._lock:
            latencies: List[float] = sorted(self._latencies)
            total = self._counters["requests_total"]
            refusals = self._counters["refusals_total"]
            return {
                "requests_total": total,
                "refusals_total": refusals,
                "errors_total": self._counters["errors_total"],
                "injection_blocks_total": self._counters["injection_blocks_total"],
                "confidential_blocks_total": self._counters["confidential_blocks_total"],
                "refusal_rate": round(refusals / total, 4) if total else 0.0,
                "latency_ms_avg": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "latency_ms_p50": round(_percentile(latencies, 50), 2),
                "latency_ms_p95": round(_percentile(latencies, 95), 2),
                "confidence_counts": dict(self._confidence),
                "refusal_reasons": dict(self._refusal_reasons),
                "feedback_positive": self._counters["feedback_positive"],
                "feedback_negative": self._counters["feedback_negative"],
            }


def _percentile(sorted_values: List[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (percentile / 100) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


metrics = MetricsRegistry()
