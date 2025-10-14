"""Evaluation utilities for ACE pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import mean
from typing import Iterable

from .schemas import Trace


@dataclass
class EvaluationResult:
    accuracy: float
    average_tokens: float
    total_traces: int


def compute_accuracy(traces: Iterable[Trace]) -> EvaluationResult:
    traces = list(traces)
    if not traces:
        return EvaluationResult(accuracy=0.0, average_tokens=0.0, total_traces=0)
    successes = sum(1 for trace in traces if trace.success)
    usage = []
    for trace in traces:
        usage_data = trace.metadata.get("usage", "{}")
        try:
            payload = json.loads(usage_data)
        except Exception:  # noqa: BLE001
            payload = {}
        total_tokens = payload.get("total_tokens", 0)
        usage.append(total_tokens)
    avg_tokens = mean(usage) if usage else 0.0
    return EvaluationResult(
        accuracy=successes / len(traces),
        average_tokens=avg_tokens,
        total_traces=len(traces),
    )
