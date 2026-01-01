"""Evaluation utilities for ACE pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency
    import jsonschema
except ImportError:  # pragma: no cover
    jsonschema = None  # type: ignore[assignment]
from statistics import mean
from typing import Iterable

from .schemas import Trace


@dataclass
class EvaluationResult:
    accuracy: float
    average_tokens: float
    total_traces: int


class BaseEvaluator:
    name: str = "base"

    def evaluate(self, expected: str, actual: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        raise NotImplementedError


class ExactMatchEvaluator(BaseEvaluator):
    name = "exact"

    def evaluate(self, expected: str, actual: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return expected.strip() == actual.strip()


class NormalizedStringEvaluator(BaseEvaluator):
    name = "normalized"

    def evaluate(self, expected: str, actual: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        return self._normalize(expected) == self._normalize(actual)

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().lower().split())


class NumericToleranceEvaluator(BaseEvaluator):
    name = "numeric"

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def evaluate(self, expected: str, actual: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        tolerance = self.tolerance
        if metadata and "tolerance" in metadata:
            tolerance = float(metadata["tolerance"])
        try:
            expected_val = float(expected)
            actual_val = float(actual)
        except (TypeError, ValueError):
            return False
        return abs(expected_val - actual_val) <= tolerance


class JSONSchemaEvaluator(BaseEvaluator):
    name = "json_schema"

    def evaluate(self, expected: str, actual: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        if jsonschema is None:
            raise RuntimeError("jsonschema is required for JSONSchemaEvaluator")
        try:
            schema = json.loads(expected) if isinstance(expected, str) else expected
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON schema: {exc}") from exc
        try:
            payload = json.loads(actual) if isinstance(actual, str) else actual
        except json.JSONDecodeError:
            return False
        try:
            jsonschema.validate(instance=payload, schema=schema)
        except jsonschema.ValidationError:
            return False
        return True


def get_evaluator(name: str, metadata: Optional[Dict[str, Any]] = None) -> BaseEvaluator:
    name = name.lower().strip()
    if name == ExactMatchEvaluator.name:
        return ExactMatchEvaluator()
    if name == NormalizedStringEvaluator.name:
        return NormalizedStringEvaluator()
    if name == NumericToleranceEvaluator.name:
        tolerance = 1e-3
        if metadata and "tolerance" in metadata:
            tolerance = float(metadata["tolerance"])
        return NumericToleranceEvaluator(tolerance=tolerance)
    if name == JSONSchemaEvaluator.name:
        return JSONSchemaEvaluator()
    raise ValueError(f"Unknown evaluator: {name}")


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
