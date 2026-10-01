"""Finite numeric regression metrics for the MVP."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return number


def _metrics(
    pairs: Sequence[tuple[float, float]],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    relative_epsilon: float,
) -> dict[str, float | int]:
    if not pairs:
        return {
            "count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "mean_relative_error": 0.0,
            "tolerance_accuracy": 0.0,
        }
    errors = [abs(prediction - target) for prediction, target in pairs]
    return {
        "count": len(pairs),
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mean_relative_error": sum(
            error / max(abs(target), relative_epsilon)
            for error, (_prediction, target) in zip(errors, pairs, strict=True)
        )
        / len(errors),
        "tolerance_accuracy": sum(
            error <= absolute_tolerance + relative_tolerance * abs(target)
            for error, (_prediction, target) in zip(errors, pairs, strict=True)
        )
        / len(errors),
    }


def evaluate_records(
    records: Sequence[dict[str, Any]],
    *,
    absolute_tolerance: float = 0.1,
    relative_tolerance: float = 0.01,
    relative_epsilon: float = 1.0e-8,
    primary_split: str = "test_compositional_ood",
    primary_metric_name: str = "Tolerance accuracy on compositional held-out-template OOD",
) -> dict[str, Any]:
    """Evaluate numeric targets only; symbolic scoring is intentionally not implemented."""
    if absolute_tolerance < 0 or relative_tolerance < 0 or relative_epsilon <= 0:
        raise ValueError("Metric tolerances must be non-negative and epsilon must be positive")
    pairs: list[tuple[float, float]] = []
    by_split: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for item in records:
        if item.get("target_mode", "numeric") != "numeric":
            raise NotImplementedError("Symbolic evaluation is a future stage")
        pair = (
            _finite_number(item["prediction"], "prediction"),
            _finite_number(item["target"], "target"),
        )
        pairs.append(pair)
        by_split[str(item.get("split", "unspecified"))].append(pair)
    result: dict[str, Any] = _metrics(
        pairs,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
        relative_epsilon=relative_epsilon,
    )
    result["by_split"] = {
        split: _metrics(
            values,
            absolute_tolerance=absolute_tolerance,
            relative_tolerance=relative_tolerance,
            relative_epsilon=relative_epsilon,
        )
        for split, values in sorted(by_split.items())
    }
    result["primary_metric_name"] = primary_metric_name
    result["primary_split"] = primary_split
    result["primary_metric"] = result["by_split"].get(primary_split, {}).get("tolerance_accuracy")
    result["tolerances"] = {
        "absolute": absolute_tolerance,
        "relative": relative_tolerance,
        "relative_epsilon": relative_epsilon,
    }
    return result
