"""Predeclared statistical procedures for later real experiment results."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


def bootstrap_confidence_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    iterations: int = 10_000,
    seed: int = 17,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean; caller declares the resampling unit."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("bootstrap requires at least two one-dimensional observations")
    if not 0 < confidence < 1 or iterations <= 0:
        raise ValueError("invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(iterations, array.size))
    means = array[indices].mean(axis=1)
    alpha = (1 - confidence) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def paired_permutation_test(
    first: Sequence[float],
    second: Sequence[float],
    *,
    iterations: int = 100_000,
    seed: int = 17,
) -> dict[str, float]:
    """Two-sided paired sign-flip permutation test with add-one correction."""
    left, right = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("paired samples must be equal-length non-empty vectors")
    differences = left - right
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(iterations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=differences.size)
        extreme += abs(float((differences * signs).mean())) >= abs(observed)
    return {
        "mean_difference": observed,
        "p_value": (extreme + 1) / (iterations + 1),
        "permutations": float(iterations),
    }


def holm_correction(p_values: Sequence[float]) -> list[float]:
    """Family-wise Holm adjusted p-values in original input order."""
    if any(value < 0 or value > 1 for value in p_values):
        raise ValueError("p-values must be in [0, 1]")
    size = len(p_values)
    order = sorted(range(size), key=lambda index: p_values[index])
    adjusted = [0.0] * size
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (size - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def aggregate_by_template(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["template_id"])].append(float(record["value"]))
    return [
        {"template_id": template, "count": len(values), "mean": sum(values) / len(values)}
        for template, values in sorted(grouped.items())
    ]


@dataclass(frozen=True, slots=True)
class SampleEfficiencyPoint:
    model: str
    seed: int
    split: str
    training_examples: int
    metric: str
    value: float

    def to_dict(self) -> dict[str, str | int | float]:
        return asdict(self)


def sample_efficiency_records(
    points: Sequence[SampleEfficiencyPoint],
) -> list[dict[str, str | int | float]]:
    if any(point.training_examples <= 0 for point in points):
        raise ValueError("training_examples must be positive")
    return [
        point.to_dict()
        for point in sorted(
            points,
            key=lambda item: (item.model, item.seed, item.split, item.training_examples),
        )
    ]
