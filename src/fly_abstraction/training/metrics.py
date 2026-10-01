"""Named metrics; intentionally no composite abstraction score."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from fly_abstraction.data.transforms import sympy_equivalent


def normalize_answer(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()


def exact_match(prediction: str, target: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(target)


def sympy_equivalent_match(prediction: str, target: str) -> bool:
    return sympy_equivalent(prediction, target)


def evaluate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0, "exact_solution_rate": 0.0, "sympy_equivalent_accuracy": 0.0}
    exact = [exact_match(str(item["prediction"]), str(item["target"])) for item in records]
    symbolic = [
        sympy_equivalent_match(str(item["prediction"]), str(item["target"])) for item in records
    ]
    by_split: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        by_split[str(item.get("split", "unspecified"))].append(index)
    split_metrics = {
        split: {
            "count": len(indices),
            "exact_solution_rate": sum(exact[index] for index in indices) / len(indices),
            "sympy_equivalent_accuracy": sum(symbolic[index] for index in indices) / len(indices),
        }
        for split, indices in sorted(by_split.items())
    }
    return {
        "count": len(records),
        "exact_solution_rate": sum(exact) / len(exact),
        "sympy_equivalent_accuracy": sum(symbolic) / len(symbolic),
        "by_split": split_metrics,
        "primary_metric_name": "Exact solution rate on compositional held-out-template OOD split",
        "primary_metric": split_metrics.get("test_compositional_ood", {}).get(
            "exact_solution_rate"
        ),
    }
