import math

import pytest

from fly_abstraction.training.metrics import evaluate_records


def test_numeric_metrics_are_named_and_finite() -> None:
    records = [
        {"prediction": 2.0, "target": 1.0, "split": "validation"},
        {"prediction": 3.0, "target": 3.0, "split": "test_compositional_ood"},
    ]
    metrics = evaluate_records(records, absolute_tolerance=0.1, relative_tolerance=0.0)
    assert metrics["mae"] == pytest.approx(0.5)
    assert metrics["rmse"] == pytest.approx(math.sqrt(0.5))
    assert metrics["tolerance_accuracy"] == pytest.approx(0.5)
    assert metrics["primary_metric"] == 1.0


def test_geometry_ood_metric_is_not_labeled_compositional() -> None:
    metrics = evaluate_records(
        [
            {
                "prediction": 3.0,
                "target": 3.0,
                "split": "test_held_out_geometry_ood",
            }
        ],
        primary_split="test_held_out_geometry_ood",
        primary_metric_name="Tolerance accuracy on held-out building-geometry OOD",
    )
    assert metrics["primary_metric"] == 1.0
    assert metrics["primary_split"] == "test_held_out_geometry_ood"
    assert "compositional" not in metrics["primary_metric_name"].lower()


@pytest.mark.parametrize("value", ["symbolic", float("nan"), float("inf")])
def test_numeric_metrics_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        evaluate_records([{"prediction": value, "target": 1.0}])
