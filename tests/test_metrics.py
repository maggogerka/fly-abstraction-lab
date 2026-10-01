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


@pytest.mark.parametrize("value", ["symbolic", float("nan"), float("inf")])
def test_numeric_metrics_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        evaluate_records([{"prediction": value, "target": 1.0}])
