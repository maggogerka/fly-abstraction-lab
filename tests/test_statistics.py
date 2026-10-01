import pytest

from fly_abstraction.statistics import (
    SampleEfficiencyPoint,
    aggregate_by_template,
    bootstrap_confidence_interval,
    holm_correction,
    paired_permutation_test,
    sample_efficiency_records,
)


def test_statistical_helpers_are_deterministic_and_well_formed() -> None:
    interval = bootstrap_confidence_interval([0.0, 0.5, 1.0, 1.0], iterations=200, seed=9)
    assert 0 <= interval[0] <= interval[1] <= 1
    result = paired_permutation_test([1, 1, 1], [0, 0, 0], iterations=200, seed=9)
    assert result["mean_difference"] == 1
    assert 0 < result["p_value"] <= 1
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_template_aggregation_and_curve_schema() -> None:
    aggregate = aggregate_by_template(
        [
            {"template_id": "a", "value": 1},
            {"template_id": "a", "value": 0},
            {"template_id": "b", "value": 1},
        ]
    )
    assert aggregate[0] == {"template_id": "a", "count": 2, "mean": 0.5}
    records = sample_efficiency_records(
        [SampleEfficiencyPoint("model", 17, "ood", 100, "exact", 0.5)]
    )
    assert records[0]["training_examples"] == 100
