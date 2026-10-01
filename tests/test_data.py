import json
from dataclasses import replace
from pathlib import Path

import pytest

from fly_abstraction.data.adapters import TinyDatasetAdapter, UCIEnergyEfficiencyAdapter
from fly_abstraction.data.schema import MathProblem
from fly_abstraction.data.transforms import (
    assert_no_template_leakage,
    compositional_split,
    equation_rearrangement,
    equivalent_expression,
    split_without_template_leakage,
    symbol_rename,
    sympy_equivalent,
)


def test_tiny_dataset_is_bounded_and_deterministic() -> None:
    first = TinyDatasetAdapter.generate(100, 17)
    second = TinyDatasetAdapter.generate(100, 17)
    assert first == second
    assert len(first) == 100
    assert len({item.source_id for item in first}) == 100


def test_symbolic_transforms_preserve_identity_and_equivalence() -> None:
    problem = MathProblem(
        source_id="source-1",
        template_id="template-a",
        prompt="Simplify (x + 1)**2",
        answer="x**2 + 2*x + 1",
        expression="(x + 1)**2",
        target_mode="symbolic_future",
        variables=("x",),
    )
    renamed = symbol_rename(problem, 3)
    equivalent = equivalent_expression(problem, 3)
    composed = compositional_split(problem, 3)
    assert renamed.source_id == problem.source_id
    assert renamed.template_id == problem.template_id
    assert sympy_equivalent(problem.expression, equivalent.expression)
    assert composed.split == "test_compositional_ood"
    assert composed.transformations[-1] == "compositional_split"


def test_equation_rearrangement_preserves_solve_prompt_semantics() -> None:
    problem = TinyDatasetAdapter.generate(4, 17)[2]
    assert "=" in problem.prompt
    transformed = equation_rearrangement(problem)
    assert transformed.prompt.startswith("Solve ")
    assert transformed.prompt.endswith(" for x")
    assert transformed.answer == problem.answer


def test_split_has_no_template_leakage() -> None:
    problems = TinyDatasetAdapter.generate(40, 17)
    train, held = split_without_template_leakage(problems, 0.25, 17)
    assert_no_template_leakage(train, held)
    assert {item.template_id for item in train}.isdisjoint({item.template_id for item in held})
    leaked = [replace(held[0], template_id=train[0].template_id)]
    try:
        assert_no_template_leakage(train, leaked)
    except ValueError:
        pass
    else:
        raise AssertionError("leakage check accepted an overlapping template")


@pytest.mark.parametrize("answer", ["not-a-number", "NaN", "inf", "-inf"])
def test_numeric_targets_reject_non_finite_values(answer: str) -> None:
    with pytest.raises(ValueError, match="Numeric target"):
        MathProblem(
            source_id="bad",
            template_id="bad",
            prompt="Predict a number",
            answer=answer,
            expression="numeric target",
        )


def test_uci_adapter_rejects_non_finite_features() -> None:
    record = {f"X{index}": "1" for index in range(1, 9)}
    record.update({"Y1": "2", "Y2": "3", "row_id": "0", "X3": "NaN"})
    with pytest.raises(ValueError, match="finite"):
        UCIEnergyEfficiencyAdapter().adapt(record)


def test_uci_preparation_records_filter_provenance(tmp_path: Path) -> None:
    source = tmp_path / "data.csv"
    source.write_text(
        "X1,X2,X3,X4,X5,X6,X7,X8,Y1,Y2\n1,2,3,4,5,6,7,8,9,10\n1,2,NaN,4,5,6,7,8,9,10\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared.jsonl"
    UCIEnergyEfficiencyAdapter.prepare(source, output)
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted_records"] == 1
    assert manifest["rejected_records"] == 1
    assert manifest["rejection_reasons"] == {"ValueError": 1}
    assert manifest["source_sha256"]
    assert manifest["prepared_sha256"]
    assert output.with_suffix(".jsonl.rejected.jsonl").is_file()
