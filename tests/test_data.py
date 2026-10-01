from dataclasses import replace

from fly_abstraction.data.adapters import TinyDatasetAdapter
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
