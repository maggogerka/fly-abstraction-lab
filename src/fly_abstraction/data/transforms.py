"""Deterministic, provenance-preserving OOD transformations."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from dataclasses import replace

import sympy

from fly_abstraction.data.schema import MathProblem


class TransformationError(ValueError):
    """Raised when a transformation cannot preserve its declared semantics."""


def _seed(problem: MathProblem, name: str, seed: int) -> int:
    value = f"{problem.source_id}|{problem.template_id}|{name}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _append(problem: MathProblem, name: str, **changes: object) -> MathProblem:
    metadata = dict(problem.metadata)
    metadata.setdefault("origin_source_id", problem.source_id)
    metadata.setdefault("origin_template_id", problem.template_id)
    return replace(
        problem,
        transformations=(*problem.transformations, name),
        metadata=metadata,
        **changes,
    )


def sympy_equivalent(left: str, right: str) -> bool:
    """Return exact symbolic equivalence; malformed expressions are not equivalent."""
    try:
        left_expr = sympy.sympify(left, evaluate=True)
        right_expr = sympy.sympify(right, evaluate=True)
        return bool(sympy.simplify(left_expr - right_expr) == 0)
    except (TypeError, ValueError, sympy.SympifyError):
        return False


def symbol_rename(problem: MathProblem, seed: int = 0) -> MathProblem:
    expression = sympy.sympify(problem.expression, evaluate=False)
    symbols = sorted(expression.free_symbols, key=str)
    if not symbols:
        return _append(problem, "symbol_rename")
    offset = _seed(problem, "symbol_rename", seed) % 13
    replacements = {
        symbol: sympy.Symbol(f"v{offset + index}") for index, symbol in enumerate(symbols)
    }
    renamed = expression.xreplace(replacements)
    prompt = problem.prompt
    for original, replacement_symbol in replacements.items():
        prompt = re.sub(rf"\b{re.escape(str(original))}\b", str(replacement_symbol), prompt)
    if not sympy_equivalent(
        str(renamed.xreplace({v: k for k, v in replacements.items()})), str(expression)
    ):
        raise TransformationError("symbol_rename failed equivalence check")
    return _append(
        problem,
        "symbol_rename",
        prompt=prompt,
        expression=str(renamed),
        variables=tuple(
            str(replacements.get(sympy.Symbol(v), sympy.Symbol(v))) for v in problem.variables
        ),
    )


def numeric_extrapolation(problem: MathProblem, seed: int = 0) -> MathProblem:
    """Scale numeric arithmetic into a disjoint range and recompute its exact answer."""
    expression = sympy.sympify(problem.expression, evaluate=False)
    if expression.free_symbols:
        raise TransformationError("numeric_extrapolation currently requires a closed expression")
    factor = 10 + (_seed(problem, "numeric_extrapolation", seed) % 7)
    numbers = sorted(expression.atoms(sympy.Integer), key=lambda item: (abs(int(item)), int(item)))
    replacements = {number: number * factor for number in numbers if number != 0}
    extrapolated = expression.xreplace(replacements)
    answer = str(sympy.simplify(extrapolated))
    prompt = f"Evaluate the extrapolated expression: {extrapolated}"
    return _append(
        problem,
        "numeric_extrapolation",
        prompt=prompt,
        expression=str(extrapolated),
        answer=answer,
        numeric_values=tuple(float(value * factor) for value in problem.numeric_values),
    )


def equivalent_expression(problem: MathProblem, seed: int = 0) -> MathProblem:
    expression = sympy.sympify(problem.expression, evaluate=False)
    candidate = (
        sympy.expand(expression)
        if _seed(problem, "equivalent_expression", seed) % 2
        else sympy.factor(expression)
    )
    if str(candidate) == str(expression):
        candidate = sympy.Add(expression, 0, evaluate=False)
    if not sympy_equivalent(str(expression), str(candidate)):
        raise TransformationError("equivalent_expression failed equivalence check")
    return _append(
        problem,
        "equivalent_expression",
        prompt=f"Compute the equivalent form {candidate}",
        expression=str(candidate),
    )


def equation_rearrangement(problem: MathProblem, seed: int = 0) -> MathProblem:
    del seed
    match = re.search(r"([^=]+)=([^=]+)", problem.prompt)
    if not match:
        return _append(
            problem,
            "equation_rearrangement",
            prompt=f"Find the value represented by the equivalent expression {problem.expression}",
        )
    left, right = match.group(1).strip(), match.group(2).strip()
    rearranged = f"{right} = {left}"
    prompt = problem.prompt[: match.start()] + rearranged + problem.prompt[match.end() :]
    return _append(problem, "equation_rearrangement", prompt=prompt)


def distractor_variables(problem: MathProblem, seed: int = 0) -> MathProblem:
    value = 2 + _seed(problem, "distractor_variables", seed) % 97
    name = f"unused_{value % 11}"
    return _append(
        problem,
        "distractor_variables",
        prompt=f"Given irrelevant {name} = {value}, {problem.prompt}",
        variables=(*problem.variables, name),
        numeric_values=(*problem.numeric_values, float(value)),
    )


def held_out_template(problem: MathProblem, seed: int = 0) -> MathProblem:
    del seed
    metadata = dict(problem.metadata)
    metadata["held_out_template"] = True
    return replace(
        _append(problem, "held_out_template", split="test_held_out_template"),
        metadata=metadata,
    )


TRANSFORMS: dict[str, Callable[[MathProblem, int], MathProblem]] = {
    "symbol_rename": symbol_rename,
    "numeric_extrapolation": numeric_extrapolation,
    "equivalent_expression": equivalent_expression,
    "equation_rearrangement": equation_rearrangement,
    "distractor_variables": distractor_variables,
    "held_out_template": held_out_template,
}


def compositional_split(problem: MathProblem, seed: int = 0) -> MathProblem:
    transformed = symbol_rename(problem, seed)
    transformed = equivalent_expression(transformed, seed)
    transformed = distractor_variables(transformed, seed)
    return _append(transformed, "compositional_split", split="test_compositional_ood")


def split_without_template_leakage(
    problems: Sequence[MathProblem], held_out_fraction: float = 0.25, seed: int = 17
) -> tuple[list[MathProblem], list[MathProblem]]:
    """Assign whole templates to train or held-out test, never both."""
    if not 0 < held_out_fraction < 1:
        raise ValueError("held_out_fraction must be between zero and one")
    templates = sorted({problem.template_id for problem in problems})
    if len(templates) < 2:
        raise ValueError("At least two templates are required for a leakage-safe split")
    ranked = sorted(
        templates,
        key=lambda item: hashlib.sha256(f"{seed}|{item}".encode()).digest(),
    )
    held_count = max(1, min(len(templates) - 1, round(len(templates) * held_out_fraction)))
    held = set(ranked[:held_count])
    train = [replace(item, split="train") for item in problems if item.template_id not in held]
    test = [held_out_template(item, seed) for item in problems if item.template_id in held]
    assert_no_template_leakage(train, test)
    return train, test


def assert_no_template_leakage(
    train: Sequence[MathProblem], held_out: Sequence[MathProblem]
) -> None:
    overlap = {item.template_id for item in train} & {item.template_id for item in held_out}
    if overlap:
        raise TransformationError(f"Template leakage detected: {sorted(overlap)}")
