"""Canonical problem schema shared by every adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class MathProblem:
    source_id: str
    template_id: str
    prompt: str
    answer: str
    expression: str
    split: str = "unspecified"
    variables: tuple[str, ...] = ()
    numeric_values: tuple[float, ...] = ()
    transformations: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id or not self.template_id:
            raise ValueError("source_id and template_id are required")
        if not self.prompt.strip() or not self.answer.strip():
            raise ValueError("prompt and answer cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("variables", "numeric_values", "transformations"):
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MathProblem:
        payload = dict(value)
        for key in ("variables", "numeric_values", "transformations"):
            payload[key] = tuple(payload.get(key, ()))
        return cls(**payload)
