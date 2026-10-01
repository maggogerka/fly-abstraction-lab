"""Adapters from source-specific records into :class:`MathProblem`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from fly_abstraction.data.schema import MathProblem
from fly_abstraction.utils import read_jsonl, write_jsonl


class DatasetAdapter(ABC):
    @abstractmethod
    def adapt(self, record: Mapping[str, Any]) -> MathProblem:
        """Convert one already-local source record."""

    def load_jsonl(self, path: Path) -> list[MathProblem]:
        return [self.adapt(record) for record in read_jsonl(path)]


class DeepMindMathematicsAdapter(DatasetAdapter):
    def adapt(self, record: Mapping[str, Any]) -> MathProblem:
        module = str(record.get("module", "unknown"))
        source_id = str(record.get("id", record.get("source_id", "deepmind-unknown")))
        return MathProblem(
            source_id=source_id,
            template_id=str(record.get("template_id", module)),
            prompt=str(record.get("question", record.get("prompt", ""))),
            answer=str(record["answer"]),
            expression=str(record.get("expression", record["answer"])),
            split=str(record.get("split", "unspecified")),
            metadata={"dataset": "deepmind_mathematics", "module": module},
        )


class SRSDFeynmanAdapter(DatasetAdapter):
    def adapt(self, record: Mapping[str, Any]) -> MathProblem:
        equation_id = str(record.get("equation_id", record.get("source_id", "srsd-unknown")))
        expression = str(record.get("equation", record.get("expression", "")))
        variables = tuple(str(value) for value in record.get("variables", ()))
        values = tuple(float(value) for value in record.get("values", ()))
        return MathProblem(
            source_id=equation_id,
            template_id=str(record.get("template_id", equation_id)),
            prompt=str(record.get("prompt", f"Evaluate {expression}")),
            answer=str(record["answer"]),
            expression=expression,
            split=str(record.get("split", "unspecified")),
            variables=variables,
            numeric_values=values,
            metadata={"dataset": "srsd_feynman"},
        )


class SciBenchAdapter(DatasetAdapter):
    def adapt(self, record: Mapping[str, Any]) -> MathProblem:
        source_id = str(record.get("id", record.get("source_id", "scibench-unknown")))
        subject = str(record.get("subject", "unknown"))
        return MathProblem(
            source_id=source_id,
            template_id=str(record.get("template_id", f"scibench:{subject}")),
            prompt=str(record.get("problem_text", record.get("prompt", ""))),
            answer=str(record.get("answer_number", record.get("answer", ""))),
            expression=str(record.get("expression", record.get("answer_number", ""))),
            split=str(record.get("split", "external")),
            metadata={"dataset": "scibench", "subject": subject, "external": True},
        )


class TinyDatasetAdapter(DatasetAdapter):
    """Deterministic generated dataset for local wiring and smoke tests."""

    def adapt(self, record: Mapping[str, Any]) -> MathProblem:
        return MathProblem.from_dict(dict(record))

    @staticmethod
    def generate(count: int = 100, seed: int = 17) -> list[MathProblem]:
        if not 1 <= count <= 100:
            raise ValueError("TinyDataset count must be between 1 and 100")
        templates = ("add", "subtract", "multiply", "linear")
        problems: list[MathProblem] = []
        for index in range(count):
            template = templates[(index + seed) % len(templates)]
            left = 1 + ((index * 7 + seed) % 19)
            right = 1 + ((index * 11 + seed) % 13)
            if template == "add":
                prompt = f"Compute {left} + {right}"
                answer, expression = str(left + right), f"{left}+{right}"
            elif template == "subtract":
                high, low = max(left, right), min(left, right)
                prompt = f"Compute {high} - {low}"
                answer, expression = str(high - low), f"{high}-{low}"
            elif template == "multiply":
                prompt = f"Compute {left} * {right}"
                answer, expression = str(left * right), f"{left}*{right}"
            else:
                coefficient = 1 + left % 5
                solution = right
                total = coefficient * solution + left
                prompt = f"Solve {coefficient}*x + {left} = {total} for x"
                answer, expression = str(solution), f"({total}-{left})/{coefficient}"
            problems.append(
                MathProblem(
                    source_id=f"tiny-{index:03d}",
                    template_id=f"tiny:{template}",
                    prompt=prompt,
                    answer=answer,
                    expression=expression,
                    split="unspecified",
                    variables=("x",) if template == "linear" else (),
                    numeric_values=(float(left), float(right)),
                    metadata={"dataset": "tiny", "generator_version": 1},
                )
            )
        return problems

    @classmethod
    def prepare(cls, path: Path, count: int = 100, seed: int = 17) -> Path:
        write_jsonl(path, (problem.to_dict() for problem in cls.generate(count, seed)))
        return path


def save_problems(path: Path, problems: Iterable[MathProblem]) -> None:
    write_jsonl(path, (problem.to_dict() for problem in problems))
