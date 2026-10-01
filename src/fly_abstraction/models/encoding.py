"""Deterministic character tokenizer with a separate aligned numeric channel."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

from fly_abstraction.data.schema import MathProblem

NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


@dataclass(frozen=True, slots=True)
class EncodedProblem:
    token_ids: torch.Tensor
    numeric: torch.Tensor
    answer: torch.Tensor
    expression_targets: torch.Tensor


class CharacterEncoder:
    PAD = 0
    BOS = 1
    EOS = 2

    def __init__(self, vocab_size: int = 128, max_length: int = 96) -> None:
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def token_id(self, character: str) -> int:
        return 3 + (ord(character) % (self.vocab_size - 3))

    def encode_text(self, text: str) -> torch.Tensor:
        content = [
            self.BOS,
            *(self.token_id(char) for char in text[: self.max_length - 2]),
            self.EOS,
        ]
        content.extend([self.PAD] * (self.max_length - len(content)))
        return torch.tensor(content, dtype=torch.long)

    def numeric_channel(self, text: str) -> torch.Tensor:
        values = torch.zeros(self.max_length, 1, dtype=torch.float32)
        offset = 1
        for match in NUMBER.finditer(text[: self.max_length - 2]):
            value = float(match.group())
            start = min(self.max_length - 1, offset + match.start())
            end = min(self.max_length - 1, offset + match.end())
            values[start:end, 0] = max(-1e6, min(1e6, value))
        return torch.sign(values) * torch.log1p(values.abs())

    def encode(self, problem: MathProblem) -> EncodedProblem:
        try:
            answer = float(problem.answer)
        except ValueError:
            answer = float("nan")
        expression = self.encode_text(problem.expression)
        expression[:-1] = expression[1:].clone()
        expression[-1] = self.PAD
        return EncodedProblem(
            token_ids=self.encode_text(problem.prompt),
            numeric=self.numeric_channel(problem.prompt),
            answer=torch.tensor(answer, dtype=torch.float32),
            expression_targets=expression,
        )
