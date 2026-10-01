"""Shared input/output contract and extension registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ModelOutput:
    answer: torch.Tensor
    expression_logits: torch.Tensor
    hidden: torch.Tensor


class MathModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, token_ids: torch.Tensor, numeric: torch.Tensor) -> ModelOutput:
        """Map tokens plus a numeric channel to the numeric MVP head.

        The expression head remains an architectural extension point and is not scored or trained.
        """

    def parameter_counts(self) -> dict[str, int]:
        trainable = sum(
            parameter.numel() for parameter in self.parameters() if parameter.requires_grad
        )
        total = sum(parameter.numel() for parameter in self.parameters())
        return {"trainable": trainable, "total": total, "frozen": total - trainable}

    @staticmethod
    def assert_finite(output: ModelOutput) -> None:
        for name, tensor in (
            ("answer", output.answer),
            ("expression_logits", output.expression_logits),
            ("hidden", output.hidden),
        ):
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Model output {name} contains NaN or Inf")


ModelFactory = Callable[..., MathModel]
MODEL_EXTENSIONS: dict[str, ModelFactory] = {}


def register_model(name: str, factory: ModelFactory) -> None:
    """Extension point intended for future LSTM and SmallTransformer baselines."""
    if name in MODEL_EXTENSIONS:
        raise ValueError(f"Model extension already registered: {name}")
    MODEL_EXTENSIONS[name] = factory


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)
