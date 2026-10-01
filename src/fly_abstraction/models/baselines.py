"""Conventional GRU and MLP baselines with identical output heads."""

from __future__ import annotations

import torch
from torch import nn

from fly_abstraction.models.base import MathModel, ModelOutput


class GRUBaseline(MathModel):
    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 16,
        hidden_size: int = 64,
        expression_vocab_size: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.recurrent = nn.GRU(embedding_dim + 1, hidden_size, batch_first=True)
        self.answer_head = nn.Linear(hidden_size, 1)
        self.expression_head = nn.Linear(hidden_size, expression_vocab_size)

    def forward(self, token_ids: torch.Tensor, numeric: torch.Tensor) -> ModelOutput:
        features = torch.cat((self.embedding(token_ids), numeric), dim=-1)
        sequence, final = self.recurrent(features)
        hidden = final[-1]
        output = ModelOutput(
            answer=self.answer_head(hidden).squeeze(-1),
            expression_logits=self.expression_head(sequence),
            hidden=hidden,
        )
        self.assert_finite(output)
        return output


class MLPBaseline(MathModel):
    def __init__(
        self,
        vocab_size: int = 128,
        embedding_dim: int = 16,
        hidden_size: int = 64,
        expression_vocab_size: int = 128,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.answer_head = nn.Linear(hidden_size, 1)
        self.expression_head = nn.Linear(hidden_size, expression_vocab_size)

    def forward(self, token_ids: torch.Tensor, numeric: torch.Tensor) -> ModelOutput:
        mask = token_ids.ne(0).unsqueeze(-1)
        features = torch.cat((self.embedding(token_ids), numeric), dim=-1)
        sequence = self.network(features)
        denominator = mask.sum(dim=1).clamp_min(1)
        hidden = (sequence * mask).sum(dim=1) / denominator
        repeated = hidden.unsqueeze(1).expand(-1, token_ids.shape[1], -1)
        output = ModelOutput(
            answer=self.answer_head(hidden).squeeze(-1),
            expression_logits=self.expression_head(repeated),
            hidden=hidden,
        )
        self.assert_finite(output)
        return output
