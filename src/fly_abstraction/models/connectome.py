"""Sparse recurrent models whose recurrent links follow graph edges exactly."""

from __future__ import annotations

import torch
from torch import nn

from fly_abstraction.graph.connectome import ConnectomeGraph
from fly_abstraction.models.base import MathModel, ModelOutput


class SparseGraphCell(nn.Module):
    """One scalar state per graph node with no off-graph recurrent parameters."""

    def __init__(self, graph: ConnectomeGraph, input_size: int, trainable_edges: bool) -> None:
        super().__init__()
        self.num_nodes = graph.num_nodes
        self.register_buffer("edge_index", graph.edge_index.clone())
        self.register_buffer("base_weight", graph.edge_weight.float().clone())
        edge_scale = torch.ones(graph.num_edges)
        if trainable_edges:
            self.edge_scale = nn.Parameter(edge_scale)
        else:
            self.register_buffer("edge_scale", edge_scale)
        self.input_projection = nn.Linear(input_size, graph.num_nodes)
        self.bias = nn.Parameter(torch.zeros(graph.num_nodes))

    def forward(self, features: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        source, target = self.edge_index
        messages = state[:, source] * (self.base_weight * self.edge_scale)
        recurrent = torch.zeros_like(state)
        recurrent.index_add_(1, target, messages)
        return torch.tanh(self.input_projection(features) + recurrent + self.bias)


class _ConnectomeSequenceModel(MathModel):
    def __init__(
        self,
        graph: ConnectomeGraph,
        vocab_size: int = 128,
        embedding_dim: int = 16,
        expression_vocab_size: int = 128,
        *,
        trainable_edges: bool,
        fixed_reservoir: bool,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.cell = SparseGraphCell(graph, embedding_dim + 1, trainable_edges)
        self.answer_head = nn.Sequential(
            nn.LayerNorm(graph.num_nodes),
            nn.Linear(graph.num_nodes, 1),
        )
        self.expression_head = nn.Linear(graph.num_nodes, expression_vocab_size)
        if fixed_reservoir:
            for parameter in self.embedding.parameters():
                parameter.requires_grad_(False)
            for parameter in self.cell.parameters():
                parameter.requires_grad_(False)

    def forward(self, token_ids: torch.Tensor, numeric: torch.Tensor) -> ModelOutput:
        if token_ids.ndim != 2 or numeric.shape != (*token_ids.shape, 1):
            raise ValueError("Expected token_ids [B,T] and numeric [B,T,1]")
        embedded = self.embedding(token_ids)
        state = torch.zeros(
            token_ids.shape[0],
            self.cell.num_nodes,
            dtype=embedded.dtype,
            device=token_ids.device,
        )
        states = []
        for step in range(token_ids.shape[1]):
            features = torch.cat((embedded[:, step], numeric[:, step]), dim=-1)
            state = self.cell(features, state)
            states.append(state)
        sequence = torch.stack(states, dim=1)
        output = ModelOutput(
            answer=self.answer_head(state).squeeze(-1),
            expression_logits=self.expression_head(sequence),
            hidden=state,
        )
        self.assert_finite(output)
        return output


class FixedConnectomeReservoir(_ConnectomeSequenceModel):
    def __init__(
        self,
        graph: ConnectomeGraph,
        vocab_size: int = 128,
        embedding_dim: int = 16,
        expression_vocab_size: int = 128,
    ) -> None:
        super().__init__(
            graph,
            vocab_size,
            embedding_dim,
            expression_vocab_size,
            trainable_edges=False,
            fixed_reservoir=True,
        )


class TrainableConnectomeRNN(_ConnectomeSequenceModel):
    def __init__(
        self,
        graph: ConnectomeGraph,
        vocab_size: int = 128,
        embedding_dim: int = 16,
        expression_vocab_size: int = 128,
    ) -> None:
        super().__init__(
            graph,
            vocab_size,
            embedding_dim,
            expression_vocab_size,
            trainable_edges=True,
            fixed_reservoir=False,
        )

    @property
    def recurrent_parameter_count(self) -> int:
        return self.cell.edge_scale.numel()


class RandomGraphReservoir(FixedConnectomeReservoir):
    """Same fixed-reservoir machinery, constructed with a matched random graph."""
