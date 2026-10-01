"""Directed weighted connectome representation and preprocessing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class ConnectomeGraph:
    num_nodes: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if self.edge_index.dtype != torch.long or self.edge_index.ndim != 2:
            raise ValueError("edge_index must be a rank-2 torch.long tensor")
        if self.edge_index.shape[0] != 2:
            raise ValueError("edge_index shape must be [2, num_edges]")
        if self.edge_weight.ndim != 1 or self.edge_weight.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_weight must contain one value per edge")
        if self.edge_index.numel():
            minimum, maximum = int(self.edge_index.min()), int(self.edge_index.max())
            if minimum < 0 or maximum >= self.num_nodes:
                raise ValueError("edge_index contains an out-of-range node")
        if not torch.isfinite(self.edge_weight).all():
            raise ValueError("edge_weight contains NaN or Inf")

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def content_hash(self) -> str:
        payload = {
            "num_nodes": self.num_nodes,
            "edge_index": self.edge_index.cpu().tolist(),
            "edge_weight": self.edge_weight.cpu().tolist(),
            "metadata": self.metadata,
            "manifest": self.manifest,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def degrees(self) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.edge_index
        out_degree = torch.bincount(source, minlength=self.num_nodes)
        in_degree = torch.bincount(target, minlength=self.num_nodes)
        return in_degree, out_degree

    def normalized(self, mode: str = "incoming") -> ConnectomeGraph:
        if mode == "none":
            return self
        source, target = self.edge_index
        if mode == "incoming":
            totals = torch.zeros(self.num_nodes, dtype=self.edge_weight.dtype)
            totals.index_add_(0, target, self.edge_weight.abs())
            denominator = totals[target].clamp_min(torch.finfo(self.edge_weight.dtype).eps)
        elif mode == "outgoing":
            totals = torch.zeros(self.num_nodes, dtype=self.edge_weight.dtype)
            totals.index_add_(0, source, self.edge_weight.abs())
            denominator = totals[source].clamp_min(torch.finfo(self.edge_weight.dtype).eps)
        elif mode == "max":
            denominator = (
                self.edge_weight.abs().max().clamp_min(torch.finfo(self.edge_weight.dtype).eps)
            )
        else:
            raise ValueError(f"Unknown normalization: {mode}")
        metadata = {**self.metadata, "normalization": mode}
        return replace(self, edge_weight=self.edge_weight / denominator, metadata=metadata)

    def filtered(
        self, minimum_weight: float = 0.0, max_nodes: int | None = None
    ) -> ConnectomeGraph:
        node_limit = min(self.num_nodes, max_nodes or self.num_nodes)
        source, target = self.edge_index
        mask = (
            (source < node_limit)
            & (target < node_limit)
            & (self.edge_weight.abs() >= minimum_weight)
        )
        return ConnectomeGraph(
            num_nodes=node_limit,
            edge_index=self.edge_index[:, mask].clone(),
            edge_weight=self.edge_weight[mask].clone(),
            metadata={**self.metadata, "minimum_weight": minimum_weight},
            manifest=dict(self.manifest),
        )

    def aggregate(self, groups: torch.Tensor) -> ConnectomeGraph:
        """Aggregate neurons by an integer group label, summing duplicate directed edges."""
        if groups.shape != (self.num_nodes,) or groups.dtype != torch.long:
            raise ValueError("groups must be a torch.long vector with one value per node")
        unique, inverse = torch.unique(groups, sorted=True, return_inverse=True)
        new_source = inverse[self.edge_index[0]]
        new_target = inverse[self.edge_index[1]]
        flat = new_source * unique.numel() + new_target
        unique_flat, edge_inverse = torch.unique(flat, sorted=True, return_inverse=True)
        weights = torch.zeros(unique_flat.numel(), dtype=self.edge_weight.dtype)
        weights.index_add_(0, edge_inverse, self.edge_weight)
        edges = torch.stack(
            (unique_flat.div(unique.numel(), rounding_mode="floor"), unique_flat % unique.numel())
        )
        return ConnectomeGraph(
            num_nodes=unique.numel(),
            edge_index=edges.long(),
            edge_weight=weights,
            metadata={**self.metadata, "aggregated_from_nodes": self.num_nodes},
            manifest={**self.manifest, "aggregation_labels": unique.tolist()},
        )


def tiny_synthetic_graph(num_nodes: int = 32, seed: int = 17) -> ConnectomeGraph:
    if not 4 <= num_nodes <= 128:
        raise ValueError("Tiny synthetic graph must contain 4..128 nodes")
    generator = torch.Generator().manual_seed(seed)
    edges: set[tuple[int, int]] = set()
    for node in range(num_nodes):
        edges.add((node, (node + 1) % num_nodes))
        edges.add((node, (node + 3 + node % 5) % num_nodes))
    ordered = sorted(edges)
    edge_index = torch.tensor(ordered, dtype=torch.long).T.contiguous()
    weights = 0.5 + torch.rand(len(ordered), generator=generator)
    return ConnectomeGraph(
        num_nodes=num_nodes,
        edge_index=edge_index,
        edge_weight=weights,
        metadata={"name": "tiny_synthetic", "directed": True, "seed": seed},
        manifest={"source": "generated", "version": 1, "max_nodes": 128},
    )


def load_graph_json(path: Path) -> ConnectomeGraph:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ConnectomeGraph(
        num_nodes=int(payload["num_nodes"]),
        edge_index=torch.tensor(payload["edge_index"], dtype=torch.long),
        edge_weight=torch.tensor(payload["edge_weight"], dtype=torch.float32),
        metadata=dict(payload.get("metadata", {})),
        manifest=dict(payload.get("manifest", {})),
    )
