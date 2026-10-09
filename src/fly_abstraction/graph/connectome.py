"""Directed weighted connectome representation and preprocessing."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ConnectomeGraph:
    num_nodes: int
    edge_index: torch.Tensor
    edge_weight: torch.Tensor
    input_nodes: torch.Tensor | None = None
    output_nodes: torch.Tensor | None = None
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
        if self.input_nodes is None:
            object.__setattr__(self, "input_nodes", torch.arange(self.num_nodes, dtype=torch.long))
        if self.output_nodes is None:
            object.__setattr__(self, "output_nodes", torch.arange(self.num_nodes, dtype=torch.long))
        for name, mapping in (
            ("input_nodes", self.input_nodes),
            ("output_nodes", self.output_nodes),
        ):
            assert mapping is not None
            if mapping.dtype != torch.long or mapping.ndim != 1:
                raise ValueError(f"{name} must be a rank-1 torch.long tensor")
            if not mapping.numel() or torch.unique(mapping).numel() != mapping.numel():
                raise ValueError(f"{name} must contain unique nodes and cannot be empty")
            if mapping.numel() and (int(mapping.min()) < 0 or int(mapping.max()) >= self.num_nodes):
                raise ValueError(f"{name} contains an out-of-range node")

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        header = {
            "num_nodes": self.num_nodes,
            "metadata": self.metadata,
            "manifest": self.manifest,
        }
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode())
        for name, tensor in (
            ("edge_index", self.edge_index),
            ("edge_weight", self.edge_weight),
            ("input_nodes", self.input_nodes),
            ("output_nodes", self.output_nodes),
        ):
            assert tensor is not None
            contiguous = tensor.detach().cpu().contiguous()
            digest.update(f"{name}|{contiguous.dtype}|{tuple(contiguous.shape)}|".encode())
            raw = memoryview(contiguous.numpy()).cast("B")
            for offset in range(0, len(raw), 1024 * 1024):
                digest.update(raw[offset : offset + 1024 * 1024])
        return digest.hexdigest()

    def degrees(self) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = self.edge_index
        out_degree = torch.bincount(source, minlength=self.num_nodes)
        in_degree = torch.bincount(target, minlength=self.num_nodes)
        return in_degree, out_degree

    def to(self, device: torch.device | str) -> ConnectomeGraph:
        return replace(
            self,
            edge_index=self.edge_index.to(device),
            edge_weight=self.edge_weight.to(device),
            input_nodes=self.input_nodes.to(device),
            output_nodes=self.output_nodes.to(device),
        )

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
            input_nodes=self.input_nodes[self.input_nodes < node_limit].clone(),
            output_nodes=self.output_nodes[self.output_nodes < node_limit].clone(),
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
            input_nodes=torch.unique(inverse[self.input_nodes], sorted=True),
            output_nodes=torch.unique(inverse[self.output_nodes], sorted=True),
            metadata={**self.metadata, "aggregated_from_nodes": self.num_nodes},
            manifest={
                **self.manifest,
                "aggregation_group_count": int(unique.numel()),
                "aggregation_groups_sha256": _tensor_sha256(unique),
            },
        )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    raw = memoryview(tensor.detach().cpu().contiguous().numpy()).cast("B")
    for offset in range(0, len(raw), 1024 * 1024):
        digest.update(raw[offset : offset + 1024 * 1024])
    return digest.hexdigest()


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


def save_graph_npz(path: Path, graph: ConnectomeGraph) -> Path:
    """Write a byte-deterministic scalable binary graph artifact without overwriting."""
    if path.suffix.lower() != ".npz":
        raise ValueError("Binary graph paths must use the .npz suffix")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial graph: {temporary}")
    arrays = {
        "num_nodes": np.asarray([graph.num_nodes], dtype=np.int64),
        "edge_index": graph.edge_index.detach().cpu().numpy(),
        "edge_weight": graph.edge_weight.detach().cpu().numpy(),
        "input_nodes": graph.input_nodes.detach().cpu().numpy(),
        "output_nodes": graph.output_nodes.detach().cpu().numpy(),
        "metadata": np.frombuffer(
            json.dumps(
                graph.metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
        "manifest": np.frombuffer(
            json.dumps(
                graph.manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
    }
    try:
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name, array in arrays.items():
                payload = io.BytesIO()
                np.save(payload, array, allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 0
                info.external_attr = 0
                archive.writestr(info, payload.getvalue(), compresslevel=9)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


def load_graph_npz(path: Path) -> ConnectomeGraph:
    """Load the primary binary graph artifact with non-pickle arrays only."""
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(payload["metadata"].tobytes().decode("utf-8"))
        manifest = json.loads(payload["manifest"].tobytes().decode("utf-8"))
        return ConnectomeGraph(
            num_nodes=int(payload["num_nodes"][0]),
            edge_index=torch.from_numpy(payload["edge_index"].copy()).long(),
            edge_weight=torch.from_numpy(payload["edge_weight"].copy()).float(),
            input_nodes=torch.from_numpy(payload["input_nodes"].copy()).long(),
            output_nodes=torch.from_numpy(payload["output_nodes"].copy()).long(),
            metadata=metadata,
            manifest=manifest,
        )


def load_graph_json_legacy(path: Path) -> ConnectomeGraph:
    """Load small legacy JSON graphs; new connectomes must use NPZ."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ConnectomeGraph(
        num_nodes=int(payload["num_nodes"]),
        edge_index=torch.tensor(payload["edge_index"], dtype=torch.long),
        edge_weight=torch.tensor(payload["edge_weight"], dtype=torch.float32),
        metadata=dict(payload.get("metadata", {})),
        manifest=dict(payload.get("manifest", {})),
    )
