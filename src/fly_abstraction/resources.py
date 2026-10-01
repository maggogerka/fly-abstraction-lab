"""Conservative pre-flight RAM/VRAM estimates for bounded experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch

GIB = 1024**3


class ResourceGuardError(RuntimeError):
    """Raised before allocation when an experiment exceeds configured headroom."""


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    nodes: int
    edges: int
    batch_size: int
    sequence_length: int
    parameter_count: int
    estimated_ram_bytes: int
    estimated_vram_bytes: int
    available_ram_bytes: int
    available_vram_bytes: int | None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["estimated_ram_gib"] = round(self.estimated_ram_bytes / GIB, 3)
        value["estimated_vram_gib"] = round(self.estimated_vram_bytes / GIB, 3)
        value["available_ram_gib"] = round(self.available_ram_bytes / GIB, 3)
        value["available_vram_gib"] = (
            None if self.available_vram_bytes is None else round(self.available_vram_bytes / GIB, 3)
        )
        return value


def _available_vram() -> int | None:
    if not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info()
    return int(free)


def estimate_resources(
    config: dict[str, Any],
    *,
    actual_nodes: int | None = None,
    actual_edges: int | None = None,
    available_ram_bytes: int | None = None,
    available_vram_bytes: int | None = None,
) -> ResourceEstimate:
    """Estimate peak allocations without loading the dataset or graph."""
    graph = config["graph"]
    model = config["model"]
    training = config["training"]
    data = config["data"]
    nodes = int(actual_nodes or graph["expected_num_nodes"])
    edges = int(actual_edges or graph["expected_num_edges"])
    batch = int(training["batch_size"])
    sequence = int(data["sequence_length"])
    embedding = int(model["embedding_dim"])
    vocabulary = int(model["vocab_size"])
    expression_vocabulary = int(model["expression_vocab_size"])

    # Embedding, sparse edge scales, input projection, bias, layer norm, and heads.
    parameters = (
        vocabulary * embedding
        + edges
        + nodes * (embedding + 1)
        + nodes
        + 3 * nodes
        + nodes * expression_vocabulary
        + expression_vocabulary
    )
    graph_bytes = edges * (2 * 8 + 4)
    training_state_bytes = parameters * 16  # fp32 params, grads, and Adam moments.
    node_activation_bytes = batch * sequence * nodes * 4 * 6
    # SparseGraphCell materializes B x E messages at every recurrent step. Autograd
    # retains multiple edge-sized intermediates, so this term dominates dense connectomes.
    edge_activation_bytes = batch * sequence * edges * 4 * 3
    input_bytes = batch * sequence * (embedding + 2) * 4
    cuda_framework_reserve = 2 * GIB if config["runtime"]["device"] == "cuda" else 0
    activation_bytes = node_activation_bytes + edge_activation_bytes + input_bytes
    vram = graph_bytes + training_state_bytes + activation_bytes
    vram += cuda_framework_reserve
    dataset_bytes = int(data["max_examples"]) * sequence * 16
    host_activation_bytes = (
        activation_bytes if config["runtime"]["device"] == "cpu" else activation_bytes // 4
    )
    ram = graph_bytes * 2 + dataset_bytes + training_state_bytes + host_activation_bytes
    ram += 512 * 1024**2

    warnings: list[str] = []
    if actual_nodes is None or actual_edges is None:
        warnings.append("estimate uses graph.expected_num_nodes/expected_num_edges")
    detected_vram = _available_vram() if available_vram_bytes is None else available_vram_bytes
    return ResourceEstimate(
        nodes=nodes,
        edges=edges,
        batch_size=batch,
        sequence_length=sequence,
        parameter_count=parameters,
        estimated_ram_bytes=ram,
        estimated_vram_bytes=vram,
        available_ram_bytes=int(
            psutil.virtual_memory().available
            if available_ram_bytes is None
            else available_ram_bytes
        ),
        available_vram_bytes=detected_vram,
        warnings=tuple(warnings),
    )


def enforce_resource_guard(
    config: dict[str, Any], estimate: ResourceEstimate, *, accepted: bool = False
) -> None:
    """Reject an unsafe confirmed run with actionable diagnostics."""
    guard = config["resource_guard"]
    problems: list[str] = []
    ram_limit = estimate.available_ram_bytes * float(guard["max_ram_fraction"])
    if estimate.estimated_ram_bytes > ram_limit:
        problems.append(
            f"estimated RAM {estimate.estimated_ram_bytes / GIB:.2f} GiB exceeds "
            f"the configured headroom ({ram_limit / GIB:.2f} GiB)"
        )
    if config["runtime"]["device"] == "cuda":
        if estimate.available_vram_bytes is None:
            problems.append("CUDA VRAM is unavailable; run doctor-gpu on the target PC")
        else:
            vram_limit = estimate.available_vram_bytes * float(guard["max_vram_fraction"])
            if estimate.estimated_vram_bytes > vram_limit:
                problems.append(
                    f"estimated VRAM {estimate.estimated_vram_bytes / GIB:.2f} GiB exceeds "
                    f"the configured headroom ({vram_limit / GIB:.2f} GiB)"
                )
    if bool(guard.get("require_explicit_acceptance")) and not accepted:
        problems.append("this profile requires --accept-resource-estimate")
    if problems:
        summary = "; ".join(problems)
        raise ResourceGuardError(
            f"Resource guard stopped the run: {summary}. Reduce nodes, edges, batch size, "
            "or sequence length and review the printed estimate."
        )
