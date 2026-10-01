"""Deterministic matched graph controls."""

from __future__ import annotations

from dataclasses import replace

import torch

from fly_abstraction.graph.connectome import ConnectomeGraph


def _tag(graph: ConnectomeGraph, control: str, seed: int) -> dict[str, object]:
    return {**graph.metadata, "control": control, "control_seed": seed}


def degree_preserving_rewired(
    graph: ConnectomeGraph, seed: int = 17, swaps_per_edge: int = 4
) -> ConnectomeGraph:
    """Tensor-only directed configuration-model control preserving both degree sequences."""
    del swaps_per_edge
    generator = torch.Generator().manual_seed(seed)
    edge_index = graph.edge_index.clone()
    edge_index[1] = edge_index[1, torch.randperm(graph.num_edges, generator=generator)]
    metadata = _tag(graph, "degree_preserving", seed)
    metadata["multigraph_control"] = True
    return replace(graph, edge_index=edge_index, metadata=metadata)


def erdos_renyi_matched(graph: ConnectomeGraph, seed: int = 17) -> ConnectomeGraph:
    """Tensor-only matched directed random multigraph with no self loops."""
    generator = torch.Generator().manual_seed(seed)
    source = torch.randint(graph.num_nodes, (graph.num_edges,), generator=generator)
    target = torch.randint(graph.num_nodes - 1, (graph.num_edges,), generator=generator)
    target += target >= source
    edge_index = torch.stack((source, target))
    order = torch.randperm(graph.num_edges, generator=generator)
    return replace(
        graph,
        edge_index=edge_index,
        edge_weight=graph.edge_weight[order].clone(),
        metadata={**_tag(graph, "er_random", seed), "multigraph_control": True},
    )


def weight_shuffled(graph: ConnectomeGraph, seed: int = 17) -> ConnectomeGraph:
    order = torch.randperm(graph.num_edges, generator=torch.Generator().manual_seed(seed))
    return replace(
        graph,
        edge_weight=graph.edge_weight[order].clone(),
        metadata=_tag(graph, "weight_shuffled", seed),
    )


def direction_shuffled(graph: ConnectomeGraph, seed: int = 17) -> ConnectomeGraph:
    generator = torch.Generator().manual_seed(seed)
    flip = torch.rand(graph.num_edges, generator=generator) < 0.5
    edge_index = graph.edge_index.clone()
    edge_index[0, flip] = graph.edge_index[1, flip]
    edge_index[1, flip] = graph.edge_index[0, flip]
    return replace(
        graph,
        edge_index=edge_index,
        metadata=_tag(graph, "direction_shuffled", seed),
    )


def assert_basic_matched_invariants(original: ConnectomeGraph, control: ConnectomeGraph) -> None:
    if original.num_nodes != control.num_nodes:
        raise AssertionError("node count changed")
    if original.num_edges != control.num_edges:
        raise AssertionError("edge count changed")
    if not torch.equal(original.input_nodes, control.input_nodes):
        raise AssertionError("input-node mapping changed")
    if not torch.equal(original.output_nodes, control.output_nodes):
        raise AssertionError("output-node mapping changed")
    if not torch.allclose(
        torch.sort(original.edge_weight).values,
        torch.sort(control.edge_weight).values,
    ):
        raise AssertionError("edge-weight multiset changed")


def apply_graph_variant(graph: ConnectomeGraph, variant: str, seed: int) -> ConnectomeGraph:
    builders = {
        "degree_preserving": degree_preserving_rewired,
        "weight_shuffled": weight_shuffled,
        "direction_shuffled": direction_shuffled,
        "er_random": erdos_renyi_matched,
    }
    if variant == "real":
        return graph
    try:
        control = builders[variant](graph, seed)
    except KeyError as exc:
        raise ValueError(f"Unknown graph variant: {variant}") from exc
    assert_basic_matched_invariants(graph, control)
    return control
