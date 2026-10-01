"""Deterministic matched graph controls."""

from __future__ import annotations

import random
from dataclasses import replace

import torch

from fly_abstraction.graph.connectome import ConnectomeGraph


def _tag(graph: ConnectomeGraph, control: str, seed: int) -> dict[str, object]:
    return {**graph.metadata, "control": control, "control_seed": seed}


def degree_preserving_rewired(
    graph: ConnectomeGraph, seed: int = 17, swaps_per_edge: int = 4
) -> ConnectomeGraph:
    """Directed double-edge swaps preserve each node's in- and out-degree."""
    rng = random.Random(seed)
    edges = [tuple(edge) for edge in graph.edge_index.T.tolist()]
    edge_set = set(edges)
    if len(edge_set) != len(edges):
        raise ValueError("degree-preserving rewiring requires unique input edges")
    target_swaps = max(1, graph.num_edges * swaps_per_edge)
    completed = 0
    for _ in range(target_swaps * 20):
        if completed >= target_swaps:
            break
        first, second = rng.sample(range(len(edges)), 2)
        a, b = edges[first]
        c, d = edges[second]
        candidate_one, candidate_two = (a, d), (c, b)
        if a == d or c == b or candidate_one == candidate_two:
            continue
        old = {edges[first], edges[second]}
        if any(candidate in edge_set - old for candidate in (candidate_one, candidate_two)):
            continue
        edge_set.difference_update(old)
        edge_set.update((candidate_one, candidate_two))
        edges[first], edges[second] = candidate_one, candidate_two
        completed += 1
    edge_index = torch.tensor(edges, dtype=torch.long).T.contiguous()
    metadata = _tag(graph, "degree_preserving_rewired", seed)
    metadata["completed_swaps"] = completed
    return replace(graph, edge_index=edge_index, metadata=metadata)


def erdos_renyi_matched(graph: ConnectomeGraph, seed: int = 17) -> ConnectomeGraph:
    """Sample exactly the same number of unique directed, non-self edges."""
    capacity = graph.num_nodes * (graph.num_nodes - 1)
    if graph.num_edges > capacity:
        raise ValueError("Too many edges for a simple directed Erdos-Renyi graph")
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < graph.num_edges:
        source = rng.randrange(graph.num_nodes)
        target = rng.randrange(graph.num_nodes - 1)
        if target >= source:
            target += 1
        edges.add((source, target))
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()
    order = torch.randperm(graph.num_edges, generator=torch.Generator().manual_seed(seed))
    return replace(
        graph,
        edge_index=edge_index,
        edge_weight=graph.edge_weight[order].clone(),
        metadata=_tag(graph, "erdos_renyi_matched", seed),
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
    if not torch.allclose(
        torch.sort(original.edge_weight).values,
        torch.sort(control.edge_weight).values,
    ):
        raise AssertionError("edge-weight multiset changed")
