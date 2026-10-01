import torch

from fly_abstraction.graph.connectome import tiny_synthetic_graph
from fly_abstraction.graph.controls import (
    assert_basic_matched_invariants,
    degree_preserving_rewired,
    direction_shuffled,
    erdos_renyi_matched,
    weight_shuffled,
)


def test_control_graph_invariants() -> None:
    graph = tiny_synthetic_graph(32, 17)
    rewired = degree_preserving_rewired(graph, 19)
    erdos = erdos_renyi_matched(graph, 19)
    weights = weight_shuffled(graph, 19)
    directions = direction_shuffled(graph, 19)
    for control in (rewired, erdos, weights, directions):
        assert_basic_matched_invariants(graph, control)
    original_in, original_out = graph.degrees()
    rewired_in, rewired_out = rewired.degrees()
    assert torch.equal(original_in, rewired_in)
    assert torch.equal(original_out, rewired_out)
    assert torch.equal(graph.edge_index, weights.edge_index)


def test_normalization_and_aggregation_are_finite() -> None:
    graph = tiny_synthetic_graph(12, 17).normalized("incoming")
    incoming = torch.zeros(graph.num_nodes)
    incoming.index_add_(0, graph.edge_index[1], graph.edge_weight.abs())
    assert torch.allclose(incoming, torch.ones_like(incoming))
    aggregated = graph.aggregate(torch.arange(graph.num_nodes) // 3)
    assert aggregated.num_nodes == 4
    assert torch.isfinite(aggregated.edge_weight).all()
