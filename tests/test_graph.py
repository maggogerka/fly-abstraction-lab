import json
from pathlib import Path

import torch

from fly_abstraction.graph.connectome import load_graph_npz, save_graph_npz, tiny_synthetic_graph
from fly_abstraction.graph.controls import (
    apply_graph_variant,
    assert_basic_matched_invariants,
    degree_preserving_rewired,
    direction_shuffled,
    erdos_renyi_matched,
    weight_shuffled,
)
from fly_abstraction.graph.flywire import convert_local_flywire_export


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


def test_binary_graph_round_trip_and_streaming_hash(tmp_path: Path) -> None:
    graph = tiny_synthetic_graph(16, 23)
    path = tmp_path / "graph.npz"
    second_path = tmp_path / "graph-copy.npz"
    save_graph_npz(path, graph)
    save_graph_npz(second_path, graph)
    loaded = load_graph_npz(path)
    assert path.read_bytes() == second_path.read_bytes()
    assert loaded.content_hash() == graph.content_hash()
    assert torch.equal(loaded.input_nodes, graph.input_nodes)
    assert torch.equal(loaded.output_nodes, graph.output_nodes)


def test_named_controls_are_deterministic_and_keep_io_mappings() -> None:
    graph = tiny_synthetic_graph(16, 17)
    for variant in (
        "real",
        "degree_preserving",
        "weight_shuffled",
        "direction_shuffled",
        "er_random",
    ):
        first = apply_graph_variant(graph, variant, 29)
        second = apply_graph_variant(graph, variant, 29)
        assert first.content_hash() == second.content_hash()
        assert torch.equal(first.input_nodes, graph.input_nodes)
        assert torch.equal(first.output_nodes, graph.output_nodes)


def test_local_flywire_conversion_aggregates_without_network(tmp_path: Path) -> None:
    source = tmp_path / "authorized.csv"
    source.write_text(
        "pre_group,post_group,synapse_count\na,b,2\na,b,3\nb,c,4\n",
        encoding="utf-8",
    )
    output = tmp_path / "flywire.npz"
    convert_local_flywire_export(source, output)
    graph = load_graph_npz(output)
    assert graph.num_nodes == 3
    assert graph.num_edges == 2
    assert torch.equal(torch.sort(graph.edge_weight).values, torch.tensor([4.0, 5.0]))
    assert graph.manifest["release"] == "FAFB v783"
    assert graph.manifest["accepted_records"] == 3
    assert graph.manifest["rejected_records"] == 0
    assert output.with_suffix(".nodes.csv").is_file()
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["prepared_sha256"]
    assert manifest["license"]
