import json
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch

from fly_abstraction.cli import main
from fly_abstraction.config import load_config
from fly_abstraction.data.adapters import TinyDatasetAdapter
from fly_abstraction.graph.connectome import load_graph_npz
from fly_abstraction.graph.controls import apply_graph_variant
from fly_abstraction.graph.flywire_v783 import (
    inspect_graph,
    preparation_plan,
    prepare_flywire_v783,
)
from fly_abstraction.training.pipeline import prepare_experiment
from fly_abstraction.utils import sha256_file

ROWS = [
    (1, 2, "A", 2.0),
    (1, 2, "B", 3.0),
    (2, 3, "A", 6.0),
    (3, 1, "A", 7.0),
    (3, 4, "A", 1.0),
    (4, 5, "A", 10.0),
    (5, 4, "A", 10.0),
    (2, 2, "A", 5.0),
    (0, 2, "A", 9.0),
    (1, 3, "A", float("nan")),
]


def _write_feather(path: Path, rows: list[tuple[int, int, str, float]] = ROWS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "pre_pt_root_id": pa.array([row[0] for row in rows], type=pa.int64()),
            "post_pt_root_id": pa.array([row[1] for row in rows], type=pa.int64()),
            "neuropil": pa.array([row[2] for row in rows], type=pa.string()),
            "syn_count": pa.array([row[3] for row in rows], type=pa.float64()),
            "unused": pa.array(range(len(rows)), type=pa.int32()),
        }
    )
    feather.write_feather(table, path, compression="uncompressed")


def _prepare(source: Path, output: Path, nodes: int = 3) -> dict[str, Path]:
    return prepare_flywire_v783(
        source,
        output,
        nodes=nodes,
        min_pair_synapses=5,
        input_node_count=1,
        output_node_count=1,
        seed=1701,
    )


def test_schema_validation_and_dry_run_create_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("data/raw/source.feather")
    _write_feather(source)
    output = Path("data/processed/core.npz")
    assert (
        main(
            [
                "graph",
                "prepare-flywire-v783",
                "--source",
                str(source),
                "--output",
                str(output),
                "--nodes",
                "3",
                "--input-node-count",
                "2",
                "--output-node-count",
                "2",
            ]
        )
        == 0
    )
    assert "DRY RUN" in capsys.readouterr().out
    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()

    bad = Path("data/raw/bad.feather")
    feather.write_feather(
        pa.table(
            {
                "pre_pt_root_id": [1],
                "post_pt_root_id": [2],
                "syn_count": [5],
            }
        ),
        bad,
    )
    with pytest.raises(ValueError, match="missing required columns"):
        preparation_plan(
            bad,
            Path("data/processed/bad.npz"),
            nodes=1,
            min_pair_synapses=5,
            input_node_count=1,
            output_node_count=1,
            seed=1701,
        )


def test_aggregation_threshold_core_mappings_manifests_and_inspect(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    source = tmp_path / "source.feather"
    output = tmp_path / "core.npz"
    _write_feather(source)
    artifacts = _prepare(source, output)
    graph = load_graph_npz(output)
    assert graph.num_nodes == 3
    assert graph.num_edges == 4
    assert graph.manifest["original_unique_pairs"] == 7
    assert graph.manifest["pairs_after_threshold"] == 6
    assert graph.manifest["accepted_records"] == 8
    assert graph.manifest["rejected_records"] == 2
    assert graph.manifest["self_loops"] == 1
    assert graph.manifest["input_root_ids"][0] == 2
    assert 2 not in graph.manifest["output_root_ids"]
    edge_weights = {
        tuple(edge): float(weight)
        for edge, weight in zip(
            graph.edge_index.T.tolist(), graph.edge_weight.tolist(), strict=True
        )
    }
    root_ids = [2, 3, 1]
    root_edges = {
        (root_ids[source_index], root_ids[target_index]): weight
        for (source_index, target_index), weight in edge_weights.items()
    }
    assert root_edges[(1, 2)] == 5.0
    assert (3, 4) not in root_edges

    manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
    assert manifest["npz_sha256"] == sha256_file(output)
    assert manifest["nodes_csv_sha256"] == sha256_file(artifacts["nodes"])
    assert manifest["algorithm"] == "weighted_connected_core_v1"
    before = output.read_bytes()
    report = inspect_graph(output)
    assert output.read_bytes() == before
    assert report["num_nodes"] == 3
    assert report["num_edges"] == 4
    assert report["self_loops"] == 1
    assert report["weak_component_count"] == 1
    assert 0.0 <= report["reachable_output_fraction_from_inputs"] <= 1.0
    monkeypatch.chdir(tmp_path)
    assert main(["graph", "inspect", "--graph", "core.npz"]) == 0
    assert '"num_nodes": 3' in capsys.readouterr().out


def test_row_order_invariance_and_byte_deterministic_npz(tmp_path: Path) -> None:
    source = tmp_path / "source.feather"
    shuffled = tmp_path / "shuffled.feather"
    _write_feather(source)
    _write_feather(shuffled, list(reversed(ROWS)))
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    reordered = tmp_path / "reordered.npz"
    _prepare(source, first)
    _prepare(source, second)
    _prepare(shuffled, reordered)
    assert first.read_bytes() == second.read_bytes()
    first_graph = load_graph_npz(first)
    reordered_graph = load_graph_npz(reordered)
    assert torch.equal(first_graph.edge_index, reordered_graph.edge_index)
    assert torch.equal(first_graph.edge_weight, reordered_graph.edge_weight)
    assert torch.equal(first_graph.input_nodes, reordered_graph.input_nodes)
    assert torch.equal(first_graph.output_nodes, reordered_graph.output_nodes)
    assert (
        first_graph.manifest["selected_root_ids_sha256"]
        == reordered_graph.manifest["selected_root_ids_sha256"]
    )


def test_controls_preserve_interface_and_overwrite_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "source.feather"
    output = tmp_path / "core.npz"
    _write_feather(source)
    _prepare(source, output)
    graph = load_graph_npz(output)
    for variant in (
        "real",
        "degree_preserving",
        "weight_shuffled",
        "direction_shuffled",
        "er_random",
    ):
        control = apply_graph_variant(graph, variant, 1701)
        assert torch.equal(control.input_nodes, graph.input_nodes)
        assert torch.equal(control.output_nodes, graph.output_nodes)
    with pytest.raises(FileExistsError, match="overwrite"):
        _prepare(source, output)


def test_pipeline_uses_exact_prepared_graph_and_cpu_forward_backward(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    source = Path("source.feather")
    graph_path = Path("data/processed/flywire_v783_core_4.npz")
    _write_feather(source)
    _prepare(source, graph_path, nodes=4)
    TinyDatasetAdapter.prepare(Path("data/processed/tiny.jsonl"), count=24, seed=1701)
    config = load_config("flywire_smoke_gpu")
    config["graph"]["path"] = str(graph_path)
    config["graph"]["max_graph_nodes"] = 4
    config["graph"]["expected_num_nodes"] = 4
    config["graph"]["expected_num_edges"] = 4
    config["model"]["hidden_size"] = 4
    config["data"]["max_examples"] = 24
    config["data"]["sequence_length"] = 12
    prepared = prepare_experiment(config)
    assert prepared.graph.num_nodes == 4
    assert prepared.graph.manifest["selection_component_count"] == 2
    batch = next(iter(prepared.train_loader))
    output = prepared.model(batch["token_ids"], batch["numeric"])
    torch.nn.functional.mse_loss(output.answer, batch["answer"]).backward()
    assert any(
        parameter.grad is not None
        for parameter in prepared.model.parameters()
        if parameter.requires_grad
    )

    config["graph"]["max_graph_nodes"] = 3
    with pytest.raises(ValueError, match="exceeding graph.max_graph_nodes"):
        prepare_experiment(config)
