import json
from pathlib import Path

import pytest
import torch

from fly_abstraction.config import load_config
from fly_abstraction.data.adapters import TinyDatasetAdapter, UCIEnergyEfficiencyAdapter
from fly_abstraction.training.pipeline import prepare_experiment, train_experiment
from fly_abstraction.utils import sha256_file, write_json, write_jsonl


def _write_test_uci(path: Path) -> None:
    problems = []
    for index in range(768):
        geometry = index % 12
        orientation = index // 12
        record = {
            "X1": 0.5 + geometry,
            "X2": 100 + geometry,
            "X3": 50 + geometry,
            "X4": 25 + geometry,
            "X5": 3 + geometry,
            "X6": 2 + orientation % 4,
            "X7": (orientation % 5) / 10,
            "X8": orientation % 6,
            "Y1": 20 + geometry + orientation / 100,
            "Y2": 25 + geometry,
            "row_id": str(index),
        }
        problems.append(UCIEnergyEfficiencyAdapter().adapt(record))
    write_jsonl(path, (problem.to_dict() for problem in problems))
    write_json(
        path.with_suffix(path.suffix + ".manifest.json"),
        {
            "version": "UCI-242-2024-02-26",
            "prepared_sha256": sha256_file(path),
            "official_row_count_complete": True,
            "accepted_records": 768,
            "rejected_records": 0,
        },
    )


def test_pilot_profile_enters_prepare_experiment_with_128_node_graph(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = Path("data/processed/uci_energy_efficiency.jsonl")
    _write_test_uci(data_path)
    config = load_config("pilot_gpu")
    prepared = prepare_experiment(config)
    assert prepared.graph.num_nodes == 128
    assert prepared.train_loader.dataset.problems
    assert all(
        problem.split == "test_held_out_geometry_ood"
        for problem in prepared.test_loader.dataset.problems
    )


def test_pilot_rejects_uci_without_preparation_manifest(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = Path("data/processed/uci_energy_efficiency.jsonl")
    problem = UCIEnergyEfficiencyAdapter().adapt(
        {
            **{f"X{index}": index for index in range(1, 9)},
            "Y1": 9,
            "Y2": 10,
            "row_id": "0",
        }
    )
    write_jsonl(data_path, [problem.to_dict()])
    with pytest.raises(FileNotFoundError, match="provenance is missing"):
        prepare_experiment(load_config("pilot_gpu"))


def test_full_cpu_epoch_writes_artifacts_and_resume_checkpoint_is_aligned(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    data_path = Path("data/processed/tiny.jsonl")
    TinyDatasetAdapter.prepare(data_path, count=16, seed=17)
    config = load_config("smoke_cpu")
    config["data"]["path"] = str(data_path)
    config["data"]["split_manifest_dir"] = "data/processed/splits"
    config["data"]["max_examples"] = 16
    config["data"]["sequence_length"] = 12
    config["graph"]["max_graph_nodes"] = 8
    config["graph"]["expected_num_nodes"] = 8
    config["graph"]["expected_num_edges"] = 16
    config["model"]["hidden_size"] = 8
    config["training"]["checkpoints"] = True
    config["output"]["root"] = "results"

    first = train_experiment(config, run_id="cpu_one_epoch")
    for name in (
        "summary.json",
        "metrics.json",
        "history.csv",
        "predictions.jsonl",
        "run_manifest.json",
    ):
        assert (first / name).is_file()
    first_checkpoint = torch.load(first / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert first_checkpoint["epoch"] == 0

    resumed_config = dict(config)
    resumed_config["training"] = dict(config["training"], max_epochs=2)
    resumed = train_experiment(
        resumed_config,
        run_id="cpu_resumed",
        resume=first / "checkpoint.pt",
    )
    checkpoint = torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False)
    summary = json.loads((resumed / "summary.json").read_text(encoding="utf-8"))
    assert checkpoint["epoch"] == summary["best_epoch"]
    assert checkpoint["best_validation_loss"] == summary["best_validation_loss"]
    batches_per_epoch = 5
    optimizer_steps = {
        int(state["step"].item())
        for state in checkpoint["optimizer"]["state"].values()
        if "step" in state
    }
    assert optimizer_steps == {batches_per_epoch * (checkpoint["epoch"] + 1)}
