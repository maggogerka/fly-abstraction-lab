"""Implemented training/validation/test pipeline. Importing this module never starts it."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fly_abstraction.data.adapters import TinyDatasetAdapter
from fly_abstraction.data.schema import MathProblem
from fly_abstraction.data.splits import (
    apply_split_manifest,
    build_split_manifest,
    materialize_split_manifest,
)
from fly_abstraction.graph.connectome import ConnectomeGraph, tiny_synthetic_graph
from fly_abstraction.graph.controls import apply_graph_variant, erdos_renyi_matched
from fly_abstraction.graph.flywire import FlyWireFAFBV783Adapter
from fly_abstraction.models.base import MODEL_EXTENSIONS, MathModel, resolve_device
from fly_abstraction.models.baselines import GRUBaseline, MLPBaseline
from fly_abstraction.models.connectome import (
    FixedConnectomeReservoir,
    RandomGraphReservoir,
    TrainableConnectomeRNN,
)
from fly_abstraction.models.encoding import CharacterEncoder
from fly_abstraction.resources import enforce_resource_guard, estimate_resources
from fly_abstraction.training.artifacts import RunArtifacts, build_manifest
from fly_abstraction.training.metrics import evaluate_records
from fly_abstraction.utils import read_jsonl, set_seeds, sha256_file


class EncodedDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, problems: list[MathProblem], encoder: CharacterEncoder) -> None:
        self.problems = problems
        self.encoded = [encoder.encode(problem) for problem in problems]

    def __len__(self) -> int:
        return len(self.problems)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.encoded[index]
        return {
            "token_ids": item.token_ids,
            "numeric": item.numeric,
            "answer": item.answer,
            "expression_targets": item.expression_targets,
            "index": torch.tensor(index),
        }


@dataclass(slots=True)
class PreparedExperiment:
    model: MathModel
    graph: ConnectomeGraph
    train_loader: DataLoader[dict[str, torch.Tensor]]
    validation_loader: DataLoader[dict[str, torch.Tensor]]
    test_loader: DataLoader[dict[str, torch.Tensor]]
    split_manifest_path: Path
    split_manifest_sha256: str


def _load_splits(
    config: dict[str, Any],
) -> tuple[list[MathProblem], list[MathProblem], list[MathProblem], Path, str]:
    path = Path(config["data"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {path}; run data prepare-tiny first")
    if config["data"]["dataset"] == "uci_energy_efficiency":
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared UCI provenance is missing: {manifest_path}; run data prepare-uci-energy"
            )
        provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
        if provenance.get("version") != config["data"]["version"]:
            raise ValueError("Prepared UCI version does not match the configured version")
        if not provenance.get("official_row_count_complete"):
            raise ValueError(
                "Prepared UCI is incomplete or contains rejected rows; inspect its manifest"
            )
        if provenance.get("prepared_sha256") != sha256_file(path):
            raise ValueError("Prepared UCI SHA256 does not match its provenance manifest")
    records = read_jsonl(path)[: config["data"]["max_examples"]]
    if config["data"]["dataset"] == "tiny":
        problems = [TinyDatasetAdapter().adapt(record) for record in records]
    else:
        problems = [MathProblem.from_dict(record) for record in records]
    manifest = build_split_manifest(
        problems,
        dataset=str(config["data"]["dataset"]),
        dataset_version=str(config["data"]["version"]),
        seed=int(config["seed"]),
        held_out_template_fraction=float(config["data"]["held_out_template_fraction"]),
        validation_fraction=float(config["data"]["validation_fraction"]),
        split_strategy=str(config["data"]["split_strategy"]),
    )
    manifest_path = materialize_split_manifest(Path(config["data"]["split_manifest_dir"]), manifest)
    train, validation, test = apply_split_manifest(problems, manifest)
    return train, validation, test, manifest_path, str(manifest["sha256"])


def _build_model(config: dict[str, Any], graph: ConnectomeGraph) -> MathModel:
    model_config = config["model"]
    common = {
        "vocab_size": int(model_config["vocab_size"]),
        "embedding_dim": int(model_config["embedding_dim"]),
        "expression_vocab_size": int(model_config["expression_vocab_size"]),
    }
    name = str(model_config["name"])
    if name == "fixed_connectome_reservoir":
        return FixedConnectomeReservoir(graph, **common)
    if name == "trainable_connectome_rnn":
        return TrainableConnectomeRNN(graph, **common)
    if name == "random_graph_reservoir":
        return RandomGraphReservoir(erdos_renyi_matched(graph, config["seed"]), **common)
    if name == "gru":
        return GRUBaseline(hidden_size=int(model_config["hidden_size"]), **common)
    if name == "mlp":
        return MLPBaseline(hidden_size=int(model_config["hidden_size"]), **common)
    if name in MODEL_EXTENSIONS:
        return MODEL_EXTENSIONS[name](graph=graph, config=model_config)
    raise ValueError(f"Unknown model: {name}")


def prepare_experiment(config: dict[str, Any]) -> PreparedExperiment:
    set_seeds(int(config["seed"]), bool(config["deterministic"]))
    torch.set_num_threads(int(config["runtime"]["torch_threads"]))
    if config["graph"]["kind"] == "tiny_synthetic":
        nodes = min(
            int(config["model"]["hidden_size"]),
            int(config["graph"]["max_graph_nodes"]),
        )
        graph = tiny_synthetic_graph(nodes, int(config["seed"]))
    elif config["graph"]["kind"] == "flywire_fafb_v783":
        graph = FlyWireFAFBV783Adapter().load(Path(config["graph"]["path"]))
        if graph.num_nodes > int(config["graph"]["max_graph_nodes"]):
            raise ValueError(
                f"Prepared FlyWire graph has {graph.num_nodes} nodes, exceeding "
                f"graph.max_graph_nodes={config['graph']['max_graph_nodes']}; "
                "prepare the intended bounded graph explicitly"
            )
    else:
        raise ValueError(f"Unknown graph kind: {config['graph']['kind']}")
    graph = apply_graph_variant(graph, str(config["graph"]["variant"]), int(config["seed"]))
    graph = graph.normalized(str(config["graph"]["normalization"]))
    model = _build_model(config, graph)
    encoder = CharacterEncoder(
        vocab_size=int(config["model"]["vocab_size"]),
        max_length=int(config["data"]["sequence_length"]),
    )
    train, validation, test, split_path, split_sha256 = _load_splits(config)
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
    }
    return PreparedExperiment(
        model=model,
        graph=graph,
        train_loader=DataLoader(
            EncodedDataset(train, encoder),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(config["seed"])),
            **loader_options,
        ),
        validation_loader=DataLoader(
            EncodedDataset(validation, encoder), shuffle=False, **loader_options
        ),
        test_loader=DataLoader(EncodedDataset(test, encoder), shuffle=False, **loader_options),
        split_manifest_path=split_path,
        split_manifest_sha256=split_sha256,
    )


def _losses(
    output_answer: torch.Tensor,
    output_expression: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del output_expression
    answer_loss = nn.functional.mse_loss(output_answer, batch["answer"])
    unused_expression_loss = answer_loss.new_zeros(())
    return answer_loss, answer_loss, unused_expression_loss


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _clone_state_to_cpu(value: Any) -> Any:
    """Detach a nested PyTorch state snapshot from later optimizer/model mutation."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _clone_state_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_to_cpu(item) for item in value)
    return copy.deepcopy(value)


def _checkpoint_snapshot(
    model: MathModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    *,
    epoch: int,
    validation_loss: float,
) -> dict[str, Any]:
    return {
        "model": _clone_state_to_cpu(model.state_dict()),
        "optimizer": _clone_state_to_cpu(optimizer.state_dict()),
        "scaler": _clone_state_to_cpu(scaler.state_dict()),
        "epoch": epoch,
        "best_validation_loss": validation_loss,
    }


@torch.no_grad()
def _evaluate_loader(
    model: MathModel,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    include_predictions: bool = False,
) -> tuple[float, list[dict[str, Any]]]:
    model.eval()
    losses: list[float] = []
    records: list[dict[str, Any]] = []
    dataset = loader.dataset
    assert isinstance(dataset, EncodedDataset)
    for raw_batch in loader:
        batch = _move(raw_batch, device)
        output = model(batch["token_ids"], batch["numeric"])
        loss, _, _ = _losses(output.answer, output.expression_logits, batch)
        losses.append(float(loss))
        if include_predictions:
            for row, prediction in zip(
                raw_batch["index"].tolist(), output.answer.detach().cpu().tolist(), strict=True
            ):
                problem = dataset.problems[row]
                records.append(
                    {
                        "source_id": problem.source_id,
                        "template_id": problem.template_id,
                        "split": problem.split,
                        "target_mode": "numeric",
                        "prediction": float(prediction),
                        "target": problem.numeric_target,
                    }
                )
    return sum(losses) / max(1, len(losses)), records


def train_experiment(
    config: dict[str, Any],
    *,
    run_id: str | None = None,
    resume: Path | None = None,
    resource_estimate_accepted: bool = False,
) -> Path:
    """Execute a confirmed run. Callers are responsible for CLI safety confirmation."""
    expected_estimate = estimate_resources(config)
    enforce_resource_guard(config, expected_estimate, accepted=resource_estimate_accepted)
    prepared = prepare_experiment(config)
    actual_estimate = estimate_resources(
        config,
        actual_nodes=prepared.graph.num_nodes,
        actual_edges=prepared.graph.num_edges,
    )
    enforce_resource_guard(config, actual_estimate, accepted=resource_estimate_accepted)
    device = resolve_device(str(config["runtime"]["device"]))
    model = prepared.model.to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    use_amp = bool(config["runtime"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_validation = float("inf")
    best_checkpoint: dict[str, Any] | None = None
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint.get("best_validation_loss", float("inf")))
        best_checkpoint = _checkpoint_snapshot(
            model,
            optimizer,
            scaler,
            epoch=int(checkpoint["epoch"]),
            validation_loss=best_validation,
        )
    if start_epoch >= int(training["max_epochs"]):
        raise ValueError(
            f"Resume checkpoint is already at epoch {start_epoch - 1}; "
            "increase training.max_epochs to continue"
        )

    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifacts = RunArtifacts(Path(config["output"]["root"]), identifier)
    artifacts.write_config(config)
    data_path = Path(config["data"]["path"])
    artifacts.write_json(
        "run_manifest.json",
        build_manifest(
            config,
            data_path,
            prepared.graph.content_hash(),
            int(config["seed"]),
            split_manifest_sha256=prepared.split_manifest_sha256,
        ),
    )

    history: list[dict[str, Any]] = []
    stale_epochs = 0
    for epoch in range(start_epoch, int(training["max_epochs"])):
        model.train()
        epoch_losses: list[float] = []
        for raw_batch in prepared.train_loader:
            batch = _move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(batch["token_ids"], batch["numeric"])
                loss, _, _ = _losses(output.answer, output.expression_logits, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(float(loss.detach()))
        validation_loss, _ = _evaluate_loader(model, prepared.validation_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": sum(epoch_losses) / max(1, len(epoch_losses)),
                "validation_loss": validation_loss,
            }
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            stale_epochs = 0
            best_checkpoint = _checkpoint_snapshot(
                model,
                optimizer,
                scaler,
                epoch=epoch,
                validation_loss=validation_loss,
            )
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stopping_patience"]):
            break

    if best_checkpoint is None:
        raise RuntimeError("Training did not produce a checkpointable epoch")
    model.load_state_dict(best_checkpoint["model"])
    optimizer.load_state_dict(best_checkpoint["optimizer"])
    scaler.load_state_dict(best_checkpoint["scaler"])
    test_loss, predictions = _evaluate_loader(model, prepared.test_loader, device, True)
    metrics = evaluate_records(
        predictions,
        primary_split=str(config["task"]["primary_split"]),
        primary_metric_name=str(config["task"]["primary_metric_name"]),
        **config["task"]["metrics"],
    )
    metrics["test_loss"] = test_loss
    artifacts.write_history(history)
    artifacts.write_predictions(predictions)
    artifacts.write_json("metrics.json", metrics)
    if bool(training["checkpoints"]):
        torch.save(best_checkpoint, artifacts.path / "checkpoint.pt")
    artifacts.write_json(
        "summary.json",
        {
            "status": "completed",
            "run_id": identifier,
            "epochs_completed": len(history),
            "best_epoch": int(best_checkpoint["epoch"]),
            "best_validation_loss": best_validation,
            "primary_metric": metrics["primary_metric"],
            "parameter_counts": model.parameter_counts(),
        },
    )
    return artifacts.path
