"""Implemented training/validation/test pipeline. Importing this module never starts it."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from fly_abstraction.data.adapters import TinyDatasetAdapter
from fly_abstraction.data.schema import MathProblem
from fly_abstraction.data.transforms import compositional_split, split_without_template_leakage
from fly_abstraction.graph.connectome import ConnectomeGraph, tiny_synthetic_graph
from fly_abstraction.graph.controls import erdos_renyi_matched
from fly_abstraction.graph.flywire import FlyWireFAFBV783Adapter
from fly_abstraction.models.base import MODEL_EXTENSIONS, MathModel, resolve_device
from fly_abstraction.models.baselines import GRUBaseline, MLPBaseline
from fly_abstraction.models.connectome import (
    FixedConnectomeReservoir,
    RandomGraphReservoir,
    TrainableConnectomeRNN,
)
from fly_abstraction.models.encoding import CharacterEncoder
from fly_abstraction.training.artifacts import RunArtifacts, build_manifest
from fly_abstraction.training.metrics import evaluate_records
from fly_abstraction.utils import read_jsonl, set_seeds


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


def _source_bucket(problem: MathProblem, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{problem.source_id}".encode()).digest()
    return int.from_bytes(digest[:2], "big") % 5


def _load_splits(
    config: dict[str, Any],
) -> tuple[list[MathProblem], list[MathProblem], list[MathProblem]]:
    path = Path(config["data"]["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Prepared data not found: {path}; run data prepare-tiny first")
    records = read_jsonl(path)[: config["data"]["max_examples"]]
    if config["data"]["dataset"] == "tiny":
        problems = [TinyDatasetAdapter().adapt(record) for record in records]
    else:
        problems = [MathProblem.from_dict(record) for record in records]
    train_pool, held_out = split_without_template_leakage(
        problems,
        float(config["data"]["held_out_template_fraction"]),
        int(config["seed"]),
    )
    validation = [item for item in train_pool if _source_bucket(item, config["seed"]) == 0]
    train = [item for item in train_pool if _source_bucket(item, config["seed"]) != 0]
    if not train or not validation or not held_out:
        raise ValueError(
            "Prepared data is too small for non-empty train/validation/held-out splits"
        )
    test = [compositional_split(item, config["seed"]) for item in held_out]
    return train, validation, test


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
        graph = (
            FlyWireFAFBV783Adapter()
            .load(Path(config["graph"]["path"]))
            .filtered(max_nodes=int(config["graph"]["max_graph_nodes"]))
        )
    else:
        raise ValueError(f"Unknown graph kind: {config['graph']['kind']}")
    graph = graph.normalized(str(config["graph"]["normalization"]))
    model = _build_model(config, graph)
    encoder = CharacterEncoder(
        vocab_size=int(config["model"]["vocab_size"]),
        max_length=96,
    )
    train, validation, test = _load_splits(config)
    loader_options = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
    }
    return PreparedExperiment(
        model=model,
        graph=graph,
        train_loader=DataLoader(EncodedDataset(train, encoder), shuffle=True, **loader_options),
        validation_loader=DataLoader(
            EncodedDataset(validation, encoder), shuffle=False, **loader_options
        ),
        test_loader=DataLoader(EncodedDataset(test, encoder), shuffle=False, **loader_options),
    )


def _losses(
    output_answer: torch.Tensor,
    output_expression: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    answer_loss = nn.functional.mse_loss(output_answer, batch["answer"])
    expression_loss = nn.functional.cross_entropy(
        output_expression.flatten(0, 1),
        batch["expression_targets"].flatten(),
        ignore_index=0,
    )
    return answer_loss + 0.1 * expression_loss, answer_loss, expression_loss


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


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
                raw_batch["index"].tolist(), output.answer.tolist(), strict=True
            ):
                problem = dataset.problems[row]
                records.append(
                    {
                        "source_id": problem.source_id,
                        "template_id": problem.template_id,
                        "split": problem.split,
                        "prediction": format(prediction, ".12g"),
                        "target": problem.answer,
                    }
                )
    return sum(losses) / max(1, len(losses)), records


def train_experiment(
    config: dict[str, Any], *, run_id: str | None = None, resume: Path | None = None
) -> Path:
    """Execute a confirmed run. Callers are responsible for CLI safety confirmation."""
    prepared = prepare_experiment(config)
    device = resolve_device(str(config["runtime"]["device"]))
    model = prepared.model.to(device)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    use_amp = bool(config["runtime"]["mixed_precision"]) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1

    identifier = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifacts = RunArtifacts(Path(config["output"]["root"]), identifier)
    artifacts.write_config(config)
    data_path = Path(config["data"]["path"])
    artifacts.write_json(
        "run_manifest.json",
        build_manifest(config, data_path, prepared.graph.content_hash(), int(config["seed"])),
    )

    history: list[dict[str, Any]] = []
    best_validation = float("inf")
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
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
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
        if stale_epochs >= int(training["early_stopping_patience"]):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, predictions = _evaluate_loader(model, prepared.test_loader, device, True)
    metrics = evaluate_records(predictions)
    metrics["test_loss"] = test_loss
    artifacts.write_history(history)
    artifacts.write_predictions(predictions)
    artifacts.write_json("metrics.json", metrics)
    if bool(training["checkpoints"]):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": history[-1]["epoch"],
            },
            artifacts.path / "checkpoint.pt",
        )
    artifacts.write_json(
        "summary.json",
        {
            "status": "completed",
            "run_id": identifier,
            "epochs_completed": len(history),
            "best_validation_loss": best_validation,
            "primary_metric": metrics["primary_metric"],
            "parameter_counts": model.parameter_counts(),
        },
    )
    return artifacts.path
