"""Safety-gated command line interface."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import psutil
import torch

from fly_abstraction import __version__
from fly_abstraction.config import (
    PROFILE_DEVICES,
    dump_config,
    load_config,
    local_safety_violations,
)
from fly_abstraction.data.adapters import TinyDatasetAdapter, UCIEnergyEfficiencyAdapter
from fly_abstraction.data.deepmind import prepare_deepmind_numeric
from fly_abstraction.data.downloads import download_registered_dataset
from fly_abstraction.data.registry import REGISTRY, get_dataset
from fly_abstraction.diagnostics import gpu_report, run_gpu_smoke
from fly_abstraction.graph.flywire import convert_local_flywire_export
from fly_abstraction.graph.flywire_v783 import (
    inspect_graph,
    preparation_plan,
    prepare_flywire_v783,
)
from fly_abstraction.resources import enforce_resource_guard, estimate_resources
from fly_abstraction.training.metrics import evaluate_records
from fly_abstraction.training.pipeline import train_experiment
from fly_abstraction.utils import read_jsonl


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise argparse.ArgumentTypeError(
            "Paths must be repository-relative and cannot contain parent traversal"
        )
    return path


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return load_config(args.profile, args.config)


def _resources(expected_bytes: int) -> dict[str, Any]:
    disk = shutil.disk_usage(Path.cwd())
    return {
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "disk_free_bytes": disk.free,
        "expected_operation_bytes": expected_bytes,
    }


def command_doctor(args: argparse.Namespace) -> int:
    del args
    checks = {
        "package_version": __version__,
        "python": platform.python_version(),
        "python_supported": sys.version_info[:2] == (3, 11),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sympy_importable": True,
        "working_directory": str(Path.cwd()),
        "resources": _resources(0),
    }
    print(json.dumps(checks, indent=2))
    return 0 if checks["python_supported"] else 1


def command_doctor_gpu(args: argparse.Namespace) -> int:
    del args
    report = gpu_report()
    print(json.dumps(report, indent=2))
    return 0 if report["blackwell_ready"] else 1


def command_smoke_gpu(args: argparse.Namespace) -> int:
    del args
    print(json.dumps(run_gpu_smoke(), indent=2))
    return 0


def command_show_config(args: argparse.Namespace) -> int:
    print(dump_config(_config(args)), end="")
    return 0


def command_data_list(args: argparse.Namespace) -> int:
    del args
    for key, record in sorted(REGISTRY.items()):
        external = " external-benchmark" if record.external_benchmark else ""
        print(
            f"{key}: {record.name} | {record.version} | {record.license} | "
            f"{record.expected_bytes} bytes{external}\n  {record.url}\n  {record.citation}"
            f"\n  checksum: {record.published_checksum or record.checksum_policy}"
        )
    return 0


def command_prepare_tiny(args: argparse.Namespace) -> int:
    path = args.output or Path("data/processed/tiny.jsonl")
    TinyDatasetAdapter.prepare(path, count=args.count, seed=args.seed)
    print(f"Prepared {args.count} deterministic examples at {path}")
    return 0


def command_data_download(args: argparse.Namespace) -> int:
    record = get_dataset(args.name)
    print(
        json.dumps(
            {"dataset": record.to_dict(), "resources": _resources(record.expected_bytes)}, indent=2
        )
    )
    if not args.confirm_download:
        print("DRY-RUN: no download performed; pass --confirm-download after reviewing metadata")
        return 0
    path, manifest = download_registered_dataset(record, args.output_root)
    print(f"Downloaded and hashed: {path}\nManifest: {manifest}")
    return 0


def command_prepare_uci(args: argparse.Namespace) -> int:
    UCIEnergyEfficiencyAdapter.prepare(args.source, args.output)
    print(f"Prepared finite numeric pilot records at {args.output}")
    return 0


def command_prepare_deepmind(args: argparse.Namespace) -> int:
    manifest = prepare_deepmind_numeric(args.source, args.output_dir)
    print(f"Prepared finite numeric DeepMind subsets; manifest: {manifest}")
    return 0


def command_convert_flywire(args: argparse.Namespace) -> int:
    convert_local_flywire_export(args.edges, args.output)
    print(f"Converted authorized local export to {args.output}")
    return 0


def command_prepare_flywire_v783(args: argparse.Namespace) -> int:
    plan = preparation_plan(
        args.source,
        args.output,
        nodes=args.nodes,
        min_pair_synapses=args.min_pair_synapses,
        input_node_count=args.input_node_count,
        output_node_count=args.output_node_count,
        seed=args.seed,
    )
    print(json.dumps(plan, indent=2))
    if not args.confirm_prepare:
        print("DRY RUN: no graph artifacts created; pass --confirm-prepare after review")
        return 0
    artifacts = prepare_flywire_v783(
        args.source,
        args.output,
        nodes=args.nodes,
        min_pair_synapses=args.min_pair_synapses,
        input_node_count=args.input_node_count,
        output_node_count=args.output_node_count,
        seed=args.seed,
    )
    print(json.dumps({key: path.as_posix() for key, path in artifacts.items()}, indent=2))
    return 0


def command_graph_inspect(args: argparse.Namespace) -> int:
    print(json.dumps(inspect_graph(args.graph), indent=2))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _config(args)
    estimate = estimate_resources(config)
    preview = {
        "mode": "confirmed" if args.confirm_train else "dry-run",
        "profile": config["profile"],
        "parameters": config,
        "resource_estimate": estimate.to_dict(),
        "results_root": config["output"]["root"],
    }
    print(json.dumps(preview, indent=2))
    if not args.confirm_train:
        print("DRY-RUN COMPLETE: training code was not entered and no result directory was created")
        return 0

    enforce_resource_guard(config, estimate, accepted=args.accept_resource_estimate)

    violations = local_safety_violations(config)
    if violations and not args.override_local_safety:
        raise RuntimeError(
            "local_cpu safety limits were exceeded; use --override-local-safety only after review: "
            + "; ".join(violations)
        )
    if config["profile"] == "paper_gpu":
        if not args.confirm_heavy_run:
            raise RuntimeError("paper_gpu requires --confirm-heavy-run")
        if not torch.cuda.is_available():
            raise RuntimeError("paper_gpu requires an available CUDA device")
    path = train_experiment(
        config,
        run_id=args.run_id,
        resume=args.resume,
        resource_estimate_accepted=args.accept_resource_estimate,
    )
    print(f"Completed run artifacts: {path}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    if args.predictions is None:
        print("DRY-RUN: pass --predictions results/<run_id>/predictions.jsonl to evaluate")
        return 0
    records = read_jsonl(args.predictions)
    config = _config(args)
    print(
        json.dumps(
            evaluate_records(
                records,
                primary_split=str(config["task"]["primary_split"]),
                primary_metric_name=str(config["task"]["primary_metric_name"]),
                **config["task"]["metrics"],
            ),
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fly_abstraction")
    parser.add_argument("--profile", choices=tuple(PROFILE_DEVICES), default="local_cpu")
    parser.add_argument("--config", type=_relative_path, help="optional YAML overrides")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect runtime without changing it")
    doctor.set_defaults(handler=command_doctor)
    doctor_gpu = commands.add_parser("doctor-gpu", help="verify CUDA, sm_120, VRAM, and AMP")
    doctor_gpu.set_defaults(handler=command_doctor_gpu)
    smoke_gpu = commands.add_parser(
        "smoke-gpu", help="one tiny CUDA forward/backward without an optimizer step"
    )
    smoke_gpu.set_defaults(handler=command_smoke_gpu)
    show = commands.add_parser("show-config", help="print the resolved YAML profile")
    show.set_defaults(handler=command_show_config)

    data = commands.add_parser("data")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    listing = data_commands.add_parser("list", help="show metadata-only registry")
    listing.set_defaults(handler=command_data_list)
    tiny = data_commands.add_parser("prepare-tiny", help="generate at most 100 local examples")
    tiny.add_argument("--count", type=int, default=100)
    tiny.add_argument("--seed", type=int, default=17)
    tiny.add_argument("--output", type=_relative_path)
    tiny.set_defaults(handler=command_prepare_tiny)
    download = data_commands.add_parser("download", help="review guarded real-data acquisition")
    download.add_argument("name", choices=tuple(sorted(REGISTRY)))
    download.add_argument("--confirm-download", action="store_true")
    download.add_argument("--output-root", type=_relative_path, default=Path("data/raw"))
    download.set_defaults(handler=command_data_download)
    uci = data_commands.add_parser("prepare-uci-energy", help="convert confirmed UCI CSV")
    uci.add_argument(
        "--source",
        type=_relative_path,
        default=Path("data/raw/uci_energy_efficiency/data.csv"),
    )
    uci.add_argument(
        "--output",
        type=_relative_path,
        default=Path("data/processed/uci_energy_efficiency.jsonl"),
    )
    uci.set_defaults(handler=command_prepare_uci)
    deepmind = data_commands.add_parser(
        "prepare-deepmind-numeric",
        help="filter a pre-downloaded official DeepMind v1.0 archive; no network access",
    )
    deepmind.add_argument("--source", type=_relative_path, required=True)
    deepmind.add_argument(
        "--output-dir",
        type=_relative_path,
        default=Path("data/processed/deepmind_mathematics_numeric_v1"),
    )
    deepmind.set_defaults(handler=command_prepare_deepmind)

    graph = commands.add_parser("graph")
    graph_commands = graph.add_subparsers(dest="graph_command", required=True)
    flywire = graph_commands.add_parser(
        "convert-flywire", help="convert an authorized local aggregated FlyWire CSV"
    )
    flywire.add_argument("--edges", type=_relative_path, required=True)
    flywire.add_argument(
        "--output", type=_relative_path, default=Path("data/processed/flywire_fafb_v783.npz")
    )
    flywire.set_defaults(handler=command_convert_flywire)
    prepare_flywire = graph_commands.add_parser(
        "prepare-flywire-v783",
        help="prepare a bounded deterministic core from the official local v783 Feather file",
    )
    prepare_flywire.add_argument(
        "--source",
        type=_relative_path,
        default=Path("data/raw/flywire_fafb_v783/proofread_connections_783.feather"),
    )
    prepare_flywire.add_argument(
        "--output",
        type=_relative_path,
        default=Path("data/processed/flywire_v783_core_1024.npz"),
    )
    prepare_flywire.add_argument("--nodes", type=int, default=1024)
    prepare_flywire.add_argument("--min-pair-synapses", type=float, default=5)
    prepare_flywire.add_argument("--input-node-count", type=int, default=64)
    prepare_flywire.add_argument("--output-node-count", type=int, default=64)
    prepare_flywire.add_argument("--seed", type=int, default=1701)
    prepare_flywire.add_argument("--confirm-prepare", action="store_true")
    prepare_flywire.set_defaults(handler=command_prepare_flywire_v783)
    inspect = graph_commands.add_parser("inspect", help="inspect a prepared graph read-only")
    inspect.add_argument("--graph", type=_relative_path, required=True)
    inspect.set_defaults(handler=command_graph_inspect)

    train = commands.add_parser("train", help="dry-run unless explicitly confirmed")
    train.add_argument("--confirm-train", action="store_true")
    train.add_argument("--confirm-heavy-run", action="store_true")
    train.add_argument("--accept-resource-estimate", action="store_true")
    train.add_argument("--override-local-safety", action="store_true")
    train.add_argument("--run-id")
    train.add_argument("--resume", type=_relative_path)
    train.set_defaults(handler=command_train)

    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--predictions", type=_relative_path)
    evaluate.set_defaults(handler=command_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileExistsError, FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
