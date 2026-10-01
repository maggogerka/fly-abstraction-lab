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
from fly_abstraction.config import dump_config, load_config, local_safety_violations
from fly_abstraction.data.adapters import TinyDatasetAdapter
from fly_abstraction.data.registry import REGISTRY, get_dataset
from fly_abstraction.training.metrics import evaluate_records
from fly_abstraction.training.pipeline import train_experiment
from fly_abstraction.utils import read_jsonl


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise argparse.ArgumentTypeError("Paths must be repository-relative")
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
    raise RuntimeError(
        "Automatic real-data acquisition is intentionally adapter-specific and not "
        "enabled in the MVP; "
        "place a license-compliant local export under data/raw"
    )


def command_train(args: argparse.Namespace) -> int:
    config = _config(args)
    estimate = int(config["data"]["max_examples"]) * 65_536
    preview = {
        "mode": "confirmed" if args.confirm_train else "dry-run",
        "profile": config["profile"],
        "parameters": config,
        "resources": _resources(estimate),
        "results_root": config["output"]["root"],
    }
    print(json.dumps(preview, indent=2))
    if not args.confirm_train:
        print("DRY-RUN COMPLETE: training code was not entered and no result directory was created")
        return 0

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
    path = train_experiment(config, run_id=args.run_id, resume=args.resume)
    print(f"Completed run artifacts: {path}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    if args.predictions is None:
        print("DRY-RUN: pass --predictions results/<run_id>/predictions.jsonl to evaluate")
        return 0
    records = read_jsonl(args.predictions)
    print(json.dumps(evaluate_records(records), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fly_abstraction")
    parser.add_argument("--profile", choices=("local_cpu", "paper_gpu"), default="local_cpu")
    parser.add_argument("--config", type=_relative_path, help="optional YAML overrides")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect runtime without changing it")
    doctor.set_defaults(handler=command_doctor)
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
    download.set_defaults(handler=command_data_download)

    train = commands.add_parser("train", help="dry-run unless explicitly confirmed")
    train.add_argument("--confirm-train", action="store_true")
    train.add_argument("--confirm-heavy-run", action="store_true")
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
