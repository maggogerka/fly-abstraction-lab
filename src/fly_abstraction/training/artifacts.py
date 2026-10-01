"""Append-only run directories and reproducibility manifests."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psutil
import torch

from fly_abstraction.config import config_hash, dump_config
from fly_abstraction.utils import sha256_file


def _git(args: list[str]) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unavailable"


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def build_manifest(
    config: dict[str, Any], data_path: Path, graph_hash: str, seed: int
) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_dirty": bool(_git(["status", "--porcelain"]) not in {"", "unavailable"}),
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "ram_bytes": psutil.virtual_memory().total,
        "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "seed": seed,
        "versions": {
            "numpy": _package_version("numpy"),
            "pyyaml": _package_version("PyYAML"),
            "sympy": _package_version("sympy"),
        },
        "hashes": {
            "data": sha256_file(data_path),
            "connectome": graph_hash,
            "config": config_hash(config),
        },
    }


class RunArtifacts:
    def __init__(self, root: Path, run_id: str) -> None:
        self.path = root / run_id
        self.path.mkdir(parents=True, exist_ok=False)

    def write_json(self, name: str, payload: Any) -> None:
        path = self.path / name
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {path}")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def write_config(self, config: dict[str, Any]) -> None:
        path = self.path / "config.resolved.yaml"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {path}")
        path.write_text(dump_config(config), encoding="utf-8")

    def write_history(self, rows: list[dict[str, Any]]) -> None:
        path = self.path / "history.csv"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {path}")
        fields = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def write_predictions(self, records: list[dict[str, Any]]) -> None:
        path = self.path / "predictions.jsonl"
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite artifact: {path}")
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
