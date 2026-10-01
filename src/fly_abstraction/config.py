"""Configuration loading, validation, hashing, and local safety limits."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _discover_config_dir() -> Path:
    """Find repository configs in editable and regular-install executions."""
    source_checkout = Path(__file__).resolve().parents[2] / "configs"
    working_checkout = Path.cwd().resolve() / "configs"
    for candidate in (source_checkout, working_checkout):
        if (candidate / "base.yaml").is_file():
            return candidate
    # Retain the expected repository-root path for a useful later ConfigError.
    return working_checkout


CONFIG_DIR = _discover_config_dir()
LOCAL_LIMITS = {
    "data.max_examples": 500,
    "training.max_epochs": 1,
    "graph.max_graph_nodes": 128,
    "training.batch_size": 4,
    "training.num_workers": 0,
    "runtime.torch_threads": 2,
}
PROFILE_DEVICES = {
    "local_cpu": "cpu",
    "smoke_cpu": "cpu",
    "smoke_gpu": "cuda",
    "pilot_gpu": "cuda",
    "paper_gpu": "cuda",
}
GRAPH_VARIANTS = {
    "real",
    "degree_preserving",
    "weight_shuffled",
    "direction_shuffled",
    "er_random",
}


class ConfigError(ValueError):
    """Raised when a configuration is missing or unsafe."""


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Configuration does not exist: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"Configuration root must be a mapping: {path}")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(profile: str = "local_cpu", config_path: Path | None = None) -> dict[str, Any]:
    """Load base + named profile + optional user override and validate it."""
    config = _merge(
        _load_yaml(CONFIG_DIR / "base.yaml"),
        _load_yaml(CONFIG_DIR / f"{profile}.yaml"),
    )
    if config_path:
        config = _merge(config, _load_yaml(config_path))
    config["profile"] = profile
    validate_config(config)
    return config


def _get(config: dict[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        value = value[part]
    return value


def validate_config(config: dict[str, Any]) -> None:
    required = ["profile", "seed", "data", "graph", "model", "training", "runtime", "output"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ConfigError(f"Missing required keys: {', '.join(missing)}")
    for path in (
        "data.max_examples",
        "graph.max_graph_nodes",
        "training.max_epochs",
        "training.batch_size",
    ):
        if not isinstance(_get(config, path), int) or _get(config, path) <= 0:
            raise ConfigError(f"{path} must be a positive integer")
    if config["profile"] not in PROFILE_DEVICES:
        raise ConfigError(f"Unknown profile: {config['profile']}")
    expected_device = PROFILE_DEVICES[config["profile"]]
    if config["runtime"]["device"] != expected_device:
        raise ConfigError(f"{config['profile']} must use device={expected_device}")
    if config.get("task", {}).get("mode") != "numeric":
        raise ConfigError(
            "Only task.mode=numeric is implemented in the MVP; symbolic targets are future work"
        )
    variant = config["graph"].get("variant")
    if variant not in GRAPH_VARIANTS:
        raise ConfigError(
            f"Unknown graph.variant={variant!r}; choose one of {sorted(GRAPH_VARIANTS)}"
        )
    for path in (
        "data.sequence_length",
        "graph.expected_num_nodes",
        "graph.expected_num_edges",
    ):
        if not isinstance(_get(config, path), int) or _get(config, path) <= 0:
            raise ConfigError(f"{path} must be a positive integer")
    if int(config["graph"]["expected_num_nodes"]) > int(config["graph"]["max_graph_nodes"]):
        raise ConfigError("graph.expected_num_nodes cannot exceed graph.max_graph_nodes")
    for path in ("data.held_out_template_fraction", "data.validation_fraction"):
        if not 0 < float(_get(config, path)) < 1:
            raise ConfigError(f"{path} must be between zero and one")
    for path in ("resource_guard.max_ram_fraction", "resource_guard.max_vram_fraction"):
        if not 0 < float(_get(config, path)) <= 1:
            raise ConfigError(f"{path} must be in (0, 1]")
    metrics = config["task"]["metrics"]
    if float(metrics["absolute_tolerance"]) < 0 or float(metrics["relative_tolerance"]) < 0:
        raise ConfigError("Metric tolerances cannot be negative")
    if float(metrics["relative_epsilon"]) <= 0:
        raise ConfigError("task.metrics.relative_epsilon must be positive")
    if config["profile"] == "paper_gpu" and not config["resource_guard"].get(
        "require_explicit_acceptance"
    ):
        raise ConfigError("paper_gpu must require explicit resource-estimate acceptance")
    for path in ("data.path", "data.split_manifest_dir", "graph.path", "output.root"):
        value = Path(str(_get(config, path)))
        if value.is_absolute() or ".." in value.parts:
            raise ConfigError(f"{path} must remain inside the repository")


def local_safety_violations(config: dict[str, Any]) -> list[str]:
    if config["profile"] != "local_cpu":
        return []
    violations = [
        f"{path}={_get(config, path)} exceeds {limit}"
        for path, limit in LOCAL_LIMITS.items()
        if _get(config, path) > limit
    ]
    if config["runtime"]["mixed_precision"]:
        violations.append("runtime.mixed_precision must be false")
    if config["training"]["checkpoints"]:
        violations.append("training.checkpoints must be false")
    return violations


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def dump_config(config: dict[str, Any]) -> str:
    return yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
