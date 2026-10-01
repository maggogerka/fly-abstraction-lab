"""Stable split manifests keyed by source IDs rather than row positions."""

from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

from fly_abstraction.data.schema import MathProblem
from fly_abstraction.data.transforms import compositional_split, held_out_template
from fly_abstraction.utils import sha256_json


def _rank(seed: int, namespace: str, value: str) -> str:
    return sha256_json({"seed": seed, "namespace": namespace, "value": value})


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")


def build_split_manifest(
    problems: list[MathProblem],
    *,
    dataset: str,
    dataset_version: str,
    seed: int,
    held_out_template_fraction: float,
    validation_fraction: float,
) -> dict[str, Any]:
    if not 0 < held_out_template_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must be between zero and one")
    ordered = sorted(problems, key=lambda item: item.source_id)
    ids = [item.source_id for item in ordered]
    if len(ids) != len(set(ids)):
        raise ValueError("source_id values must be unique before splitting")
    templates = sorted({item.template_id for item in ordered})
    if len(templates) < 2:
        raise ValueError("At least two templates are required for held-out splitting")
    ranked_templates = sorted(templates, key=lambda value: _rank(seed, "template", value))
    held_count = max(1, min(len(templates) - 1, round(len(templates) * held_out_template_fraction)))
    held_templates = set(ranked_templates[:held_count])
    test_ids = [item.source_id for item in ordered if item.template_id in held_templates]
    train_pool = [item.source_id for item in ordered if item.template_id not in held_templates]
    if len(train_pool) < 2 or not test_ids:
        raise ValueError("Dataset is too small for non-empty train/validation/test splits")
    ranked_train = sorted(train_pool, key=lambda value: _rank(seed, "validation", value))
    validation_count = max(
        1, min(len(ranked_train) - 1, round(len(ranked_train) * validation_fraction))
    )
    validation_ids = sorted(ranked_train[:validation_count])
    validation_set = set(validation_ids)
    train_ids = sorted(value for value in train_pool if value not in validation_set)
    rules = {
        "held_out": "whole template IDs ranked by SHA256(seed, template_id)",
        "held_out_template_fraction": held_out_template_fraction,
        "validation": "source IDs in the remaining templates ranked by SHA256(seed, source_id)",
        "validation_fraction": validation_fraction,
        "test_transform": "compositional_split",
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "dataset_version": dataset_version,
        "seed": seed,
        "rules": rules,
        "source_ids_sha256": sha256_json(ids),
        "splits": {
            "train": train_ids,
            "validation": validation_ids,
            "test_compositional_ood": sorted(test_ids),
        },
    }
    payload["sha256"] = sha256_json(payload)
    return payload


def materialize_split_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    rules_hash = sha256_json(manifest["rules"])[:12]
    name = (
        f"{_safe_name(str(manifest['dataset']))}-{_safe_name(str(manifest['dataset_version']))}"
        f"-seed{manifest['seed']}-{rules_hash}-{manifest['sha256'][:12]}.json"
    )
    path = directory / name
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != manifest or current.get("sha256") != sha256_json(
            {key: value for key, value in current.items() if key != "sha256"}
        ):
            raise ValueError(f"Existing split manifest does not match requested split: {path}")
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    os.replace(temporary, path)
    return path


def apply_split_manifest(
    problems: list[MathProblem], manifest: dict[str, Any]
) -> tuple[list[MathProblem], list[MathProblem], list[MathProblem]]:
    by_id = {item.source_id: item for item in problems}
    assigned_ids = [source_id for values in manifest["splits"].values() for source_id in values]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("Split manifest assigns a source ID more than once")
    expected = set(assigned_ids)
    if set(by_id) != expected:
        raise ValueError("Split manifest source IDs do not match the prepared dataset")
    seed = int(manifest["seed"])
    train = [replace(by_id[source_id], split="train") for source_id in manifest["splits"]["train"]]
    validation = [
        replace(by_id[source_id], split="validation")
        for source_id in manifest["splits"]["validation"]
    ]
    test = [
        compositional_split(held_out_template(by_id[source_id], seed), seed)
        for source_id in manifest["splits"]["test_compositional_ood"]
    ]
    return train, validation, test
