import json
import random
from pathlib import Path

from fly_abstraction.data.adapters import TinyDatasetAdapter, UCIEnergyEfficiencyAdapter
from fly_abstraction.data.splits import (
    apply_split_manifest,
    build_split_manifest,
    materialize_split_manifest,
)


def test_split_manifest_is_order_independent_and_reusable(tmp_path: Path) -> None:
    problems = TinyDatasetAdapter.generate(80, 17)
    kwargs = {
        "dataset": "tiny",
        "dataset_version": "1",
        "seed": 17,
        "held_out_template_fraction": 0.25,
        "validation_fraction": 0.2,
    }
    first = build_split_manifest(problems, **kwargs)
    shuffled = list(problems)
    random.Random(99).shuffle(shuffled)
    second = build_split_manifest(shuffled, **kwargs)
    assert first == second
    path = materialize_split_manifest(tmp_path, first)
    assert materialize_split_manifest(tmp_path, second) == path
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["sha256"] == first["sha256"]
    assert set(stored["splits"]) == {"train", "validation", "test_compositional_ood"}


def _uci_problems() -> list:
    problems = []
    for geometry in range(4):
        for orientation in range(4):
            record = {
                "X1": 0.5 + geometry,
                "X2": 100 + geometry,
                "X3": 50 + geometry,
                "X4": 25 + geometry,
                "X5": 3 + geometry,
                "X6": orientation + 2,
                "X7": 0.1,
                "X8": orientation,
                "Y1": 9000 + geometry * 10 + orientation,
                "Y2": 20 + geometry,
                "row_id": f"{geometry}-{orientation}",
            }
            problems.append(UCIEnergyEfficiencyAdapter().adapt(record))
    return problems


def test_uci_geometry_split_is_deterministic_and_never_injects_target() -> None:
    problems = _uci_problems()
    kwargs = {
        "dataset": "uci_energy_efficiency",
        "dataset_version": "UCI-242-2024-02-26",
        "seed": 1701,
        "held_out_template_fraction": 0.25,
        "validation_fraction": 0.2,
        "split_strategy": "held_out_geometry",
    }
    manifest = build_split_manifest(problems, **kwargs)
    shuffled = list(reversed(problems))
    assert build_split_manifest(shuffled, **kwargs) == manifest
    train, validation, test = apply_split_manifest(problems, manifest)
    originals = {problem.source_id: problem.prompt for problem in problems}
    assert set(manifest["splits"]) == {
        "train",
        "validation",
        "test_held_out_geometry_ood",
    }
    assert {item.template_id for item in train + validation}.isdisjoint(
        {item.template_id for item in test}
    )
    assert all(item.prompt == originals[item.source_id] for item in test)
    assert all(item.answer not in item.prompt for item in test)
    assert all(item.expression == "numeric_target_unexposed" for item in test)
    assert all(item.split == "test_held_out_geometry_ood" for item in test)
