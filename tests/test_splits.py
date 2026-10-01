import json
import random
from pathlib import Path

from fly_abstraction.data.adapters import TinyDatasetAdapter
from fly_abstraction.data.splits import build_split_manifest, materialize_split_manifest


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
