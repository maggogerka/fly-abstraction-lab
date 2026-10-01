import json
from pathlib import Path

from fly_abstraction.cli import main
from fly_abstraction.utils import read_jsonl


def test_deepmind_numeric_cli_preserves_official_splits_and_filter_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "official"
    (source / "train-easy").mkdir(parents=True)
    (source / "interpolate").mkdir()
    (source / "train-easy" / "arithmetic__add.txt").write_text(
        "What is 1 + 2?\n3\nWhat is one third?\n1/3\nBad finite value?\nNaN\n",
        encoding="utf-8",
    )
    (source / "interpolate" / "numbers__round.txt").write_text(
        "Round 2.25.\n2.0\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "data",
                "prepare-deepmind-numeric",
                "--source",
                "official",
                "--output-dir",
                "prepared",
            ]
        )
        == 0
    )
    manifest = json.loads((tmp_path / "prepared" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["outputs"]) == {"train-easy", "interpolate"}
    assert manifest["official_splits_kept_separate"] is True
    assert manifest["accepted_records"] == 2
    assert manifest["rejected_records"] == 2
    assert manifest["rejection_reasons"] == {
        "non_finite_numeric": 1,
        "not_plain_numeric": 1,
    }
    assert manifest["source_commit"] == "427f45075f84b8b9774950196ad63867ca20ffb3"
    train = read_jsonl(tmp_path / "prepared" / "train-easy.jsonl")
    interpolation = read_jsonl(tmp_path / "prepared" / "interpolate.jsonl")
    assert {row["split"] for row in train} == {"train-easy"}
    assert {row["split"] for row in interpolation} == {"interpolate"}
    assert len(read_jsonl(tmp_path / "prepared" / "rejected.jsonl")) == 2
