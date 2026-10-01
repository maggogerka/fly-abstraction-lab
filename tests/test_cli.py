from pathlib import Path

import pytest

from fly_abstraction.cli import main
from fly_abstraction.data.adapters import TinyDatasetAdapter


def test_train_defaults_to_side_effect_free_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["--profile", "local_cpu", "train"]) == 0
    assert "DRY-RUN COMPLETE" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


def test_tiny_prepare_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "tiny.jsonl"
    TinyDatasetAdapter.prepare(path, count=10, seed=17)
    with pytest.raises(FileExistsError):
        TinyDatasetAdapter.prepare(path, count=10, seed=17)


def test_evaluate_without_predictions_is_a_dry_run(capsys) -> None:
    assert main(["evaluate"]) == 0
    assert "DRY-RUN" in capsys.readouterr().out
