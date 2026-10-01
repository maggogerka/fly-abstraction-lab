import argparse
from pathlib import Path

import pytest

from fly_abstraction.cli import _relative_path, main
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


def test_doctor_and_gpu_doctor_are_read_only(capsys) -> None:
    assert main(["doctor"]) == 0
    assert '"python"' in capsys.readouterr().out
    result = main(["doctor-gpu"])
    assert result in {0, 1}
    assert '"sm_120_compiled"' in capsys.readouterr().out


def test_registered_download_is_dry_without_confirmation(capsys) -> None:
    assert main(["data", "download", "uci_energy_efficiency"]) == 0
    assert "no download performed" in capsys.readouterr().out


def test_cli_paths_cannot_escape_repository() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        _relative_path("../outside")
