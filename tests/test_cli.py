"""CLI commands write artifacts and return exit codes without a network."""

import json
import os
from pathlib import Path

import pytest

from churn.cli import main
from churn.data.generate import generate_churn_data, write_dataset


def test_help_exits_zero():
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_generate_and_reject_zero_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "customers.csv"
    assert main(["generate", "--rows", "50", "--seed", "2", "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "50" in text
    assert output.is_file()
    assert main(["generate", "--rows", "0", "--output", str(tmp_path / "nope.csv")]) == 1
    assert "error:" in capsys.readouterr().err


def test_train_and_drift_commands(tmp_path: Path):
    data = tmp_path / "train.csv"
    write_dataset(generate_churn_data(360, seed=2), data)
    current = tmp_path / "shifted.csv"
    write_dataset(generate_churn_data(200, seed=2, scenario="shifted"), current)
    registry = tmp_path / "registry"
    experiments = tmp_path / "experiments"
    report = tmp_path / "training_report.json"
    code = main(
        [
            "train",
            "--data",
            str(data),
            "--registry",
            str(registry),
            "--experiments",
            str(experiments),
            "--report",
            str(report),
            "--cv-folds",
            "3",
            "--estimators",
            "12",
            "--seed",
            "2",
            "--tracking",
            "file",
        ]
    )
    assert code == 0
    summary = json.loads(report.read_text(encoding="utf-8"))
    assert summary["champion"]["version"].startswith("v")
    assert any(path.name == "metrics.json" for path in experiments.rglob("metrics.json"))

    drift_report = tmp_path / "drift.json"
    drift_code = main(
        [
            "drift",
            "--current",
            str(current),
            "--registry",
            str(registry),
            "--output",
            str(drift_report),
        ]
    )
    assert drift_code == 0
    drift = json.loads(drift_report.read_text(encoding="utf-8"))
    assert drift["rows"] == 200
    assert "max_psi" in drift

    reference_report = tmp_path / "drift_reference.json"
    assert (
        main(
            [
                "drift",
                "--current",
                str(current),
                "--reference",
                str(data),
                "--output",
                str(reference_report),
            ]
        )
        == 0
    )
    assert json.loads(reference_report.read_text(encoding="utf-8"))["reference"] == str(data)


def test_train_missing_file_and_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    missing = tmp_path / "missing.csv"
    assert main(["train", "--data", str(missing), "--registry", str(tmp_path)]) == 1
    captured: dict = {}

    def fake_run(target, host, port):
        captured["target"] = target
        captured["host"] = host
        captured["port"] = port

    import uvicorn

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.delenv("CHURN_REGISTRY_DIR", raising=False)
    code = main(["serve", "--host", "127.0.0.1", "--port", "9002"])
    assert code == 0
    assert captured["port"] == 9002
    code = main(
        ["serve", "--host", "0.0.0.0", "--port", "9001", "--registry", str(tmp_path / "registry")]
    )
    assert code == 0
    assert captured == {
        "target": "churn.serving.app:app",
        "host": "0.0.0.0",
        "port": 9001,
    }
    assert os.environ["CHURN_REGISTRY_DIR"] == str(tmp_path / "registry")


def test_unknown_tracking_choice_is_an_argparse_error(tmp_path: Path):
    data = tmp_path / "train.csv"
    write_dataset(generate_churn_data(20, seed=1), data)
    with pytest.raises(SystemExit) as caught:
        main(["train", "--data", str(data), "--tracking", "nope"])
    assert caught.value.code == 2
