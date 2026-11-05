"""File tracker, MLflow adapter, and the offline fallback."""

import builtins
import json
from pathlib import Path

import numpy as np
import pytest

from churn.tracking.tracker import (
    FileExperimentTracker,
    MLflowExperimentTracker,
    get_tracker,
)


class _Info:
    run_id = "abc123"


class _Run:
    info = _Info()


class FakeMLflow:
    def __init__(self):
        self.uri = None
        self.experiment = None
        self.run_name = None
        self.params = None
        self.metrics = None
        self.ended = False

    def set_tracking_uri(self, uri):
        self.uri = uri

    def set_experiment(self, name):
        self.experiment = name

    def start_run(self, run_name):
        self.run_name = run_name
        return _Run()

    def log_params(self, params):
        self.params = params

    def log_metrics(self, metrics):
        self.metrics = metrics

    def end_run(self):
        self.ended = True


def test_file_tracker_writes_a_finished_run(tmp_path: Path):
    tracker = FileExperimentTracker(tmp_path)
    with pytest.raises(RuntimeError):
        tracker.log_params({"seed": 1})
    run_id = tracker.start_run("logistic regression")
    tracker.log_params({"seed": np.int64(1), "note": Path("local")})
    tracker.log_metrics({"test_roc_auc": np.float64(0.8)})
    tracker.end_run()
    run_dir = tmp_path / run_id
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert meta["status"] == "finished"
    assert meta["backend"] == "file"
    assert meta["ended_at"]
    assert params["seed"] == 1
    assert metrics["test_roc_auc"] == pytest.approx(0.8)
    with pytest.raises(RuntimeError):
        tracker.log_metrics({"test_roc_auc": 0.1})
    second = tracker.start_run("gradient_boosting")
    assert second != run_id


def test_mlflow_adapter_with_injected_module():
    fake = FakeMLflow()
    tracker = MLflowExperimentTracker(tracking_uri="file:/tmp/mlruns", mlflow_module=fake)
    assert fake.experiment == "churn"
    with pytest.raises(RuntimeError):
        tracker.log_metrics({"test_f1": 0.5})
    assert tracker.start_run("gradient_boosting") == "abc123"
    tracker.log_params({"flags": ["a", "b"], "note": "x" * 800})
    tracker.log_metrics({"test_f1": 1})
    tracker.end_run()
    assert fake.ended
    assert fake.params["flags"] == '["a", "b"]'
    assert len(fake.params["note"]) == 500
    assert fake.metrics["test_f1"] == pytest.approx(1.0)
    assert isinstance(fake.metrics["test_f1"], float)


def test_get_tracker_backends(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    assert isinstance(get_tracker("file", tmp_path), FileExperimentTracker)
    monkeypatch.delenv("CHURN_TRACKING", raising=False)
    assert isinstance(get_tracker("auto", tmp_path), FileExperimentTracker)

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise ImportError("blocked for the offline test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="not installed"):
        get_tracker("mlflow", tmp_path)
    monkeypatch.setenv("CHURN_TRACKING", "mlflow")
    assert isinstance(get_tracker("auto", tmp_path), FileExperimentTracker)
    monkeypatch.setenv("CHURN_TRACKING", "somewhere-else")
    with pytest.raises(ValueError):
        get_tracker("auto", tmp_path)
    with pytest.raises(ValueError):
        get_tracker("s3", tmp_path)
