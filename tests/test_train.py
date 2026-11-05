"""Offline training loop, champion selection, and input guards."""

import json
from pathlib import Path

import pytest

from churn.data.generate import generate_churn_data
from churn.registry.store import ModelRegistry
from churn.tracking.tracker import FileExperimentTracker
from churn.training.train import TrainConfig, load_training_frame, run_training, write_report


def _fast(**overrides) -> TrainConfig:
    values = {"seed": 1, "cv_folds": 3, "gb_n_estimators": 5, "gb_max_depth": 2}
    values.update(overrides)
    return TrainConfig(**values)


def test_run_training_registers_both_models_and_promotes_the_better_auc(trained_bundle):
    report = trained_bundle["report"]
    registry: ModelRegistry = trained_bundle["registry"]
    assert {model["model_name"] for model in report["models"]} == {
        "logistic_regression",
        "gradient_boosting",
    }
    best = max(
        report["models"],
        key=lambda model: (model["metrics"]["cv_roc_auc_mean"], model["metrics"]["oof_f1"]),
    )
    assert report["champion"]["version"] == best["version"]
    assert report["champion"]["metrics"]["cv_roc_auc_mean"] > 0.6
    assert 0.0 < report["champion"]["threshold"] < 1.0
    for model in report["models"]:
        metrics = model["metrics"]
        assert 0.0 < metrics["test_roc_auc"] <= 1.0
        assert 0.0 <= metrics["test_f1"] <= 1.0
        assert len(metrics["cv_roc_auc_folds"]) == 3
        assert metrics["test_brier"] >= 0.0
    model, metadata = registry.load(alias="champion")
    assert metadata["version"] == best["version"]
    assert metadata["tracking_run_id"]
    row = trained_bundle["frame"].drop(columns=["churn"]).iloc[[0]]
    probability = float(model.predict_proba(row)[0, 1])
    assert 0.0 <= probability <= 1.0
    baseline_path = registry.root / best["version"] / "drift_baseline.json"
    baseline_text = baseline_path.read_text(encoding="utf-8")
    assert "Infinity" not in baseline_text
    assert "-inf" in baseline_text
    experiments = list((trained_bundle["tracker"].root).iterdir())
    assert len(experiments) == 2
    assert (experiments[0] / "metrics.json").is_file()


def test_training_guards(tmp_path: Path):
    with pytest.raises(ValueError):
        TrainConfig(cv_folds=1)
    with pytest.raises(ValueError):
        TrainConfig(test_size=0.8)
    with pytest.raises(ValueError):
        TrainConfig(gb_n_estimators=0)
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_training_frame(missing)
    frame = generate_churn_data(40, seed=2)
    csv_path = tmp_path / "no_target.csv"
    frame.drop(columns=["churn"]).to_csv(csv_path, index=False)
    with pytest.raises(ValueError):
        load_training_frame(csv_path)

    registry = ModelRegistry(tmp_path / "registry")
    tracker = FileExperimentTracker(tmp_path / "experiments")
    single = frame.copy()
    single["churn"] = 0
    with pytest.raises(ValueError, match="both classes"):
        run_training(single, _fast(cv_folds=2), registry, tracker)
    rare = frame.copy()
    rare["churn"] = 0
    rare.loc[rare.index[0], "churn"] = 1
    with pytest.raises(ValueError, match="minority"):
        run_training(rare, _fast(), registry, tracker)
    unlabeled = frame.drop(columns=["churn"])
    with pytest.raises(ValueError, match="churn"):
        run_training(unlabeled, _fast(cv_folds=2), registry, tracker)


def test_write_report_roundtrip(tmp_path: Path):
    destination = tmp_path / "nested" / "report.json"
    write_report({"champion": {"version": "v001"}, "rows": 3}, destination)
    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["champion"]["version"] == "v001"
    assert loaded["rows"] == 3
