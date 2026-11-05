"""The committed registry and reports are the run documented in the README."""

import json

import pandas as pd
import pytest

from churn.cli import _portable
from churn.config import project_root
from churn.registry.store import ModelRegistry


def test_portable_path_inside_the_repo():
    assert _portable(project_root() / "data" / "churn.csv") == "data/churn.csv"


def test_committed_champion_matches_the_training_report():
    root = project_root()
    model, metadata = ModelRegistry(root / "model_registry").load(alias="champion")
    report = json.loads((root / "reports" / "training_report.json").read_text(encoding="utf-8"))
    drift = json.loads((root / "reports" / "drift_report.json").read_text(encoding="utf-8"))

    assert metadata["version"] == "v001"
    assert metadata["model_name"] == "logistic_regression"
    assert report["champion"]["version"] == metadata["version"]
    assert report["champion"]["model_name"] == metadata["model_name"]
    assert metadata["metrics"]["test_roc_auc"] == pytest.approx(
        report["champion"]["metrics"]["test_roc_auc"]
    )
    assert report["rows"] == 8000
    assert report["seed"] == 42
    assert drift["rows"] == 8000
    assert drift["reference_rows"] == 6400
    assert drift["n_major"] == 9
    assert drift["reference"] == "model_registry alias=champion"
    assert drift["max_psi"] == pytest.approx(0.9882574665790824)

    frame = pd.read_csv(root / "data" / "churn.csv").drop(columns=["churn"]).head(1)
    probability = float(model.predict_proba(frame)[0, 1])
    assert probability == pytest.approx(0.17795492162382548, abs=1e-5)
