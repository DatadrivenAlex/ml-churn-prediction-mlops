"""Shared fixtures. Training fixtures stay small so the suite remains offline and quick."""

from __future__ import annotations

import pytest

from churn.data.generate import generate_churn_data
from churn.registry.store import ModelRegistry
from churn.tracking.tracker import FileExperimentTracker
from churn.training.train import TrainConfig, run_training


def fast_config(**overrides) -> TrainConfig:
    values = {
        "seed": 1,
        "cv_folds": 3,
        "gb_n_estimators": 20,
        "gb_max_depth": 2,
        "gb_learning_rate": 0.1,
    }
    values.update(overrides)
    return TrainConfig(**values)


@pytest.fixture(scope="module")
def trained_bundle(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("trained")
    frame = generate_churn_data(700, seed=1)
    registry = ModelRegistry(tmp_path / "registry")
    tracker = FileExperimentTracker(tmp_path / "experiments")
    report = run_training(frame, fast_config(), registry, tracker)
    return {"frame": frame, "registry": registry, "tracker": tracker, "report": report}
