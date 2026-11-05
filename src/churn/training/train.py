"""Compare logistic regression and gradient boosting, then register both."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from churn.columns import ID_COLUMN, TARGET
from churn.drift.psi import build_baseline
from churn.features.engineering import ChurnFeatureBuilder, build_preprocessor
from churn.registry.store import ModelRegistry
from churn.tracking.tracker import ExperimentTracker
from churn.training.metrics import evaluate_at_threshold, select_threshold
from churn.utils import jsonable, utc_now


@dataclass
class TrainConfig:
    """Knobs for the offline training entrypoint. Tests use a smaller forest."""

    seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    gb_n_estimators: int = 120
    gb_learning_rate: float = 0.08
    gb_max_depth: int = 3

    def __post_init__(self) -> None:
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if not 0.0 < self.test_size < 0.5:
            raise ValueError("test_size must be between 0 and 0.5")
        if self.gb_n_estimators < 1:
            raise ValueError("gb_n_estimators must be positive")


def load_training_frame(path: Path) -> pd.DataFrame:
    """Read a customer CSV produced by the generator (or the same schema)."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Training data not found: {source}")
    frame = pd.read_csv(source)
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must include a '{TARGET}' column")
    return frame


def run_training(
    frame: pd.DataFrame,
    config: TrainConfig,
    registry: ModelRegistry,
    tracker: ExperimentTracker,
) -> dict:
    """Fit both candidates, log them, and promote the better CV ROC-AUC."""
    features, labels = _split_features_labels(frame)
    if labels.nunique() < 2:
        raise ValueError("Training data must contain both classes")
    minority = int(labels.value_counts().min())
    if minority < config.cv_folds:
        raise ValueError(
            f"The minority class has {minority} rows, which is fewer than cv_folds="
            f"{config.cv_folds}"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=config.test_size,
        random_state=config.seed,
        stratify=labels,
    )
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    baseline = build_baseline(x_train)

    model_reports: list[dict] = []
    for name, estimator in _candidates(config).items():
        model_reports.append(
            _fit_one(
                name=name,
                estimator=estimator,
                config=config,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                baseline=baseline,
                registry=registry,
                tracker=tracker,
            )
        )

    champion = max(
        model_reports,
        key=lambda item: (item["metrics"]["cv_roc_auc_mean"], item["metrics"]["oof_f1"]),
    )
    registry.promote(champion["version"], alias="champion")
    champion = {**champion, "alias": "champion"}

    return {
        "seed": config.seed,
        "rows": int(len(frame)),
        "churn_rate": float(labels.mean()),
        "test_size": config.test_size,
        "cv_folds": config.cv_folds,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "trained_at": utc_now(),
        "models": model_reports,
        "champion": {
            "model_name": champion["model_name"],
            "version": champion["version"],
            "reason": "highest mean stratified CV ROC-AUC; ties break on out-of-fold F1",
            "metrics": champion["metrics"],
            "threshold": champion["threshold"],
        },
    }


def write_report(report: dict, path: Path) -> None:
    """Persist the training summary as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")


def _candidates(config: TrainConfig) -> dict:
    return {
        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=config.seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=config.gb_n_estimators,
            learning_rate=config.gb_learning_rate,
            max_depth=config.gb_max_depth,
            subsample=0.9,
            random_state=config.seed,
        ),
    }


def _fit_one(
    name: str,
    estimator,
    config: TrainConfig,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    baseline: dict,
    registry: ModelRegistry,
    tracker: ExperimentTracker,
) -> dict:
    template = _pipeline(estimator)
    folder = StratifiedKFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    oof = np.zeros(len(y_train), dtype=float)
    fold_aucs: list[float] = []
    for train_index, valid_index in folder.split(x_train, y_train):
        fold_model = clone(template)
        fold_model.fit(x_train.iloc[train_index], y_train.iloc[train_index])
        probabilities = fold_model.predict_proba(x_train.iloc[valid_index])[:, 1]
        oof[valid_index] = probabilities
        fold_aucs.append(float(roc_auc_score(y_train.iloc[valid_index], probabilities)))

    choice = select_threshold(y_train.to_numpy(), oof)
    model = clone(template)
    model.fit(x_train, y_train)
    test_scores = model.predict_proba(x_test)[:, 1]
    test_stats = evaluate_at_threshold(y_test.to_numpy(), test_scores, choice.threshold)
    hyperparameters = _hyperparameters(estimator)

    metrics = {
        "cv_roc_auc_mean": float(np.mean(fold_aucs)),
        "cv_roc_auc_std": float(np.std(fold_aucs)),
        "cv_roc_auc_folds": [float(score) for score in fold_aucs],
        "oof_f1": choice.f1,
        "oof_precision": choice.precision,
        "oof_recall": choice.recall,
        "test_roc_auc": float(roc_auc_score(y_test, test_scores)),
        "test_f1": test_stats["f1"],
        "test_precision": test_stats["precision"],
        "test_recall": test_stats["recall"],
        "test_positive_rate": test_stats["positive_rate"],
        "test_brier": float(brier_score_loss(y_test, test_scores)),
        "tp": test_stats["tp"],
        "fp": test_stats["fp"],
        "tn": test_stats["tn"],
        "fn": test_stats["fn"],
    }
    run_id = tracker.start_run(name)
    tracker.log_params(
        {
            "model_name": name,
            "seed": config.seed,
            "cv_folds": config.cv_folds,
            "test_size": config.test_size,
            "threshold": choice.threshold,
            **hyperparameters,
        }
    )
    tracker.log_metrics(
        {
            "cv_roc_auc_mean": metrics["cv_roc_auc_mean"],
            "cv_roc_auc_std": metrics["cv_roc_auc_std"],
            "oof_f1": metrics["oof_f1"],
            "test_roc_auc": metrics["test_roc_auc"],
            "test_f1": metrics["test_f1"],
            "test_precision": metrics["test_precision"],
            "test_recall": metrics["test_recall"],
            "test_brier": metrics["test_brier"],
        }
    )
    tracker.end_run()

    metadata = {
        "model_name": name,
        "algorithm": f"{type(estimator).__module__}.{type(estimator).__name__}",
        "seed": config.seed,
        "trained_at": utc_now(),
        "threshold": choice.threshold,
        "hyperparameters": hyperparameters,
        "metrics": metrics,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "tracking_run_id": run_id,
        "target": TARGET,
    }
    version = registry.register(model, metadata, baseline)
    return {
        "model_name": name,
        "version": version,
        "threshold": choice.threshold,
        "metrics": metrics,
        "tracking_run_id": run_id,
        "hyperparameters": hyperparameters,
    }


def _pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", ChurnFeatureBuilder()),
            ("preprocess", build_preprocessor()),
            ("model", estimator),
        ]
    )


def _hyperparameters(estimator) -> dict:
    interesting = (
        "C",
        "max_iter",
        "class_weight",
        "solver",
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
    )
    params = estimator.get_params()
    return {key: params[key] for key in interesting if key in params}


def _split_features_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in frame.columns:
        raise ValueError(f"Training data must include a '{TARGET}' column")
    labels = frame[TARGET].astype(int)
    features = frame.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    return features, labels
