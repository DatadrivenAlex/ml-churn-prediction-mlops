"""Ranking metrics and a validation-set threshold for F1.

Thresholds are chosen on out-of-fold training scores. The test set is only
used after the threshold is fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ThresholdChoice:
    """Operating point selected on out-of-fold scores."""

    threshold: float
    f1: float
    precision: float
    recall: float


def select_threshold(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    thresholds: np.ndarray | None = None,
) -> ThresholdChoice:
    """Pick the threshold with the highest F1.

    Ties break toward the cutoff closest to 0.5, then toward the lower cutoff,
    so the choice stays stable when several nearby thresholds look the same.
    """
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if labels.shape[0] == 0:
        raise ValueError("Cannot select a threshold on an empty sample")
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 91), 2)

    best_key: tuple[float, float, float] | None = None
    best: ThresholdChoice | None = None
    for threshold in thresholds:
        stats = evaluate_at_threshold(labels, scores, float(threshold))
        key = (stats["f1"], -abs(float(threshold) - 0.5), -float(threshold))
        if best_key is None or key > best_key:
            best_key = key
            best = ThresholdChoice(
                threshold=float(threshold),
                f1=stats["f1"],
                precision=stats["precision"],
                recall=stats["recall"],
            )
    if best is None:
        raise ValueError("threshold grid is empty")
    return best


def evaluate_at_threshold(
    y_true: np.ndarray | list[int],
    y_score: np.ndarray | list[float],
    threshold: float,
) -> dict[str, float]:
    """Precision, recall, F1, and confusion counts at one probability cutoff."""
    labels = np.asarray(y_true).astype(int)
    scores = np.asarray(y_score, dtype=float)
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("y_true and y_score must have the same length")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    predicted = (scores >= float(threshold)).astype(int)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "positive_rate": float(predicted.mean()) if len(predicted) else 0.0,
        "tp": float(true_positive),
        "fp": float(false_positive),
        "tn": float(true_negative),
        "fn": float(false_negative),
    }


def safe_roc_auc(y_true: np.ndarray | list[int], y_score: np.ndarray | list[float]) -> float:
    """ROC-AUC, or ``nan`` when the slice contains a single class."""
    labels = np.asarray(y_true).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(y_score, dtype=float)))
