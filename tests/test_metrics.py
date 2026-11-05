"""Threshold selection and metric edge cases."""

import numpy as np
import pytest

from churn.training.metrics import evaluate_at_threshold, safe_roc_auc, select_threshold


def test_perfect_ranking_selects_cutoff_near_half():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    choice = select_threshold(labels, scores)
    assert choice.f1 == pytest.approx(1.0)
    assert choice.threshold == pytest.approx(0.5)


def test_tie_breaks_toward_half_then_lower_threshold():
    labels = np.array([0, 1])
    scores = np.array([0.1, 0.9])
    closer = select_threshold(labels, scores, thresholds=np.array([0.4, 0.9]))
    assert closer.threshold == pytest.approx(0.4)
    lower = select_threshold(labels, scores, thresholds=np.array([0.2, 0.8]))
    assert lower.threshold == pytest.approx(0.2)


def test_confusion_counts_at_threshold():
    stats = evaluate_at_threshold([0, 0, 1, 1], [0.2, 0.8, 0.7, 0.1], 0.5)
    assert stats["tp"] == pytest.approx(1)
    assert stats["fp"] == pytest.approx(1)
    assert stats["tn"] == pytest.approx(1)
    assert stats["fn"] == pytest.approx(1)
    assert stats["positive_rate"] == pytest.approx(0.5)
    assert 0.0 < stats["f1"] < 1.0


def test_metric_guards():
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.2])
    with pytest.raises(ValueError):
        select_threshold([], [])
    with pytest.raises(ValueError):
        select_threshold([0, 1], [0.2, 0.8], thresholds=np.array([]))
    with pytest.raises(ValueError):
        evaluate_at_threshold([0, 1], [0.2], 0.5)
    with pytest.raises(ValueError):
        evaluate_at_threshold([0, 1], [0.2, 0.8], 1.4)
    assert np.isnan(safe_roc_auc([0, 0, 0], [0.2, 0.4, 0.6]))
    assert safe_roc_auc([0, 1], [0.1, 0.9]) == pytest.approx(1.0)
