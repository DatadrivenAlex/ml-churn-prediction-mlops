"""PSI formula, identical-population baseline, and the shifted scenario."""

import json

import numpy as np
import pytest

from churn.data.generate import generate_churn_data
from churn.drift.psi import build_baseline, population_stability_index, score_drift


def test_psi_matches_smoothed_definition_and_grows_with_shift():
    expected = np.array([0.5, 0.5])
    actual = np.array([0.1, 0.9])
    eps = 1e-6
    smoothed_expected = expected + eps
    smoothed_actual = actual + eps
    smoothed_expected = smoothed_expected / smoothed_expected.sum()
    smoothed_actual = smoothed_actual / smoothed_actual.sum()
    manual = float(
        np.sum((smoothed_actual - smoothed_expected) * np.log(smoothed_actual / smoothed_expected))
    )
    assert population_stability_index(expected, actual, eps=eps) == pytest.approx(manual)
    assert population_stability_index(expected, expected) == pytest.approx(0.0)
    mild = population_stability_index([0.5, 0.5], [0.4, 0.6])
    strong = population_stability_index([0.5, 0.5], [0.1, 0.9])
    assert strong > mild > 0
    # Negative mass is clipped before smoothing, so the call still returns a finite PSI.
    assert population_stability_index([-0.2, 0.8], [0.5, 0.5]) >= 0


def test_psi_rejects_bad_inputs():
    with pytest.raises(ValueError):
        population_stability_index([0.5, 0.5], [1.0])
    with pytest.raises(ValueError):
        population_stability_index([], [])
    with pytest.raises(ValueError):
        population_stability_index([1.0], [1.0], eps=0)


def test_identical_frame_has_zero_psi_including_after_json_roundtrip():
    frame = generate_churn_data(400, seed=9).drop(columns=["churn", "customer_id"])
    baseline = build_baseline(frame)
    report = score_drift(frame, baseline)
    assert report["max_psi"] == pytest.approx(0.0)
    assert report["n_major"] == 0
    assert "stable" in report["recommendation"]
    restored = json.loads(json.dumps(baseline))
    assert "Infinity" not in json.dumps(restored)
    assert score_drift(frame, restored)["max_psi"] == pytest.approx(0.0)


def test_shifted_population_is_flagged():
    reference = generate_churn_data(2500, seed=4, scenario="baseline")
    current = generate_churn_data(2500, seed=4, scenario="shifted")
    features = reference.drop(columns=["churn", "customer_id"])
    report = score_drift(current.drop(columns=["churn", "customer_id"]), build_baseline(features))
    by_name = {row["feature"]: row for row in report["features"]}
    assert by_name["contract"]["psi"] > 0.1
    assert by_name["contract"]["status"] in {"minor_shift", "major_shift"}
    assert report["n_minor_or_worse"] >= 1
    assert report["max_psi"] == pytest.approx(report["features"][0]["psi"])
    assert "does not retrain" in report["recommendation"] or report["n_major"] == 0


def test_boolean_frame_matches_integer_encoding():
    frame = generate_churn_data(200, seed=5).drop(columns=["churn", "customer_id"])
    baseline = build_baseline(frame)
    booleaned = frame.copy()
    booleaned["autopay"] = booleaned["autopay"].astype(bool)
    assert score_drift(booleaned, baseline)["features"]
    scored = score_drift(booleaned, baseline)
    autopay = next(row for row in scored["features"] if row["feature"] == "autopay")
    assert autopay["psi"] == pytest.approx(0.0)


def test_constant_numeric_and_unseen_category():
    frame = generate_churn_data(120, seed=6).drop(columns=["churn", "customer_id"])
    frame["satisfaction_score"] = 3
    baseline = build_baseline(frame, n_bins=4)
    same = score_drift(frame, baseline)
    satisfaction = next(row for row in same["features"] if row["feature"] == "satisfaction_score")
    assert satisfaction["psi"] == pytest.approx(0.0)
    moved = frame.copy()
    moved["region"] = "central"
    scored = score_drift(moved, baseline)
    region = next(row for row in scored["features"] if row["feature"] == "region")
    assert region["status"] == "major_shift"


def test_drift_guards():
    frame = generate_churn_data(30, seed=7).drop(columns=["churn", "customer_id"])
    with pytest.raises(ValueError):
        build_baseline(frame, n_bins=1)
    with pytest.raises(ValueError):
        build_baseline(frame.iloc[0:0])
    with pytest.raises(ValueError):
        score_drift(frame.iloc[0:0], build_baseline(frame))
    with pytest.raises(ValueError, match="Missing columns"):
        build_baseline(frame.drop(columns=["region"]))
    broken = frame.copy()
    broken["tenure_months"] = np.nan
    with pytest.raises(ValueError, match="no numeric"):
        build_baseline(broken)
    with pytest.raises(ValueError):
        score_drift(frame, build_baseline(frame), minor=0.3, major=0.2)


def test_minor_shift_recommendation():
    frame = generate_churn_data(80, seed=8).drop(columns=["churn", "customer_id"])
    baseline = build_baseline(frame)
    tweaked = frame.copy()
    tweaked.loc[tweaked.index[:8], "contract"] = "two_year"
    # Widen the major band so a real but moderate move stays in the minor bucket.
    report = score_drift(tweaked, baseline, minor=1e-9, major=10)
    assert report["n_major"] == 0
    assert report["n_minor_or_worse"] > 0
    assert "minor" in report["recommendation"].lower()
