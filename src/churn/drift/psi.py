"""Population Stability Index on numeric and categorical inputs.

Numeric features use quantile bins learned from the reference sample, with
open ends so new extremes still land in a bin. Categorical features compare
level shares. Unseen levels in the new batch collapse to ``__other__``.

Shares are smoothed by ``eps`` before the log so empty bins do not explode,
and so a feature compared with itself has PSI 0. Bands follow the usual
scorecard reading: below 0.10 stable, below 0.25 a minor shift, otherwise major.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.columns import DRIFT_FEATURES, RAW_BOOLEAN, RAW_CATEGORICAL, RAW_NUMERIC

PSI_MINOR = 0.10
PSI_MAJOR = 0.25
_EPS = 1e-6


def population_stability_index(
    expected: np.ndarray | list[float],
    actual: np.ndarray | list[float],
    eps: float = _EPS,
) -> float:
    """PSI between two non-negative distributions of the same length."""
    reference = np.asarray(expected, dtype=float)
    current = np.asarray(actual, dtype=float)
    if reference.ndim != 1 or current.ndim != 1 or reference.shape != current.shape:
        raise ValueError("expected and actual must be 1-d arrays of the same length")
    if reference.size == 0:
        raise ValueError("distributions must be non-empty")
    if eps <= 0:
        raise ValueError("eps must be positive")
    reference = np.clip(reference, 0, None) + eps
    current = np.clip(current, 0, None) + eps
    reference = reference / reference.sum()
    current = current / current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def build_baseline(frame: pd.DataFrame, n_bins: int = 10) -> dict:
    """Quantile edges and category shares from a reference frame (the train split)."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    _require_columns(frame)
    if len(frame) == 0:
        raise ValueError("Cannot build a drift baseline from zero rows")
    features: dict[str, dict] = {}
    for column in RAW_NUMERIC:
        values = _numeric_values(frame[column])
        edges = _quantile_edges(values, n_bins)
        features[column] = {
            "kind": "numeric",
            "edges": edges,
            "expected": _numeric_shares(values, edges),
        }
    for column in list(RAW_CATEGORICAL) + list(RAW_BOOLEAN):
        categories, shares = _categorical_reference(frame[column], boolean=column in RAW_BOOLEAN)
        features[column] = {
            "kind": "categorical",
            "categories": categories,
            "expected": shares,
        }
    return {"rows": int(len(frame)), "n_bins": n_bins, "features": features}


def score_drift(
    frame: pd.DataFrame,
    baseline: dict,
    minor: float = PSI_MINOR,
    major: float = PSI_MAJOR,
) -> dict:
    """Score each monitored feature and summarize the worst PSI."""
    _require_columns(frame)
    if len(frame) == 0:
        raise ValueError("Cannot score drift on zero rows")
    if major <= minor:
        raise ValueError("major PSI band must be greater than the minor band")
    rows: list[dict] = []
    for name in DRIFT_FEATURES:
        spec = baseline["features"][name]
        if spec["kind"] == "numeric":
            values = _numeric_values(frame[name])
            actual = _numeric_shares(values, spec["edges"])
        else:
            actual = _categorical_shares(
                frame[name],
                spec["categories"],
                boolean=name in RAW_BOOLEAN,
            )
        psi = population_stability_index(spec["expected"], actual)
        rows.append(
            {
                "feature": name,
                "kind": spec["kind"],
                "psi": psi,
                "status": _status(psi, minor, major),
            }
        )
    rows.sort(key=lambda item: item["psi"], reverse=True)
    max_psi = max(item["psi"] for item in rows)
    n_major = sum(item["status"] == "major_shift" for item in rows)
    n_shifted = sum(item["status"] != "stable" for item in rows)
    if n_major:
        recommendation = (
            "Review this batch before treating scores as comparable to training. "
            "At least one feature has a major marginal shift. "
            "This check does not retrain the model."
        )
    elif n_shifted:
        recommendation = (
            "No major marginal shift. Inspect the minor shifts before the next scheduled retrain."
        )
    else:
        recommendation = "Input marginals are stable against the reference baseline."
    return {
        "rows": int(len(frame)),
        "reference_rows": int(baseline.get("rows", 0)),
        "max_psi": max_psi,
        "n_minor_or_worse": n_shifted,
        "n_major": n_major,
        "recommendation": recommendation,
        "features": rows,
    }


def _status(psi: float, minor: float, major: float) -> str:
    if psi < minor:
        return "stable"
    if psi < major:
        return "minor_shift"
    return "major_shift"


def _require_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in DRIFT_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def _numeric_values(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise ValueError(f"Column '{series.name}' has no numeric values")
    return values


def _quantile_edges(values: np.ndarray, n_bins: int) -> list[float | str]:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles).astype(float))
    if edges.size < 2:
        center = float(values[0])
        edges = np.array([center - 1.0, center + 1.0])
    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    encoded: list[float | str] = []
    for edge in edges:
        if np.isneginf(edge):
            encoded.append("-inf")
        elif np.isposinf(edge):
            encoded.append("inf")
        else:
            encoded.append(float(edge))
    return encoded


def _as_edges(edges: list[float | str]) -> np.ndarray:
    converted: list[float] = []
    for edge in edges:
        if edge == "-inf":
            converted.append(-np.inf)
        elif edge == "inf":
            converted.append(np.inf)
        else:
            converted.append(float(edge))
    return np.asarray(converted, dtype=float)


def _numeric_shares(values: np.ndarray, edges: list[float | str]) -> list[float]:
    bins = _as_edges(edges)
    # digitize returns 1..n-1 for finite values when the ends are infinite.
    index = np.digitize(values, bins) - 1
    index = np.clip(index, 0, len(bins) - 2)
    counts = np.bincount(index, minlength=len(bins) - 1).astype(float)
    return (counts / counts.sum()).tolist()


def _as_category(series: pd.Series, boolean: bool) -> pd.Series:
    if boolean:
        if pd.api.types.is_bool_dtype(series):
            return series.astype(int).astype(str)
        numeric = pd.to_numeric(series, errors="coerce").fillna(0)
        return numeric.astype(int).astype(str)
    return series.astype(str)


def _categorical_reference(series: pd.Series, boolean: bool) -> tuple[list[str], list[float]]:
    values = _as_category(series, boolean)
    categories = sorted(values.unique().tolist())
    shares = _categorical_shares(series, categories, boolean)
    return categories, shares


def _categorical_shares(series: pd.Series, categories: list[str], boolean: bool) -> list[float]:
    values = _as_category(series, boolean)
    known = set(categories)
    collapsed = values.where(values.isin(list(known)), other="__other__")
    levels = list(categories) + ["__other__"]
    counts = collapsed.value_counts()
    total = float(len(collapsed))
    return [float(counts.get(level, 0)) / total for level in levels]
