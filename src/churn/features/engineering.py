"""Train-safe feature construction for raw subscription snapshots.

The builder imputes with training medians and adds a short list of ratios and
flags a tree or a linear model can use directly. Category encoding and scaling
live in the column transformer so they are fit on the training fold only.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from churn.columns import RAW_BOOLEAN, RAW_CATEGORICAL, RAW_NUMERIC, REQUIRED_COLUMNS

CONTRACT_RISK = {"month_to_month": 2.0, "one_year": 1.0, "two_year": 0.0}

ENGINEERED_NUMERIC = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_90d",
    "avg_monthly_gb",
    "satisfaction_score",
    "charges_per_tenure",
    "billing_gap",
    "tickets_per_tenure",
    "gb_per_dollar",
    "new_customer",
    "low_satisfaction",
    "contract_risk",
    "has_fiber",
    "unprotected",
    "addon_count",
    "senior_citizen",
    "paperless_billing",
    "partner",
    "dependents",
    "autopay",
]
ENGINEERED_CATEGORICAL = list(RAW_CATEGORICAL)


class ChurnFeatureBuilder(BaseEstimator, TransformerMixin):
    """Impute raw fields and append domain features. Does not look at the label."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> ChurnFeatureBuilder:
        frame = _require_frame(X)
        medians: dict[str, float] = {}
        for column in RAW_NUMERIC:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            median = float(numeric.median())
            medians[column] = 0.0 if pd.isna(median) else median
        self.numeric_medians_ = medians
        self.n_features_in_ = len(REQUIRED_COLUMNS)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "numeric_medians_")
        frame = _require_frame(X).copy()
        for column in RAW_NUMERIC:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = numeric.fillna(self.numeric_medians_[column]).astype(float)
        for column in RAW_BOOLEAN:
            frame[column] = _coerce_bool(frame[column])

        tenure = frame["tenure_months"].clip(lower=0)
        monthly = frame["monthly_charges"].clip(lower=0)
        charges_per_tenure = frame["total_charges"] / tenure.clip(lower=1.0)
        has_internet = frame["internet_service"].astype(str).ne("none")

        engineered = pd.DataFrame(index=frame.index)
        engineered["tenure_months"] = tenure
        engineered["monthly_charges"] = monthly
        engineered["total_charges"] = frame["total_charges"]
        engineered["support_tickets_90d"] = frame["support_tickets_90d"]
        engineered["avg_monthly_gb"] = frame["avg_monthly_gb"]
        engineered["satisfaction_score"] = frame["satisfaction_score"]
        engineered["charges_per_tenure"] = charges_per_tenure
        engineered["billing_gap"] = monthly - charges_per_tenure
        engineered["tickets_per_tenure"] = frame["support_tickets_90d"] / tenure.clip(lower=1.0)
        engineered["gb_per_dollar"] = frame["avg_monthly_gb"] / monthly.clip(lower=1.0)
        engineered["new_customer"] = tenure.lt(6).astype(float)
        engineered["low_satisfaction"] = frame["satisfaction_score"].le(2).astype(float)
        engineered["contract_risk"] = (
            frame["contract"].astype(str).map(CONTRACT_RISK).fillna(1.0).astype(float)
        )
        engineered["has_fiber"] = frame["internet_service"].astype(str).eq("fiber").astype(float)
        engineered["unprotected"] = (
            has_internet
            & frame["online_security"].astype(str).ne("yes")
            & frame["tech_support"].astype(str).ne("yes")
        ).astype(float)
        engineered["addon_count"] = (
            frame["online_security"].astype(str).eq("yes").astype(float)
            + frame["tech_support"].astype(str).eq("yes").astype(float)
            + frame["streaming_tv"].astype(str).eq("yes").astype(float)
        )
        for column in RAW_BOOLEAN:
            engineered[column] = frame[column].astype(float)
        for column in RAW_CATEGORICAL:
            engineered[column] = frame[column].astype(str).fillna("missing")
        return engineered[ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL].reset_index(drop=True)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        return ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric columns and one-hot encode categories, ignoring new levels."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ENGINEERED_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ENGINEERED_CATEGORICAL,
            ),
        ],
        remainder="drop",
    )


def _require_frame(X: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("ChurnFeatureBuilder expects a pandas DataFrame of raw customer fields")
    missing = [column for column in REQUIRED_COLUMNS if column not in X.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return X


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y"}).astype(int)
