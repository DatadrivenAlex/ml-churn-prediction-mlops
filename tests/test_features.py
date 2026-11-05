"""Feature builder: imputation, domain features, and unknown categories."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import NotFittedError

from churn.columns import TARGET
from churn.data.generate import generate_churn_data
from churn.features.engineering import (
    ENGINEERED_CATEGORICAL,
    ENGINEERED_NUMERIC,
    ChurnFeatureBuilder,
    build_preprocessor,
)


def _raw(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[TARGET, "customer_id"])


def test_engineered_columns_and_ratios():
    frame = generate_churn_data(80, seed=2)
    builder = ChurnFeatureBuilder().fit(_raw(frame))
    row = _raw(frame).iloc[[0]].copy()
    row["tenure_months"] = 10
    row["monthly_charges"] = 100.0
    row["total_charges"] = 500.0
    row["support_tickets_90d"] = 4
    row["contract"] = "month_to_month"
    row["internet_service"] = "fiber"
    row["online_security"] = "no"
    row["tech_support"] = "no"
    row["streaming_tv"] = "yes"
    row["satisfaction_score"] = 1
    transformed = builder.transform(row)
    assert list(transformed.columns) == ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL
    assert builder.get_feature_names_out() == ENGINEERED_NUMERIC + ENGINEERED_CATEGORICAL
    assert transformed.loc[0, "charges_per_tenure"] == pytest.approx(50.0)
    assert transformed.loc[0, "billing_gap"] == pytest.approx(50.0)
    assert transformed.loc[0, "tickets_per_tenure"] == pytest.approx(0.4)
    assert transformed.loc[0, "contract_risk"] == pytest.approx(2.0)
    assert transformed.loc[0, "has_fiber"] == pytest.approx(1.0)
    assert transformed.loc[0, "unprotected"] == pytest.approx(1.0)
    assert transformed.loc[0, "addon_count"] == pytest.approx(1.0)
    assert transformed.loc[0, "new_customer"] == pytest.approx(0.0)
    assert transformed.loc[0, "low_satisfaction"] == pytest.approx(1.0)


def test_new_customer_flag_and_unknown_contract_risk():
    frame = generate_churn_data(40, seed=3)
    builder = ChurnFeatureBuilder().fit(_raw(frame))
    row = _raw(frame).iloc[[0]].copy()
    row["tenure_months"] = 3
    row["contract"] = "enterprise"
    transformed = builder.transform(row)
    assert transformed.loc[0, "new_customer"] == pytest.approx(1.0)
    assert transformed.loc[0, "contract_risk"] == pytest.approx(1.0)


def test_imputes_numeric_and_coerces_bool_strings():
    frame = generate_churn_data(40, seed=4)
    raw = _raw(frame)
    raw.loc[raw.index[0], "tenure_months"] = np.nan
    builder = ChurnFeatureBuilder().fit(raw)
    row = raw.iloc[[1]].copy()
    row["tenure_months"] = np.nan
    row["paperless_billing"] = "yes"
    row["autopay"] = "false"
    transformed = builder.transform(row)
    assert np.isfinite(transformed["tenure_months"]).all()
    assert transformed.loc[0, "paperless_billing"] == pytest.approx(1.0)
    assert transformed.loc[0, "autopay"] == pytest.approx(0.0)


def test_missing_column_and_unfitted_and_non_frame():
    frame = _raw(generate_churn_data(20, seed=5))
    builder = ChurnFeatureBuilder()
    with pytest.raises(NotFittedError):
        builder.transform(frame)
    with pytest.raises(ValueError, match="Missing columns"):
        builder.fit(frame.drop(columns=["region"]))
    with pytest.raises(TypeError):
        builder.fit(frame.to_numpy())


def test_preprocessor_accepts_unknown_category():
    frame = generate_churn_data(120, seed=6)
    raw = _raw(frame)
    pipeline = Pipeline(
        [
            ("features", ChurnFeatureBuilder()),
            ("preprocess", build_preprocessor()),
            ("identity", FunctionTransformer()),
        ]
    )
    pipeline.fit(raw)
    changed = raw.iloc[[0]].copy()
    changed["region"] = "central"
    encoded = pipeline.transform(changed)
    assert encoded.shape[0] == 1
    assert np.isfinite(encoded).all()
