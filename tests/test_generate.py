"""Seeded generator: schema, reproducibility, and the shifted population."""

from pathlib import Path

import pandas as pd
import pytest

from churn.columns import ID_COLUMN, REQUIRED_COLUMNS, TARGET
from churn.data.generate import OUTPUT_COLUMNS, generate_churn_data, write_dataset


def test_baseline_schema_and_rate():
    frame = generate_churn_data(2000, seed=42)
    assert list(frame.columns) == OUTPUT_COLUMNS
    assert set(OUTPUT_COLUMNS) == {ID_COLUMN, *REQUIRED_COLUMNS, TARGET}
    assert frame[ID_COLUMN].is_unique
    assert set(frame[TARGET].unique()).issubset({0, 1})
    rate = float(frame[TARGET].mean())
    assert 0.15 < rate < 0.45
    assert frame.isna().sum().sum() == 0
    assert frame.loc[frame["internet_service"].eq("none"), "tech_support"].eq("no_internet").all()
    assert frame["monthly_charges"].ge(18).all()
    assert frame["tenure_months"].between(1, 72).all()


def test_same_seed_is_identical_and_other_seed_is_not():
    first = generate_churn_data(300, seed=7)
    second = generate_churn_data(300, seed=7)
    other = generate_churn_data(300, seed=8)
    pd.testing.assert_frame_equal(first, second)
    assert not first.equals(other)


def test_shifted_scenario_moves_contract_mix():
    baseline = generate_churn_data(3000, seed=3, scenario="baseline")
    shifted = generate_churn_data(3000, seed=3, scenario="shifted")
    base_share = float(baseline["contract"].eq("month_to_month").mean())
    shifted_share = float(shifted["contract"].eq("month_to_month").mean())
    assert shifted_share > base_share + 0.15
    assert float(shifted["monthly_charges"].mean()) > float(baseline["monthly_charges"].mean())
    assert not baseline.drop(columns=[ID_COLUMN]).equals(shifted.drop(columns=[ID_COLUMN]))


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        generate_churn_data(0, seed=1)
    with pytest.raises(ValueError):
        generate_churn_data(10, seed=1, scenario="future")


def test_write_dataset_creates_parent(tmp_path: Path):
    frame = generate_churn_data(20, seed=1)
    destination = tmp_path / "nested" / "customers.csv"
    write_dataset(frame, destination)
    loaded = pd.read_csv(destination)
    assert len(loaded) == 20
    assert TARGET in loaded.columns
