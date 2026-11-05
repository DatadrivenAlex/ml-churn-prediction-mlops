"""Seeded synthetic subscription customers with a noisy, known churn process.

The table is shaped like a telecom/SaaS billing extract: contract, tenure,
charges, support load, and a few service add-ons. Labels are Bernoulli draws
from a logistic score of those fields, plus Gaussian noise, so the problem is
learnable and not separable. ``scenario="shifted"`` keeps that same labeling
rule and changes the covariate mix, which is what the PSI check is for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from churn.columns import (
    CONTRACTS,
    ID_COLUMN,
    INTERNET_SERVICES,
    PAYMENT_METHODS,
    REGIONS,
    TARGET,
)

SCENARIOS = ("baseline", "shifted")

OUTPUT_COLUMNS = [
    ID_COLUMN,
    "tenure_months",
    "contract",
    "internet_service",
    "payment_method",
    "monthly_charges",
    "total_charges",
    "paperless_billing",
    "senior_citizen",
    "partner",
    "dependents",
    "online_security",
    "tech_support",
    "streaming_tv",
    "support_tickets_90d",
    "avg_monthly_gb",
    "satisfaction_score",
    "autopay",
    "region",
    TARGET,
]


def generate_churn_data(
    n_rows: int = 8000,
    seed: int = 42,
    scenario: str = "baseline",
) -> pd.DataFrame:
    """Return a deterministic customer table for ``seed`` and ``scenario``."""
    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")

    rng = np.random.default_rng(seed)
    shifted = scenario == "shifted"

    contract = _choice(
        rng,
        CONTRACTS,
        [0.80, 0.13, 0.07] if shifted else [0.54, 0.24, 0.22],
        n_rows,
    )
    internet = _choice(
        rng,
        INTERNET_SERVICES,
        [0.25, 0.62, 0.13] if shifted else [0.34, 0.44, 0.22],
        n_rows,
    )
    payment = _choice(
        rng,
        PAYMENT_METHODS,
        [0.62, 0.14, 0.12, 0.12] if shifted else [0.34, 0.22, 0.22, 0.22],
        n_rows,
    )
    region = _choice(rng, REGIONS, [0.25, 0.25, 0.25, 0.25], n_rows)

    tenure_shape, tenure_scale = (1.6, 8.0) if shifted else (2.2, 12.0)
    tenure = np.clip(rng.gamma(tenure_shape, tenure_scale, n_rows), 1, 72).round().astype(int)

    senior = rng.random(n_rows) < 0.16
    partner = rng.random(n_rows) < 0.48
    dependents = partner & (rng.random(n_rows) < 0.45)
    paperless = rng.random(n_rows) < (0.72 if shifted else 0.58)
    autopay = rng.random(n_rows) < (0.32 if shifted else 0.57)

    no_internet = internet == "none"
    security_yes = rng.random(n_rows) < (0.22 if shifted else 0.36)
    support_yes = rng.random(n_rows) < (0.20 if shifted else 0.34)
    streaming_yes = rng.random(n_rows) < 0.40
    online_security = np.where(no_internet, "no_internet", np.where(security_yes, "yes", "no"))
    tech_support = np.where(no_internet, "no_internet", np.where(support_yes, "yes", "no"))
    streaming_tv = np.where(no_internet, "no_internet", np.where(streaming_yes, "yes", "no"))

    ticket_rate = 2.4 if shifted else 1.05
    tickets = rng.poisson(ticket_rate, n_rows).astype(int)

    fiber = internet == "fiber"
    dsl = internet == "dsl"
    monthly = (
        28.0
        + np.where(fiber, 42.0, 0.0)
        + np.where(dsl, 16.0, 0.0)
        + np.where(streaming_tv == "yes", 12.0, 0.0)
        + np.where(online_security == "yes", 6.0, 0.0)
        + np.where(tech_support == "yes", 7.0, 0.0)
        + rng.normal(0.0, 6.5, n_rows)
    )
    if shifted:
        monthly = monthly * 1.18 + 4.0
    monthly = np.round(np.clip(monthly, 18.0, None), 2)

    list_total = monthly * tenure
    total = list_total * rng.uniform(0.96, 1.03, n_rows)
    credit = rng.random(n_rows) < (0.14 if shifted else 0.07)
    total = np.where(credit, total * rng.uniform(0.55, 0.90, n_rows), total)
    total = np.round(total, 2)

    gb = np.where(
        fiber,
        rng.normal(150 if shifted else 210, 45, n_rows),
        np.where(dsl, rng.normal(75, 22, n_rows), rng.normal(1.5, 0.8, n_rows)),
    )
    gb = np.round(np.clip(gb, 0.0, None), 1)

    satisfaction = _choice(
        rng,
        (1, 2, 3, 4, 5),
        [0.24, 0.30, 0.24, 0.15, 0.07] if shifted else [0.08, 0.14, 0.28, 0.32, 0.18],
        n_rows,
    ).astype(int)

    logit = (
        -1.35
        + 1.20 * (contract == "month_to_month")
        + 0.32 * (contract == "one_year")
        - 0.030 * tenure
        + 0.52 * (payment == "electronic_check")
        + 0.38 * fiber
        + 0.16 * tickets
        - 0.40 * (satisfaction - 3)
        + 0.010 * (monthly - 70.0)
        - 0.50 * autopay.astype(float)
        - 0.38 * (tech_support == "yes")
        - 0.22 * (online_security == "yes")
        + 0.22 * senior.astype(float)
        - 0.12 * partner.astype(float)
        + 0.40 * credit.astype(float)
        - 0.003 * np.where(no_internet, 0.0, gb - 80.0)
        + rng.normal(0.0, 0.55, n_rows)
    )
    probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, probability).astype(int)

    frame = pd.DataFrame(
        {
            ID_COLUMN: [f"C{index:06d}" for index in range(1, n_rows + 1)],
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "monthly_charges": monthly,
            "total_charges": total,
            "paperless_billing": paperless.astype(int),
            "senior_citizen": senior.astype(int),
            "partner": partner.astype(int),
            "dependents": dependents.astype(int),
            "online_security": online_security,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "support_tickets_90d": tickets,
            "avg_monthly_gb": gb,
            "satisfaction_score": satisfaction,
            "autopay": autopay.astype(int),
            "region": region,
            TARGET: churn,
        }
    )
    return frame[OUTPUT_COLUMNS]


def write_dataset(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` to CSV, creating parent directories as needed."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def _choice(
    rng: np.random.Generator,
    options: tuple,
    probabilities: list[float],
    n_rows: int,
) -> np.ndarray:
    return rng.choice(np.array(options, dtype=object), size=n_rows, p=probabilities)
