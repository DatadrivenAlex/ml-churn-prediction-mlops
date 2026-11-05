"""Request and response models for the scoring API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from churn.columns import (
    ADDON_VALUES,
    CONTRACTS,
    INTERNET_SERVICES,
    PAYMENT_METHODS,
    REGIONS,
)

Contract = Literal["month_to_month", "one_year", "two_year"]
InternetService = Literal["dsl", "fiber", "none"]
PaymentMethod = Literal["electronic_check", "mailed_check", "bank_transfer", "credit_card"]
Addon = Literal["yes", "no", "no_internet"]
Region = Literal["north", "south", "east", "west"]


class CustomerFeatures(BaseModel):
    """One customer snapshot. ``customer_id`` is echoed and is not a model feature."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = None
    tenure_months: int = Field(ge=0, le=240)
    contract: Contract
    internet_service: InternetService
    payment_method: PaymentMethod
    monthly_charges: float = Field(ge=0, le=1000)
    total_charges: float = Field(ge=0, le=100_000)
    paperless_billing: bool
    senior_citizen: bool
    partner: bool
    dependents: bool
    online_security: Addon
    tech_support: Addon
    streaming_tv: Addon
    support_tickets_90d: int = Field(ge=0, le=100)
    avg_monthly_gb: float = Field(ge=0, le=5000)
    satisfaction_score: int = Field(ge=1, le=5)
    autopay: bool
    region: Region


class PredictionResponse(BaseModel):
    """Score for one customer at the champion threshold."""

    customer_id: str | None = None
    churn_probability: float
    churn_prediction: bool
    threshold: float
    model_version: str


class BatchPredictRequest(BaseModel):
    """Up to 500 customers scored in one call."""

    model_config = ConfigDict(extra="forbid")

    customers: list[CustomerFeatures] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Batch scores in request order."""

    model_version: str
    threshold: float
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Process liveness. ``model_loaded`` is the readiness flag."""

    status: Literal["ok"]
    model_loaded: bool
    model_version: str | None = None


class ModelInfoResponse(BaseModel):
    """Champion metadata written at registration time."""

    model_name: str
    version: str
    alias: str
    algorithm: str
    threshold: float
    trained_at: str
    train_rows: int
    test_rows: int
    hyperparameters: dict[str, Any]
    metrics: dict[str, Any]


# Kept so a test can fail if the API literals drift from the generator.
SCHEMA_DOMAINS = {
    "contract": CONTRACTS,
    "internet_service": INTERNET_SERVICES,
    "payment_method": PAYMENT_METHODS,
    "online_security": ADDON_VALUES,
    "tech_support": ADDON_VALUES,
    "streaming_tv": ADDON_VALUES,
    "region": REGIONS,
}
