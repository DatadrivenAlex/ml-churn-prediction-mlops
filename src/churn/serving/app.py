"""FastAPI app: health, single score, batch score, and champion metadata."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from churn import __version__
from churn.registry.store import ModelRegistry
from churn.serving.schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)


def create_app(registry_dir: Path | None = None) -> FastAPI:
    """Build an app bound to one registry directory."""
    resolved = _resolve_registry(registry_dir)
    model = None
    metadata = None
    try:
        model, metadata = ModelRegistry(resolved).load(alias="champion")
    except (LookupError, FileNotFoundError):
        model = None
        metadata = None

    app = FastAPI(
        title="Churn scoring API",
        version=__version__,
        description=(
            "Scores subscription customers with the champion pipeline in the local "
            "model registry. This is a portfolio service: it does not authenticate "
            "callers and it is not a retention decision system."
        ),
    )
    app.state.registry_dir = resolved
    app.state.model = model
    app.state.metadata = metadata

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        loaded = app.state.model is not None and app.state.metadata is not None
        version = app.state.metadata.get("version") if loaded else None
        return HealthResponse(status="ok", model_loaded=loaded, model_version=version)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        metadata = _require_model(app)
        return ModelInfoResponse(
            model_name=metadata["model_name"],
            version=metadata["version"],
            alias="champion",
            algorithm=metadata["algorithm"],
            threshold=float(metadata["threshold"]),
            trained_at=metadata["trained_at"],
            train_rows=int(metadata["train_rows"]),
            test_rows=int(metadata["test_rows"]),
            hyperparameters=metadata["hyperparameters"],
            metrics=metadata["metrics"],
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: CustomerFeatures) -> PredictionResponse:
        _require_model(app)
        frame = pd.DataFrame([payload.model_dump()])
        probability = _probabilities(app, frame)[0]
        return _prediction(payload.customer_id, probability, app.state.metadata)

    @app.post("/predict/batch", response_model=BatchPredictionResponse)
    def predict_batch(payload: BatchPredictRequest) -> BatchPredictionResponse:
        metadata = _require_model(app)
        frame = pd.DataFrame([customer.model_dump() for customer in payload.customers])
        probabilities = _probabilities(app, frame)
        predictions = [
            _prediction(customer.customer_id, probability, metadata)
            for customer, probability in zip(payload.customers, probabilities, strict=True)
        ]
        return BatchPredictionResponse(
            model_version=metadata["version"],
            threshold=float(metadata["threshold"]),
            count=len(predictions),
            predictions=predictions,
        )

    return app


def _resolve_registry(registry_dir: Path | None) -> Path:
    if registry_dir is not None:
        return Path(registry_dir)
    override = os.environ.get("CHURN_REGISTRY_DIR")
    if override:
        return Path(override)
    from churn.config import project_root

    return project_root() / "model_registry"


def _require_model(app: FastAPI) -> dict:
    if app.state.model is None or app.state.metadata is None:
        raise HTTPException(status_code=503, detail="No champion model is registered")
    return app.state.metadata


def _probabilities(app: FastAPI, frame: pd.DataFrame) -> list[float]:
    try:
        scores = app.state.model.predict_proba(frame)[:, 1]
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return [float(score) for score in scores]


def _prediction(customer_id: str | None, probability: float, metadata: dict) -> PredictionResponse:
    threshold = float(metadata["threshold"])
    return PredictionResponse(
        customer_id=customer_id,
        churn_probability=probability,
        churn_prediction=probability >= threshold,
        threshold=threshold,
        model_version=str(metadata["version"]),
    )


app = create_app()
