"""Scoring API, including validation and a registry with no champion."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from churn.serving.app import create_app
from churn.serving.schemas import SCHEMA_DOMAINS, CustomerFeatures
from tests.samples import customer_payload, risky_payload


def test_schema_literals_match_the_generator_domain():
    for field, allowed in SCHEMA_DOMAINS.items():
        annotation = CustomerFeatures.model_fields[field].annotation
        assert set(annotation.__args__) == set(allowed)


def test_health_and_scoring_require_a_champion(tmp_path: Path):
    client = TestClient(create_app(tmp_path))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model_loaded": False, "model_version": None}
    assert client.post("/predict", json=customer_payload()).status_code == 503
    assert client.get("/model-info").status_code == 503


def test_predict_batch_and_model_info(trained_bundle):
    client = TestClient(create_app(trained_bundle["registry"].root))
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model_loaded"] is True
    version = health.json()["model_version"]

    single = client.post("/predict", json=customer_payload())
    assert single.status_code == 200
    body = single.json()
    assert body["model_version"] == version
    assert body["customer_id"] == "C000001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] == (body["churn_probability"] >= body["threshold"])

    risky = client.post("/predict", json=risky_payload())
    assert risky.status_code == 200
    assert risky.json()["churn_probability"] > body["churn_probability"]

    batch = client.post(
        "/predict/batch",
        json={"customers": [customer_payload(customer_id="A"), risky_payload()]},
    )
    assert batch.status_code == 200
    payload = batch.json()
    assert payload["count"] == 2
    assert payload["model_version"] == version
    assert [row["customer_id"] for row in payload["predictions"]] == ["A", "C999999"]

    info = client.get("/model-info")
    assert info.status_code == 200
    info_body = info.json()
    assert info_body["version"] == version
    assert info_body["alias"] == "champion"
    assert info_body["threshold"] == pytest.approx(body["threshold"])
    assert "test_roc_auc" in info_body["metrics"]
    assert info_body["train_rows"] > 0


def test_validation_errors(trained_bundle):
    client = TestClient(create_app(trained_bundle["registry"].root))
    rejected = customer_payload(contract="lifetime")
    assert client.post("/predict", json=rejected).status_code == 422
    negative = customer_payload(tenure_months=-1)
    assert client.post("/predict", json=negative).status_code == 422
    extra = customer_payload(unexpected=1)
    assert client.post("/predict", json=extra).status_code == 422
    assert client.post("/predict/batch", json={"customers": []}).status_code == 422


def test_registry_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CHURN_REGISTRY_DIR", str(tmp_path))
    from_env = create_app()
    assert from_env.state.registry_dir == tmp_path
    monkeypatch.delenv("CHURN_REGISTRY_DIR", raising=False)
    default = create_app()
    assert default.state.registry_dir.name == "model_registry"


def test_predict_turns_pipeline_errors_into_400(trained_bundle):
    app = create_app(trained_bundle["registry"].root)

    class Broken:
        def predict_proba(self, frame):
            raise ValueError("score failed")

    app.state.model = Broken()
    client = TestClient(app)
    response = client.post("/predict", json=customer_payload())
    assert response.status_code == 400
    assert "score failed" in response.json()["detail"]
