"""Day 2 API tests.

We mock the model loader so tests don't require a running MLflow server.
On Day 4 the CI workflow will run these in an isolated container.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """TestClient with a mocked model bundle."""
    fake_model = MagicMock()
    fake_model.predict_proba.return_value = np.array([[0.2, 0.8]])

    from app.model_loader import LoadedModel

    fake_bundle = LoadedModel(
        model=fake_model,
        name="fraud-detector",
        version="2",
        alias="staging",
        run_id="fake-run-id",
    )

    with patch("app.main.load_model_by_alias", return_value=fake_bundle):
        from app.main import app

        with TestClient(app) as c:
            yield c


def _valid_payload() -> dict:
    return {
        "amount": 142.50,
        "hour_of_day": 14,
        "merchant_risk_score": 0.3,
        "user_txn_count_24h": 5,
        "avg_txn_amount_30d": 87.20,
        "distance_from_home_km": 12.5,
        "is_foreign_txn": False,
        "card_age_days": 380,
    }


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "fraud-detector"
    assert body["model_version"] == "2"
    assert body["model_alias"] == "staging"


def test_predict_happy_path(client):
    r = client.post("/predict", json=_valid_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1  # 0.8 >= 0.5
    assert body["fraud_probability"] == pytest.approx(0.8)
    assert body["model_version"] == "2"
    assert body["model_alias"] == "staging"


def test_predict_rejects_missing_field(client):
    payload = _valid_payload()
    del payload["amount"]
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_extra_field(client):
    payload = _valid_payload()
    payload["surprise_field"] = "boo"
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_rejects_out_of_range(client):
    payload = _valid_payload()
    payload["merchant_risk_score"] = 1.5  # > 1
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_metrics_endpoint_exists(client):
    # Hit predict first so the counter has something
    client.post("/predict", json=_valid_payload())
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "fraud_predictions_total" in r.text
