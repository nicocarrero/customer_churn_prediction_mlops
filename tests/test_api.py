import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.api import app, risk_label

client = TestClient(app)

VALID_PAYLOAD = {
    "tenure_months": 12,
    "monthly_charge": 65.5,
    "total_charges": 786.0,
    "support_tickets": 1,
    "late_payments": 0,
    "avg_monthly_usage_gb": 95.3,
    "contract_type": "anual",
    "payment_method": "debito",
    "internet_service": "fibra",
    "has_streaming": 1,
    "has_security_pack": 0,
    "num_products": 2,
    "region": "centro",
    "customer_age": 35,
    "is_promo": 0,
}

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_pipeline.joblib"


# ---- health ----

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert "model_path" in body


# ---- metrics ----

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "api_requests_total" in response.text


# ---- predict (con modelo real) ----

def test_predict():
    if not MODEL_PATH.exists():
        pytest.fail("No se encontro el modelo. Ejecutar: dvc pull")

    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert body["risk_level"] in ("bajo", "medio", "alto")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_batch():
    if not MODEL_PATH.exists():
        pytest.fail("No se encontro el modelo. Ejecutar: dvc pull")

    payload = {"customers": [VALID_PAYLOAD, VALID_PAYLOAD]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert len(body["predictions"]) == 2
    assert 0.0 <= body["churn_rate"] <= 1.0


# ---- validacion ----

def test_missing_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure_months"}
    assert client.post("/predict", json=payload).status_code == 422

def test_invalid_contract():
    payload = {**VALID_PAYLOAD, "contract_type": "semestral"}
    assert client.post("/predict", json=payload).status_code == 422

def test_tenure_below_min():
    payload = {**VALID_PAYLOAD, "tenure_months": 0}
    assert client.post("/predict", json=payload).status_code == 422

def test_tenure_above_max():
    payload = {**VALID_PAYLOAD, "tenure_months": 73}
    assert client.post("/predict", json=payload).status_code == 422

def test_empty_body():
    assert client.post("/predict", json={}).status_code == 422


# ---- schema ----

def test_schema():
    response = client.get("/schema")
    assert response.status_code == 200
    body = response.json()
    assert "properties" in body
    expected = {
        "tenure_months", "monthly_charge", "total_charges", "support_tickets",
        "late_payments", "avg_monthly_usage_gb", "contract_type", "payment_method",
        "internet_service", "has_streaming", "has_security_pack", "num_products",
        "region", "customer_age", "is_promo",
    }
    assert set(body["properties"].keys()) == expected


# ---- risk_label (sin HTTP) ----

def test_risk_label_bajo():
    assert risk_label(0.0) == "bajo"
    assert risk_label(0.34) == "bajo"

def test_risk_label_medio():
    assert risk_label(0.35) == "medio"
    assert risk_label(0.64) == "medio"

def test_risk_label_alto():
    assert risk_label(0.65) == "alto"
    assert risk_label(1.0) == "alto"


# ---- modelo no disponible ----

def test_model_unavailable():
    from src.api import api as api_module
    original = api_module.model
    api_module.model = None
    try:
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 503
    finally:
        api_module.model = original