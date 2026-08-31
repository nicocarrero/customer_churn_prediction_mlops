import pytest
import requests

API_URL = "http://localhost:8000"
GUI_URL = "http://localhost:8501"


def _service_up(url):
    try:
        return requests.get(url, timeout=2).status_code == 200
    except:
        return False


@pytest.fixture(scope="module")
def docker_up():
    if not _service_up(f"{API_URL}/health"):
        pytest.skip("Docker Compose no esta levantado. Ejecutar: docker-compose up -d")


def test_api_health(docker_up):
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_api_predict(docker_up):
    payload = {
        "tenure_months": 12, "monthly_charge": 65.5, "total_charges": 786.0,
        "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 95.3,
        "contract_type": "anual", "payment_method": "debito",
        "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 0,
        "num_products": 2, "region": "centro", "customer_age": 35, "is_promo": 0,
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["risk_level"] in ("bajo", "medio", "alto")


def test_gui_responde(docker_up):
    response = requests.get(GUI_URL)
    assert response.status_code == 200