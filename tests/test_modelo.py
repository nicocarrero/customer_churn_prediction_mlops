from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.train import build_final_pipeline
from src.preprocessing import get_preprocessor

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model_pipeline.joblib"


@pytest.fixture(scope="module")
def modelo():
    if not MODEL_PATH.exists():
        pytest.fail("No se encontro el modelo en models/model_pipeline.joblib. Ejecutar: dvc pull")
    import joblib
    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        "tenure_months": 12, "monthly_charge": 65.5, "total_charges": 786.0,
        "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 95.3,
        "contract_type": "anual", "payment_method": "debito",
        "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 0,
        "num_products": 2, "region": "centro", "customer_age": 35, "is_promo": 0,
    }])


# ---- el modelo carga ----

def test_modelo_carga(modelo):
    assert modelo is not None


# ---- predict devuelve lo esperado ----

def test_predict_retorna_array(modelo, sample_input):
    preds = modelo.predict(sample_input)
    assert isinstance(preds, np.ndarray)
    assert len(preds) == 1


def test_predict_valores_binarios(modelo, sample_input):
    preds = modelo.predict(sample_input)
    assert set(np.unique(preds)).issubset({0, 1})


# ---- predict_proba devuelve lo esperado ----

def test_predict_proba_shape(modelo, sample_input):
    proba = modelo.predict_proba(sample_input)
    assert proba.shape == (1, 2)


def test_predict_proba_suma_1(modelo, sample_input):
    proba = modelo.predict_proba(sample_input)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_predict_proba_entre_0_y_1(modelo, sample_input):
    proba = modelo.predict_proba(sample_input)
    assert (proba >= 0).all() and (proba <= 1).all()


# ---- consistencia entre predict y predict_proba ----

def test_predict_coincide_con_umbral_de_proba(modelo, sample_input):
    proba = modelo.predict_proba(sample_input)[0][1]
    pred = modelo.predict(sample_input)[0]
    expected = 1 if proba >= 0.5 else 0
    assert pred == expected


# ---- el modelo acepta las columnas de la API ----

def test_acepta_columnas_api(modelo):
    sample = pd.DataFrame([{
        "tenure_months": 12, "monthly_charge": 65.5, "total_charges": 786.0,
        "support_tickets": 1, "late_payments": 0, "avg_monthly_usage_gb": 95.3,
        "contract_type": "anual", "payment_method": "debito",
        "internet_service": "fibra", "has_streaming": 1, "has_security_pack": 0,
        "num_products": 2, "region": "centro", "customer_age": 35, "is_promo": 0,
    }])
    try:
        modelo.predict(sample)
    except Exception as e:
        pytest.fail(f"El modelo rechazo las columnas: {e}")


# ---- el pipeline tiene la estructura esperada ----

def test_pipeline_tres_pasos():
    preprocessor = get_preprocessor()
    pipeline = build_final_pipeline(preprocessor)
    assert len(pipeline.steps) >= 3


def test_feature_engineering_primero():
    preprocessor = get_preprocessor()
    pipeline = build_final_pipeline(preprocessor)
    assert pipeline.steps[0][0] == "feature_engineering"


def test_classifier_ultimo():
    preprocessor = get_preprocessor()
    pipeline = build_final_pipeline(preprocessor)
    assert pipeline.steps[-1][0] == "classifier"


def test_classifier_default_es_logistic():
    from sklearn.linear_model import LogisticRegression
    preprocessor = get_preprocessor()
    pipeline = build_final_pipeline(preprocessor)
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)


def test_classifier_custom_inyectable():
    from sklearn.ensemble import RandomForestClassifier
    preprocessor = get_preprocessor()
    clf = RandomForestClassifier(n_estimators=5, random_state=0)
    pipeline = build_final_pipeline(preprocessor, classifier=clf)
    assert isinstance(pipeline.named_steps["classifier"], RandomForestClassifier)