import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware

from . import metrics
from .inference_logger import log_inference
from .schemas import (
    BatchPredictionResponse,
    BatchRequest,
    CustomerFeatures,
    PredictionResponse,
)

# app 
app = FastAPI(
    title="Churn Prediction API",
    description="API de inferencia para prediccion de churn",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path
        method = request.method
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:
            duration = time.perf_counter() - start
            metrics.record_request(endpoint, method, 500, duration)
            metrics.record_exception(endpoint, type(exc).__name__)
            raise
        duration = time.perf_counter() - start
        metrics.record_request(endpoint, method, response.status_code, duration)
        return response

app.add_middleware(PrometheusMiddleware)

# modelo
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_pipeline.joblib")

try:
    model = joblib.load(MODEL_PATH)
    print(f"[OK] Modelo cargado desde {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"[ERROR] No se pudo cargar el modelo: {e}")

metrics.set_model_availability(model is not None)

FEATURE_ORDER = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "contract_type",
    "payment_method",
    "internet_service",
    "has_streaming",
    "has_security_pack",
    "num_products",
    "region",
    "customer_age",
    "is_promo",
]

def risk_label(prob: float) -> str:
    if prob < 0.35:
        return "bajo"
    elif prob < 0.65:
        return "medio"
    return "alto"

def predict_one(customer: CustomerFeatures) -> PredictionResponse:
    df = pd.DataFrame([customer.model_dump()])[FEATURE_ORDER]

    inference_start = time.perf_counter()
    pred = int(model.predict(df)[0])
    prob = float(model.predict_proba(df)[0][1])
    metrics.INFERENCE_DURATION.observe(time.perf_counter() - inference_start)

    risk = risk_label(prob)

    metrics.record_prediction(churn_prediction=pred, churn_probability=prob, risk_level=risk)
    log_inference(customer, pred, prob, risk)

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk,
    )

def check_model():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verifica que el archivo .joblib exista en el contenedor.",
        )

# endpoints 
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    check_model()
    return predict_one(customer)

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchRequest):
    check_model()
    results = [predict_one(c) for c in request.customers]
    churn_count = sum(r.churn_prediction for r in results)

    all_probs = [r.churn_probability for r in results]
    all_preds = [r.churn_prediction for r in results]
    metrics.update_batch_stats(all_probs)

    return BatchPredictionResponse(
        predictions=results,
        total=len(results),
        churn_count=churn_count,
        churn_rate=round(churn_count / len(results), 4) if results else 0.0,
    )

@app.get("/schema")
def schema():
    return CustomerFeatures.model_json_schema()

@app.get("/metrics", include_in_schema=False)
def metrics_endpoint():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)