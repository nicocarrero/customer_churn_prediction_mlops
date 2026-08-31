"""Exporter de métricas Evidently en formato Prometheus."""

import os
import time
import threading
import traceback
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Gauge,
    CollectorRegistry,
    generate_latest,
)

import pandas as pd

from src.monitoring import evidently_monitor as em

REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/raw/churn_sintetico.csv")
CURRENT_PATH = os.getenv("CURRENT_PATH", "data/processed/current_data.csv")
RUN_INTERVAL = int(os.getenv("RUN_INTERVAL", "30"))

REGISTRY = CollectorRegistry()

mlops_drifted_features_ratio = Gauge(
    "mlops_drifted_features_ratio", "Ratio de features con drift", registry=REGISTRY,
)
mlops_feature_drift_score = Gauge(
    "mlops_feature_drift_score", "Score de drift por feature", ["feature"], registry=REGISTRY,
)
mlops_current_rows = Gauge(
    "mlops_current_rows", "Filas en dataset actual", registry=REGISTRY,
)
mlops_high_risk_ratio = Gauge(
    "mlops_high_risk_ratio", "Proporcion de predicciones de riesgo alto", registry=REGISTRY,
)
mlops_evidently_report_success = Gauge(
    "mlops_evidently_report_success", "1 si el ultimo analisis fue exitoso", registry=REGISTRY,
)
mlops_evidently_last_run_timestamp = Gauge(
    "mlops_evidently_last_run_timestamp", "Timestamp del ultimo analisis", registry=REGISTRY,
)


def _reset_metrics():
    mlops_drifted_features_ratio.set(0)
    mlops_high_risk_ratio.set(0)
    mlops_current_rows.set(0)


def analyze():
    now = datetime.now(timezone.utc).timestamp()
    try:
        if not os.path.exists(REFERENCE_PATH) or not os.path.exists(CURRENT_PATH):
            print("[WARN] Falta referencia o dataset actual")
            _reset_metrics()
            mlops_evidently_report_success.set(0)
            mlops_evidently_last_run_timestamp.set(now)
            return

        current = em.load_current_data(CURRENT_PATH)
        current_raw = pd.read_csv(CURRENT_PATH).tail(100).reset_index(drop=True)
        reference = em.load_reference_data(REFERENCE_PATH)

        if current.empty or reference.empty:
            print("[WARN] Dataset vacio")
            _reset_metrics()
            mlops_evidently_report_success.set(1)
            mlops_evidently_last_run_timestamp.set(now)
            return

        try:
            drift = em.extract_drift_summary(em.build_drift_report(reference, current))
            n_total = drift.get("number_of_columns", 1)
            n_drifted = drift.get("number_of_drifted_columns", 0)
            mlops_drifted_features_ratio.set(n_drifted / n_total if n_total > 0 else 0)
            for col, info in drift.get("per_column", {}).items():
                mlops_feature_drift_score.labels(feature=col).set(info.get("drift_score", 0))
        except Exception as e:
            print(f"[ERROR] Drift failed: {e}")
            mlops_drifted_features_ratio.set(0)

        mlops_current_rows.set(len(current_raw))
        if "risk_level" in current_raw.columns and len(current_raw) > 0:
            mlops_high_risk_ratio.set((current_raw["risk_level"] == "alto").mean())

        mlops_evidently_report_success.set(1)
        mlops_evidently_last_run_timestamp.set(now)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Evidently OK | rows={len(current_raw)}")

    except Exception as e:
        print(f"[ERROR] Evidently failed: {e}")
        traceback.print_exc()
        _reset_metrics()
        mlops_evidently_report_success.set(0)
        mlops_evidently_last_run_timestamp.set(now)


def run_periodically():
    while True:
        analyze()
        time.sleep(RUN_INTERVAL)


app = FastAPI(title="Evidently Exporter", version="1.2.0")


@app.on_event("startup")
def startup():
    analyze()
    threading.Thread(target=run_periodically, daemon=True).start()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "last_run": mlops_evidently_last_run_timestamp._value.get(),
        "current_rows": mlops_current_rows._value.get(),
        "success": mlops_evidently_report_success._value.get(),
    }


@app.get("/metrics", include_in_schema=False)
def metrics_endpoint():
    return Response(content=generate_latest(REGISTRY), media_type=CONTENT_TYPE_LATEST)