import os
import time

import numpy as np
import pandas as pd
import requests

API_URL = os.getenv("API_URL", "http://api:8000")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
REFERENCE_PATH = os.getenv("REFERENCE_PATH", "data/raw/churn_sintetico.csv")
DATA_DRIFT_PROB = float(os.getenv("DATA_DRIFT_PROB", "0.05"))
STARTUP_DELAY = int(os.getenv("STARTUP_DELAY", "10"))

INITIAL_INTERVAL = float(os.getenv("INITIAL_INTERVAL", "5"))
MIN_INTERVAL = float(os.getenv("MIN_INTERVAL", "0.2"))
INTERVAL_STEP = float(os.getenv("INTERVAL_STEP", "0.2"))
RAMP_INTERVAL = int(os.getenv("RAMP_INTERVAL", "30"))

FEATURE_COLUMNS = [
    "tenure_months", "monthly_charge", "total_charges", "support_tickets",
    "late_payments", "avg_monthly_usage_gb", "contract_type", "payment_method",
    "internet_service", "has_streaming", "has_security_pack", "num_products",
    "region", "customer_age", "is_promo",
]

_reference_df = None


def _load_reference():
    global _reference_df
    if _reference_df is not None:
        return _reference_df
    _reference_df = pd.read_csv(REFERENCE_PATH)
    print(f"[INFO] Dataset cargado: {len(_reference_df)} filas")
    return _reference_df


def _generate_customers(n):
    weights = np.where(_reference_df["churn"] == 0, 0.8, 0.2)
    sampled = _reference_df.sample(n=n, replace=True, weights=weights)
    return sampled[FEATURE_COLUMNS].reset_index(drop=True)


def _apply_drift(df, drift_prob):
    if drift_prob <= 0 or df.empty:
        return df
    rng = np.random.default_rng()
    mask = rng.random(len(df)) < drift_prob
    idx = df.index[mask]
    if len(idx) == 0:
        return df
    k = len(idx)
    df.loc[idx, "monthly_charge"] = np.clip(
        df.loc[idx, "monthly_charge"] * rng.uniform(1.3, 1.8, k), 15.0, 130.0
    ).round(2)
    df.loc[idx, "tenure_months"] = np.clip(
        (df.loc[idx, "tenure_months"] * rng.uniform(0.2, 0.5, k)).astype(int), 1, 72
    )
    df.loc[idx, "support_tickets"] = np.clip(
        df.loc[idx, "support_tickets"] + rng.integers(2, 5, k), 0, 8
    )
    df.loc[idx, "late_payments"] = np.clip(
        df.loc[idx, "late_payments"] + rng.integers(1, 3, k), 0, 5
    )
    df.loc[idx, "avg_monthly_usage_gb"] = (
        df.loc[idx, "avg_monthly_usage_gb"] * rng.uniform(1.3, 2.0, k)
    ).round(1)
    df.loc[idx, "contract_type"] = "mensual"
    df.loc[idx, "internet_service"] = "fibra"
    df.loc[idx, "is_promo"] = 1
    return df


def _send_batch():
    customers = _generate_customers(BATCH_SIZE)
    customers = _apply_drift(customers, DATA_DRIFT_PROB)
    payload = {"customers": customers.to_dict(orient="records")}
    try:
        resp = requests.post(f"{API_URL}/predict/batch", json=payload, timeout=15)
        resp.raise_for_status()
        print(f"[OK] Batch de {BATCH_SIZE} enviado")
    except Exception as e:
        print(f"[ERROR] {e}")


def _next_interval(elapsed):
    steps = int(elapsed // RAMP_INTERVAL)
    return max(INITIAL_INTERVAL - steps * INTERVAL_STEP, MIN_INTERVAL)


def main():
    print(f"[INFO] Esperando {STARTUP_DELAY}s...")
    time.sleep(STARTUP_DELAY)
    _load_reference()
    start = time.monotonic()
    while True:
        _send_batch()
        interval = _next_interval(time.monotonic() - start)
        time.sleep(interval)


if __name__ == "__main__":
    main()