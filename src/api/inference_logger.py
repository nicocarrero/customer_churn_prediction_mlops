#lógica de logging de inferencias a CSV (current_data.csv) para poder hacer monitoreo de drift con Evidently

import csv
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

PATH = os.getenv("CURRENT_DATA_PATH", "data/processed/current_data.csv")
MAX_ROWS = int(os.getenv("MAX_CSV_ROWS", "2000"))
MARGIN = 500

_lock = threading.Lock()


def _rotate(path, keep):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open(f"{path}.tmp", "w", encoding="utf-8") as f:
        f.writelines(lines[:1] + lines[-keep:])
    os.replace(f"{path}.tmp", path)


def log_inference(customer, pred, prob, risk):
    try:
        row = customer.model_dump() | {
            "churn_prediction": pred,
            "risk_level": risk,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        Path(PATH).parent.mkdir(parents=True, exist_ok=True)

        with _lock:
            exists = Path(PATH).exists() and Path(PATH).stat().st_size > 0

            with open(PATH, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if not exists:
                    w.writeheader()
                w.writerow(row)

            # cuenta filas reales del archivo (no en memoria)
            with open(PATH, "r", encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1

            if rows > MAX_ROWS + MARGIN:
                _rotate(PATH, MAX_ROWS)

    except Exception as e:
        print(f"[WARN] No se pudo registrar inferencia: {e}")