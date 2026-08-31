import os

import pandas as pd
from evidently.legacy.metric_preset import DataDriftPreset
from evidently.legacy.pipeline.column_mapping import ColumnMapping
from evidently.legacy.report import Report

NUMERICAL_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "total_charges",
    "support_tickets",
    "late_payments",
    "avg_monthly_usage_gb",
    "num_products",
    "customer_age",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "internet_service",
    "region",
    "has_streaming",
    "has_security_pack",
    "is_promo",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
NON_ANALYSIS_COLUMNS = ["churn_prediction", "risk_level", "timestamp"]


def load_reference_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro referencia en '{path}'")
    df = pd.read_csv(path)
    if "churn" in df.columns:
        df = df.drop(columns=["churn"])
    return df


def load_current_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontro '{path}'")
    df = pd.read_csv(path)
    drop_cols = [c for c in NON_ANALYSIS_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)
    if len(df) > 100:
        df = df.tail(100).reset_index(drop=True)
    return df


def get_column_mapping():
    return ColumnMapping(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def _align_columns(reference, current):
    cols = [c for c in FEATURE_COLUMNS if c in reference.columns and c in current.columns]
    return reference[cols], current[cols]


def build_drift_report(reference, current):
    ref, cur = _align_columns(reference, current)
    report = Report(metrics=[DataDriftPreset(columns=list(ref.columns))])
    report.run(reference_data=ref, current_data=cur, column_mapping=get_column_mapping())
    return report


def extract_drift_summary(report):
    result = report.as_dict()
    dataset_drift = next(m["result"] for m in result["metrics"] if m["metric"] == "DatasetDriftMetric")
    drift_table = next(m["result"] for m in result["metrics"] if m["metric"] == "DataDriftTable")
    per_column = {
        col: {
            "drift_detected": bool(info.get("drift_detected")),
            "drift_score": round(float(info.get("drift_score", 0.0)), 4),
        }
        for col, info in drift_table.get("drift_by_columns", {}).items()
    }
    return {**dataset_drift, "per_column": per_column}