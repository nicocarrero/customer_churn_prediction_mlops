# src/train.py
import json
import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# Importamos el nuevo Transformer
from src.feature_engineering import FeatureEngineeringTransformer


def build_final_pipeline(preprocessor, classifier=None):

    if classifier is None:
        classifier = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=0.5,
            class_weight={0: 1, 1: 1.6},
            max_iter=1000,
            random_state=42,
        )

    steps = [
        ("feature_engineering", FeatureEngineeringTransformer()),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]

    return Pipeline(steps)


def train_evaluate_save(pipeline, X_train, X_test, y_train, y_test, model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)

    # El fit ahora arranca desde los datos crudos, pasa por FE, escala, selecciona y entrena
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    cm = confusion_matrix(y_test, y_pred)

    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(model_dir, "model_pipeline.joblib")
    joblib.dump(pipeline, model_path)

    print("\n" + "=" * 60)
    print("RESULTADOS DEL MODELO FINAL (Logistic Regression con FE Integrado)")
    print("=" * 60)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"\n[OK] Métricas guardadas en {metrics_path}")
    print(f"[OK] Pipeline completo serializado en {model_path}")

    return metrics, cm
