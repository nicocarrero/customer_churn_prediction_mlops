# tracking/experiments.py
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn
import warnings
import logging

mlflow.set_tracking_uri(
    "https://dagshub.com/carreronicoo/proyecto_ml.mlflow"
)

# =========================================================================
# ESCUDO CONTRA WARNINGS DE POWERSHELL / WINDOWS
# =========================================================================
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

class HideSkopsWarning(logging.Filter):
    def filter(self, record):
        return "skops" not in record.getMessage() and "pickle" not in record.getMessage()

logging.getLogger("mlflow").addFilter(HideSkopsWarning())
logging.getLogger("mlflow.sklearn").addFilter(HideSkopsWarning())
# =========================================================================

# Inyección de paths para importar desde la raíz de tu proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import split_and_preprocess

# IMPORTAMOS TUS FUNCIONES OFICIALES DE ENTRÁNAMIENTO
from src.train import build_final_pipeline, train_evaluate_save

os.makedirs("reports", exist_ok=True)

# Función auxiliar para la fase de exploración (Baselines y Tuning)
def evaluate_and_track_baseline(pipeline, X_test, y_test, run_name, stage):
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1_score": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs))
    }
    mlflow.log_metrics(metrics)
    mlflow.set_tags({"stage": stage, "framework": "scikit-learn"})

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Matriz de Confusion - {run_name}')
    plt.tight_layout()
    
    plot_filename = f"reports/cm_exp_{run_name.lower().replace(' ', '_')}.png"
    plt.savefig(plot_filename)
    plt.close()
    
    mlflow.log_artifact(plot_filename, artifact_path="plots")
    mlflow.sklearn.log_model(pipeline, artifact_path="model_pipeline")


# =====================================================
# PREPARACIÓN DE DATOS REUTILIZABLE
# =====================================================
df = pd.read_csv("data/raw/churn_sintetico.csv")

# El Feature Engineering ya vive dentro del pipeline.
# No debemos ejecutarlo acá para evitar duplicarlo.
X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df)


# =====================================================
# ETAPA 1: BASELINES (Exploratorios)
# =====================================================
RUN_BASELINES = True

if RUN_BASELINES:
    mlflow.set_experiment("customer_churn_baselines")
    models = {
        "Logistic Regression Base": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest Base": RandomForestClassifier(random_state=42),
        "XGBoost Base": XGBClassifier(eval_metric="logloss", random_state=42)
    }

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: str(v) for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool))})
            pipeline_baseline = build_final_pipeline(preprocessor=preprocessor, classifier=model, )
            pipeline_baseline.fit(X_train, y_train)
            evaluate_and_track_baseline(pipeline_baseline, X_test, y_test, name, "baseline")


# =====================================================
# ETAPA 2: TUNING (Exploratorio)
# =====================================================

RUN_RF_TUNED = True
RUN_LR_TUNED = True
RUN_XGB_TUNED = True

# --- RANDOM FOREST TUNED ---
if RUN_RF_TUNED:

    mlflow.set_experiment("customer_churn_tuning")

    rf_params = {
        "n_estimators": 300,
        "min_samples_split": 2,
        "min_samples_leaf": 4,
        "max_features": "log2",
        "max_depth": 10,
        "bootstrap": True,
        "random_state": 42
    }

    with mlflow.start_run(run_name="RandomForest_Tuned"):

        mlflow.log_params(rf_params)

        pipeline_rf = build_final_pipeline(
            preprocessor=preprocessor,
            classifier=RandomForestClassifier(**rf_params),
        )

        pipeline_rf.fit(X_train, y_train)

        evaluate_and_track_baseline(
            pipeline_rf,
            X_test,
            y_test,
            "RandomForest_Tuned",
            "tuning"
        )

# --- LOGISTIC REGRESSION TUNED ---
if RUN_LR_TUNED:

    mlflow.set_experiment("customer_churn_tuning")

    lr_params = {
        "penalty": "l1",
        "C": 0.03359818286283781,
        "class_weight": "balanced",
        "solver": "saga",
        "max_iter": 5000,
        "random_state": 42
    }

    with mlflow.start_run(run_name="LogisticRegression_Tuned"):

        mlflow.log_params(lr_params)

        pipeline_lr = build_final_pipeline(
            preprocessor=preprocessor,
            classifier=LogisticRegression(**lr_params),
        )

        pipeline_lr.fit(X_train, y_train)

        evaluate_and_track_baseline(
            pipeline_lr,
            X_test,
            y_test,
            "LogisticRegression_Tuned",
            "tuning"
        )


# --- XGBOOST TUNED ---
if RUN_XGB_TUNED:

    mlflow.set_experiment("customer_churn_tuning")

    xgb_params = {
        "subsample": 0.8,
        "n_estimators": 200,
        "min_child_weight": 10,
        "max_depth": 3,
        "learning_rate": 0.05,
        "gamma": 0,
        "colsample_bytree": 0.7,
        "objective": "binary:logistic",
        "random_state": 42,
        "eval_metric": "logloss"
    }

    with mlflow.start_run(run_name="XGBoost_Tuned"):

        mlflow.log_params(xgb_params)

        pipeline_xgb = build_final_pipeline(
            preprocessor=preprocessor,
            classifier=XGBClassifier(**xgb_params),   
        )

        pipeline_xgb.fit(X_train, y_train)

        evaluate_and_track_baseline(
            pipeline_xgb,
            X_test,
            y_test,
            "XGBoost_Tuned",
            "tuning"
        )
# =====================================================
# ETAPA 3: MODELO FINAL DE PRODUCCIÓN (Trazabilidad Oficial)
# =====================================================
# =====================================================
# FINAL MODEL PRODUCCIÓN
# =====================================================

RUN_FINAL_MODEL = True

if RUN_FINAL_MODEL:

    mlflow.set_experiment(
        "customer_churn_final_model"
    )

    with mlflow.start_run(
        run_name="Final_Logistic_Regression_Production"
    ):
        print("- Entrenando y registrando modelo final de producción...")
        
        # 1. Construimos una instancia limpia del pipeline
        pipeline_produccion = build_final_pipeline(
        preprocessor=preprocessor,
        )
        
        # 2. Entrena, evalúa y genera los archivos locales (.joblib y .json)
        metrics, cm = train_evaluate_save(pipeline_produccion, X_train, X_test, y_train, y_test)

        # 3. Registramos los parámetros oficiales
        mlflow.log_params({
            "penalty": "l1",
            "solver": "saga",
            "C": 0.5,
            "class_weight": "{0:1,1:1.6}",
            "max_iter": 1000
        })

        # 4. Registramos las métricas mapeadas que devolvió la función
        mlflow.log_metrics({
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1'],
            "roc_auc": metrics['roc_auc']
        })
        
        mlflow.log_artifact("models/confusion_matrix.png", artifact_path="plots")

        mlflow.set_tags({
            "stage": "production",
            "status": "active"
        })

        # 5. ¡EL PASO FALTANTE! Sube el binario del modelo entrenado a MLflow
        # Usamos obligatoriamente 'artifact_path' para indicar la carpeta en MLflow
        mlflow.sklearn.log_model(
        sk_model=pipeline_produccion,
        artifact_path="model_pipeline",
        registered_model_name="CustomerChurn"
    )
        
        print(" [OK] ¡Modelo final y artefactos guardados con éxito en MLflow!")