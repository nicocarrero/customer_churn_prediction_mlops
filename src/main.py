# src/main.py
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import FeatureEngineeringTransformer
from src.preprocessing import split_and_preprocess
from src.train import build_final_pipeline, train_evaluate_save


def main():
    print("- Iniciando pipeline de entrenamiento integral con FE acoplado...")

    raw_path = "data/raw/churn_sintetico.csv"
    processed_path = "data/processed/data_final.csv"

    try:
        print(f"- Cargando datos crudos desde {raw_path}...")
        df_raw = pd.read_csv(raw_path)
    except FileNotFoundError:
        print(f"[ERROR] Error: No se encontró el archivo en {raw_path}.")
        return

    # --- PASO 2: Split + Preprocessing directo sobre los datos RAW ---
    print("- Preparando particiones de datos crudos (Split)...")
    # Al pasar df_raw, X_train y X_test mantendrán las columnas nativas ('total_charges', etc.)
    X_train, X_test, y_train, y_test, preprocessor = split_and_preprocess(df_raw)

    # --- PASO 3: Guardar evidencia de Feature Engineering (Opcional / Trazabilidad) ---
    print("- Guardando copia local de datos transformados para trazabilidad...")
    try:
        os.makedirs("data/processed", exist_ok=True)
        fe_visual = FeatureEngineeringTransformer()
        df_processed_visual = fe_visual.fit_transform(df_raw)
        df_processed_visual.to_csv(processed_path, index=False)
        print(f"[OK] Data procesada guardada para control en: {processed_path}")
    except Exception as e:
        print(f"[WARNING] No se pudo guardar la copia estática: {e}")

    # --- PASO 4: Construir pipeline integral ---
    print("- Construyendo pipeline final (FE + Preprocesamiento + Modelo)...")
    pipeline = build_final_pipeline(preprocessor)

    # --- PASO 5: Entrenar, evaluar y serializar ---
    print("- Entrenando y evaluando modelo integral...")
    metrics, cm = train_evaluate_save(pipeline, X_train, X_test, y_train, y_test)

    # --- PASO 6: Generar evidencia gráfica ---
    print("- Generando evidencia gráfica...")
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Matriz de Confusión - Modelo Final (Logistic Regression)")
    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.tight_layout()

    os.makedirs("models", exist_ok=True)
    plt.savefig("models/confusion_matrix.png", dpi=300)
    plt.close()

    print("\n" + "=" * 50)
    print("¡PIPELINE EJECUTADO EXITOSAMENTE!")
    print("=" * 50)
    print("Artefacto agnóstico y completo guardado en 'models/model_pipeline.joblib'")


if __name__ == "__main__":
    main()
