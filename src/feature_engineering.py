# src/feature_engineering.py

import os

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineeringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Guardaremos los cuartiles aquí para evitar Data Leakage
        self.q1 = None
        self.q2 = None
        self.q3 = None

    def fit(self, X, y=None):
        # Calculamos los cuartiles basados únicamente en el set que entra al .fit()
        if "total_charges" in X.columns:
            self.q1 = X["total_charges"].quantile(0.25)
            self.q2 = X["total_charges"].quantile(0.50)
            self.q3 = X["total_charges"].quantile(0.75)
        return self

    def transform(self, X):
        # Evitamos modificar el DataFrame original por referencia
        X = X.copy()

        # 1. Segmentación de Total Charges en categorías usando los cuartiles del fit
        if "total_charges" in X.columns and self.q1 is not None:

            def segmentar_total_charges(x):
                if x <= self.q1:
                    return "Bajo"
                if x <= self.q2:
                    return "Medio-Bajo"
                if x <= self.q3:
                    return "Medio-Alto"
                return "Alto/VIP"

            X["total_charges_cat"] = X["total_charges"].apply(segmentar_total_charges)

        # 2. Agrupación de Tickets de Soporte
        if "support_tickets" in X.columns:
            X["tickets_grouped"] = X["support_tickets"].copy()
            X.loc[X["tickets_grouped"] >= 5, "tickets_grouped"] = 5

        # 3. Riesgo de Contrato
        if "contract_type" in X.columns and "tenure_months" in X.columns:
            es_mensual = (X["contract_type"] == "mensual").astype(int)
            X["riesgo_contrato"] = es_mensual / (X["tenure_months"] + 1)

        # 4. Conteo de Servicios
        cols_servicios = [col for col in X.columns if col.startswith("has_")]
        if cols_servicios:
            X["num_servicios"] = X[cols_servicios].sum(axis=1)
        else:
            X["num_servicios"] = 0

        # 5. Cliente Problemático y Anchor Score
        if "late_payments" in X.columns and "tickets_grouped" in X.columns:
            X["cliente_problematico"] = (
                (X["late_payments"] + X["tickets_grouped"]) > 0
            ).astype(int)

        if "num_products" in X.columns and "tenure_months" in X.columns:
            X["anchor_score"] = X["tenure_months"] * X["num_products"]

        # 6. Drop de columnas originales modificadas
        cols_to_drop = ["total_charges", "support_tickets"]
        X.drop(columns=[c for c in cols_to_drop if c in X.columns], inplace=True)

        return X


# Mantenemos la función original envolviendo al Transformer para no romper compatibilidades externas
def run_feature_engineering(df):
    fe = FeatureEngineeringTransformer()
    return fe.fit_transform(df)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_dir, "data", "raw", "churn_sintetico.csv")
    output_path = os.path.join(base_dir, "data", "processed", "data_final.csv")

    try:
        data = pd.read_csv(input_path)
        df_final = run_feature_engineering(data)
        df_final.to_csv(output_path, index=False)
        print(
            f"[OK] Feature Engineering completado. Columnas finales: {list(df_final.columns)}"
        )
        print(f"--> Shape: {df_final.shape}")
    except Exception as e:
        print(f"[ERROR] Error al procesar: {e}")
