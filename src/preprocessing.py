# src/preprocessing.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# CONFIGURACIÓN DE COLUMNAS

TARGET_COL = "churn"

COL_ORDINAL = ["total_charges_cat"]
ORDINAL_CATEGORIAS = [["Bajo", "Medio-Bajo", "Medio-Alto", "Alto/VIP"]]

COL_NOMINALES = ["contract_type", "payment_method", "internet_service", "region"]

COL_NUMERICAS = [
    "tenure_months",
    "monthly_charge",
    "avg_monthly_usage_gb",
    "customer_age",
    "num_products",
    "tickets_grouped",
    "riesgo_contrato",
    "late_payments",
    "anchor_score",
]
COL_BINARIAS = [
    "is_promo",
    "has_streaming",
    "has_security_pack",
    "num_servicios",
    "cliente_problematico",
]


def get_preprocessor():
    """
    Construye el ColumnTransformer.
    """
    transformer_ordinal = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=ORDINAL_CATEGORIAS,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        ]
    )

    transformer_nominal = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore", drop="first", sparse_output=False
                ),
            )
        ]
    )

    transformer_numerico = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", transformer_ordinal, COL_ORDINAL),
            ("nom", transformer_nominal, COL_NOMINALES),
            ("num", transformer_numerico, COL_NUMERICAS),
            ("bin", "passthrough", COL_BINARIAS),  # ← pasan sin transformar
        ],
        remainder="drop",
    )
    return preprocessor


def split_and_preprocess(df, test_size=0.2, random_state=123):
    """
    1. Separa X/y
    2. Divide train/test
    3. Instancia el preprocessor
    4. Devuelve DataFrames para que el Pipeline los procese
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # 1. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 2. Solo instanciar preprocessor
    # La transformación la hará el Pipeline en train.py
    preprocessor = get_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor
