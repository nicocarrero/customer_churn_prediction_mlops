# -*- coding: utf-8 -*-
"""
tests/test_preprocessing.py
----------------------------
Tests de preprocessing y feature engineering.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from src.feature_engineering import FeatureEngineeringTransformer
from src.preprocessing import TARGET_COL, get_preprocessor, split_and_preprocess


@pytest.fixture
def df_raw():
    np.random.seed(42)
    n = 120
    return pd.DataFrame(
        {
            "tenure_months": np.random.randint(1, 73, n),
            "monthly_charge": np.random.uniform(15, 130, n),
            "total_charges": np.random.uniform(50, 9000, n),
            "support_tickets": np.random.randint(0, 9, n),
            "late_payments": np.random.randint(0, 6, n),
            "avg_monthly_usage_gb": np.random.uniform(5, 300, n),
            "contract_type": np.random.choice(["mensual", "anual", "bianual"], n),
            "payment_method": np.random.choice(
                ["transferencia", "debito", "efectivo", "credito"], n
            ),
            "internet_service": np.random.choice(
                ["cable", "fibra", "movil", "ninguno"], n
            ),
            "has_streaming": np.random.randint(0, 2, n),
            "has_security_pack": np.random.randint(0, 2, n),
            "num_products": np.random.randint(1, 5, n),
            "region": np.random.choice(["centro", "norte", "oeste", "sur"], n),
            "customer_age": np.random.randint(18, 79, n),
            "is_promo": np.random.randint(0, 2, n),
            "churn": np.random.randint(0, 2, n),
        }
    )


class TestSinDataLeakage:

    def test_cuartiles_se_calculan_solo_en_fit(self, df_raw):
        fe = FeatureEngineeringTransformer()
        fe.fit(df_raw)
        q1_original = fe.q1

        df_test = df_raw.copy()
        df_test["total_charges"] = df_test["total_charges"] * 10
        fe.transform(df_test)

        assert fe.q1 == q1_original, (
            "q1 cambió al transformar el test set — hay data leakage. "
            "Los cuartiles deben calcularse solo en fit()."
        )

    def test_transform_usa_cuartiles_del_train(self):
        df_train = pd.DataFrame(
            {
                "total_charges": [100.0, 200.0, 300.0, 400.0],
                "support_tickets": [0, 1, 2, 3],
                "late_payments": [0, 0, 1, 1],
                "tenure_months": [12, 24, 36, 48],
                "contract_type": ["mensual", "anual", "mensual", "bianual"],
                "has_streaming": [1, 0, 1, 0],
                "has_security_pack": [0, 1, 0, 1],
                "num_products": [1, 2, 3, 4],
            }
        )
        fe = FeatureEngineeringTransformer()
        fe.fit(df_train)
        q1_tras_fit = fe.q1

        df_test = df_train.copy()
        df_test["total_charges"] = [5000.0, 6000.0, 7000.0, 8000.0]
        fe.transform(df_test)

        assert (
            fe.q1 == q1_tras_fit
        ), "q1 cambió al transformar test — los cuartiles se están recalculando en transform()."


class TestFeatureEngineeringContrato:

    def test_crea_todas_las_features_esperadas(self, df_raw):
        fe = FeatureEngineeringTransformer()
        result = fe.fit_transform(df_raw)
        features_requeridas = [
            "total_charges_cat",
            "tickets_grouped",
            "riesgo_contrato",
            "num_servicios",
            "cliente_problematico",
            "anchor_score",
        ]
        for col in features_requeridas:
            assert (
                col in result.columns
            ), f"Falta '{col}' — el preprocessor va a fallar al buscarla."

    def test_elimina_columnas_crudas(self, df_raw):
        fe = FeatureEngineeringTransformer()
        result = fe.fit_transform(df_raw)
        assert "total_charges" not in result.columns
        assert "support_tickets" not in result.columns

    def test_no_modifica_dataframe_original(self, df_raw):
        columnas_originales = set(df_raw.columns)
        fe = FeatureEngineeringTransformer()
        fe.fit_transform(df_raw)
        assert set(df_raw.columns) == columnas_originales

    def test_tickets_grouped_capped_at_5(self, df_raw):
        fe = FeatureEngineeringTransformer()
        result = fe.fit_transform(df_raw)
        assert result["tickets_grouped"].max() <= 5

    def test_riesgo_contrato_cero_para_no_mensual(self, df_raw):
        fe = FeatureEngineeringTransformer()
        result = fe.fit_transform(df_raw)
        mask_no_mensual = df_raw["contract_type"] != "mensual"
        assert (result.loc[mask_no_mensual, "riesgo_contrato"] == 0).all()

    def test_cliente_problematico_es_binario(self, df_raw):
        fe = FeatureEngineeringTransformer()
        result = fe.fit_transform(df_raw)
        assert set(result["cliente_problematico"].unique()).issubset({0, 1})

    def test_conserva_cantidad_de_filas(self, df_raw):
        fe = FeatureEngineeringTransformer()
        result = fe.fit_transform(df_raw)
        assert len(result) == len(df_raw)


class TestPreprocessorContrato:

    def test_retorna_column_transformer(self):
        assert isinstance(get_preprocessor(), ColumnTransformer)

    def test_tiene_cuatro_transformers(self):
        assert len(get_preprocessor().transformers) == 4

    def test_nombres_de_transformers(self):
        nombres = {t[0] for t in get_preprocessor().transformers}
        assert nombres == {"ord", "nom", "num", "bin"}

    def test_binarias_son_passthrough(self):
        bin_step = next(t[1] for t in get_preprocessor().transformers if t[0] == "bin")
        assert bin_step == "passthrough"

    def test_instancias_independientes(self):
        assert get_preprocessor() is not get_preprocessor()


class TestSplitContrato:

    def test_target_no_en_X_train(self, df_raw):
        X_train, _, _, _, _ = split_and_preprocess(df_raw)
        assert TARGET_COL not in X_train.columns

    def test_suma_train_test_igual_total(self, df_raw):
        X_train, X_test, _, _, _ = split_and_preprocess(df_raw)
        assert len(X_train) + len(X_test) == len(df_raw)

    def test_reproducibilidad_con_misma_semilla(self, df_raw):
        X_a, _, _, _, _ = split_and_preprocess(df_raw, random_state=42)
        X_b, _, _, _, _ = split_and_preprocess(df_raw, random_state=42)
        pd.testing.assert_frame_equal(
            X_a.reset_index(drop=True), X_b.reset_index(drop=True)
        )

    def test_estratificacion_preserva_proporcion_de_churn(self, df_raw):
        _, _, y_train, y_test, _ = split_and_preprocess(df_raw)
        assert abs(y_train.mean() - y_test.mean()) < 0.05