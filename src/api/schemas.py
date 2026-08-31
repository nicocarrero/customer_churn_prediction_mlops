"""
schemas.py
----------
Esquemas Pydantic para la API de predicción de churn
"""

from typing import List, Literal

from pydantic import BaseModel, Field


# Entrada
class CustomerFeatures(BaseModel):

    tenure_months: int = Field(
        ...,
        ge=1,
        le=72,
        description="Meses que el cliente lleva con el servicio (1–72).",
    )
    monthly_charge: float = Field(
        ..., ge=15.0, le=130.0, description="Cargo mensual en USD."
    )
    total_charges: float = Field(
        ..., ge=50.0, description="Cargos totales acumulados en USD."
    )
    support_tickets: int = Field(
        ..., ge=0, le=8, description="Número de tickets de soporte abiertos."
    )
    late_payments: int = Field(
        ..., ge=0, le=5, description="Número de pagos tardíos registrados."
    )
    avg_monthly_usage_gb: float = Field(
        ..., ge=5.0, description="Uso promedio mensual en GB."
    )
    contract_type: Literal["mensual", "anual", "bianual"] = Field(
        ..., description="Tipo de contrato."
    )
    payment_method: Literal["transferencia", "debito", "efectivo", "credito"] = Field(
        ..., description="Método de pago."
    )
    internet_service: Literal["cable", "fibra", "movil", "ninguno"] = Field(
        ..., description="Tipo de servicio de internet."
    )
    has_streaming: Literal[0, 1] = Field(
        ..., description="1 si el cliente tiene servicio de streaming."
    )
    has_security_pack: Literal[0, 1] = Field(
        ..., description="1 si el cliente tiene paquete de seguridad."
    )
    num_products: int = Field(
        ..., ge=1, le=4, description="Cantidad de productos contratados."
    )
    region: Literal["centro", "norte", "oeste", "sur"] = Field(
        ..., description="Región del cliente."
    )
    customer_age: int = Field(
        ..., ge=18, le=78, description="Edad del cliente en años."
    )
    is_promo: Literal[0, 1] = Field(
        ..., description="1 si el cliente está en una promoción activa."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure_months": 12,
                "monthly_charge": 65.50,
                "total_charges": 786.0,
                "support_tickets": 1,
                "late_payments": 0,
                "avg_monthly_usage_gb": 95.3,
                "contract_type": "anual",
                "payment_method": "debito",
                "internet_service": "fibra",
                "has_streaming": 1,
                "has_security_pack": 0,
                "num_products": 2,
                "region": "centro",
                "customer_age": 35,
                "is_promo": 0,
            }
        }
    }


class BatchRequest(BaseModel):
    """Petición de predicción por lote."""

    customers: List[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Lista de clientes a evaluar (máx. 1000 por request).",
    )



# Salida
class PredictionResponse(BaseModel):
    """Resultado de predicción para un cliente."""

    churn_prediction: int = Field(..., description="0 = no churn, 1 = churn.")
    churn_probability: float = Field(..., description="Probabilidad de churn (0–1).")
    risk_level: Literal["bajo", "medio", "alto"] = Field(
        ..., description="Nivel de riesgo: bajo (<35%) | medio (35–65%) | alto (>65%)."
    )


class BatchPredictionResponse(BaseModel):
    """Resultado de predicción para un lote de clientes."""

    predictions: List[PredictionResponse]
    total: int = Field(..., description="Total de clientes evaluados.")
    churn_count: int = Field(
        ..., description="Cantidad de clientes con churn predicho."
    )
    churn_rate: float = Field(..., description="Tasa de churn sobre el total (0–1).")
