# Informe Técnico — Predicción de Churn

**AndesLink Servicios Digitales S.A.**
Proyecto MLOps Local — Laboratorio de Minería de Datos, ISTEA
Entregas 1 (Entrenamiento) · 2 (Despliegue) · 3 (Monitoreo)

---

## 1. Problema de negocio

AndesLink enfrenta un incremento sostenido en la tasa de cancelación voluntaria de clientes (*churn*), con impacto directo en los ingresos recurrentes, el costo de adquisición y la eficiencia de las campañas comerciales.

Se aborda como una **clasificación binaria supervisada**: dado un cliente, predecir si cancelará su suscripción (`churn = 1`) o continuará activo (`churn = 0`). El modelo estima una probabilidad de churn, habilitando segmentación por riesgo y decisiones basadas en umbrales configurables.

---

## 2. Dataset

| Característica | Detalle |
|---|---|
| Archivo | `churn_sintetico.csv` |
| Filas / Columnas | 5.000 registros / 16 variables originales |
| Nulos / Duplicados | Ninguno |
| Balance | Variable objetivo desbalanceada |

Variables más relevantes del EDA: `contract_type`, `payment_method` e `internet_service` muestran alta incidencia en churn; `tenure_months` es el principal factor de retención (correlación ≈ -0.17); `support_tickets` es fuerte indicador de fuga (correlación ≈ +0.10); `total_charges` y `tenure_months` están fuertemente correlacionadas entre sí (r ≈ 0.87); `region` aporta relativamente poco frente al resto.

---

## 3. Preparación de datos y feature engineering

`total_charges` se transformó en variable ordinal por cuartiles (Bajo / Medio-Bajo / Medio-Alto / Alto-VIP) dada su asimetría y su alta correlación con `tenure_months`; la variable original fue eliminada. `support_tickets` se agrupó en una categoría `5+` para valores extremos, reduciendo dispersión.

Se generaron variables derivadas: `riesgo_contrato` (`contract_type_mensual / (tenure_months+1)`), `num_servicios`, `cliente_problematico` y `anchor_score` (`tenure_months × num_products`).

| Paso | Decisión |
|---|---|
| Encoding ordinal | `total_charges_cat` |
| Encoding nominal | One-Hot Encoding |
| Escalado | StandardScaler |
| División train/test | 80/20 estratificada |
| Selección de variables | SelectFromModel + Logistic Regression L1 |

---

## 4. Modelos y métricas

Se evaluaron tres algoritmos: **Logistic Regression** (mejor desempeño general), **Random Forest** (inferior) y **XGBoost** (resultados similares, sin ventajas que justifiquen su complejidad). La optimización se realizó con `RandomizedSearchCV` y `StratifiedKFold` (5 folds).

### Modelo final

```python
LogisticRegression(
    penalty="l1",
    solver="saga",
    C=0.5,
    class_weight={0:1, 1:1.6}
)
```

| Métrica | Validación cruzada | Test set |
|---|---|---|
| F1 | 0.59 | 0.64 |
| ROC-AUC | 0.76 | 0.80 |
| Precision | 0.55 | 0.60 |
| Recall | 0.64 | 0.69 |

El modelo prioriza la detección de clientes en riesgo de abandono manteniendo una precisión razonable para campañas de retención.

---

## 5. Arquitectura de la solución

La solución final integra, sobre un mismo entorno local orquestado con Docker Compose, la capa de servicio desarrollada en la Entrega 2 (GUI + API + modelo) y la capa de observabilidad incorporada en la Entrega 3 (Prometheus, Evidently y Grafana), junto con el traffic generator que genera la carga necesaria para poner a prueba ambas capas:

```text
Usuario → Streamlit (GUI) ─┐
                            ├─→ FastAPI (API de inferencia) → Modelo ML     
                            |                                  (Joblib)
Traffic Generator ──────────┘              ↓
                                   current_data.csv
                                            ↓
                                       Evidently (compara vs. dataset de referencia)
                                            ↓
                                       Prometheus (recolecta métricas de API + Evidently)
                                            ↓
                                       Grafana (dashboard)
```

Toda la solución está contenerizada con Docker y orquestada con Docker Compose. Los seis servicios (`api`, `gui`, `prometheus`, `grafana`, `evidently` y `traffic generator`) se comunican dentro de una misma red de Docker (`churn_net`).

**Componentes:**
- **GUI (Streamlit):** permite ingresar datos de un cliente, invocar la API y visualizar la predicción y su probabilidad.
- **API (FastAPI):** expone POST /predict y /predict/batch, ejecuta el modelo serializado (joblib), además de GET /health para verificar el estado del servicio y GET /metrics para exponer métricas a Prometheus. 
- **Modelo ML:** Logistic Regression con su pipeline de preprocesamiento, cargado una única vez al iniciar el servicio.
- **Traffic Generator:** simula clientes reales contra la API con carga creciente y drift sintético controlado.
- **Evidently:** compara el dataset de referencia contra los datos recibidos en producción y publica métricas de drift.
- **Prometheus:** recolecta las métricas expuestas por la API y por Evidently.
- **Grafana:** visualiza las métricas recolectadas en un dashboard que centraliza toda la información operativa.

---

## 6. Reproducibilidad y herramientas MLOps

| Herramienta | Propósito |
|---|---|
| Git | Versionado de código |
| DVC | Versionado de datasets y modelos |
| MLflow | Tracking de experimentos (parámetros, métricas, artefactos) |
| Docker / Compose | Empaquetado y despliegue reproducible |
| Pytest | Testing automatizado |

---

## 7. Servicio de inferencia (API)

Desarrollada con **FastAPI**. El endpoint principal `POST /predict` recibe los datos del cliente, ejecuta el pipeline completo (preprocesamiento + modelo) y devuelve la predicción y la probabilidad de churn. Documentación interactiva disponible vía Swagger UI.

---

## 8. Interfaz gráfica

Se desarrolló una GUI en **Streamlit** que permite ingresar datos del cliente, consumir la API de inferencia y visualizar el resultado junto con la probabilidad asociada, desacoplando al usuario final de la API.

---

## 9. Testing

Se implementaron pruebas automatizadas con **Pytest**, cubriendo la API, el preprocesamiento, el modelo, la validación de respuestas y la integración de los servicios en docker (`pytest -v`).

test_api.py — que la API responda bien: endpoints devuelven lo esperado, valida datos de entrada (422 si algo está mal) y no se cae si el modelo no cargó (503).

test_modelo.py — que el modelo serializado funcione: carga bien, predice 0/1, las probabilidades son coherentes (suman 1, entre 0-1) y el pipeline tiene la estructura correcta (FE → clasificador).

test_preprocessing.py — el más crítico: que no haya data leakage (los cuartiles no se recalculan con datos de test) y que el feature engineering y el split hagan lo que tienen que hacer.

test_integration.py — prueba todo ya levantado en Docker: que la API y la GUI respondan de verdad, no en memoria.

---

## 10. Monitoreo y observabilidad (Entrega 3)

Se incorporó una capa de observabilidad basada en tres ejes: **salud técnica del servicio** (disponibilidad, latencia, modelo cargado), **comportamiento del modelo** (variación de la probabilidad media de churn devuelta) y **drift de datos** (comparación estadística, variable por variable, entre los datos de entrenamiento y los datos recibidos en producción, mediante Evidently).

Para observar estas señales en movimiento se desarrolló un **traffic generator** que simula clientes reales contra la API (lotes de 10 vía `/predict/batch`), acelerando gradualmente la frecuencia de envío para poder ver en el dashboard cómo se degrada la latencia bajo carga creciente. Con 5% de probabilidad por cliente inyecta drift sintético (sube facturación y consumo, baja antigüedad, suma tickets y atrasos, fuerza contrato mensual e internet fibra), simulando un cambio real en el perfil de clientes.

Todas las consultas (reales y simuladas) se registran en `current_data.csv`, que Evidently compara contra el dataset original para calcular drift cada 30 segundos. Prometheus recolecta todas las métricas expuestas y Grafana las visualiza en dashboards.

### Dashboard de Grafana

| Bloque | Métricas clave |
|---|---|
| Estado general | `api_requests_total`, `up{job="churn_api"}`, % de error |
| Drift de datos | `mlops_drifted_features_ratio`, `mlops_feature_drift_score` por variable |
| Comportamiento del modelo | `mlops_churn_probability_mean`, `churn_predictions_total` por `risk_level` |
| Latencia | p95 de `api_request_latency_seconds` y de `model_inference_duration_seconds` |
| Salud del sistema | `mlops_current_rows` y alertas activas de Prometheus |

### Alertas (`monitoring/rules.yml`)

Las alertas se agrupan en cuatro categorías: **disponibilidad** (`APIDown`, `ModelUnavailable`, `EvidentlyExporterDown`, caído 1 min), **latencia** (p95 > 1s en API, p95 > 0.5s en inferencia), **errores** (`HighErrorRate` > 5%) y **drift/calidad** (`DataDriftDetected` > 30% de features). Se usó p95 en lugar de promedio porque este último oculta los picos que afectan a usuarios puntuales; el umbral de 30% de drift se consideró óptimo para considerar el deterioro del modelo.

### Acciones correctivas por señal

- **Disponibilidad caída:** se revisa siempre primero, ya que el resto de las métricas puede estar "congelado" y no reflejar la realidad.
- **Data drift sostenido:** evaluar reentrenamiento con datos recientes; revisar `mlops_feature_drift_score` por columna para decidir si también ajustar el feature engineering.
- **Probabilidad media de churn corrida sin subir el drift ratio:** indica *concept drift* (cambió la relación del modelo con el negocio, no los datos) — requiere revisar cambios de negocio, no solo más datos.
- **Latencia de API alta sin latencia de modelo alta:** cuello de botella en API/red — revisar tamaño de batch.
- **Latencia de inferencia alta:** el cuello de botella está en el modelo o en los recursos del contenedor.
- **Tasa de error alta:** desagregar por tipo — mayoría 422 es problema de contrato del cliente, mayoría 500 es un bug en la API.

### Limitaciones de esta entrega

No se llegó a incorporar en el dashboard métricas de calidad de datos en tiempo real (nulos, duplicados, fuera de rango); actualmente Pydantic bloquea estos casos en la API con un 422 antes de llegar al modelo, pero no se cuantifican en Grafana. Tampoco se incorporaron métricas de desempeño del modelo en producción (Precision, Recall, F1), ya que requieren la etiqueta real del cliente, disponible solo tiempo después de la predicción — limitación típica de monitoreo con feedback demorado.

---

## 11. Conclusiones y limitaciones generales

La Regresión Logística con regularización L1 y pesos de clase ajustados resultó la alternativa más adecuada: buen equilibrio entre precisión y recall, alta interpretabilidad, estabilidad en validación cruzada y menor complejidad operativa que alternativas de mayor complejidad como Random Forest o XGBoost.

### Insights de negocio

- Contrato mensual + pago en efectivo + internet móvil representan el perfil de mayor riesgo.
- A partir del tercer ticket de soporte aumenta significativamente la probabilidad de abandono.
- Los clientes Alto/VIP presentan la mayor estabilidad.
- Las estrategias de retención deberían focalizarse en clientes de bajo compromiso contractual.

### Limitaciones

- Dataset sintético y desbalance natural de clases.
- Supuesto de linealidad de la Regresión Logística.
- Ausencia de variables temporales y umbral de clasificación fijo en 0.5.
- Monitoreo de calidad de datos en tiempo real y métricas de performance en producción (Precision/Recall/F1) pendientes, por requerir la etiqueta real del cliente.

### Trabajo futuro

Incorporar métricas de calidad de datos en tiempo real en Grafana, cerrar el ciclo de feedback para medir Precision/Recall/F1 en producción, y automatizar el reentrenamiento ante drift sostenido.