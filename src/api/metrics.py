from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "api_requests_total",
    "Cantidad total de requests recibidos por la API",
    ["endpoint", "method", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Latencia total de un request HTTP",
    ["endpoint", "method"],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 10],
)

REQUEST_ERRORS = Counter(
    "api_request_errors_total",
    "Cantidad total de requests que terminaron en error",
    ["endpoint", "error_type"],
)

MODEL_AVAILABILITY = Gauge(
    "model_availability",
    "1 si el modelo esta cargado, 0 si no",
)

INFERENCE_DURATION = Histogram(
    "model_inference_duration_seconds",
    "Tiempo de inferencia puro del modelo",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5],
)

PREDICTIONS_TOTAL = Counter(
    "churn_predictions_total",
    "Predicciones de churn por nivel de riesgo y resultado",
    ["risk_level", "churn_prediction"],
)

CHURN_PROBABILITY_SCORE = Histogram(
    "churn_probability_score",
    "Distribucion de la probabilidad de churn",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

CHURN_PROBABILITY_MEAN = Gauge(
    "mlops_churn_probability_mean",
    "Probabilidad media de churn del ultimo batch",
)


def record_request(endpoint, method, status_code, duration_seconds):
    REQUEST_COUNT.labels(endpoint=endpoint, method=method, status_code=str(status_code)).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint, method=method).observe(duration_seconds)
    if status_code >= 400:
        REQUEST_ERRORS.labels(endpoint=endpoint, error_type=f"http_{status_code}").inc()


def record_exception(endpoint, error_type):
    REQUEST_ERRORS.labels(endpoint=endpoint, error_type=error_type).inc()


def record_prediction(churn_prediction, churn_probability, risk_level):
    PREDICTIONS_TOTAL.labels(risk_level=risk_level, churn_prediction=str(churn_prediction)).inc()
    CHURN_PROBABILITY_SCORE.observe(churn_probability)


def set_model_availability(is_available):
    MODEL_AVAILABILITY.set(1 if is_available else 0)


def update_batch_stats(probabilities):
    if probabilities:
        CHURN_PROBABILITY_MEAN.set(sum(probabilities) / len(probabilities))