"""Prometheus metrics stub. Fleshed out on Day 6."""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Day 6 will add: latency histogram per endpoint, error counter,
# fraud-rate gauge, model-version label, etc.
prediction_counter = Counter(
    "fraud_predictions_total",
    "Total predictions served",
    labelnames=("model_version", "outcome"),
)

prediction_latency = Histogram(
    "fraud_prediction_duration_seconds",
    "Prediction request latency",
    labelnames=("model_version",),
)


def render_metrics() -> tuple[bytes, str]:
    """Return (payload, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
