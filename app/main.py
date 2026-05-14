"""FastAPI service for fraud-detector."""

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Response

from app.metrics import (
    prediction_counter,
    prediction_latency,
    render_metrics,
)
from app.model_loader import (
    MLFLOW_TRACKING_URI,
    LoadedModel,
    load_model_by_alias,
)
from app.schemas import HealthResponse, PredictionResponse, Transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, attach to app state."""
    logger.info("Starting up, loading model...")
    app.state.model_bundle = load_model_by_alias()
    logger.info(
        "Model loaded: %s v%s (alias=%s)",
        app.state.model_bundle.name,
        app.state.model_bundle.version,
        app.state.model_bundle.alias,
    )
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Fraud Detection API",
    description="Serves fraud-detector model via MLflow registry alias.",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    bundle: LoadedModel = app.state.model_bundle
    return HealthResponse(
        status="ok",
        model_name=bundle.name,
        model_version=bundle.version,
        model_alias=bundle.alias,
        mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(txn: Transaction) -> PredictionResponse:
    bundle: LoadedModel = app.state.model_bundle
    model = bundle.model

    df = pd.DataFrame([txn.to_feature_dict()])

    start = time.perf_counter()
    try:
        proba = float(model.predict_proba(df)[0, 1])
        pred = int(proba >= 0.5)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Inference error: {e}") from e
    finally:
        prediction_latency.labels(model_version=bundle.version).observe(time.perf_counter() - start)

    prediction_counter.labels(
        model_version=bundle.version,
        outcome="fraud" if pred == 1 else "legit",
    ).inc()

    return PredictionResponse(
        prediction=pred,
        fraud_probability=proba,
        model_version=bundle.version,
        model_alias=bundle.alias,
    )


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)
