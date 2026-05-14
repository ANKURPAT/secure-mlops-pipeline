"""Model loading with explicit alias resolution.

Why explicit resolution: we want the version number available at request
time so /health and Prometheus labels can expose it. Loading via
`models:/name@alias` works but hides the version from the application.
"""

import logging
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@dataclass
class LoadedModel:
    model: object  # sklearn estimator wrapped by mlflow
    name: str
    version: str
    alias: str
    run_id: str


def load_model_by_alias(
    name: str = MODEL_NAME,
    alias: str = MODEL_ALIAS,
) -> LoadedModel:
    """Resolve alias -> version explicitly, then load.

    Raises if the alias is not set on any version of the model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    mv = client.get_model_version_by_alias(name=name, alias=alias)
    model_uri = f"models:/{name}/{mv.version}"
    logger.info(
        "Loading model name=%s alias=%s -> version=%s run_id=%s",
        name,
        alias,
        mv.version,
        mv.run_id,
    )
    model = mlflow.sklearn.load_model(model_uri)

    return LoadedModel(
        model=model,
        name=name,
        version=mv.version,
        alias=alias,
        run_id=mv.run_id,
    )
