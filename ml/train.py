"""Train a baseline fraud detection model and log it to MLflow.

Day 1 goals:
    1. Produce a logged MLflow run with params, metrics, and a model artifact.
    2. Register the model in the MLflow Model Registry under a stable name.
    3. Promote the new version to the 'staging' alias so the API (Day 2) has
       a deterministic pointer to load.

Run:
    # In one terminal:
    mlflow server --host 127.0.0.1 --port 5000 \\
        --backend-store-uri sqlite:///mlflow.db \\
        --default-artifact-root ./mlartifacts

    # In another terminal:
    python -m ml.train
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass

import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml.data import FEATURE_NAMES, load_synthetic_fraud

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("train")

MODEL_NAME = "fraud-detector"
EXPERIMENT_NAME = "fraud-detection"
STAGING_ALIAS = "staging"


@dataclass
class TrainConfig:
    n_estimators: int = 200
    max_depth: int = 10
    min_samples_leaf: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_samples: int = 10_000
    fraud_rate: float = 0.05


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-samples", type=int, default=10_000)
    p.add_argument("--fraud-rate", type=float, default=0.05)
    args = p.parse_args()
    return TrainConfig(**vars(args))


def evaluate(model, X_test, y_test) -> dict[str, float]:
    """Compute the metrics that actually matter for an imbalanced problem."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "roc_auc": roc_auc_score(y_test, y_proba),
        "pr_auc": average_precision_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }


def promote_to_staging(client: mlflow.MlflowClient, version: str) -> None:
    """Set the 'staging' alias on this version.

    Aliases (introduced in MLflow 2.9) replace the deprecated 'stages' API.
    The API server can then load `models:/fraud-detector@staging` and always
    get the current staging model without code changes.
    """
    try:
        client.set_registered_model_alias(name=MODEL_NAME, alias=STAGING_ALIAS, version=version)
        log.info("Set alias '%s' -> %s version %s", STAGING_ALIAS, MODEL_NAME, version)
    except MlflowException as e:
        log.error("Failed to set alias: %s", e)
        raise


def main() -> None:
    cfg = parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    log.info("Tracking URI: %s", tracking_uri)

    log.info("Generating synthetic fraud dataset (n=%d)", cfg.n_samples)
    X, y = load_synthetic_fraud(
        n_samples=cfg.n_samples,
        fraud_rate=cfg.fraud_rate,
        random_state=cfg.random_state,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,  # critical for imbalanced data
    )

    with mlflow.start_run() as run:
        log.info("MLflow run_id: %s", run.info.run_id)

        # Params
        mlflow.log_params(
            {
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "min_samples_leaf": cfg.min_samples_leaf,
                "test_size": cfg.test_size,
                "random_state": cfg.random_state,
                "n_samples": cfg.n_samples,
                "fraud_rate": cfg.fraud_rate,
                "model_type": "RandomForestClassifier",
                "dataset": "synthetic_fraud_v1",
            }
        )

        # Train
        model = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced",
            random_state=cfg.random_state,
            n_jobs=-1,
        )
        log.info("Fitting model...")
        model.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        log.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

        # Log model with signature + input example
        # Signature gives the API server (Day 2) a schema to validate against.
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(3)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=MODEL_NAME,
        )
        log.info("Logged + registered model as '%s'", MODEL_NAME)

        # Log feature names as a tag for traceability
        mlflow.set_tag("feature_names", ",".join(FEATURE_NAMES))
        mlflow.set_tag("git_commit", os.getenv("GIT_COMMIT", "local"))

    # Promote the just-registered version to staging
    # client = mlflow.MlflowClient()
    # latest = client.get_latest_versions(MODEL_NAME)
    # if not latest:
    #     raise RuntimeError(f"No versions found for {MODEL_NAME}")
    # newest_version = max(latest, key=lambda v: int(v.version)).version
    # promote_to_staging(client, newest_version)

    # Promote the just-registered version to staging.
    # search_model_versions is the supported replacement for get_latest_versions,
    # which was deprecated in MLflow 2.9 alongside the legacy stages API.
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(f"No versions found for {MODEL_NAME}")
    newest_version = max(versions, key=lambda v: int(v.version)).version
    promote_to_staging(client, newest_version)

    log.info("Done. Model URI for serving: models:/%s@%s", MODEL_NAME, STAGING_ALIAS)


if __name__ == "__main__":
    main()
