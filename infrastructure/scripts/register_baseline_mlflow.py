"""Register Phase 2 Baseline Model (Model v1) into MLflow Model Registry as Production."""

import argparse
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict
import joblib
import requests
import yaml

# Ensure project root is in sys.path for direct CLI script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def wait_for_mlflow(tracking_uri: str, max_retries: int = 30, retry_delay: float = 2.0) -> bool:
    """Poll MLflow tracking server until healthy or max retries exceeded."""
    health_url = f"{tracking_uri.rstrip('/')}/health"
    logger.info("Checking MLflow health at %s...", health_url)

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(health_url, timeout=3.0)
            if resp.status_code == 200:
                logger.info("MLflow server is healthy and responding.")
                return True
        except requests.exceptions.RequestException:
            pass

        logger.info("MLflow not ready yet (attempt %d/%d). Retrying in %.1fs...", attempt, max_retries, retry_delay)
        time.sleep(retry_delay)

    logger.error("Timed out waiting for MLflow server at %s", tracking_uri)
    return False


def register_model_v1(
    tracking_uri: str = "http://localhost:5001",
    model_path: str = "ml_core/models/model_v1.joblib",
    experiment_name: str = "weather-hazard-early-warning",
    model_name: str = "weather-hazard-classifier",
    alias: str = "Production",
) -> Dict[str, Any]:
    """Load model_v1.joblib, log run to MLflow, register model, and assign Production alias."""
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    project_root = Path(__file__).resolve().parent.parent.parent
    full_model_path = project_root / model_path

    if not full_model_path.is_file():
        raise FileNotFoundError(f"Baseline model artifact not found at {full_model_path}")

    # Ensure MLflow is ready
    if not wait_for_mlflow(tracking_uri):
        raise ConnectionError(f"Cannot connect to MLflow server at {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    logger.info("Loading baseline model artifact bundle from %s...", full_model_path)
    bundle = joblib.load(full_model_path)

    model = bundle["model"]
    metrics = bundle.get("metrics", {})
    hyperparameters = bundle.get("hyperparameters", {})
    feature_cols = bundle.get("feature_cols", [])

    logger.info("Setting MLflow experiment: '%s'...", experiment_name)
    mlflow.set_experiment(experiment_name)

    logger.info("Logging baseline run and registering model under name '%s'...", model_name)
    with mlflow.start_run(run_name="baseline_model_v1_registration") as run:
        run_id = run.info.run_id

        # Log parameters, metrics, tags
        mlflow.log_params(hyperparameters)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "model_version": "v1",
                "phase": "phase_3_registration",
                "author": "senior-mlops-engineer",
                "features_count": len(feature_cols),
            }
        )

        # Log XGBoost model artifact and register to Model Registry
        model_info = mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

    # Retrieve registered version
    logger.info("Retrieving registered model version for '%s'...", model_name)
    versions = client.search_model_versions(f"name = '{model_name}'")
    if not versions:
        raise RuntimeError(f"No versions found for registered model '{model_name}'")

    target_version = max(int(v.version) for v in versions)
    version_str = str(target_version)
    logger.info("Identified registered model version: %s", version_str)

    # 1. Assign Production alias (MLflow 2.x standard)
    logger.info("Assigning alias '%s' to version %s...", alias, version_str)
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=version_str,
    )

    # 2. Transition stage to Production (backwards compatibility)
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage="Production",
            archive_existing_versions=True,
        )
        logger.info("Transitioned version %s stage to 'Production'", version_str)
    except Exception as e:
        logger.debug("Stage transition note: %s", e)

    logger.info(
        "Successfully registered '%s' Version %s under alias '%s' with F1=%.4f, ROC-AUC=%.4f",
        model_name,
        version_str,
        alias,
        metrics.get("f1_score", 0.0),
        metrics.get("roc_auc", 0.0),
    )

    return {
        "model_name": model_name,
        "version": version_str,
        "alias": alias,
        "run_id": run_id,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Register Phase 2 baseline model to MLflow Registry")
    parser.add_argument("--tracking-uri", default="http://localhost:5001", help="MLflow Tracking Server URI")
    parser.add_argument("--model-path", default="ml_core/models/model_v1.joblib", help="Path to model bundle")
    parser.add_argument("--experiment-name", default="weather-hazard-early-warning", help="Experiment name")
    parser.add_argument("--model-name", default="weather-hazard-classifier", help="Registered model name")
    parser.add_argument("--alias", default="Production", help="Model alias to assign")
    args = parser.parse_args()

    register_model_v1(
        tracking_uri=args.tracking_uri,
        model_path=args.model_path,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        alias=args.alias,
    )


if __name__ == "__main__":
    main()
