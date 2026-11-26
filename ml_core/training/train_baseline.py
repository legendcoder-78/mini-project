"""Baseline Model Training & Validation Pipeline for Extreme-Weather Early Warning.

Trains Model v1 (XGBoost Classifier) with class-imbalance weighting, evaluates performance
metrics (F1, ROC-AUC, Precision, Recall, Accuracy), serializes the model bundle to
ml_core/models/model_v1.joblib, supports optional MLflow logging, and triggers
baseline statistical distribution profiling.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import yaml

# Ensure project root is in sys.path for direct CLI script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml_core.training.profiler import generate_and_save_profile

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_settings(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load settings from settings.yaml."""
    if config_path is None:
        project_root = Path(__file__).resolve().parent.parent.parent
        config_path = project_root / "config" / "settings.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Settings file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_baseline_model(
    csv_path: Optional[Union[str, Path]] = None,
    model_output_path: Optional[Union[str, Path]] = None,
    settings_path: Optional[Union[str, Path]] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train, evaluate, and save Model v1 XGBoost classifier.

    Args:
        csv_path: Path to baseline_train.csv.
        model_output_path: Destination path for model_v1.joblib.
        settings_path: Path to settings.yaml.
        test_size: Validation holdout split fraction.
        random_state: Deterministic random seed.

    Returns:
        Dict containing model artifact metadata, metrics, and file paths.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    settings = load_settings(settings_path)

    if csv_path is None:
        csv_path = project_root / settings["data"]["baseline_train_path"]
    else:
        csv_path = Path(csv_path)

    if model_output_path is None:
        model_output_path = project_root / "ml_core" / "models" / "model_v1.joblib"
    else:
        model_output_path = Path(model_output_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Baseline training data not found at {csv_path}")

    logger.info("Loading baseline training data from %s...", csv_path)
    df = pd.read_csv(csv_path)

    feature_cols = settings.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    target_col = settings.get("features", {}).get("target", "hazard_label")

    X = df[feature_cols]
    y = df[target_col].astype(int)

    num_samples = len(df)
    num_positives = int(y.sum())
    logger.info(
        "Dataset loaded: %d samples, %d positive hazard events (%.2f%%)",
        num_samples,
        num_positives,
        (num_positives / num_samples) * 100.0,
    )

    # Stratified split to ensure validation set reflects severe hazard distribution
    stratify = y if num_positives >= 5 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    # Class imbalance compensation
    pos_train = max(int(y_train.sum()), 1)
    neg_train = len(y_train) - pos_train
    scale_pos_weight = float(neg_train / pos_train)
    logger.info("Computed scale_pos_weight: %.2f", scale_pos_weight)

    hyperparameters = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.08,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": 2,
    }

    logger.info("Training XGBoost Classifier (Model v1)...")
    clf = XGBClassifier(**hyperparameters)
    clf.fit(X_train, y_train)

    # Model evaluation on validation split
    y_pred = clf.predict(X_val)
    y_prob = clf.predict_proba(X_val)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "precision": round(float(precision_score(y_val, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_val, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_val, y_pred, zero_division=0)), 4),
        "roc_auc": round(
            float(roc_auc_score(y_val, y_prob)) if len(np.unique(y_val)) > 1 else 1.0, 4
        ),
    }

    logger.info(
        "Validation Metrics: Accuracy=%.4f, Precision=%.4f, Recall=%.4f, F1=%.4f, ROC-AUC=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    logger.info("Classification Report:\n%s", classification_report(y_val, y_pred, zero_division=0))

    # Save artifact bundle with complete metadata
    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": clf,
        "model_version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_cols": feature_cols,
        "target_col": target_col,
        "metrics": metrics,
        "hyperparameters": hyperparameters,
        "validation_samples": len(y_val),
    }
    joblib.dump(artifact, model_output_path)
    logger.info("Saved trained model artifact to %s", model_output_path)

    # Optional MLflow tracking (safe try/except for offline execution)
    try:
        import mlflow
        import mlflow.xgboost

        tracking_uri = settings.get("orchestration", {}).get("mlflow_tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        with mlflow.start_run(run_name="baseline_model_v1"):
            mlflow.log_params(hyperparameters)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(clf, artifact_path="model")
            logger.info("Logged parameters and metrics to MLflow Tracking Server")
    except Exception as e:
        logger.debug("MLflow logging skipped (offline/Phase 2): %s", e)

    # Trigger reference distribution profiling
    logger.info("Triggering reference baseline profiling...")
    profile = generate_and_save_profile(csv_path=csv_path, settings_path=settings_path)

    return {
        "model": clf,
        "metrics": metrics,
        "artifact_path": str(model_output_path),
        "profile": profile,
    }


if __name__ == "__main__":
    train_baseline_model()
