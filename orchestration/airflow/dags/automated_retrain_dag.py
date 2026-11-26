"""Automated Retraining DAG for MLOps Weather Early Warning System.

Triggered via Airflow REST API by the Phase 6 Spatial-Temporal Drift Monitor Worker
when statistical distribution drift (PSI > 0.20 or KS p < 0.05) is detected.

4-Task Pipeline:
1. extract_drifted_window: Blends Redis buffer + historical baseline into training/val sets.
2. train_challenger_model: Fits new XGBoost model, logs metrics/params to MLflow, registers version.
3. evaluate_and_gate: Evaluates Champion (Production) vs. Challenger on recent validation split.
4. promote_and_hot_reload: Promotes Challenger to 'Production' alias if gate passes,
   updates baseline_stats.json atomically, triggers zero-downtime hot-reload on FastAPI,
   and clears the Redis debounce lock (retrain_in_progress).
"""

from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from airflow import DAG
from airflow.operators.python import PythonOperator
import joblib
import numpy as np
import pandas as pd
import redis
import requests
import yaml

# Ensure project root & ml_core are in sys.path
def _setup_sys_path():
    candidates = [
        Path("/opt/airflow"),
        Path("/opt/airflow/ml_core"),
        Path("/opt/airflow/services"),
        Path(__file__).resolve().parent.parent.parent.parent,
        Path(__file__).resolve().parent.parent.parent.parent / "ml_core",
        Path(__file__).resolve().parent.parent.parent.parent / "services",
    ]
    for p in candidates:
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))

_setup_sys_path()

try:
    from ml_core.training.retrain_pipeline import (
        extract_and_blend_training_data,
        train_challenger_model as run_train_challenger,
    )
    from ml_core.training.profiler import compute_baseline_profile
    from ml_core.evaluation.comparator import evaluate_and_compare
except ImportError as exc:
    logging.warning("Could not import ml_core modules: %s", exc)

try:
    from services.common.redis_buffer import SpatialTemporalBuffer
except ImportError:
    SpatialTemporalBuffer = None  # type: ignore

logger = logging.getLogger("airflow.task")


def get_project_root() -> Path:
    """Resolve project root directory across container and local environments."""
    if Path("/opt/airflow/data").is_dir():
        return Path("/opt/airflow")
    return Path(__file__).resolve().parent.parent.parent.parent


def get_data_dir() -> Path:
    """Resolve data directory."""
    if Path("/opt/airflow/data").is_dir():
        return Path("/opt/airflow/data")
    return get_project_root() / "data"


def get_settings() -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    cfg_file = get_project_root() / "config" / "settings.yaml"
    if cfg_file.is_file():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Error loading settings from %s: %s", cfg_file, e)
    return {}


def get_mlflow_uri() -> str:
    """Resolve active MLflow tracking URI."""
    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        return env_uri

    settings = get_settings()
    mlflow_cfg = settings.get("orchestration", {}).get("mlflow", {})
    if Path("/opt/airflow").is_dir():
        return mlflow_cfg.get("internal_tracking_uri", "http://mlflow:5000")
    return mlflow_cfg.get("tracking_uri", "http://localhost:5001")


def get_redis_client() -> redis.Redis:
    """Connect to Redis container."""
    host = os.environ.get("REDIS_HOST") or "localhost"
    port = int(os.environ.get("REDIS_PORT") or 6379)
    return redis.Redis(host=host, port=port, db=0, decode_responses=True)


# =========================================================================
# Task 1: Extract Drifted Window & Blend Training Data
# =========================================================================

def extract_drifted_window(**context) -> Dict[str, Any]:
    """Extract recent telemetry from Redis buffer, blend with historical data, write parquet."""
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}
    triggered_by_zone = conf.get("triggered_by_zone")
    max_psi = conf.get("max_psi")
    alert_id = conf.get("alert_id")

    logger.info(
        "Task 1: extract_drifted_window started. Triggered by zone: '%s' (max_psi=%.4f, alert_id=%s)",
        triggered_by_zone,
        max_psi or 0.0,
        alert_id,
    )

    settings = get_settings()
    feature_cols = settings.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    target_col = settings.get("features", {}).get("target", "hazard_label")

    # Connect buffer if available
    buffer_obj = None
    if SpatialTemporalBuffer is not None:
        try:
            r_client = get_redis_client()
            buffer_obj = SpatialTemporalBuffer(client=r_client)
        except Exception as e:
            logger.warning("Could not instantiate SpatialTemporalBuffer: %s", e)

    base_csv = get_data_dir() / "processed" / "baseline_train.csv"
    train_df, val_df = extract_and_blend_training_data(
        redis_buffer=buffer_obj,
        baseline_csv_path=base_csv,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    data_dir = get_data_dir() / "processed"
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "retrain_train.parquet"
    val_path = data_dir / "retrain_val.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    logger.info(
        "Task 1 complete: wrote %s (%d rows) and %s (%d rows)",
        train_path,
        len(train_df),
        val_path,
        len(val_df),
    )

    return {
        "train_parquet": str(train_path),
        "val_parquet": str(val_path),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "triggered_by_zone": triggered_by_zone,
        "alert_id": alert_id,
        "max_psi": max_psi,
    }


# =========================================================================
# Task 2: Train Challenger Model & Log to MLflow
# =========================================================================

def train_challenger_model(**context) -> Dict[str, Any]:
    """Train Challenger XGBoost classifier and register version in MLflow."""
    import mlflow
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    logger.info("Task 2: train_challenger_model started.")
    ti = context.get("ti")
    extract_info = ti.xcom_pull(task_ids="extract_drifted_window") if ti else {}
    train_path = (extract_info or {}).get("train_parquet") or str(
        get_data_dir() / "processed" / "retrain_train.parquet"
    )

    train_df = pd.read_parquet(train_path)

    settings = get_settings()
    feature_cols = settings.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    target_col = settings.get("features", {}).get("target", "hazard_label")
    mlflow_cfg = settings.get("orchestration", {}).get("mlflow", {})
    model_name = mlflow_cfg.get("model_name", "weather-hazard-classifier")
    experiment_name = mlflow_cfg.get("experiment_name", "weather-hazard-early-warning")

    # Fit Challenger estimator
    clf, metadata = run_train_challenger(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    # MLflow tracking
    tracking_uri = get_mlflow_uri()
    logger.info("Connecting to MLflow Tracking Server at %s...", tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    try:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            client.create_experiment(experiment_name)
    except Exception:
        pass

    mlflow.set_experiment(experiment_name)

    run_name = f"challenger_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log params & metadata
        mlflow.log_params(metadata.get("hyperparameters", {}))
        mlflow.log_params({
            "n_train_samples": metadata.get("n_samples"),
            "pos_samples": metadata.get("pos_samples"),
            "neg_samples": metadata.get("neg_samples"),
        })
        mlflow.set_tags({
            "role": "challenger",
            "triggered_by_zone": str((extract_info or {}).get("triggered_by_zone")),
            "alert_id": str((extract_info or {}).get("alert_id")),
            "retrain_timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Register model artifact
        logger.info("Logging Challenger model to MLflow and registering under '%s'...", model_name)
        mlflow.xgboost.log_model(
            xgb_model=clf,
            artifact_path="model",
            registered_model_name=model_name,
        )

    # Find the registered model version number
    versions = client.search_model_versions(f"name = '{model_name}'")
    latest_version = str(max(int(v.version) for v in versions))
    logger.info(
        "Task 2 complete: Registered Challenger Version %s in MLflow (run_id: %s)",
        latest_version,
        run_id,
    )

    return {
        "challenger_version": latest_version,
        "run_id": run_id,
        "model_name": model_name,
    }


# =========================================================================
# Task 3: Evaluate Champion vs. Challenger Gate
# =========================================================================

def evaluate_and_gate(**context) -> Dict[str, Any]:
    """Evaluate current Champion (Production) vs Challenger on holdout validation data."""
    import mlflow.xgboost
    from mlflow.tracking import MlflowClient

    logger.info("Task 3: evaluate_and_gate started.")
    ti = context.get("ti")
    extract_info = ti.xcom_pull(task_ids="extract_drifted_window") if ti else {}
    train_info = ti.xcom_pull(task_ids="train_challenger_model") if ti else {}

    val_path = (extract_info or {}).get("val_parquet") or str(
        get_data_dir() / "processed" / "retrain_val.parquet"
    )
    challenger_version = str((train_info or {}).get("challenger_version", "2"))
    model_name = (train_info or {}).get("model_name", "weather-hazard-classifier")

    val_df = pd.read_parquet(val_path)

    settings = get_settings()
    feature_cols = settings.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )
    target_col = settings.get("features", {}).get("target", "hazard_label")

    tracking_uri = get_mlflow_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # 1. Load current Production Champion
    champion_model = None
    champion_version_str = "v1"
    try:
        champion_uri = f"models:/{model_name}@Production"
        logger.info("Attempting to load Champion from %s...", champion_uri)
        champion_model = mlflow.xgboost.load_model(champion_uri)
        try:
            prod_model = client.get_model_version_by_alias(name=model_name, alias="Production")
            champion_version_str = f"v{prod_model.version}"
        except Exception:
            pass
    except Exception as exc:
        logger.warning(
            "Could not load Champion from %s (%s). Attempting local fallback.",
            champion_uri,
            exc,
        )
        local_model_path = get_project_root() / "ml_core" / "models" / "model_v1.joblib"
        if local_model_path.is_file():
            bundle = joblib.load(local_model_path)
            champion_model = bundle.get("model")

    # 2. Load newly trained Challenger
    challenger_uri = f"models:/{model_name}/{challenger_version}"
    logger.info("Loading Challenger model from %s...", challenger_uri)
    challenger_model = mlflow.xgboost.load_model(challenger_uri)

    # 3. Compare models
    decision = evaluate_and_compare(
        champion_model=champion_model,
        challenger_model=challenger_model,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col=target_col,
        min_improvement=0.01,
    )

    logger.info(
        "Task 3 complete: Promotion Decision -> promote=%s | Improvement: %+.4f F1 (Reason: %s)",
        decision["promote"],
        decision["improvement"],
        decision["reason"],
    )

    decision_ts = datetime.now(timezone.utc).isoformat()
    decision["challenger_version"] = f"v{challenger_version}"
    decision["champion_version"] = champion_version_str
    decision["decision_timestamp"] = decision_ts
    decision["model_name"] = model_name

    # Persist evaluation record to data/reference/evaluation_history.json
    try:
        history_path = get_data_dir() / "reference" / "evaluation_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        eval_history = []
        if history_path.is_file():
            with open(history_path, "r", encoding="utf-8") as f:
                eval_history = json.load(f)
        eval_record = {
            "evaluation_id": f"eval_retrain_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "timestamp": decision_ts,
            "champion_version": champion_version_str,
            "challenger_version": f"v{challenger_version}",
            "champion_metrics": decision.get("champion_metrics"),
            "challenger_metrics": decision.get("challenger_metrics"),
            "f1_improvement": round(float(decision.get("improvement", 0.0)), 4),
            "promoted": bool(decision.get("promote", False)),
            "status": "APPROVED" if decision.get("promote") else "REJECTED",
            "reason": decision.get("reason", ""),
        }
        eval_history.append(eval_record)
        tmp_history = history_path.with_suffix(".json.tmp")
        with open(tmp_history, "w", encoding="utf-8") as f:
            json.dump(eval_history, f, indent=2)
        os.replace(tmp_history, history_path)
        logger.info("Persisted evaluation record to %s", history_path)
    except Exception as exc:
        logger.warning("Could not persist evaluation history: %s", exc)

    return decision


# =========================================================================
# Task 4: Promote, Hot-Reload & Release Debounce Lock
# =========================================================================

def promote_and_hot_reload(**context) -> Dict[str, Any]:
    """Promote Challenger alias, update reference stats atomically, call /reload-model, clear lock."""
    from mlflow.tracking import MlflowClient

    logger.info("Task 4: promote_and_hot_reload started.")
    ti = context.get("ti")
    decision = ti.xcom_pull(task_ids="evaluate_and_gate") if ti else {}
    dag_run = context.get("dag_run")
    conf = dag_run.conf if dag_run and dag_run.conf else {}

    promote = bool((decision or {}).get("promote", False))
    raw_challenger = str((decision or {}).get("challenger_version", "2"))
    challenger_version = raw_challenger if raw_challenger.startswith("v") else f"v{raw_challenger}"
    clean_version = challenger_version.lstrip("v")
    champion_version = str((decision or {}).get("champion_version", "v1"))
    decision_timestamp = (decision or {}).get("decision_timestamp") or datetime.now(timezone.utc).isoformat()
    model_name = (decision or {}).get("model_name", "weather-hazard-classifier")

    event_id = conf.get("event_id") or f"drift_evt_{int(time.time())}"
    trigger_timestamp = conf.get("trigger_timestamp") or conf.get("triggered_at") or decision_timestamp
    affected_cities = conf.get("affected_cities") or conf.get("drifted_zones", [conf.get("triggered_by_zone", "ZONE_KODAGU")])
    city_psi_values = conf.get("city_psi_values", {conf.get("triggered_by_zone", "ZONE_KODAGU"): float(conf.get("max_psi", 0.25))})
    psi_threshold = float(conf.get("psi_threshold", 0.20))
    affected_city_count = int(conf.get("drifted_city_count", len(affected_cities)))
    active_model_at_trigger = str(conf.get("active_model_version", champion_version))

    settings = get_settings()
    feature_cols = settings.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )

    tracking_uri = get_mlflow_uri()
    client = MlflowClient(tracking_uri=tracking_uri)

    hot_reload_success = False
    deployment_timestamp = None
    final_production_version = champion_version
    decision_str = "rejected"

    try:
        if promote:
            decision_str = "approved"
            final_production_version = challenger_version
            logger.info(
                "GATE APPROVED: Promoting Challenger Version %s to 'Production' alias in MLflow...",
                clean_version,
            )
            # 1. Assign Production alias in MLflow
            client.set_registered_model_alias(
                name=model_name,
                alias="Production",
                version=clean_version,
            )

            # 2. Update baseline_stats.json atomically
            train_path = get_data_dir() / "processed" / "retrain_train.parquet"
            if train_path.is_file():
                logger.info("Recomputing reference baseline statistics from %s...", train_path)
                train_df = pd.read_parquet(train_path)
                new_profile = compute_baseline_profile(
                    train_df,
                    feature_cols=feature_cols,
                    zone_col="zone_id",
                    num_bins=10,
                )
                stats_path = get_data_dir() / "reference" / "baseline_stats.json"
                tmp_path = stats_path.with_suffix(".json.tmp")

                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(new_profile, f, indent=2)

                os.replace(tmp_path, stats_path)
                logger.info("Atomically updated reference stats at %s (mtime=%.2f)", stats_path, stats_path.stat().st_mtime)

            # 3. Call FastAPI /reload-model endpoint
            candidate_reload_urls = [
                "http://inference-service:8000/reload-model",
                "http://localhost:8000/reload-model",
                "http://127.0.0.1:8000/reload-model",
            ]
            for url in candidate_reload_urls:
                try:
                    resp = requests.post(url, timeout=5.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        logger.info("Inference service reloaded successfully via %s: %s", url, data)
                        hot_reload_success = True
                        deployment_timestamp = datetime.now(timezone.utc).isoformat()
                        break
                except Exception as e:
                    logger.debug("Could not reach reload endpoint at %s: %s", url, e)

            if not hot_reload_success:
                logger.warning("Could not reach any inference service URL for hot-reload.")
        else:
            decision_str = "rejected"
            final_production_version = champion_version
            deployment_timestamp = None
            logger.info(
                "GATE REJECTED: Retaining current Production Champion (%s). Reason: %s",
                champion_version,
                (decision or {}).get("reason"),
            )

        # 4. Persist / update drift event record in data/reference/drift_event_history.json
        try:
            drift_history_path = get_data_dir() / "reference" / "drift_event_history.json"
            drift_history_path.parent.mkdir(parents=True, exist_ok=True)
            drift_events = []
            if drift_history_path.is_file():
                with open(drift_history_path, "r", encoding="utf-8") as f:
                    try:
                        drift_events = json.load(f)
                    except Exception:
                        drift_events = []

            event_record = {
                "event_id": event_id,
                "trigger_timestamp": trigger_timestamp,
                "affected_cities": affected_cities,
                "city_psi_values": city_psi_values,
                "psi_threshold": psi_threshold,
                "affected_city_count": affected_city_count,
                "active_model_version": active_model_at_trigger,
                "challenger_version": challenger_version,
                "decision": decision_str,
                "champion_challenger_decision": decision_str,
                "final_production_model_version": final_production_version,
                "decision_timestamp": decision_timestamp,
                "deployment_timestamp": deployment_timestamp,
                "f1_improvement": round(float((decision or {}).get("improvement", 0.0)), 4),
                "reason": (decision or {}).get("reason", ""),
                "status": "PROMOTED_AND_DEPLOYED" if promote else "REJECTED_CHAMPION_RETAINED",
            }

            found = False
            for idx, e in enumerate(drift_events):
                if e.get("event_id") == event_id:
                    drift_events[idx].update(event_record)
                    found = True
                    break
            if not found:
                drift_events.append(event_record)

            tmp_drift = drift_history_path.with_suffix(".json.tmp")
            with open(tmp_drift, "w", encoding="utf-8") as f:
                json.dump(drift_events, f, indent=2)
            os.replace(tmp_drift, drift_history_path)
            logger.info("Persisted drift event record %s to %s", event_id, drift_history_path)
        except Exception as exc:
            logger.warning("Could not persist drift event history: %s", exc)

        # 5. Notify FastAPI inference service of final decision & deployment outcome
        outcome_payload = {
            "event_id": event_id,
            "decision": decision_str,
            "promoted": promote,
            "champion_version": champion_version,
            "challenger_version": challenger_version,
            "final_production_model_version": final_production_version,
            "decision_timestamp": decision_timestamp,
            "deployment_timestamp": deployment_timestamp,
            "trigger_timestamp": trigger_timestamp,
            "affected_cities": affected_cities,
            "city_psi_values": city_psi_values,
            "psi_threshold": psi_threshold,
            "affected_city_count": affected_city_count,
            "active_model_version": active_model_at_trigger,
            "f1_improvement": round(float((decision or {}).get("improvement", 0.0)), 4),
            "reason": (decision or {}).get("reason", ""),
        }
        candidate_outcome_urls = [
            "http://inference-service:8000/api/retrain/outcome",
            "http://localhost:8000/api/retrain/outcome",
            "http://127.0.0.1:8000/api/retrain/outcome",
        ]
        for url in candidate_outcome_urls:
            try:
                resp = requests.post(url, json=outcome_payload, timeout=5.0)
                if resp.status_code == 200:
                    logger.info("Reported retraining outcome to %s: %s", url, resp.json())
                    break
            except Exception as e:
                logger.debug("Could not reach outcome endpoint at %s: %s", url, e)

    finally:
        # Clear evaluated sliding window buffer so the historical batch is not re-evaluated continuously
        try:
            r_client = get_redis_client()
            if SpatialTemporalBuffer is not None:
                buf = SpatialTemporalBuffer(client=r_client)
                cleared_keys = buf.clear_all_windows()
                logger.info("Cleared %d Redis sliding window keys after retraining evaluation.", cleared_keys)
        except Exception as exc:
            logger.warning("Could not clear Redis sliding windows: %s", exc)

        # In all branches, clear the Redis debounce lock so subsequent drift events can trigger
        try:
            r = get_redis_client()
            lock_deleted = r.delete("retrain_in_progress")
            logger.info("Released Redis debounce lock 'retrain_in_progress' (deleted=%s)", bool(lock_deleted))
        except Exception as exc:
            logger.error("Failed to release Redis debounce lock: %s", exc)

    return {
        "promoted": promote,
        "promoted_version": challenger_version if promote else None,
        "hot_reload_triggered": hot_reload_success,
    }


# =========================================================================
# DAG Specification
# =========================================================================

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="automated_retrain_dag",
    default_args=default_args,
    description="Automated model retraining triggered by real-time spatial-temporal drift",
    schedule=None,  # Triggered externally via REST API or worker
    start_date=datetime(2023, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "drift-retraining"],
) as dag:

    t1_extract = PythonOperator(
        task_id="extract_drifted_window",
        python_callable=extract_drifted_window,
        provide_context=True,
    )

    t2_train = PythonOperator(
        task_id="train_challenger_model",
        python_callable=train_challenger_model,
        provide_context=True,
    )

    t3_gate = PythonOperator(
        task_id="evaluate_and_gate",
        python_callable=evaluate_and_gate,
        provide_context=True,
    )

    t4_promote = PythonOperator(
        task_id="promote_and_hot_reload",
        python_callable=promote_and_hot_reload,
        trigger_rule="all_done",
        provide_context=True,
    )

    t1_extract >> t2_train >> t3_gate >> t4_promote
