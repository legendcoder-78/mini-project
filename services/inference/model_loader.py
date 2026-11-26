"""Thread-Safe MLflow Production Model Loader & Hot-Reload Manager.

Manages runtime model weights under threading.RLock protection, ensuring
concurrent real-time prediction requests execute without interruption while
allowing atomic hot-reloads when new model versions are promoted in MLflow.
"""

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("model_loader")


def _load_settings() -> Dict[str, Any]:
    """Load system configuration from config/settings.yaml if available."""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    if config_path.is_file():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not read settings from %s: %s", config_path, e)
    return {}


class ModelManager:
    """Thread-safe model manager supporting live inference and atomic hot-reloads."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        model_name: Optional[str] = None,
        alias: str = "Production",
        fallback_path: Optional[str] = None,
        feature_cols: Optional[List[str]] = None,
    ):
        settings = _load_settings()
        mlflow_cfg = settings.get("orchestration", {}).get("mlflow", {})
        features_cfg = settings.get("features", {})

        # Resolve tracking URI (env var -> argument -> settings host/internal -> defaults)
        self.tracking_uri = (
            os.environ.get("MLFLOW_TRACKING_URI")
            or tracking_uri
            or mlflow_cfg.get("tracking_uri", "http://localhost:5001")
        )
        self.model_name = (
            model_name
            or mlflow_cfg.get("model_name", "weather-hazard-classifier")
        )
        self.alias = alias

        # Feature column ordering strict contract
        self.feature_cols = (
            feature_cols
            or features_cfg.get("numeric_features")
            or [
                "rainfall_mm",
                "soil_moisture",
                "temperature_c",
                "humidity_pct",
                "wind_speed_kmh",
            ]
        )

        # Fallback local artifact path
        if fallback_path:
            p = Path(fallback_path)
            self.fallback_path = p if p.is_file() else PROJECT_ROOT / fallback_path
        else:
            self.fallback_path = PROJECT_ROOT / "ml_core" / "models" / "model_v1.joblib"

        self._lock = threading.RLock()
        self._model: Any = None
        self._model_version: str = "unknown"
        self._loaded_at: Optional[str] = None
        self._trained_at: Optional[str] = None
        self._metrics: Optional[Dict[str, float]] = None
        self._source: str = "unloaded"

    @property
    def model_version(self) -> str:
        """Currently active model version string."""
        with self._lock:
            return self._model_version

    @property
    def is_loaded(self) -> bool:
        """Check if an active model instance is loaded."""
        with self._lock:
            return self._model is not None

    def load_production_model(self) -> str:
        """Attempt to load Production model from MLflow Registry with fallback to local joblib.

        Returns:
            Active model version string.
        """
        with self._lock:
            # 1. Attempt MLflow registry load
            mlflow_uris = [
                self.tracking_uri,
                "http://localhost:5001",
                "http://127.0.0.1:5001",
                "http://mlflow:5000",
                "http://localhost:5000",
            ]
            # Deduplicate while preserving order
            seen = set()
            candidate_uris = [u for u in mlflow_uris if u and not (u in seen or seen.add(u))]

            mlflow_loaded = False
            for uri in candidate_uris:
                try:
                    import requests
                    # Fast connectivity probe before invoking heavy MLflow client
                    probe = requests.get(f"{uri.rstrip('/')}/health", timeout=1.5)
                    if probe.status_code != 200:
                        continue

                    import mlflow
                    import mlflow.xgboost
                    from mlflow.tracking import MlflowClient

                    logger.info("Attempting to connect to MLflow at %s...", uri)
                    mlflow.set_tracking_uri(uri)
                    client = MlflowClient(tracking_uri=uri)

                    # Lookup model version under target alias
                    model_version_info = client.get_model_version_by_alias(
                        name=self.model_name,
                        alias=self.alias,
                    )
                    v_str = f"v{model_version_info.version}"

                    model_uri = f"models:/{self.model_name}@{self.alias}"
                    logger.info("Loading model artifact from %s...", model_uri)
                    loaded_xgb = mlflow.xgboost.load_model(model_uri)

                    self._model = loaded_xgb
                    self._model_version = v_str
                    self._loaded_at = datetime.now(timezone.utc).isoformat()
                    self._source = f"mlflow:{uri}@{self.alias}"

                    # Extract actual metrics and training timestamp from MLflow run
                    try:
                        run_info = client.get_run(model_version_info.run_id)
                        raw_metrics = run_info.data.metrics or {}
                        self._metrics = {k: round(float(v), 4) for k, v in raw_metrics.items()}
                        self._trained_at = datetime.fromtimestamp(
                            run_info.info.start_time / 1000.0, tz=timezone.utc
                        ).isoformat()
                    except Exception as e:
                        logger.debug("Could not fetch metrics for MLflow run: %s", e)
                        self._metrics = None

                    # If MLflow run didn't have metrics logged directly, check local joblib
                    if not self._metrics and self.fallback_path.is_file():
                        try:
                            b = joblib.load(self.fallback_path)
                            if isinstance(b, dict) and "metrics" in b:
                                self._metrics = b["metrics"]
                                self._trained_at = b.get("trained_at", self._trained_at)
                        except Exception:
                            pass

                    mlflow_loaded = True
                    logger.info(
                        "Successfully loaded model '%s' (version=%s) from MLflow (%s)",
                        self.model_name,
                        self._model_version,
                        uri,
                    )
                    break
                except Exception as exc:
                    logger.debug("Could not load from MLflow (%s): %s", uri, exc)

            if mlflow_loaded:
                return self._model_version

            # 2. Graceful fallback to local model artifact
            logger.warning(
                "MLflow server unavailable. Falling back to local model artifact: %s",
                self.fallback_path,
            )
            if not self.fallback_path.is_file():
                raise FileNotFoundError(
                    f"Neither MLflow nor local fallback model found at {self.fallback_path}"
                )

            bundle = joblib.load(self.fallback_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self._model = bundle["model"]
                self._model_version = bundle.get("model_version", "v1")
                self._metrics = bundle.get("metrics")
                self._trained_at = bundle.get("trained_at")
            else:
                self._model = bundle
                self._model_version = "v1"
                self._metrics = None
                self._trained_at = None

            self._loaded_at = datetime.now(timezone.utc).isoformat()
            self._source = f"local_fallback:{self.fallback_path.name}"
            logger.info(
                "Fallback model loaded successfully from %s (version=%s)",
                self.fallback_path,
                self._model_version,
            )
            return self._model_version

    def predict(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Execute model inference with strict feature order enforcement.

        Args:
            features: Dictionary containing required feature columns.

        Returns:
            Tuple of (predicted_label: int, hazard_probability: float).
        """
        with self._lock:
            if self._model is None:
                self.load_production_model()

            # Strict ordering according to self.feature_cols
            missing = set(self.feature_cols) - set(features.keys())
            if missing:
                raise ValueError(f"Features missing required columns: {sorted(list(missing))}")

            ordered_row = [float(features[col]) for col in self.feature_cols]
            df = pd.DataFrame([ordered_row], columns=self.feature_cols)

            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(df)[0]
                hazard_prob = float(proba[1])
                pred_label = int(hazard_prob >= 0.5)
            else:
                pred_label = int(self._model.predict(df)[0])
                hazard_prob = float(pred_label)

            return pred_label, hazard_prob

    def reload(self) -> Dict[str, Any]:
        """Atomically hot-reload model from MLflow Registry without dropping concurrent requests."""
        with self._lock:
            prev_version = self._model_version
            prev_source = self._source
            logger.info("Initiating hot-reload (current_version=%s)...", prev_version)

            new_version = self.load_production_model()
            logger.info(
                "Hot-reload complete. Version transitioned: %s -> %s (source=%s)",
                prev_version,
                new_version,
                self._source,
            )

            return {
                "status": "reloaded",
                "previous_version": prev_version,
                "previous_source": prev_source,
                "active_version": new_version,
                "active_source": self._source,
                "reloaded_at": self._loaded_at,
            }

    def get_metadata(self) -> Dict[str, Any]:
        """Return runtime metadata and health properties of active model."""
        with self._lock:
            return {
                "model_name": self.model_name,
                "model_version": self._model_version,
                "alias": self.alias,
                "source": self._source,
                "loaded_at": self._loaded_at,
                "trained_at": self._trained_at,
                "metrics": self._metrics,
                "feature_cols": list(self.feature_cols),
                "is_loaded": self._model is not None,
            }

    def get_metrics(self) -> Optional[Dict[str, float]]:
        """Return actual evaluation metrics for active model."""
        with self._lock:
            return self._metrics
