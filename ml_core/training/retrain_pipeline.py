"""Modular Retraining Pipeline for Automated Model Retraining.

Extracts live sliding-window telemetry from Redis buffers, filters for verified
ground-truth labels, blends with historical baseline training data to prevent
catastrophic forgetting, and trains a new Challenger XGBoost model with class-imbalance
compensation.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from services.common.redis_buffer import SpatialTemporalBuffer
except ImportError:
    SpatialTemporalBuffer = Any  # type: ignore

logger = logging.getLogger("retrain_pipeline")


def _load_settings(settings_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from settings.yaml."""
    if settings_path:
        path = Path(settings_path)
    else:
        path = PROJECT_ROOT / "config" / "settings.yaml"

    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning("Could not read settings from %s: %s", path, e)
    return {}


def extract_and_blend_training_data(
    redis_buffer: Optional[Any] = None,
    baseline_csv_path: Optional[Union[str, Path]] = None,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "hazard_label",
    historical_sample_size: int = 2500,
    val_split_ratio: float = 0.25,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Extract recent drifted records from Redis and blend with historical baseline.

    Args:
        redis_buffer: SpatialTemporalBuffer instance or None (uses default if None).
        baseline_csv_path: Path to baseline_train.csv.
        feature_cols: List of numeric meteorological features.
        target_col: Target binary label column name.
        historical_sample_size: Maximum historical baseline samples to blend.
        val_split_ratio: Fraction of recent drifted regime held out for validation.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (train_df: pd.DataFrame, val_df: pd.DataFrame).
    """
    settings = _load_settings()
    if feature_cols is None:
        feature_cols = settings.get("features", {}).get(
            "numeric_features",
            ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
        )

    # 1. Resolve path to historical baseline training data
    if baseline_csv_path is None:
        raw_base_path = settings.get("data", {}).get(
            "baseline_train_path", "data/processed/baseline_train.csv"
        )
        base_path = (
            Path("/opt/airflow") / raw_base_path
            if Path("/opt/airflow/data").is_dir()
            else PROJECT_ROOT / raw_base_path
        )
    else:
        base_path = Path(baseline_csv_path)

    # 2. Extract recent window records from Redis buffer
    drifted_records: List[Dict[str, Any]] = []

    if redis_buffer is not None:
        try:
            active_zones = redis_buffer.get_all_active_zones()
            logger.info("Found %d active spatial zones in Redis buffer", len(active_zones))
            for zone_id in active_zones:
                zone_df = redis_buffer.get_zone_dataframe(zone_id)
                if not zone_df.empty:
                    drifted_records.extend(zone_df.to_dict(orient="records"))
        except Exception as exc:
            logger.warning("Error reading from Redis buffer: %s. Proceeding with fallback.", exc)

    drifted_df = pd.DataFrame(drifted_records) if drifted_records else pd.DataFrame()

    # 3. Filter for records with verified ground truth label
    if not drifted_df.empty and "delayed_label" in drifted_df.columns:
        # Keep non-null binary labels
        valid_mask = drifted_df["delayed_label"].notna() & drifted_df["delayed_label"].isin([0, 1, 0.0, 1.0])
        drifted_df = drifted_df[valid_mask].copy()
        drifted_df[target_col] = drifted_df["delayed_label"].astype(int)

    # Fallback to stream simulation if buffer is empty or contains insufficient labeled rows
    if drifted_df.empty or len(drifted_df) < 50:
        sim_path = (
            Path("/opt/airflow/data/processed/stream_simulation.csv")
            if Path("/opt/airflow/data").is_dir()
            else PROJECT_ROOT / "data" / "processed" / "stream_simulation.csv"
        )
        if sim_path.is_file():
            logger.info("Augmenting drifted dataset from stream simulation archive: %s", sim_path)
            sim_df = pd.read_csv(sim_path)
            if target_col in sim_df.columns:
                # Sample recent drifted regime from monsoon simulation
                sample_n = min(1500, len(sim_df))
                drifted_df = sim_df.sample(n=sample_n, random_state=random_state).copy()

    if drifted_df.empty:
        raise ValueError("No drifted or stream telemetry data available for retraining.")

    # Clean missing feature columns in drifted dataset
    for col in feature_cols:
        if col not in drifted_df.columns:
            drifted_df[col] = 0.0
        drifted_df[col] = pd.to_numeric(drifted_df[col], errors="coerce").fillna(0.0)

    drifted_df[target_col] = drifted_df[target_col].astype(int)

    # 4. Split recent drifted regime into training and validation sets
    pos_count = int(drifted_df[target_col].sum())
    stratify = drifted_df[target_col] if (pos_count >= 5 and (len(drifted_df) - pos_count) >= 5) else None

    drift_train, val_df = train_test_split(
        drifted_df,
        test_size=val_split_ratio,
        random_state=random_state,
        stratify=stratify,
    )
    logger.info(
        "Drift regime split: %d training records, %d holdout validation records (pos=%d)",
        len(drift_train),
        len(val_df),
        int(val_df[target_col].sum()),
    )

    # 5. Blend training set with stratified historical baseline to prevent catastrophic forgetting
    if base_path.is_file():
        logger.info("Blending with historical baseline dataset from %s...", base_path)
        baseline_df = pd.read_csv(base_path)
        for col in feature_cols:
            if col in baseline_df.columns:
                baseline_df[col] = pd.to_numeric(baseline_df[col], errors="coerce").fillna(0.0)

        if target_col in baseline_df.columns:
            baseline_df[target_col] = baseline_df[target_col].astype(int)
            b_pos = int(baseline_df[target_col].sum())
            b_strat = baseline_df[target_col] if (b_pos >= 5 and (len(baseline_df) - b_pos) >= 5) else None

            sample_size = min(historical_sample_size, len(baseline_df))
            if b_strat is not None:
                _, hist_sample = train_test_split(
                    baseline_df,
                    test_size=sample_size,
                    random_state=random_state,
                    stratify=b_strat,
                )
            else:
                hist_sample = baseline_df.sample(n=sample_size, random_state=random_state)

            train_df = pd.concat([drift_train, hist_sample], ignore_index=True)
        else:
            train_df = drift_train
    else:
        logger.warning("Historical baseline CSV not found at %s. Using pure drift dataset.", base_path)
        train_df = drift_train

    # Shuffle combined training set
    train_df = train_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    logger.info(
        "Data extraction complete -> Blended Train: %d records (pos=%d, %.1f%%) | Val: %d records (pos=%d, %.1f%%)",
        len(train_df),
        int(train_df[target_col].sum()),
        (train_df[target_col].sum() / len(train_df)) * 100.0,
        len(val_df),
        int(val_df[target_col].sum()),
        (val_df[target_col].sum() / len(val_df)) * 100.0 if len(val_df) > 0 else 0.0,
    )

    return train_df, val_df


def train_challenger_model(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "hazard_label",
    n_estimators: int = 120,
    max_depth: int = 5,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any]]:
    """Train new Challenger XGBoost model with class-imbalance compensation.

    Args:
        train_df: Blended training DataFrame.
        feature_cols: Numeric meteorological feature columns.
        target_col: Target binary hazard label column name.
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth.
        learning_rate: Boosting learning rate.
        random_state: Random state seed.

    Returns:
        Tuple of (trained_model: XGBClassifier, metadata: Dict[str, Any]).
    """
    settings = _load_settings()
    if feature_cols is None:
        feature_cols = settings.get("features", {}).get(
            "numeric_features",
            ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
        )

    X = train_df[feature_cols].copy()
    y = train_df[target_col].astype(int).copy()

    num_samples = len(train_df)
    pos_samples = max(int(y.sum()), 1)
    neg_samples = num_samples - pos_samples

    scale_pos_weight = float(neg_samples / pos_samples)
    logger.info(
        "Challenger training: %d samples, %d positive (%d negative), scale_pos_weight=%.2f",
        num_samples,
        pos_samples,
        neg_samples,
        scale_pos_weight,
    )

    hyperparameters = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "learning_rate": learning_rate,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
        "eval_metric": "logloss",
        "n_jobs": 2,
    }

    logger.info("Fitting Challenger XGBoost model...")
    clf = XGBClassifier(**hyperparameters)
    clf.fit(X, y)

    metadata = {
        "model_role": "challenger",
        "n_samples": num_samples,
        "pos_samples": pos_samples,
        "neg_samples": neg_samples,
        "scale_pos_weight": round(scale_pos_weight, 4),
        "feature_cols": feature_cols,
        "hyperparameters": hyperparameters,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    return clf, metadata
