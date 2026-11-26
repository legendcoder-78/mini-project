"""Data Cleaner & Temporal Splitter for Environmental Telemetry.

Performs schema enforcement, missing value handling, physical boundary clamping,
geospatial zone encoding, chronological sorting, and temporal splitting into:
- data/processed/baseline_train.csv (normal / pre-monsoon baseline)
- data/processed/stream_simulation.csv (seasonal transition + extreme monsoon drift)
"""

import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import yaml

# Ensure project root is in sys.path for direct CLI script execution
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from ml_core.preprocessing.geo_encoder import encode_geospatial_zones, load_spatial_zones

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


class DataCleaner:
    """Production-grade data cleaning, validation, and temporal splitting."""

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        self.settings = settings or load_settings()

        features_cfg = self.settings.get("features", {})
        self.numeric_features = features_cfg.get(
            "numeric_features",
            ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
        )
        self.spatial_features = features_cfg.get("spatial_features", ["latitude", "longitude"])
        self.target_column = features_cfg.get("target", "hazard_label")
        self.zone_column = "zone_id"

        temporal_cfg = self.settings.get("temporal", {})
        self.baseline_ratio = float(temporal_cfg.get("baseline_ratio", 0.60))
        self.split_date_cutoff = temporal_cfg.get("split_date_cutoff", "2023-06-01T00:00:00Z")

        # Physical constraints for anomaly detection & clamping
        self.clamping_bounds = {
            "rainfall_mm": (0.0, 300.0),
            "soil_moisture": (0.0, 1.0),
            "temperature_c": (-10.0, 60.0),
            "humidity_pct": (0.0, 100.0),
            "wind_speed_kmh": (0.0, 200.0),
        }

    def clean_and_validate(self, df: pd.DataFrame, zones_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Validate, clean, and enrich raw environmental telemetry DataFrame.

        Steps:
        1. Parse and format timestamp to ISO-8601 UTC string.
        2. Cast numeric features and coordinates to float.
        3. Impute any missing values (forward fill + backward fill + median).
        4. Apply physical boundary clamping to eliminate sensor glitches.
        5. Encode geospatial zones using geo_encoder.
        6. Cast target to int (0 or 1).
        7. Sort chronologically by timestamp.
        """
        if df.empty:
            raise ValueError("Input DataFrame is empty")

        df_clean = df.copy()

        # 1. Timestamp validation & parsing
        if "timestamp" not in df_clean.columns:
            raise KeyError("DataFrame missing mandatory 'timestamp' column")

        df_clean["dt_parsed"] = pd.to_datetime(df_clean["timestamp"], utc=True, errors="coerce")
        # Drop rows with totally unparseable timestamps
        df_clean = df_clean.dropna(subset=["dt_parsed"]).sort_values("dt_parsed").reset_index(drop=True)
        # Format explicitly as ISO-8601 UTC
        df_clean["timestamp"] = df_clean["dt_parsed"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # 2. Schema type enforcement & imputation for numeric & spatial columns
        all_numeric = self.numeric_features + self.spatial_features
        for col in all_numeric:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
                # Handle NaNs: ffill, bfill, then median
                if df_clean[col].isna().any():
                    median_val = df_clean[col].median()
                    median_val = 0.0 if pd.isna(median_val) else median_val
                    df_clean[col] = df_clean[col].ffill().bfill().fillna(median_val)
            else:
                raise KeyError(f"Expected column '{col}' missing from dataset")

        # 3. Physical boundary clamping
        for col, (min_v, max_v) in self.clamping_bounds.items():
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].clip(lower=min_v, upper=max_v).round(4)

        # 4. Target validation
        if self.target_column in df_clean.columns:
            df_clean[self.target_column] = (
                pd.to_numeric(df_clean[self.target_column], errors="coerce")
                .fillna(0)
                .clip(0, 1)
                .astype(int)
            )
        else:
            raise KeyError(f"Target column '{self.target_column}' missing from dataset")

        # 5. Geospatial zone encoding
        df_clean = encode_geospatial_zones(
            df_clean,
            zones_config=zones_config,
            lat_col=self.spatial_features[0],
            lon_col=self.spatial_features[1],
            target_col=self.zone_column,
        )

        # 6. Final cleanup & column ordering
        expected_cols = (
            ["timestamp"]
            + self.spatial_features
            + [self.zone_column]
            + self.numeric_features
            + [self.target_column]
        )
        # Ensure all expected columns exist
        final_cols = [c for c in expected_cols if c in df_clean.columns]
        df_clean = df_clean[final_cols].drop(columns=["dt_parsed"], errors="ignore").reset_index(drop=True)

        return df_clean

    def temporal_split(
        self, df_clean: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform chronological temporal split into baseline_train and stream_simulation.

        Uses either split_date_cutoff or baseline_ratio.
        """
        if df_clean.empty:
            raise ValueError("Cannot split an empty DataFrame")

        # Chronological split using date cutoff if viable
        if self.split_date_cutoff:
            mask_baseline = df_clean["timestamp"] < self.split_date_cutoff
            baseline_df = df_clean[mask_baseline].reset_index(drop=True)
            stream_df = df_clean[~mask_baseline].reset_index(drop=True)

            # Fallback to ratio if split produces empty slice
            if len(baseline_df) >= 100 and len(stream_df) >= 100:
                logger.info(
                    "Split by date cutoff '%s': baseline=%d rows, stream=%d rows",
                    self.split_date_cutoff,
                    len(baseline_df),
                    len(stream_df),
                )
                return baseline_df, stream_df

        # Ratio-based split fallback
        split_idx = int(len(df_clean) * self.baseline_ratio)
        baseline_df = df_clean.iloc[:split_idx].reset_index(drop=True)
        stream_df = df_clean.iloc[split_idx:].reset_index(drop=True)
        logger.info(
            "Split by ratio (%.2f / %.2f): baseline=%d rows, stream=%d rows",
            self.baseline_ratio,
            1.0 - self.baseline_ratio,
            len(baseline_df),
            len(stream_df),
        )
        return baseline_df, stream_df

    def process_and_save(
        self,
        raw_csv_path: Optional[Union[str, Path]] = None,
        baseline_out_path: Optional[Union[str, Path]] = None,
        stream_out_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Read raw CSV, clean, encode zones, split temporally, and save output CSVs."""
        project_root = Path(__file__).resolve().parent.parent.parent

        if raw_csv_path is None:
            raw_csv_path = project_root / self.settings["data"]["raw_data_path"]
        else:
            raw_csv_path = Path(raw_csv_path)

        if baseline_out_path is None:
            baseline_out_path = project_root / self.settings["data"]["baseline_train_path"]
        else:
            baseline_out_path = Path(baseline_out_path)

        if stream_out_path is None:
            stream_out_path = project_root / self.settings["data"]["stream_simulation_path"]
        else:
            stream_out_path = Path(stream_out_path)

        if not raw_csv_path.is_file():
            logger.info("Raw data file %s not found. Triggering data fetcher...", raw_csv_path)
            from ml_core.preprocessing.data_fetcher import fetch_or_generate_dataset
            fetch_or_generate_dataset(output_path=raw_csv_path, num_records=15000, seed=42)

        logger.info("Reading raw telemetry from %s", raw_csv_path)
        df_raw = pd.read_csv(raw_csv_path)

        df_clean = self.clean_and_validate(df_raw)
        baseline_df, stream_df = self.temporal_split(df_clean)

        baseline_out_path.parent.mkdir(parents=True, exist_ok=True)
        stream_out_path.parent.mkdir(parents=True, exist_ok=True)

        baseline_df.to_csv(baseline_out_path, index=False)
        stream_df.to_csv(stream_out_path, index=False)

        logger.info("Saved baseline training data (%d rows) to %s", len(baseline_df), baseline_out_path)
        logger.info("Saved stream simulation data (%d rows) to %s", len(stream_df), stream_out_path)

        return baseline_df, stream_df


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end Phase 1 data cleaning & splitting."""
    cleaner = DataCleaner()
    return cleaner.process_and_save()


if __name__ == "__main__":
    run_pipeline()
