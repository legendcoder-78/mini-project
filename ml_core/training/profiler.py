"""Statistical Distribution & Reference Baseline Profiler.

Computes baseline reference distributions globally and per spatial zone,
generating histogram bin edges, normalized proportions for PSI calculations,
summary statistics, and sample values for two-sample Kolmogorov-Smirnov tests.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_feature_statistics(
    series: pd.Series,
    num_bins: int = 10,
    max_sample_size: int = 500,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute comprehensive statistical profile and histogram for a single numeric series.

    Args:
        series: Pandas Series with numeric observations.
        num_bins: Number of histogram bins.
        max_sample_size: Maximum sample values to persist for KS tests.
        seed: Random seed for deterministic sample extraction.

    Returns:
        Dict with mean, std, min, max, median, iqr, bin_edges, bin_proportions, sample_values.
    """
    clean_series = pd.to_numeric(series, errors="coerce").dropna()
    vals = clean_series.to_numpy(dtype=float)

    if len(vals) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "iqr": 0.0,
            "bin_edges": [0.0, 1.0],
            "bin_proportions": [1.0],
            "sample_values": [],
        }

    min_v = float(np.min(vals))
    max_v = float(np.max(vals))
    mean_v = float(np.mean(vals))
    std_v = float(np.std(vals))
    median_v = float(np.median(vals))
    q75, q25 = np.percentile(vals, [75, 25])
    iqr_v = float(q75 - q25)

    # Histogram binning strategy:
    # If the feature has uniform values (min == max)
    if np.isclose(min_v, max_v):
        bin_edges = np.array([min_v - 0.1, min_v + 0.1], dtype=float)
    else:
        bin_edges = np.linspace(min_v, max_v, num_bins + 1)

    counts, _ = np.histogram(vals, bins=bin_edges)

    # Laplace / epsilon smoothing so proportions are strictly positive (>0)
    # and sum strictly to 1.0, preventing zero division and log(0) in PSI
    eps = 1e-4
    smoothed_counts = counts.astype(float) + eps
    raw_proportions = smoothed_counts / np.sum(smoothed_counts)

    # Ensure strictly positive and adjust rounding so sum is precisely 1.0
    rounded_proportions = [max(round(float(p), 6), 1e-6) for p in raw_proportions]
    diff = 1.0 - sum(rounded_proportions)
    if rounded_proportions:
        rounded_proportions[0] = round(rounded_proportions[0] + diff, 6)

    # Extract capped sample for two-sample Kolmogorov-Smirnov testing
    rng = np.random.default_rng(seed)
    if len(vals) <= max_sample_size:
        sample = vals
    else:
        sample = rng.choice(vals, size=max_sample_size, replace=False)

    return {
        "mean": round(mean_v, 4),
        "std": round(std_v, 4),
        "min": round(min_v, 4),
        "max": round(max_v, 4),
        "median": round(median_v, 4),
        "iqr": round(iqr_v, 4),
        "bin_edges": [round(float(b), 5) for b in bin_edges],
        "bin_proportions": rounded_proportions,
        "sample_values": [round(float(s), 4) for s in sample],
    }


def compute_baseline_profile(
    df: pd.DataFrame,
    feature_cols: List[str],
    num_bins: int = 10,
    max_sample_size: int = 500,
    zone_col: str = "zone_id",
) -> Dict[str, Any]:
    """Compute baseline reference distributions globally and partitioned per spatial zone.

    Args:
        df: Processed baseline DataFrame.
        feature_cols: Numeric feature column names to profile.
        num_bins: Number of histogram bins per feature.
        max_sample_size: Max sample values retained per feature for KS tests.
        zone_col: Column containing discrete spatial zone identifiers.

    Returns:
        Structured profile dictionary with global and zone-specific reference statistics.
    """
    if df.empty:
        raise ValueError("Cannot profile an empty DataFrame")

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns {missing} not found in DataFrame")

    profile: Dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "num_records": len(df),
            "feature_cols": feature_cols,
            "num_bins": num_bins,
        },
        "global": {},
        "zones": {},
    }

    # 1. Global distribution profiles
    for col in feature_cols:
        profile["global"][col] = compute_feature_statistics(
            df[col], num_bins=num_bins, max_sample_size=max_sample_size, seed=42
        )

    # 2. Zone-specific distribution profiles
    if zone_col in df.columns:
        unique_zones = sorted(df[zone_col].dropna().unique())
        for zone_id in unique_zones:
            zone_df = df[df[zone_col] == zone_id]
            profile["zones"][zone_id] = {
                "num_records": len(zone_df),
            }
            for col in feature_cols:
                profile["zones"][zone_id][col] = compute_feature_statistics(
                    zone_df[col],
                    num_bins=num_bins,
                    max_sample_size=max_sample_size,
                    seed=hash(zone_id) % (2**31),
                )

    return profile


def generate_and_save_profile(
    csv_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    settings_path: Optional[Union[str, Path]] = None,
    num_bins: int = 10,
) -> Dict[str, Any]:
    """Generate baseline profile from baseline_train.csv and serialize to baseline_stats.json."""
    project_root = Path(__file__).resolve().parent.parent.parent

    if settings_path is None:
        settings_path = project_root / "config" / "settings.yaml"
    else:
        settings_path = Path(settings_path)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = yaml.safe_load(f)

    if csv_path is None:
        csv_path = project_root / settings["data"]["baseline_train_path"]
    else:
        csv_path = Path(csv_path)

    if output_path is None:
        output_path = project_root / settings["data"]["baseline_stats_path"]
    else:
        output_path = Path(output_path)

    if not csv_path.is_file():
        raise FileNotFoundError(f"Baseline CSV not found at {csv_path}")

    feature_cols = settings.get("features", {}).get(
        "numeric_features",
        ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
    )

    logger.info("Profiling baseline reference distributions from %s...", csv_path)
    df = pd.read_csv(csv_path)
    profile = compute_baseline_profile(df, feature_cols=feature_cols, num_bins=num_bins)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info("Successfully serialized baseline profile to %s", output_path)
    return profile


if __name__ == "__main__":
    generate_and_save_profile()
