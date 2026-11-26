"""Unit tests for Phase 1: Project Scaffolding, Data Preparation & Geospatial Encoding."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from ml_core.preprocessing.geo_encoder import (
    assign_spatial_zone,
    encode_geospatial_zones,
    load_spatial_zones,
)
from ml_core.preprocessing.data_fetcher import generate_synthetic_telemetry
from ml_core.preprocessing.cleaner import DataCleaner, load_settings


@pytest.fixture(scope="session")
def project_paths():
    """Return dictionary of essential project paths."""
    root = Path(__file__).resolve().parent.parent.parent
    return {
        "root": root,
        "config_settings": root / "config" / "settings.yaml",
        "config_zones": root / "config" / "spatial_zones.json",
        "raw_csv": root / "data" / "raw" / "weather_telemetry_raw.csv",
        "baseline_csv": root / "data" / "processed" / "baseline_train.csv",
        "stream_csv": root / "data" / "processed" / "stream_simulation.csv",
    }


@pytest.fixture(scope="session", autouse=True)
def ensure_data_artifacts(project_paths):
    """Ensure raw and processed datasets exist before tests run."""
    if not project_paths["raw_csv"].is_file():
        from ml_core.preprocessing.data_fetcher import fetch_or_generate_dataset
        fetch_or_generate_dataset(output_path=project_paths["raw_csv"], num_records=15000, seed=42)

    if not project_paths["baseline_csv"].is_file() or not project_paths["stream_csv"].is_file():
        from ml_core.preprocessing.cleaner import DataCleaner
        cleaner = DataCleaner()
        cleaner.process_and_save(
            raw_csv_path=project_paths["raw_csv"],
            baseline_out_path=project_paths["baseline_csv"],
            stream_out_path=project_paths["stream_csv"],
        )


@pytest.fixture(scope="session")
def spatial_zones_config(project_paths):
    """Load spatial zones configuration."""
    return load_spatial_zones(project_paths["config_zones"])


@pytest.fixture(scope="session")
def project_settings(project_paths):
    """Load settings.yaml configuration."""
    return load_settings(project_paths["config_settings"])


# =========================================================================
# 1. Tests for Spatial Zone Assignment & Vectorized Encoding
# =========================================================================

def test_assign_spatial_zone_known_locations(spatial_zones_config):
    """Verify deterministic mapping for coordinates within known Karnataka/Western Ghats zones."""
    # Kodagu (Coorg Highlands) ~ lat 12.2, lon 75.8
    assert assign_spatial_zone(12.20, 75.80, spatial_zones_config) == "ZONE_KODAGU"

    # Dakshina Kannada (Mangaluru coast) ~ lat 12.87, lon 75.00
    assert assign_spatial_zone(12.87, 75.00, spatial_zones_config) == "ZONE_DAKSHINA_KANNADA"

    # Chikmagalur ~ lat 13.40, lon 75.70
    assert assign_spatial_zone(13.40, 75.70, spatial_zones_config) == "ZONE_CHIKMAGALUR"

    # Shimoga / Agumbe ~ lat 13.90, lon 75.20
    assert assign_spatial_zone(13.90, 75.20, spatial_zones_config) == "ZONE_SHIMOGA_AGUMBE"

    # Uttara Kannada ~ lat 14.60, lon 74.50
    assert assign_spatial_zone(14.60, 74.50, spatial_zones_config) == "ZONE_UTTARA_KANNADA"

    # Wayanad-Nilgiris border ~ lat 11.60, lon 76.20
    assert assign_spatial_zone(11.60, 76.20, spatial_zones_config) == "ZONE_WAYANAD_NILGIRIS"


def test_assign_spatial_zone_out_of_bounds_and_invalid(spatial_zones_config):
    """Verify fallback behavior for coordinates outside Karnataka regions and invalid inputs."""
    # Delhi
    assert assign_spatial_zone(28.6139, 77.2090, spatial_zones_config) == "OUT_OF_BOUNDS"
    # Null island
    assert assign_spatial_zone(0.0, 0.0, spatial_zones_config) == "OUT_OF_BOUNDS"
    # NaN inputs
    assert assign_spatial_zone(np.nan, 75.0, spatial_zones_config) == "OUT_OF_BOUNDS"
    assert assign_spatial_zone(12.5, np.nan, spatial_zones_config) == "OUT_OF_BOUNDS"


def test_assign_spatial_zone_boundaries(spatial_zones_config):
    """Verify exact boundary inclusion."""
    # Boundary of ZONE_KODAGU: min_lat 11.90, min_lon 75.50
    assert assign_spatial_zone(11.90, 75.50, spatial_zones_config) == "ZONE_KODAGU"
    assert assign_spatial_zone(12.60, 76.15, spatial_zones_config) == "ZONE_KODAGU"


def test_encode_geospatial_zones_vectorized(spatial_zones_config):
    """Verify vectorized DataFrame geospatial encoding produces identical results to single assignments."""
    test_df = pd.DataFrame({
        "latitude": [12.20, 12.87, 13.40, 28.61, np.nan],
        "longitude": [75.80, 75.00, 75.70, 77.20, 75.00],
    })

    encoded_df = encode_geospatial_zones(test_df, zones_config=spatial_zones_config)

    expected = [
        "ZONE_KODAGU",
        "ZONE_DAKSHINA_KANNADA",
        "ZONE_CHIKMAGALUR",
        "OUT_OF_BOUNDS",
        "OUT_OF_BOUNDS",
    ]
    assert list(encoded_df["zone_id"]) == expected


# =========================================================================
# 2. Tests for Telemetry Data Generation & Preprocessing
# =========================================================================

def test_deterministic_synthetic_telemetry(spatial_zones_config):
    """Verify synthetic telemetry generator is reproducible given identical random seeds."""
    df1 = generate_synthetic_telemetry(num_records=120, seed=42, zones_config=spatial_zones_config)
    df2 = generate_synthetic_telemetry(num_records=120, seed=42, zones_config=spatial_zones_config)

    pd.testing.assert_frame_equal(df1, df2)
    assert len(df1) == 120

    expected_cols = {
        "timestamp", "latitude", "longitude", "rainfall_mm",
        "soil_moisture", "temperature_c", "humidity_pct",
        "wind_speed_kmh", "hazard_label"
    }
    assert expected_cols.issubset(set(df1.columns))


def test_data_cleaner_schema_and_clamping(project_settings, spatial_zones_config):
    """Verify cleaner handles invalid numbers, clamping, and generates valid schema."""
    cleaner = DataCleaner(settings=project_settings)

    dirty_df = pd.DataFrame({
        "timestamp": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
        "latitude": [12.2, 12.3, 12.4],
        "longitude": [75.8, 75.8, 75.8],
        "rainfall_mm": [-5.0, 500.0, np.nan],  # negative and extreme outlier and NaN
        "soil_moisture": [1.5, -0.2, 0.45],    # out-of-range bounds
        "temperature_c": [28.0, 30.0, 32.0],
        "humidity_pct": [120.0, -10.0, 85.0],  # out-of-range bounds
        "wind_speed_kmh": [15.0, 20.0, -5.0],
        "hazard_label": [0, 1, 99],            # 99 should clamp to 1
    })

    cleaned = cleaner.clean_and_validate(dirty_df, zones_config=spatial_zones_config)

    # 1. Zero NaNs
    assert cleaned.isna().sum().sum() == 0

    # 2. Clamped bounds
    assert (cleaned["rainfall_mm"] >= 0.0).all()
    assert (cleaned["rainfall_mm"] <= 300.0).all()
    assert (cleaned["soil_moisture"] >= 0.0).all()
    assert (cleaned["soil_moisture"] <= 1.0).all()
    assert (cleaned["humidity_pct"] >= 0.0).all()
    assert (cleaned["humidity_pct"] <= 100.0).all()
    assert (cleaned["wind_speed_kmh"] >= 0.0).all()
    assert set(cleaned["hazard_label"].unique()).issubset({0, 1})
    assert "zone_id" in cleaned.columns


# =========================================================================
# 3. Tests for Processed Artifacts Integrity (baseline & stream CSVs)
# =========================================================================

def test_processed_artifacts_exist_and_no_nans(project_paths):
    """Verify baseline_train.csv and stream_simulation.csv exist, have zero NaNs, and strict schema."""
    baseline_path = project_paths["baseline_csv"]
    stream_path = project_paths["stream_csv"]

    assert baseline_path.is_file(), f"Expected artifact {baseline_path} not found"
    assert stream_path.is_file(), f"Expected artifact {stream_path} not found"

    df_base = pd.read_csv(baseline_path)
    df_stream = pd.read_csv(stream_path)

    # Both must have significant row count
    assert len(df_base) >= 1000, f"baseline_train.csv too small: {len(df_base)} rows"
    assert len(df_stream) >= 1000, f"stream_simulation.csv too small: {len(df_stream)} rows"

    # Absolute zero NaNs
    assert df_base.isna().sum().sum() == 0, "Found NaNs in baseline_train.csv"
    assert df_stream.isna().sum().sum() == 0, "Found NaNs in stream_simulation.csv"

    expected_columns = [
        "timestamp",
        "latitude",
        "longitude",
        "zone_id",
        "rainfall_mm",
        "soil_moisture",
        "temperature_c",
        "humidity_pct",
        "wind_speed_kmh",
        "hazard_label",
    ]
    assert list(df_base.columns) == expected_columns, f"Columns mismatch in {baseline_path.name}"
    assert list(df_stream.columns) == expected_columns, f"Columns mismatch in {stream_path.name}"


# =========================================================================
# 4. Tests for Verifiable Statistical Shift (Spatial-Temporal Drift)
# =========================================================================

def _compute_psi(baseline: np.ndarray, target: np.ndarray, num_bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) between two 1D distributions.

    Handles zero-inflated variables (such as precipitation) by separating
    zero values into an isolated bin to accurately measure statistical drift.
    """
    b = baseline[~np.isnan(baseline)]
    t = target[~np.isnan(target)]
    min_v, max_v = float(min(b.min(), t.min())), float(max(b.max(), t.max()))
    if min_v == max_v:
        return 0.0

    # For zero-inflated variables, isolate zero in its own discrete bin
    if np.mean(b == min_v) > 0.3:
        non_zero = b[b > min_v]
        non_zero_min = float(np.min(non_zero)) if len(non_zero) > 0 else min_v + 1e-3
        bin_edges = np.concatenate((
            [min_v - 1e-5, min_v + 1e-5],
            np.linspace(non_zero_min, max_v, num_bins)
        ))
    else:
        bin_edges = np.linspace(min_v, max_v, num_bins + 1)

    b_counts, _ = np.histogram(b, bins=bin_edges)
    t_counts, _ = np.histogram(t, bins=bin_edges)

    # Add small epsilon to prevent division by zero or log(0)
    eps = 1e-4
    b_prop = (b_counts + eps) / (len(b) + eps * len(b_counts))
    t_prop = (t_counts + eps) / (len(t) + eps * len(t_counts))

    psi_val = np.sum((t_prop - b_prop) * np.log(t_prop / b_prop))
    return float(psi_val)


def test_verifiable_statistical_shift(project_paths, project_settings):
    """Verify that stream_simulation.csv exhibits statistically significant drift vs baseline."""
    df_base = pd.read_csv(project_paths["baseline_csv"])
    df_stream = pd.read_csv(project_paths["stream_csv"])

    # 1. Two-sample Kolmogorov-Smirnov test on rainfall_mm
    ks_stat, ks_pvalue = ks_2samp(df_base["rainfall_mm"], df_stream["rainfall_mm"])

    ks_threshold = project_settings.get("drift_thresholds", {}).get("ks_pvalue_threshold", 0.05)
    assert ks_pvalue < ks_threshold, (
        f"Expected statistically significant shift in rainfall (p-value {ks_pvalue} >= {ks_threshold})"
    )

    # 2. Two-sample KS test on soil_moisture
    _, ks_pvalue_sm = ks_2samp(df_base["soil_moisture"], df_stream["soil_moisture"])
    assert ks_pvalue_sm < ks_threshold, (
        f"Expected statistically significant shift in soil moisture (p-value {ks_pvalue_sm} >= {ks_threshold})"
    )

    # 3. Population Stability Index (PSI) check
    rainfall_psi = _compute_psi(df_base["rainfall_mm"].to_numpy(), df_stream["rainfall_mm"].to_numpy())
    psi_threshold = project_settings.get("drift_thresholds", {}).get("psi_threshold", 0.20)

    # Stream should show marked distribution shift (at least moderate/substantial drift)
    assert rainfall_psi >= 0.10, (
        f"Rainfall PSI is {rainfall_psi:.4f}, expected substantial distributional shift (>= 0.10)"
    )

    # 4. Elevated hazard occurrence during extreme monsoon stream simulation
    hazard_rate_baseline = df_base["hazard_label"].mean()
    hazard_rate_stream = df_stream["hazard_label"].mean()

    assert hazard_rate_stream > hazard_rate_baseline, (
        f"Monsoon stream hazard rate ({hazard_rate_stream:.4f}) should exceed baseline ({hazard_rate_baseline:.4f})"
    )
