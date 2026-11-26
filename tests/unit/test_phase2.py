"""Unit tests for Phase 2: Offline Baseline Modeling & Reference Distribution Profiling."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
import yaml

from ml_core.training.profiler import compute_baseline_profile, compute_feature_statistics


@pytest.fixture(scope="session")
def project_paths():
    """Return dictionary of essential project paths."""
    root = Path(__file__).resolve().parent.parent.parent
    return {
        "root": root,
        "config_settings": root / "config" / "settings.yaml",
        "config_zones": root / "config" / "spatial_zones.json",
        "baseline_csv": root / "data" / "processed" / "baseline_train.csv",
        "baseline_stats": root / "data" / "reference" / "baseline_stats.json",
        "model_v1": root / "ml_core" / "models" / "model_v1.joblib",
    }


@pytest.fixture(scope="session")
def project_settings(project_paths):
    """Load settings.yaml configuration."""
    with open(project_paths["config_settings"], "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def spatial_zones_config(project_paths):
    """Load spatial_zones.json configuration."""
    with open(project_paths["config_zones"], "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session", autouse=True)
def ensure_model_and_stats(project_paths):
    """Ensure Model v1 artifact and baseline_stats.json exist before running tests."""
    if not project_paths["model_v1"].is_file() or not project_paths["baseline_stats"].is_file():
        from ml_core.training.train_baseline import train_baseline_model
        train_baseline_model(
            csv_path=project_paths["baseline_csv"],
            model_output_path=project_paths["model_v1"],
            settings_path=project_paths["config_settings"],
        )


# =========================================================================
# 1. Tests for Model v1 Artifact & Inference Contract
# =========================================================================

def test_model_v1_artifact_integrity(project_paths, project_settings):
    """Verify that model_v1.joblib exists, loads cleanly, and carries complete metadata."""
    model_path = project_paths["model_v1"]
    assert model_path.is_file(), f"Model artifact not found at {model_path}"

    bundle = joblib.load(model_path)
    assert isinstance(bundle, dict), "Model artifact must be saved as a dictionary bundle"

    required_keys = {"model", "model_version", "trained_at", "feature_cols", "target_col", "metrics"}
    assert required_keys.issubset(set(bundle.keys())), f"Missing keys in bundle: {required_keys - set(bundle.keys())}"

    assert bundle["model_version"] == "v1"
    assert bundle["target_col"] == "hazard_label"

    expected_features = project_settings.get("features", {}).get("numeric_features")
    assert bundle["feature_cols"] == expected_features, "Model feature columns mismatch settings.yaml"


def test_model_v1_inference_contract(project_paths):
    """Verify Model v1 accepts feature DataFrame and outputs valid probabilities in [0.0, 1.0]."""
    bundle = joblib.load(project_paths["model_v1"])
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Synthetic test points:
    # Point 0: Benign dry day (low rain, low soil moisture)
    # Point 1: Extreme storm event (heavy cloudburst rain, saturated soil, high wind)
    test_df = pd.DataFrame(
        [
            {
                "rainfall_mm": 0.0,
                "soil_moisture": 0.22,
                "temperature_c": 29.5,
                "humidity_pct": 55.0,
                "wind_speed_kmh": 8.5,
            },
            {
                "rainfall_mm": 55.0,
                "soil_moisture": 0.85,
                "temperature_c": 21.0,
                "humidity_pct": 98.0,
                "wind_speed_kmh": 62.0,
            },
        ]
    )[feature_cols]

    # Predict discrete class
    preds = model.predict(test_df)
    assert set(preds).issubset({0, 1}), f"Predictions must be binary 0 or 1, got {preds}"
    assert preds[0] == 0, "Benign weather condition should predict 0 (Normal)"
    assert preds[1] == 1, "Extreme convective storm condition should predict 1 (Hazard)"

    # Predict probabilities
    probs = model.predict_proba(test_df)
    assert probs.shape == (2, 2), f"Expected proba shape (2, 2), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Probabilities must lie strictly in [0.0, 1.0]"
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1e-5)

    # Risk score for extreme event should be significantly higher than benign condition
    assert probs[1, 1] > probs[0, 1], f"Extreme prob ({probs[1, 1]}) should exceed benign prob ({probs[0, 1]})"


def test_model_v1_baseline_performance(project_paths):
    """Verify Model v1 achieves target baseline metrics (F1 > 0.70, ROC-AUC > 0.80)."""
    bundle = joblib.load(project_paths["model_v1"])
    metrics = bundle["metrics"]

    assert metrics["f1_score"] > 0.70, f"F1 score {metrics['f1_score']} is below target threshold of 0.70"
    assert metrics["roc_auc"] > 0.80, f"ROC-AUC {metrics['roc_auc']} is below target threshold of 0.80"
    assert metrics["accuracy"] > 0.95, f"Accuracy {metrics['accuracy']} is unexpectedly low"
    assert metrics["recall"] > 0.70, f"Recall {metrics['recall']} is below 0.70"


# =========================================================================
# 2. Tests for Baseline Reference Statistics & Profiler
# =========================================================================

def test_profiler_unit_feature_statistics():
    """Verify single-feature statistics computation and smoothed histogram proportions."""
    series = pd.Series([0.0, 0.0, 0.0, 1.2, 5.5, 12.0, 18.5, 25.0])
    stats = compute_feature_statistics(series, num_bins=5, max_sample_size=10)

    assert "mean" in stats
    assert "std" in stats
    assert "bin_edges" in stats
    assert "bin_proportions" in stats
    assert "sample_values" in stats

    # Bin proportions must sum to 1.0 (+/- 1e-5)
    assert abs(sum(stats["bin_proportions"]) - 1.0) <= 1e-5
    # Smoothed proportions must be strictly positive
    assert all(p > 0.0 for p in stats["bin_proportions"])
    assert len(stats["bin_edges"]) == len(stats["bin_proportions"]) + 1


def test_baseline_stats_json_schema(project_paths, project_settings):
    """Verify data/reference/baseline_stats.json exists, is valid JSON, and matches strict schema."""
    stats_path = project_paths["baseline_stats"]
    assert stats_path.is_file(), f"Expected baseline profile at {stats_path}"

    with open(stats_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    assert "metadata" in profile
    assert "global" in profile
    assert "zones" in profile

    expected_features = project_settings.get("features", {}).get("numeric_features")
    for feat in expected_features:
        assert feat in profile["global"], f"Feature '{feat}' missing from global baseline stats"
        feat_stats = profile["global"][feat]
        for key in ["mean", "std", "min", "max", "median", "iqr", "bin_edges", "bin_proportions", "sample_values"]:
            assert key in feat_stats, f"Key '{key}' missing from global stats for feature '{feat}'"

        # Check proportions sum to 1.0
        prop_sum = sum(feat_stats["bin_proportions"])
        assert abs(prop_sum - 1.0) <= 1e-5, f"Feature '{feat}' proportions sum to {prop_sum}, expected 1.0 +/- 1e-5"

        # Check sample values capped and non-empty
        assert 1 <= len(feat_stats["sample_values"]) <= 500


def test_baseline_stats_zones_coverage(project_paths, spatial_zones_config, project_settings):
    """Verify baseline_stats.json contains complete profiles for every configured spatial zone."""
    with open(project_paths["baseline_stats"], "r", encoding="utf-8") as f:
        profile = json.load(f)

    expected_features = project_settings.get("features", {}).get("numeric_features")
    zone_ids = [z["zone_id"] for z in spatial_zones_config["zones"]]

    for zone_id in zone_ids:
        assert zone_id in profile["zones"], f"Zone '{zone_id}' missing from baseline_stats.json zones"
        zone_data = profile["zones"][zone_id]
        assert "num_records" in zone_data

        for feat in expected_features:
            assert feat in zone_data, f"Feature '{feat}' missing from zone '{zone_id}' stats"
            feat_stats = zone_data[feat]
            prop_sum = sum(feat_stats["bin_proportions"])
            assert abs(prop_sum - 1.0) <= 1e-5, (
                f"Zone '{zone_id}' feature '{feat}' proportions sum to {prop_sum}, expected 1.0"
            )
