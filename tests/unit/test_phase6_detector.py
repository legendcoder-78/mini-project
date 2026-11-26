"""Unit Tests for Phase 6: Statistical Drift Detector.

Verifies:
1. PSI and KS calculations for identical vs shifted distributions.
2. Nominal baseline samples yield PSI < 0.05, KS p-value > 0.05, and drift_detected == False.
3. Monsoon-shifted rainfall/soil moisture samples yield PSI > 0.20, KS p-value < 0.05,
   and drift_detected == True.
4. Insufficient sample size (< min_samples) safely skips drift triggering.
5. Automatic reloading of baseline_stats.json when disk mtime changes.
"""

import json
from pathlib import Path
import sys
import numpy as np
import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.common.schemas import DriftAlertEvent
from services.drift_monitor.detector import SpatialTemporalDriftDetector


@pytest.fixture(scope="module")
def detector():
    """Instantiate standard SpatialTemporalDriftDetector."""
    return SpatialTemporalDriftDetector()


@pytest.fixture(scope="module")
def baseline_data(detector):
    """Return parsed baseline_stats dict."""
    return detector.get_baseline_stats()


# =========================================================================
# 1. PSI & KS Mathematical Verification Tests
# =========================================================================

def test_calculate_psi_identical_distribution():
    """Verify PSI between identical distributions is essentially zero (< 0.001)."""
    bin_edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    expected_props = [0.2, 0.2, 0.2, 0.2, 0.2]

    # Uniform sample across the 5 bins
    sample = np.array([1.0, 3.0, 5.0, 7.0, 9.0] * 100)
    psi = SpatialTemporalDriftDetector.calculate_psi(bin_edges, expected_props, sample)
    assert psi < 0.01, f"Expected PSI near 0.0, got {psi:.6f}"


def test_calculate_psi_extreme_drift_and_clamping():
    """Verify PSI detects extreme shift and correctly clamps outer outliers."""
    bin_edges = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    expected_props = [0.8, 0.1, 0.05, 0.03, 0.02]

    # Extreme spike far beyond max edge (100.0, 200.0)
    drifted_sample = np.array([50.0, 100.0, 200.0, 500.0] * 50)
    psi = SpatialTemporalDriftDetector.calculate_psi(bin_edges, expected_props, drifted_sample)
    assert psi > 2.0, f"Expected massive PSI (> 2.0), got {psi:.4f}"


def test_calculate_ks_identical_vs_drifted():
    """Verify KS-test distinguishes identical samples from shifted distributions."""
    np.random.seed(42)
    sample_a = np.random.normal(loc=20.0, scale=3.0, size=300).tolist()
    sample_b_identical = list(sample_a)

    # Identical: p-value must be 1.0
    stat_ident, pval_ident = SpatialTemporalDriftDetector.calculate_ks(sample_a, np.array(sample_b_identical))
    assert stat_ident == 0.0
    assert pval_ident == 1.0

    # Drifting: mean shifted by 15 units
    sample_c_drifted = np.random.normal(loc=35.0, scale=3.0, size=300)
    stat_drift, pval_drift = SpatialTemporalDriftDetector.calculate_ks(sample_a, sample_c_drifted)
    assert stat_drift > 0.5
    assert pval_drift < 1e-10


# =========================================================================
# 2. Zone-Level Baseline vs. Drifted Evaluation Tests
# =========================================================================

def test_evaluate_zone_drift_nominal_baseline(detector, baseline_data):
    """Verify records drawn from baseline distribution yield PSI < 0.05, p_val > 0.05, and no drift."""
    zone_id = "ZONE_KODAGU"
    zone_stats = baseline_data["zones"][zone_id]

    # Construct nominal records from baseline representative samples
    baseline_records = []
    sample_len = len(zone_stats["temperature_c"]["sample_values"])
    assert sample_len >= 100

    for i in range(sample_len):
        features = {}
        for feat in detector.features:
            features[feat] = float(zone_stats[feat]["sample_values"][i])
        baseline_records.append({"features": features})

    alert_event = detector.evaluate_zone_drift(zone_id, baseline_records)

    assert isinstance(alert_event, DriftAlertEvent)
    assert alert_event.zone_id == zone_id
    assert alert_event.drift_detected is False, (
        f"Unexpected drift detected for baseline records: max_psi={alert_event.max_psi}, min_pvalue={alert_event.min_pvalue}"
    )
    # Continuous features (soil moisture, temperature, humidity) match reference profile closely
    assert alert_event.psi_scores["soil_moisture"] < 0.05
    assert alert_event.psi_scores["temperature_c"] < 0.05
    assert alert_event.psi_scores["humidity_pct"] < 0.05
    assert alert_event.min_pvalue > 0.05
    assert alert_event.retrain_triggered is False


def test_evaluate_zone_drift_monsoon_shifted(detector, baseline_data):
    """Verify monsoon-shifted rainfall & soil moisture triggers drift alert (PSI > 0.20, p_val < 0.05)."""
    zone_id = "ZONE_KODAGU"
    zone_stats = baseline_data["zones"][zone_id]

    shifted_records = []
    sample_len = len(zone_stats["rainfall_mm"]["sample_values"])

    for i in range(sample_len):
        shifted_features = {
            # Substantial monsoon shift: +30mm rainfall, +0.40 soil moisture
            "rainfall_mm": float(zone_stats["rainfall_mm"]["sample_values"][i]) + 30.0,
            "soil_moisture": min(1.0, float(zone_stats["soil_moisture"]["sample_values"][i]) + 0.40),
            "temperature_c": float(zone_stats["temperature_c"]["sample_values"][i]) - 5.0,
            "humidity_pct": min(100.0, float(zone_stats["humidity_pct"]["sample_values"][i]) + 15.0),
            "wind_speed_kmh": float(zone_stats["wind_speed_kmh"]["sample_values"][i]) + 20.0,
        }
        shifted_records.append({"features": shifted_features})

    alert_event = detector.evaluate_zone_drift(zone_id, shifted_records)

    assert isinstance(alert_event, DriftAlertEvent)
    assert alert_event.zone_id == zone_id
    assert alert_event.drift_detected is True, "Drift not detected for heavily shifted monsoon stream"
    assert alert_event.max_psi > 0.20, f"Expected max_psi > 0.20, got {alert_event.max_psi}"
    assert alert_event.min_pvalue < 0.05, f"Expected min_pvalue < 0.05, got {alert_event.min_pvalue}"

    # Specific feature drift checks
    assert alert_event.psi_scores["rainfall_mm"] > 0.20
    assert alert_event.ks_pvalues["rainfall_mm"] < 0.001
    assert alert_event.psi_scores["soil_moisture"] > 0.20


def test_evaluate_zone_drift_insufficient_samples(detector):
    """Verify detector safely suppresses drift check when window has fewer than min_samples."""
    zone_id = "ZONE_KODAGU"
    sparse_records = [
        {"features": {"rainfall_mm": 50.0, "soil_moisture": 0.9, "temperature_c": 22.0, "humidity_pct": 98.0, "wind_speed_kmh": 20.0}}
        for _ in range(10)  # Only 10 records < 30
    ]

    alert_event = detector.evaluate_zone_drift(zone_id, sparse_records)
    assert alert_event.drift_detected is False
    assert alert_event.max_psi == 0.0
    assert alert_event.min_pvalue == 1.0


def test_detector_auto_reloads_on_file_mtime_change(tmp_path):
    """Verify detector reloads stats when baseline_stats.json mtime changes."""
    stats_file = tmp_path / "test_baseline_stats.json"

    initial_stats = {
        "metadata": {"version": "v1"},
        "global": {
            "rainfall_mm": {
                "bin_edges": [0.0, 5.0, 10.0],
                "bin_proportions": [0.5, 0.5],
                "sample_values": [1.0, 2.0, 7.0, 8.0],
            }
        },
        "zones": {},
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(initial_stats, f)

    det = SpatialTemporalDriftDetector(baseline_stats_path=stats_file, min_samples=2)
    s1 = det.get_baseline_stats()
    assert s1["metadata"]["version"] == "v1"

    # Modify file with v2 and update mtime
    import time
    time.sleep(0.05)
    updated_stats = dict(initial_stats)
    updated_stats["metadata"] = {"version": "v2"}
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(updated_stats, f)

    s2 = det.get_baseline_stats()
    assert s2["metadata"]["version"] == "v2", "Detector failed to hot-reload stats on mtime change"
