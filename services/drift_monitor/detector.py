"""Statistical Spatial-Temporal Drift Detector.

Calculates Population Stability Index (PSI) and two-sample Kolmogorov-Smirnov (KS) tests
for real-time micro-climate feature windows against reference baseline distributions.
Automatically hot-reloads data/reference/baseline_stats.json upon disk mtime updates.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import numpy as np
from scipy.stats import ks_2samp
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.common.schemas import DriftAlertEvent

logger = logging.getLogger("drift_detector")


def _load_settings(settings_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load configuration from config/settings.yaml if available."""
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


class SpatialTemporalDriftDetector:
    """Evaluates multivariate and per-zone statistical drift across sliding windows."""

    def __init__(
        self,
        baseline_stats_path: Optional[Union[str, Path]] = None,
        settings_path: Optional[Union[str, Path]] = None,
        psi_threshold: Optional[float] = None,
        ks_pvalue_threshold: Optional[float] = None,
        min_samples: Optional[int] = None,
    ):
        self.settings = _load_settings(settings_path)
        drift_cfg = self.settings.get("drift_thresholds", {})
        features_cfg = self.settings.get("features", {})

        self.psi_threshold = float(
            psi_threshold if psi_threshold is not None else drift_cfg.get("psi_threshold", 0.20)
        )
        self.ks_pvalue_threshold = float(
            ks_pvalue_threshold
            if ks_pvalue_threshold is not None
            else drift_cfg.get("ks_pvalue_threshold", 0.05)
        )
        self.min_samples = int(
            min_samples if min_samples is not None else drift_cfg.get("min_samples", 30)
        )
        self.features = features_cfg.get(
            "numeric_features",
            ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"],
        )

        if baseline_stats_path:
            p = Path(baseline_stats_path)
            self.baseline_stats_path = p if p.is_file() else PROJECT_ROOT / baseline_stats_path
        else:
            self.baseline_stats_path = (
                PROJECT_ROOT / "data" / "reference" / "baseline_stats.json"
            )

        self._stats_mtime: Optional[float] = None
        self._stats_data: Optional[Dict[str, Any]] = None

        # Initial load
        self.get_baseline_stats()

    def get_baseline_stats(self) -> Dict[str, Any]:
        """Load baseline stats JSON, auto-reloading if disk mtime has changed."""
        if not self.baseline_stats_path.is_file():
            raise FileNotFoundError(
                f"Baseline reference stats file not found at {self.baseline_stats_path}"
            )

        current_mtime = self.baseline_stats_path.stat().st_mtime
        if self._stats_data is None or self._stats_mtime != current_mtime:
            logger.info(
                "Loading reference baseline distributions from %s (mtime=%.2f)",
                self.baseline_stats_path,
                current_mtime,
            )
            with open(self.baseline_stats_path, "r", encoding="utf-8") as f:
                self._stats_data = json.load(f)
            self._stats_mtime = current_mtime

        return self._stats_data

    @staticmethod
    def calculate_psi(
        expected_bin_edges: List[float],
        expected_proportions: List[float],
        current_values: np.ndarray,
        eps: float = 1e-6,
    ) -> float:
        """Calculate Population Stability Index (PSI) with outer-edge bin clamping.

        Args:
            expected_bin_edges: Strictly monotonic bin edge cutoffs from baseline profile.
            expected_proportions: Expected probability mass per bin summing to 1.0.
            current_values: 1D array of observed real-time telemetry values.
            eps: Epsilon factor to prevent zero-division or log(0).

        Returns:
            Calculated PSI score (float >= 0.0).
        """
        arr = np.asarray(current_values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2 or len(expected_bin_edges) < 2 or len(expected_proportions) < 1:
            return 0.0

        # Clamp extreme values into outer bins to properly account for distribution mass
        min_edge = expected_bin_edges[0]
        max_edge = expected_bin_edges[-1]
        clamped = np.clip(arr, min_edge, max_edge)

        # Histogram count using baseline bin edges
        counts, _ = np.histogram(clamped, bins=expected_bin_edges)

        # Proportions with Laplace epsilon smoothing
        p_actual = (counts + eps) / (len(clamped) + eps * len(counts))
        p_expected = np.array(expected_proportions, dtype=float)
        p_expected_norm = (p_expected + eps) / (np.sum(p_expected) + eps * len(p_expected))

        # PSI sum: (P_act - P_exp) * ln(P_act / P_exp)
        psi_val = np.sum((p_actual - p_expected_norm) * np.log(p_actual / p_expected_norm))
        return float(max(0.0, psi_val))

    @staticmethod
    def calculate_ks(
        baseline_sample: List[float],
        current_values: np.ndarray,
    ) -> Tuple[float, float]:
        """Execute Two-Sample Kolmogorov-Smirnov test.

        Args:
            baseline_sample: Representative baseline sample vector (e.g. 500 records).
            current_values: 1D array of observed real-time values.

        Returns:
            Tuple of (ks_statistic: float, p_value: float).
        """
        arr = np.asarray(current_values, dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) < 2 or len(baseline_sample) < 2:
            return 0.0, 1.0

        stat, pval = ks_2samp(baseline_sample, arr)
        return float(stat), float(pval)

    def evaluate_zone_drift(
        self,
        zone_id: str,
        window_records: List[Dict[str, Any]],
    ) -> DriftAlertEvent:
        """Evaluate real-time sliding window records for a spatial zone against reference profile.

        Args:
            zone_id: Micro-climate spatial zone ID.
            window_records: List of telemetry event dictionaries.

        Returns:
            Validated DriftAlertEvent with feature-level PSI, KS p-values, and drift decision.
        """
        stats = self.get_baseline_stats()
        alert_id = str(uuid.uuid4())
        now_ts = datetime.now(timezone.utc).isoformat()

        # Check sample size adequacy
        if len(window_records) < self.min_samples:
            return DriftAlertEvent(
                alert_id=alert_id,
                timestamp=now_ts,
                zone_id=zone_id,
                psi_scores={feat: 0.0 for feat in self.features},
                ks_pvalues={feat: 1.0 for feat in self.features},
                max_psi=0.0,
                min_pvalue=1.0,
                drift_detected=False,
                retrain_triggered=False,
            )

        # Resolve zone-specific profile with fallback to global reference
        zones_data = stats.get("zones") or stats.get("per_zone", {})
        zone_profile = zones_data.get(zone_id) or stats.get("global", {})
        global_profile = stats.get("global", {})

        psi_scores: Dict[str, float] = {}
        ks_pvalues: Dict[str, float] = {}

        for feat in self.features:
            # Extract feature values from records (handles nested 'features' or flat structure)
            values = []
            for r in window_records:
                feat_dict = r.get("features")
                if isinstance(feat_dict, dict) and feat in feat_dict:
                    val = feat_dict[feat]
                else:
                    val = r.get(feat, np.nan)
                if val is not None and not np.isnan(float(val)):
                    values.append(float(val))

            val_arr = np.array(values, dtype=float)
            if len(val_arr) < self.min_samples:
                psi_scores[feat] = 0.0
                ks_pvalues[feat] = 1.0
                continue

            # Feature baseline stats lookup
            feat_stats = zone_profile.get(feat) or global_profile.get(feat, {})

            # Bin edges and expected proportions
            bin_edges = (
                feat_stats.get("bin_edges")
                or feat_stats.get("histogram", {}).get("bin_edges")
                or global_profile.get(feat, {}).get("bin_edges", [])
            )
            bin_proportions = (
                feat_stats.get("bin_proportions")
                or feat_stats.get("histogram", {}).get("proportions")
                or global_profile.get(feat, {}).get("bin_proportions", [])
            )
            ks_sample = (
                feat_stats.get("sample_values")
                or feat_stats.get("ks_sample")
                or global_profile.get(feat, {}).get("sample_values", [])
            )

            # 1. Compute PSI
            if bin_edges and bin_proportions:
                psi = self.calculate_psi(bin_edges, bin_proportions, val_arr)
            else:
                psi = 0.0
            psi_scores[feat] = round(psi, 4)

            # 2. Compute KS
            if ks_sample:
                _, p_val = self.calculate_ks(ks_sample, val_arr)
            else:
                p_val = 1.0
            ks_pvalues[feat] = round(p_val, 6)

        max_psi = float(max(psi_scores.values())) if psi_scores else 0.0
        min_pvalue = float(min(ks_pvalues.values())) if ks_pvalues else 1.0

        # Drift trigger condition: substantial distributional shift in any feature
        drift_detected = bool(
            max_psi > self.psi_threshold or min_pvalue < self.ks_pvalue_threshold
        )

        return DriftAlertEvent(
            alert_id=alert_id,
            timestamp=now_ts,
            zone_id=zone_id,
            psi_scores=psi_scores,
            ks_pvalues=ks_pvalues,
            max_psi=round(max_psi, 4),
            min_pvalue=round(min_pvalue, 6),
            drift_detected=drift_detected,
            retrain_triggered=False,
        )
