"""Unit Tests for Phase 7: Champion vs. Challenger Model Comparator.

Verifies:
1. Comparator approves promotion when Challenger F1 exceeds Champion F1 by >= min_improvement.
2. Comparator rejects promotion when Challenger performs worse than Champion.
3. Comparator rejects promotion when Challenger improvement is below min_improvement threshold.
4. Comparator approves recovery promotion when Champion F1 has severely degraded under drift (< 0.60).
5. Comparator handles initial unseeded state (champion_model is None) with unconditional promotion.
6. End-to-end evaluation using real XGBoost models on monsoon-shifted validation distributions.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml_core.evaluation.comparator import compute_model_metrics, evaluate_and_compare
from ml_core.training.retrain_pipeline import train_challenger_model


class DummyModel:
    """Mock model returning pre-configured binary predictions."""

    def __init__(self, predictions: list, probabilities: list = None):
        self._preds = np.array(predictions)
        self._probs = np.array(probabilities) if probabilities is not None else None

    def predict(self, X):
        return self._preds[: len(X)]

    def predict_proba(self, X):
        if self._probs is not None:
            p1 = self._probs[: len(X)]
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        p1 = self._preds[: len(X)].astype(float)
        return np.column_stack([1.0 - p1, p1])


@pytest.fixture
def sample_validation_df():
    """Create synthetic validation dataframe with 100 rows."""
    np.random.seed(42)
    n = 100
    y = np.array([1] * 30 + [0] * 70)  # 30% positive hazard events
    np.random.shuffle(y)

    return pd.DataFrame({
        "rainfall_mm": np.random.uniform(10.0, 80.0, size=n),
        "soil_moisture": np.random.uniform(0.4, 0.95, size=n),
        "temperature_c": np.random.uniform(18.0, 32.0, size=n),
        "humidity_pct": np.random.uniform(60.0, 99.0, size=n),
        "wind_speed_kmh": np.random.uniform(5.0, 45.0, size=n),
        "hazard_label": y,
    })


@pytest.fixture
def feature_cols():
    return ["rainfall_mm", "soil_moisture", "temperature_c", "humidity_pct", "wind_speed_kmh"]


# =========================================================================
# 1. Decision Logic Unit Tests
# =========================================================================

def test_comparator_approves_when_challenger_improves(sample_validation_df, feature_cols):
    """Verify promotion approved when Challenger F1 exceeds Champion by >= 0.01."""
    y_true = sample_validation_df["hazard_label"].to_numpy()

    # Champion: mediocre predictions (F1 ~0.70)
    champ_preds = y_true.copy()
    champ_preds[np.where(y_true == 1)[0][:10]] = 0  # Miss 10 positive hazards
    champion = DummyModel(champ_preds)

    # Challenger: near perfect predictions (F1 ~0.95)
    chall_preds = y_true.copy()
    chall_preds[np.where(y_true == 1)[0][:2]] = 0  # Miss only 2
    challenger = DummyModel(chall_preds)

    decision = evaluate_and_compare(
        champion_model=champion,
        challenger_model=challenger,
        val_df=sample_validation_df,
        feature_cols=feature_cols,
        target_col="hazard_label",
        min_improvement=0.01,
    )

    assert decision["promote"] is True
    assert decision["challenger_f1"] > decision["champion_f1"]
    assert decision["improvement"] >= 0.01
    assert "exceeding threshold" in decision["reason"]


def test_comparator_rejects_when_challenger_is_worse(sample_validation_df, feature_cols):
    """Verify promotion rejected when Challenger performs worse than Champion."""
    y_true = sample_validation_df["hazard_label"].to_numpy()

    # Champion: excellent predictions
    champ_preds = y_true.copy()
    champion = DummyModel(champ_preds)

    # Challenger: poor predictions
    chall_preds = np.zeros_like(y_true)
    challenger = DummyModel(chall_preds)

    decision = evaluate_and_compare(
        champion_model=champion,
        challenger_model=challenger,
        val_df=sample_validation_df,
        feature_cols=feature_cols,
        target_col="hazard_label",
        min_improvement=0.01,
    )

    assert decision["promote"] is False
    assert decision["improvement"] < 0
    assert "Existing Production Champion retained" in decision["reason"]


def test_comparator_rejects_when_improvement_is_below_threshold(sample_validation_df, feature_cols):
    """Verify promotion rejected when Challenger improves by less than min_improvement."""
    y_true = sample_validation_df["hazard_label"].to_numpy()

    # Champion and Challenger with almost identical performance (difference < 0.01)
    champ_preds = y_true.copy()
    champ_preds[0] = 1 - champ_preds[0]
    champ_preds[1] = 1 - champ_preds[1]
    champion = DummyModel(champ_preds)

    chall_preds = y_true.copy()
    chall_preds[0] = 1 - chall_preds[0]  # Only 1 error vs 2 errors
    challenger = DummyModel(chall_preds)

    decision = evaluate_and_compare(
        champion_model=champion,
        challenger_model=challenger,
        val_df=sample_validation_df,
        feature_cols=feature_cols,
        target_col="hazard_label",
        min_improvement=0.05,  # Requires at least +5% improvement
    )

    assert decision["promote"] is False
    assert decision["improvement"] < 0.05
    assert "failed to achieve required" in decision["reason"]


def test_comparator_approves_when_champion_severely_degraded(sample_validation_df, feature_cols):
    """Verify promotion approved when Champion F1 < 0.60 and Challenger is strictly better."""
    y_true = sample_validation_df["hazard_label"].to_numpy()

    # Champion severely degraded under drift (F1 ~0.40)
    champ_preds = np.zeros_like(y_true)
    champ_preds[np.where(y_true == 1)[0][:8]] = 1
    champion = DummyModel(champ_preds)

    # Challenger noticeably better (F1 ~0.50)
    chall_preds = np.zeros_like(y_true)
    chall_preds[np.where(y_true == 1)[0][:15]] = 1
    challenger = DummyModel(chall_preds)

    decision = evaluate_and_compare(
        champion_model=champion,
        challenger_model=challenger,
        val_df=sample_validation_df,
        feature_cols=feature_cols,
        target_col="hazard_label",
        min_improvement=0.30,  # Strict threshold (delta is ~0.2456), so degraded gate triggers
    )

    assert decision["champion_f1"] < 0.60
    assert decision["challenger_f1"] > decision["champion_f1"]
    assert decision["promote"] is True
    assert "Champion severely degraded" in decision["reason"]


def test_comparator_unconditional_promotion_without_champion(sample_validation_df, feature_cols):
    """Verify initial cold-start promotion when champion_model is None."""
    y_true = sample_validation_df["hazard_label"].to_numpy()
    challenger = DummyModel(y_true)

    decision = evaluate_and_compare(
        champion_model=None,
        challenger_model=challenger,
        val_df=sample_validation_df,
        feature_cols=feature_cols,
        target_col="hazard_label",
    )

    assert decision["promote"] is True
    assert "No Champion model currently active" in decision["reason"]


# =========================================================================
# 2. End-to-End Real XGBoost Model Comparison Test
# =========================================================================

def test_comparator_with_real_xgboost_models(feature_cols):
    """Verify comparator with actual trained XGBoost classifiers under simulated drift."""
    np.random.seed(42)

    # 1. Historical baseline: low rainfall, hazard triggers at rainfall > 25.0
    n_base = 500
    rain_base = np.random.exponential(scale=5.0, size=n_base)
    soil_base = np.random.uniform(0.2, 0.5, size=n_base)
    temp_base = np.random.uniform(22.0, 32.0, size=n_base)
    hum_base = np.random.uniform(60.0, 85.0, size=n_base)
    wind_base = np.random.uniform(5.0, 20.0, size=n_base)
    y_base = ((rain_base > 20.0) | (soil_base > 0.45)).astype(int)

    base_train = pd.DataFrame({
        "rainfall_mm": rain_base,
        "soil_moisture": soil_base,
        "temperature_c": temp_base,
        "humidity_pct": hum_base,
        "wind_speed_kmh": wind_base,
        "hazard_label": y_base,
    })

    # Train Champion on baseline
    champion = XGBClassifier(n_estimators=30, max_depth=3, random_state=42)
    champion.fit(base_train[feature_cols], base_train["hazard_label"])

    # 2. Monsoon regime: heavy rainfall, high moisture, compound hazards
    n_drift = 300
    rain_drift = np.random.uniform(30.0, 100.0, size=n_drift)
    soil_drift = np.random.uniform(0.65, 0.98, size=n_drift)
    temp_drift = np.random.uniform(18.0, 25.0, size=n_drift)
    hum_drift = np.random.uniform(90.0, 100.0, size=n_drift)
    wind_drift = np.random.uniform(25.0, 50.0, size=n_drift)
    # Monsoon compound hazard: high rain + wind + moisture
    y_drift = ((rain_drift > 50.0) & (soil_drift > 0.80) & (wind_drift > 30.0)).astype(int)

    drift_df = pd.DataFrame({
        "rainfall_mm": rain_drift,
        "soil_moisture": soil_drift,
        "temperature_c": temp_drift,
        "humidity_pct": hum_drift,
        "wind_speed_kmh": wind_drift,
        "hazard_label": y_drift,
    })

    # Validation holdout on monsoon regime
    val_df = drift_df.iloc[:100].copy()
    drift_train = drift_df.iloc[100:].copy()

    # Train Challenger on blended drift data
    blended_train = pd.concat([drift_train, base_train.iloc[:200]], ignore_index=True)
    challenger, _ = train_challenger_model(
        blended_train,
        feature_cols=feature_cols,
        target_col="hazard_label",
        n_estimators=50,
        max_depth=4,
    )

    decision = evaluate_and_compare(
        champion_model=champion,
        challenger_model=challenger,
        val_df=val_df,
        feature_cols=feature_cols,
        target_col="hazard_label",
        min_improvement=0.01,
    )

    assert isinstance(decision, dict)
    assert "promote" in decision
    assert "champion_f1" in decision
    assert "challenger_f1" in decision
    assert decision["promote"] is True
