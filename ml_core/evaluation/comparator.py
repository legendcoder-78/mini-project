"""Champion vs. Challenger Model Evaluation & Gatekeeper.

Computes classification metrics (F1, Precision, Recall, ROC-AUC) across holdout
validation data representing current environmental conditions, and implements
an automated promotion gate to safely promote Challenger over Champion only when
measurable statistical improvement or recovery from performance degradation is confirmed.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("comparator")


def compute_model_metrics(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
) -> Dict[str, float]:
    """Calculate classification performance metrics for a candidate model.

    Args:
        model: Trained estimator with predict (and optionally predict_proba).
        X: Feature matrix DataFrame.
        y: Binary ground-truth vector (0 or 1).

    Returns:
        Dict with accuracy, precision, recall, f1_score, and roc_auc.
    """
    if len(y) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "roc_auc": 0.5,
        }

    y_pred = model.predict(X)
    # Ensure binary integer predictions
    y_pred = np.asarray(y_pred, dtype=int)

    # Probability estimation for ROC-AUC
    roc_auc = 0.5
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            if probs.shape[1] > 1:
                prob_pos = probs[:, 1]
                if len(np.unique(y)) > 1:
                    roc_auc = float(roc_auc_score(y, prob_pos))
                else:
                    roc_auc = 1.0
        except Exception as e:
            logger.debug("ROC-AUC calculation exception: %s", e)

    acc = float(accuracy_score(y, y_pred))
    prec = float(precision_score(y, y_pred, zero_division=0))
    rec = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
    }


def evaluate_and_compare(
    champion_model: Optional[Any],
    challenger_model: Any,
    val_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "hazard_label",
    min_improvement: float = 0.01,
) -> Dict[str, Any]:
    """Execute Champion vs. Challenger evaluation gate on current validation dataset.

    Args:
        champion_model: Currently deployed Production model instance (or None if unseeded).
        challenger_model: Newly trained candidate model.
        val_df: Holdout validation DataFrame reflecting recent environmental regime.
        feature_cols: Meteorological feature column names.
        target_col: Target binary label column name.
        min_improvement: Minimum delta in F1 score required to dethrone Champion.

    Returns:
        Structured decision dict:
        {
            "promote": bool,
            "champion_f1": float,
            "challenger_f1": float,
            "improvement": float,
            "champion_metrics": dict,
            "challenger_metrics": dict,
            "reason": str,
        }
    """
    if val_df.empty:
        raise ValueError("Validation DataFrame is empty. Cannot perform model comparison.")

    X_val = val_df[feature_cols].copy()
    y_val = val_df[target_col].astype(int).to_numpy()

    # 1. Evaluate Challenger
    challenger_metrics = compute_model_metrics(challenger_model, X_val, y_val)
    chall_f1 = challenger_metrics["f1_score"]

    # 2. Evaluate Champion (if present)
    if champion_model is None:
        return {
            "promote": True,
            "champion_f1": 0.0,
            "challenger_f1": chall_f1,
            "improvement": round(chall_f1, 4),
            "champion_metrics": {},
            "challenger_metrics": challenger_metrics,
            "reason": "No Champion model currently active in Production. Unconditional promotion.",
        }

    champion_metrics = compute_model_metrics(champion_model, X_val, y_val)
    champ_f1 = champion_metrics["f1_score"]

    improvement = chall_f1 - champ_f1

    # 3. Decision Gate:
    # Promote if:
    # a) Challenger improves upon Champion by at least min_improvement
    # OR
    # b) Champion F1 has severely degraded (< 0.60 under drifted conditions) and Challenger is strictly better
    if improvement >= min_improvement:
        promote = True
        reason = (
            f"Challenger outperformed Champion by {improvement:+.4f} F1 score "
            f"(Challenger: {chall_f1:.4f} vs Champion: {champ_f1:.4f}), exceeding threshold +{min_improvement:.2%}."
        )
    elif champ_f1 < 0.60 and chall_f1 > champ_f1:
        promote = True
        reason = (
            f"Champion severely degraded under new environmental regime (F1: {champ_f1:.4f} < 0.60). "
            f"Challenger showed strictly better performance (F1: {chall_f1:.4f}, improvement: {improvement:+.4f})."
        )
    else:
        promote = False
        reason = (
            f"Challenger failed to achieve required +{min_improvement:.2%} F1 improvement "
            f"(improvement: {improvement:+.4f}, Challenger: {chall_f1:.4f} vs Champion: {champ_f1:.4f}). "
            f"Existing Production Champion retained."
        )

    logger.info("Evaluation Decision: promote=%s | Reason: %s", promote, reason)

    return {
        "promote": promote,
        "champion_f1": round(champ_f1, 4),
        "challenger_f1": round(chall_f1, 4),
        "improvement": round(improvement, 4),
        "champion_metrics": champion_metrics,
        "challenger_metrics": challenger_metrics,
        "reason": reason,
    }
