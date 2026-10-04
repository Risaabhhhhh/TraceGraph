"""
Evaluation metrics for ChainGuard illicit transaction detection.

Per specification (Section 4.3):
  - Primary metric is PR-AUC (Average Precision) on the illicit (positive) class.
  - F1 score on the illicit class.
  - Recall at fixed precision thresholds (80% and 90%).
  - Accuracy and ROC-AUC must NOT be headlining metrics due to extreme class imbalance (~9.8% illicit).
  - All metrics must filter out unknown nodes (y == -1 or label_mask == False).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def recall_at_precision(
    y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.80
) -> float:
    """
    Compute maximum recall achievable on positive class subject to Precision >= min_precision.

    If precision never reaches min_precision at any threshold, returns 0.0.
    """
    if len(y_true) == 0 or np.sum(y_true == 1) == 0:
        return 0.0

    precisions, recalls, _ = precision_recall_curve(y_true, y_prob)

    # Filter where precision satisfies threshold
    valid_mask = precisions >= min_precision
    if not np.any(valid_mask):
        return 0.0

    return float(np.max(recalls[valid_mask]))


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics on labeled subset.

    Args:
        y_true: True binary labels (1=illicit, 0=licit, -1=unknown).
        y_prob: Predicted probabilities for illicit class (1).
        threshold: Decision threshold for discrete classification (default 0.5).
        mask: Optional boolean array selecting labeled nodes to evaluate.
              If None, automatically filters out y_true == -1.

    Returns:
        Dictionary of computed metrics for the illicit class.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if mask is not None:
        eval_mask = np.asarray(mask, dtype=bool) & (y_true != -1)
    else:
        eval_mask = y_true != -1

    y_t = y_true[eval_mask].astype(int)
    y_p = y_prob[eval_mask]

    n_samples = int(len(y_t))
    n_illicit = int(np.sum(y_t == 1))
    n_licit = int(np.sum(y_t == 0))

    if n_samples == 0 or n_illicit == 0:
        return {
            "n_samples": n_samples,
            "n_illicit": n_illicit,
            "n_licit": n_licit,
            "pr_auc": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "recall_at_prec_80": 0.0,
            "recall_at_prec_90": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "threshold": threshold,
        }

    # PR-AUC
    pr_auc = float(average_precision_score(y_t, y_p))

    # Discrete predictions at threshold
    y_pred = (y_p >= threshold).astype(int)

    f1 = float(f1_score(y_t, y_pred, zero_division=0))
    prec = float(precision_score(y_t, y_pred, zero_division=0))
    rec = float(recall_score(y_t, y_pred, zero_division=0))

    # Recall at fixed precision
    rec_at_80 = recall_at_precision(y_t, y_p, min_precision=0.80)
    rec_at_90 = recall_at_precision(y_t, y_p, min_precision=0.90)

    # Confusion matrix
    cm = confusion_matrix(y_t, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    return {
        "n_samples": n_samples,
        "n_illicit": n_illicit,
        "n_licit": n_licit,
        "pr_auc": round(pr_auc, 5),
        "f1": round(f1, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "recall_at_prec_80": round(rec_at_80, 5),
        "recall_at_prec_90": round(rec_at_90, 5),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "threshold": threshold,
    }


def aggregate_seed_metrics(seed_results: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate metric dictionaries across multiple random seeds, reporting mean and std.
    """
    if not seed_results:
        return {}

    metric_keys = [
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "recall_at_prec_80",
        "recall_at_prec_90",
    ]

    agg = {
        "n_seeds": len(seed_results),
        "n_samples": seed_results[0].get("n_samples", 0),
        "n_illicit": seed_results[0].get("n_illicit", 0),
        "n_licit": seed_results[0].get("n_licit", 0),
    }

    for k in metric_keys:
        values = [res[k] for res in seed_results if k in res]
        if values:
            agg[f"{k}_mean"] = round(float(np.mean(values)), 5)
            agg[f"{k}_std"] = round(float(np.std(values)), 5)

    return agg
