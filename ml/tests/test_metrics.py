"""
Test: Evaluation metrics correctness and masking behavior.

Per specification (Phase 2):
  - Correct PR-AUC calculation.
  - F1 calculation on illicit class.
  - Recall at fixed precision (80% and 90%).
  - Unknown label masking (-1) handling.
  - Edge cases (no positives, precision threshold not reachable).
"""

import numpy as np
import pytest

from src.eval.metrics import aggregate_seed_metrics, compute_metrics, recall_at_precision


def test_recall_at_precision_perfect():
    """Test recall at precision when predictions are perfectly separable."""
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    rec_80 = recall_at_precision(y_true, y_prob, min_precision=0.80)
    rec_90 = recall_at_precision(y_true, y_prob, min_precision=0.90)

    assert rec_80 == 1.0
    assert rec_90 == 1.0


def test_recall_at_precision_unreachable():
    """Test recall at precision when required precision is never achieved."""
    # Inverted predictions: higher scores for negative class
    y_true = np.array([1, 0, 0, 0, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.8, 0.7, 0.6, 0.5])

    rec_90 = recall_at_precision(y_true, y_prob, min_precision=0.90)
    assert rec_90 == 0.0


def test_compute_metrics_with_unknown_mask():
    """Test compute_metrics properly filters unknown (-1) labels."""
    y_true = np.array([1, 0, -1, 1, -1, 0])
    y_prob = np.array([0.8, 0.2, 0.99, 0.7, 0.01, 0.1])
    mask = np.array([True, True, True, True, True, True])

    metrics = compute_metrics(y_true, y_prob, threshold=0.5, mask=mask)

    assert metrics["n_samples"] == 4  # 4 labeled samples
    assert metrics["n_illicit"] == 2
    assert metrics["n_licit"] == 2
    assert metrics["pr_auc"] > 0.9
    assert metrics["f1"] == 1.0
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["tn"] == 2
    assert metrics["fn"] == 0


def test_aggregate_seed_metrics():
    """Test multi-seed metrics aggregation produces mean and std."""
    seed_res = [
        {"pr_auc": 0.80, "f1": 0.75, "recall_at_prec_80": 0.60, "recall_at_prec_90": 0.40, "n_samples": 100},
        {"pr_auc": 0.82, "f1": 0.77, "recall_at_prec_80": 0.62, "recall_at_prec_90": 0.42, "n_samples": 100},
        {"pr_auc": 0.84, "f1": 0.79, "recall_at_prec_80": 0.64, "recall_at_prec_90": 0.44, "n_samples": 100},
    ]

    agg = aggregate_seed_metrics(seed_res)

    assert agg["n_seeds"] == 3
    assert agg["pr_auc_mean"] == pytest.approx(0.82, abs=1e-4)
    assert agg["f1_mean"] == pytest.approx(0.77, abs=1e-4)
    assert agg["pr_auc_std"] > 0.0
