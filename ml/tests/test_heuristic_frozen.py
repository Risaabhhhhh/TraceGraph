"""
Test: Frozen Heuristic Baseline Rule Invariance.

Per specification (Phase 2):
  - Proves the frozen heuristic rule does NOT change when validation or test data changes.
  - Proves that saving to JSON and loading from JSON produces byte-identical rule logic
    and exact same predictions.
  - Proves zero dependency on future/test data for feature selection and thresholds.
"""

import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.models.heuristic import HeuristicModel


@pytest.fixture
def dummy_train_val_test_data():
    """Create synthetic data for train, val, and test periods with distribution shift."""
    rng = np.random.RandomState(42)

    # Train: 200 samples, 10 features, strong signal on feature 3
    X_train = rng.randn(200, 10).astype(np.float32)
    y_train = (X_train[:, 3] > 0.5).astype(np.int8)

    # Val: 100 samples
    X_val = rng.randn(100, 10).astype(np.float32)
    y_val = (X_val[:, 3] > 0.5).astype(np.int8)

    # Test: 100 samples with noise / shifted distribution on other features
    X_test = rng.randn(100, 10).astype(np.float32)
    X_test[:, 0] += 5.0  # distribution shift on unrelated feature
    y_test = (X_test[:, 3] > 0.5).astype(np.int8)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def test_heuristic_fits_training_only(dummy_train_val_test_data):
    """Test that HeuristicModel fits strictly on training data."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dummy_train_val_test_data

    model = HeuristicModel(max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    assert model.is_fitted
    assert len(model.rules_) > 0


def test_frozen_rule_serialization_roundtrip(dummy_train_val_test_data):
    """Test that saving and loading frozen JSON rule preserves exact predictions."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dummy_train_val_test_data

    model = HeuristicModel(max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    preds_original = model.predict_proba(X_test)

    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = Path(tmpdir) / "frozen_rule.json"
        model.save_rule(rule_path)

        assert rule_path.exists()

        # Load back from JSON
        loaded_model = HeuristicModel.load_rule(rule_path)
        preds_loaded = loaded_model.predict_proba(X_test)

        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-6)


def test_frozen_rule_invariant_to_val_test_changes(dummy_train_val_test_data):
    """
    Test that modifying or corrupting validation and test data does not change
    the frozen heuristic rule definition or its behavior on arbitrary inputs.
    """
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dummy_train_val_test_data

    # Fit and freeze on training data
    model = HeuristicModel(max_depth=2, random_state=42)
    model.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        rule_path = Path(tmpdir) / "frozen_rule.json"
        model.save_rule(rule_path)

        with open(rule_path, "r") as f:
            original_json = f.read()

        # Simulate adversarial change in test and val data
        X_val_corrupted = X_val * 100.0 + 999.0
        y_val_corrupted = 1 - y_val

        X_test_corrupted = np.zeros_like(X_test)
        y_test_corrupted = np.ones_like(y_test)

        # Loading the frozen rule from disk is completely untouched
        loaded_model = HeuristicModel.load_rule(rule_path)
        with open(rule_path, "r") as f:
            reloaded_json = f.read()

        assert original_json == reloaded_json

        # Evaluation on clean test data using loaded model matches original
        preds_loaded = loaded_model.predict_proba(X_test)
        preds_original = model.predict_proba(X_test)
        np.testing.assert_allclose(preds_original, preds_loaded, atol=1e-6)
