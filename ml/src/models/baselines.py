"""
Learned Tabular Baselines for ChainGuard (Logistic Regression & XGBoost).

Per specification (Section 4.2 & Phase 2):
  - Logistic Regression: standardizes features using training-set statistics only.
  - XGBoost: addresses class imbalance (scale_pos_weight) and tunes a small hyperparameter
    grid strictly on the validation window (never touching test).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class LogisticRegressionBaseline:
    """
    Tier 1 Logistic Regression baseline with training-set-only feature standardization.
    """

    def __init__(
        self,
        C: float = 1.0,
        class_weight: str = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs",
        )
        self.is_fitted = False

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegressionBaseline":
        """
        Fit scaler and logistic regression on training set only.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int8)

        # Standardize strictly on training data
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_fitted:
            raise RuntimeError("LogisticRegressionBaseline must be fitted before predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict discrete class for X at given threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(np.int8)


class XGBoostBaseline:
    """
    Tier 1 Gradient Boosted Decision Tree baseline (XGBoost).

    Handles class imbalance via scale_pos_weight and performs small hyperparameter
    tuning on the validation window only.
    """

    def __init__(
        self,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.best_params_: Dict[str, Any] = {}
        self.model: Optional[xgb.XGBClassifier] = None
        self.is_fitted = False

    def _compute_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """Compute negative / positive ratio for class weighting."""
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        return float(n_neg / max(n_pos, 1))

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "XGBoostBaseline":
        """Fit a single XGBoost model with default/initialized hyperparameters."""
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int8)
        scale_pos_weight = self._compute_scale_pos_weight(y_train)

        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
        )
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def tune_and_fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        param_grid: Optional[Dict[str, List[Any]]] = None,
    ) -> "XGBoostBaseline":
        """
        Tune hyperparameters using validation window PR-AUC and fit the best model.
        NEVER touches test data.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int8)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_val = np.asarray(y_val, dtype=np.int8)

        scale_pos_weight = self._compute_scale_pos_weight(y_train)

        if param_grid is None:
            param_grid = {
                "max_depth": [3, 5],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 150],
            }

        # Grid search over validation PR-AUC
        best_val_score = -1.0
        best_params = {}
        best_model = None

        from itertools import product
        keys, values = zip(*param_grid.items())
        permutations = [dict(zip(keys, v)) for v in product(*values)]

        for p in permutations:
            candidate = xgb.XGBClassifier(
                max_depth=p.get("max_depth", self.max_depth),
                learning_rate=p.get("learning_rate", self.learning_rate),
                n_estimators=p.get("n_estimators", self.n_estimators),
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=4,
            )
            candidate.fit(X_train, y_train)
            val_probs = candidate.predict_proba(X_val)[:, 1]
            val_pr_auc = float(average_precision_score(y_val, val_probs))

            if val_pr_auc > best_val_score:
                best_val_score = val_pr_auc
                best_params = p
                best_model = candidate

        self.best_params_ = best_params
        self.model = best_model
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("XGBoostBaseline must be fitted before predict_proba.")
        return self.model.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict discrete class for X at given threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(np.int8)
