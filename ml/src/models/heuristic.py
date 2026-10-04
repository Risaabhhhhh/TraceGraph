"""
Tier 0 Heuristic Baseline for ChainGuard.

Per specification (Section 4.2 & Phase 2):
  - Uses the 1 to 2 strongest features and threshold(s) learned from TRAINING DATA ONLY.
  - Frozen to a JSON artifact so inference is 100% deterministic and zero-computation.
  - Frozen rule must never change when validation or test data changes.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class HeuristicModel:
    """
    Tier 0 Interpretable Heuristic Rule Model.

    A simple, fast 1-to-2 feature decision rule fitted strictly on training data
    and serialized to JSON for reproducible, zero-leakage baselines.
    """

    def __init__(
        self,
        max_depth: int = 2,
        criterion: str = "gini",
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.criterion = criterion
        self.random_state = random_state
        self.tree_: Optional[DecisionTreeClassifier] = None
        self.rules_: List[Dict[str, Any]] = []
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> "HeuristicModel":
        """
        Fit a 1-to-2 feature decision rule on training data only.

        Args:
            X: Training feature matrix (N, D).
            y: Training binary labels (N,).
            feature_names: Optional list of feature names.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int8)

        if feature_names is None:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names_ = list(feature_names)

        # Train a constrained decision tree (depth <= 2, max 1-2 features)
        dt = DecisionTreeClassifier(
            max_depth=self.max_depth,
            max_leaf_nodes=4,
            criterion=self.criterion,
            class_weight="balanced",
            random_state=self.random_state,
        )
        dt.fit(X, y)
        self.tree_ = dt
        self.is_fitted = True

        # Extract structured rules
        self._extract_rules()
        return self

    def _extract_rules(self):
        """Extract interpretable rule metadata from the fitted tree."""
        if self.tree_ is None:
            return

        tree = self.tree_.tree_
        feature = tree.feature
        threshold = tree.threshold
        value = tree.value

        rules = []

        def recurse(node_id: int, conditions: List[Dict[str, Any]]):
            if feature[node_id] != -2:  # Not a leaf node
                feat_idx = int(feature[node_id])
                feat_name = self.feature_names_[feat_idx]
                thresh = float(threshold[node_id])

                # Left branch: feature <= threshold
                recurse(
                    int(tree.children_left[node_id]),
                    conditions + [{"feature_idx": feat_idx, "feature_name": feat_name, "op": "<=", "threshold": thresh}],
                )
                # Right branch: feature > threshold
                recurse(
                    int(tree.children_right[node_id]),
                    conditions + [{"feature_idx": feat_idx, "feature_name": feat_name, "op": ">", "threshold": thresh}],
                )
            else:
                # Leaf node
                counts = value[node_id][0]
                prob_illicit = float(counts[1] / np.sum(counts)) if np.sum(counts) > 0 else 0.0
                pred_class = int(np.argmax(counts))
                rules.append({
                    "leaf_id": int(node_id),
                    "conditions": conditions,
                    "prob_illicit": round(prob_illicit, 6),
                    "prediction": pred_class,
                    "class_counts": [int(c) for c in counts],
                })

        recurse(0, [])
        self.rules_ = rules

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for X. Returns shape (N, 2)."""
        if not self.is_fitted or self.tree_ is None:
            raise RuntimeError("HeuristicModel must be fitted or loaded before predict_proba.")
        return self.tree_.predict_proba(np.asarray(X, dtype=np.float32))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary class for X at given threshold."""
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(np.int8)

    def save_rule(self, output_path: Union[str, Path]):
        """Save the frozen rule to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted HeuristicModel.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model_type": "HeuristicModel",
            "tier": 0,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
            "rules": self.rules_,
            "feature_names": self.feature_names_,
        }

        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_rule(cls, rule_path: Union[str, Path]) -> "HeuristicModel":
        """Load a frozen heuristic rule from JSON."""
        rule_path = Path(rule_path)
        if not rule_path.exists():
            raise FileNotFoundError(f"Frozen rule not found at {rule_path}")

        with open(rule_path, "r") as f:
            data = json.load(f)

        model = cls(
            max_depth=data.get("max_depth", 2),
            random_state=data.get("random_state", 42),
        )
        model.rules_ = data["rules"]
        model.feature_names_ = data.get("feature_names", [])
        model.is_fitted = True

        # Reconstruct an inference function from rules
        class RuleInferenceWrapper:
            def __init__(self, rules):
                self.rules = rules

            def predict_proba(self, X):
                X = np.asarray(X, dtype=np.float32)
                N = X.shape[0]
                probs = np.zeros((N, 2), dtype=np.float32)

                for r in self.rules:
                    conds = r["conditions"]
                    match = np.ones(N, dtype=bool)
                    for c in conds:
                        f_idx = c["feature_idx"]
                        op = c["op"]
                        th = c["threshold"]
                        if op == "<=":
                            match &= X[:, f_idx] <= th
                        elif op == ">":
                            match &= X[:, f_idx] > th
                    p1 = r["prob_illicit"]
                    probs[match, 1] = p1
                    probs[match, 0] = 1.0 - p1

                return probs

        model.tree_ = RuleInferenceWrapper(model.rules_)
        return model

    def get_summary(self) -> str:
        """Return a human-readable text summary of the rules."""
        if not self.rules_:
            return "No rules fitted."
        lines = ["--- Frozen Heuristic Rules ---"]
        for r in self.rules_:
            cond_strs = [f"{c['feature_name']} {c['op']} {c['threshold']:.4f}" for c in r["conditions"]]
            cond_expr = " AND ".join(cond_strs) if cond_strs else "TRUE"
            lines.append(f"  IF {cond_expr} => P(illicit) = {r['prob_illicit']:.4f} (pred: {r['prediction']})")
        return "\n".join(lines)
