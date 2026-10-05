"""
Model Exporter and Manifest Generator for ChainGuard.

Exports the production XGBoost baseline model (seed 42, all 165 features),
calculates cryptographic SHA-256 hashes of model binaries, configuration,
and metrics, and generates a signed/tamper-evident MANIFEST.json for
on-chain registration in ModelRegistry.sol.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np

# Ensure path includes ML root
ML_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import compute_metrics
from src.eval.temporal_split import TemporalSplitter, load_graph
from src.models.baselines import XGBoostBaseline
from src.registry.snapshot_hash import compute_sha256_bytes, compute_sha256_file


def export_best_model(
    output_dir: Optional[Path] = None,
    version: str = "v1.0.0",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train and export the production XGBoost model on primary split,
    writing artifacts and MANIFEST.json to output_dir.
    """
    if output_dir is None:
        output_dir = ML_DIR / "artifacts" / version
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_path = REPO_ROOT / "ml" / "data" / "processed" / "elliptic_graph.pkl"
    split_config_path = REPO_ROOT / "ml" / "experiments" / "configs" / "split_config.yaml"

    print(f"[1/4] Loading graph data from {graph_path}...")
    graph = load_graph(graph_path)
    splitter = TemporalSplitter.from_config(split_config_path)
    primary_split = splitter.get_primary_split(graph)

    node_features = graph["node_features"]
    labels = graph["labels"]

    train_idx = primary_split.train_node_idx
    train_sup_mask = primary_split.train_supervised_mask[train_idx]
    X_train = node_features[train_idx][train_sup_mask]
    y_train = labels[train_idx][train_sup_mask]

    val_idx = primary_split.val_node_idx
    val_sup_mask = primary_split.val_supervised_mask[val_idx]
    X_val = node_features[val_idx][val_sup_mask]
    y_val = labels[val_idx][val_sup_mask]

    test_idx = primary_split.test_node_idx
    test_sup_mask = primary_split.test_supervised_mask[test_idx]
    X_test = node_features[test_idx][test_sup_mask]
    y_test = labels[test_idx][test_sup_mask]

    print(f"[2/4] Tuning and training XGBoost model (seed={seed}, all 165 features)...")
    baseline = XGBoostBaseline(random_state=seed)
    baseline.tune_and_fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        param_grid={
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "n_estimators": [100, 150],
        },
    )

    print(f"[3/4] Evaluating model on primary test split...")
    test_probs = baseline.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(
        y_true=y_test,
        y_prob=test_probs,
        threshold=0.5,
    )

    # Save model binary
    model_path = output_dir / "xgboost_model.json"
    baseline.model.save_model(str(model_path))

    # Save config
    config_dict = {
        "model_type": "xgboost_all_features",
        "version": version,
        "seed": seed,
        "feature_dim": int(node_features.shape[1]),
        "feature_type": "all_features_165",
        "hyperparameters": baseline.best_params_,
        "train_steps": "1-34",
        "val_steps": "35-39",
        "test_steps": "40-49",
        "data_split_id": "primary_steps_1_34_val_35_39_test_40_49",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_dict = {
        "pr_auc": float(metrics["pr_auc"]),
        "f1": float(metrics["f1"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "recall_at_80_precision": float(metrics["recall_at_prec_80"]),
        "recall_at_90_precision": float(metrics["recall_at_prec_90"]),
        "n_test_samples": int(len(y_test)),
        "n_test_illicit": int(np.sum(y_test == 1)),
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)

    print(f"[4/4] Generating cryptographic hashes and MANIFEST.json...")
    model_hash = compute_sha256_file(str(model_path))
    config_hash = compute_sha256_file(str(config_path))
    metrics_hash = compute_sha256_file(str(metrics_path))

    manifest_payload = {
        "version": version,
        "model_type": "xgboost_all_features",
        "model_hash": model_hash,
        "config_hash": config_hash,
        "metrics_hash": metrics_hash,
        "data_split_id": "primary_steps_1_34_val_35_39_test_40_49",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "metrics": metrics_dict,
        "hyperparameters": baseline.best_params_,
        "artifacts": {
            "model_file": model_path.name,
            "config_file": config_path.name,
            "metrics_file": metrics_path.name,
        },
    }

    # Compute hash of manifest content (excluding manifest_hash itself)
    manifest_bytes = json.dumps(manifest_payload, indent=2, sort_keys=True).encode("utf-8")
    manifest_hash = compute_sha256_bytes(manifest_bytes)
    manifest_payload["manifest_hash"] = manifest_hash

    manifest_path = output_dir / "MANIFEST.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_payload, f, indent=2)

    print(f"\nSuccessfully exported model artifact:")
    print(f"  Artifact Directory: {output_dir}")
    print(f"  Model Hash:         {model_hash}")
    print(f"  Config Hash:        {config_hash}")
    print(f"  Metrics Hash:       {metrics_hash}")
    print(f"  Manifest Hash:      {manifest_hash}")
    print(f"  PR-AUC:             {metrics_dict['pr_auc']:.4f}")
    print(f"  F1:                 {metrics_dict['f1']:.4f}")
    print(f"  Recall @ 80% Prec:  {metrics_dict['recall_at_80_precision']:.4f}")

    return manifest_payload


if __name__ == "__main__":
    export_best_model()
