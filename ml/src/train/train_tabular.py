"""
Tabular & Heuristic Baseline Training & Evaluation Runner.

Per specification (Phase 2):
  - Fits Heuristic (tier 0), Logistic Regression (tier 1), and XGBoost (tier 1).
  - Evaluates on Primary Split (1-34 train, 35-39 val, 40-49 test) AND on all 5 Rolling-Origin Windows.
  - Excludes unknown nodes from loss and metrics via supervised mask.
  - Runs 5 seeds for learned models; reports mean and std. Heuristic runs once (deterministic).
  - Saves per-node predicted probabilities for every model, window, and seed.
  - Freezes heuristic rule to JSON.
  - Outputs structured results JSON for documentation & report generation.
"""

import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# Add repository root and ML dir to path
ML_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import aggregate_seed_metrics, compute_metrics
from src.eval.temporal_split import SplitResult, TemporalSplitter, load_graph
from src.models.baselines import LogisticRegressionBaseline, XGBoostBaseline
from src.models.heuristic import HeuristicModel

DEFAULT_CONFIG_PATH = ML_DIR / "experiments" / "configs" / "baseline_config.yaml"


def train_and_eval_heuristic(
    split: SplitResult,
    graph: dict,
    cfg: dict,
    is_primary: bool = False,
    frozen_rule_path: Optional[Path] = None,
    pred_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train/evaluate Heuristic model on a split."""
    time_steps = graph["time_steps"]
    node_features = graph["node_features"]
    labels = graph["labels"]

    # Training labeled nodes
    train_idx = split.train_node_idx
    train_sup_mask = split.train_supervised_mask[train_idx]
    X_train = node_features[train_idx][train_sup_mask]
    y_train = labels[train_idx][train_sup_mask]

    # Test nodes
    test_idx = split.test_node_idx
    test_sup_mask = split.test_supervised_mask[test_idx]
    X_test_all = node_features[test_idx]
    y_test_all = labels[test_idx]

    if is_primary:
        # Fit heuristic on primary training data and freeze
        model = HeuristicModel(
            max_depth=cfg.get("max_depth", 2),
            criterion=cfg.get("criterion", "gini"),
            random_state=cfg.get("random_state", 42),
        )
        model.fit(X_train, y_train)
        if frozen_rule_path:
            model.save_rule(frozen_rule_path)
            print(f"  Frozen heuristic rule saved to: {frozen_rule_path}")
            print(f"  Rule summary:\n{model.get_summary()}")
    else:
        # Load frozen rule or fit if not available
        if frozen_rule_path and frozen_rule_path.exists():
            model = HeuristicModel.load_rule(frozen_rule_path)
        else:
            model = HeuristicModel(
                max_depth=cfg.get("max_depth", 2),
                criterion=cfg.get("criterion", "gini"),
                random_state=cfg.get("random_state", 42),
            )
            model.fit(X_train, y_train)

    # Predict on ALL test nodes in window
    probs_all = model.predict_proba(X_test_all)[:, 1]

    # Compute metrics strictly on supervised/labeled subset
    metrics = compute_metrics(
        y_true=y_test_all,
        y_prob=probs_all,
        threshold=0.5,
        mask=test_sup_mask,
    )

    # Save per-node predictions
    if pred_dir:
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_file = pred_dir / f"heuristic_{split.name}_seed42.npz"
        np.savez_compressed(
            pred_file,
            node_idx=test_idx,
            y_prob=probs_all,
            y_true=y_test_all,
            supervised_mask=test_sup_mask,
        )

    return metrics


def train_and_eval_logistic_regression(
    split: SplitResult,
    graph: dict,
    cfg: dict,
    seeds: List[int],
    pred_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train/evaluate Logistic Regression across multiple seeds on a split."""
    node_features = graph["node_features"]
    labels = graph["labels"]

    train_idx = split.train_node_idx
    train_sup_mask = split.train_supervised_mask[train_idx]
    X_train = node_features[train_idx][train_sup_mask]
    y_train = labels[train_idx][train_sup_mask]

    test_idx = split.test_node_idx
    test_sup_mask = split.test_supervised_mask[test_idx]
    X_test_all = node_features[test_idx]
    y_test_all = labels[test_idx]

    seed_metrics = []

    for seed in seeds:
        model = LogisticRegressionBaseline(
            C=cfg.get("C", 1.0),
            class_weight=cfg.get("class_weight", "balanced"),
            max_iter=cfg.get("max_iter", 1000),
            random_state=seed,
        )
        model.fit(X_train, y_train)

        probs_all = model.predict_proba(X_test_all)[:, 1]

        m = compute_metrics(
            y_true=y_test_all,
            y_prob=probs_all,
            threshold=0.5,
            mask=test_sup_mask,
        )
        m["seed"] = seed
        seed_metrics.append(m)

        if pred_dir:
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_file = pred_dir / f"logistic_regression_{split.name}_seed{seed}.npz"
            np.savez_compressed(
                pred_file,
                node_idx=test_idx,
                y_prob=probs_all,
                y_true=y_test_all,
                supervised_mask=test_sup_mask,
            )

    aggregated = aggregate_seed_metrics(seed_metrics)
    aggregated["seeds_raw"] = seed_metrics
    return aggregated


def train_and_eval_xgboost(
    split: SplitResult,
    graph: dict,
    cfg: dict,
    seeds: List[int],
    pred_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Train/evaluate XGBoost with validation tuning across multiple seeds on a split."""
    node_features = graph["node_features"]
    labels = graph["labels"]

    # Training labeled nodes
    train_idx = split.train_node_idx
    train_sup_mask = split.train_supervised_mask[train_idx]
    X_train = node_features[train_idx][train_sup_mask]
    y_train = labels[train_idx][train_sup_mask]

    # Validation labeled nodes
    val_idx = split.val_node_idx
    val_sup_mask = split.val_supervised_mask[val_idx]
    X_val = node_features[val_idx][val_sup_mask]
    y_val = labels[val_idx][val_sup_mask]

    # Test nodes
    test_idx = split.test_node_idx
    test_sup_mask = split.test_supervised_mask[test_idx]
    X_test_all = node_features[test_idx]
    y_test_all = labels[test_idx]

    param_grid = cfg.get("param_grid", {
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 150],
    })

    seed_metrics = []
    best_params_per_seed = []

    for seed in seeds:
        model = XGBoostBaseline(
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            random_state=seed,
        )

        if len(y_val) > 0:
            model.tune_and_fit(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                param_grid=param_grid,
            )
        else:
            model.fit(X_train, y_train)

        best_params_per_seed.append(model.best_params_)

        probs_all = model.predict_proba(X_test_all)[:, 1]

        m = compute_metrics(
            y_true=y_test_all,
            y_prob=probs_all,
            threshold=0.5,
            mask=test_sup_mask,
        )
        m["seed"] = seed
        m["best_params"] = model.best_params_
        seed_metrics.append(m)

        if pred_dir:
            pred_dir.mkdir(parents=True, exist_ok=True)
            pred_file = pred_dir / f"xgboost_{split.name}_seed{seed}.npz"
            np.savez_compressed(
                pred_file,
                node_idx=test_idx,
                y_prob=probs_all,
                y_true=y_test_all,
                supervised_mask=test_sup_mask,
            )

    aggregated = aggregate_seed_metrics(seed_metrics)
    aggregated["best_params"] = best_params_per_seed[0] if best_params_per_seed else {}
    aggregated["seeds_raw"] = seed_metrics
    return aggregated


def run_all_baselines(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Execute all Phase 2 baseline runs and save structured results."""
    print("=" * 80)
    print("CHAINGUARD PHASE 2: BASELINE MODELS EVALUATION")
    print("=" * 80)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    graph_path = REPO_ROOT / config.get("graph_path", "ml/data/processed/elliptic_graph.pkl")
    split_config_path = REPO_ROOT / config.get("split_config_path", "ml/experiments/configs/split_config.yaml")

    frozen_rule_path = REPO_ROOT / config.get("frozen_heuristic_path", "ml/artifacts/heuristic/frozen_rule.json")
    pred_dir = REPO_ROOT / config.get("predictions_dir", "ml/experiments/results/predictions")
    results_file = REPO_ROOT / config.get("results_file", "ml/experiments/results/baseline_results.json")

    print(f"Loading graph from {graph_path}...")
    graph = load_graph(graph_path)

    print(f"Loading splitter from {split_config_path}...")
    splitter = TemporalSplitter.from_config(split_config_path)

    primary_split = splitter.get_primary_split(graph)
    rolling_splits = splitter.get_rolling_origin_splits(graph)

    seeds = config.get("seeds", [42, 123, 456, 789, 1337])
    models_cfg = config.get("models", {})

    all_results = {
        "primary_split": {},
        "rolling_origin": {},
        "metadata": {
            "experiment_name": config.get("experiment_name", "phase2_baselines"),
            "seeds": seeds,
            "total_nodes": len(graph["node_ids"]),
            "total_edges": graph["edge_index"].shape[1],
        },
    }

    # ==========================================
    # 1. PRIMARY SPLIT EVALUATION
    # ==========================================
    print("\n" + "=" * 80)
    print("1. EVALUATING ON PRIMARY SPLIT (Train: 1-34, Val: 35-39, Test: 40-49)")
    print("=" * 80)

    print("\n--- Model: Heuristic (Tier 0) ---")
    h_prim = train_and_eval_heuristic(
        split=primary_split,
        graph=graph,
        cfg=models_cfg.get("heuristic", {}),
        is_primary=True,
        frozen_rule_path=frozen_rule_path,
        pred_dir=pred_dir,
    )
    all_results["primary_split"]["heuristic"] = h_prim
    print(f"  Heuristic -> PR-AUC: {h_prim['pr_auc']:.4f} | F1: {h_prim['f1']:.4f} | "
          f"Rec@80%Prec: {h_prim['recall_at_prec_80']:.4f} | Rec@90%Prec: {h_prim['recall_at_prec_90']:.4f}")

    print("\n--- Model: Logistic Regression (Tier 1) [5 seeds] ---")
    lr_prim = train_and_eval_logistic_regression(
        split=primary_split,
        graph=graph,
        cfg=models_cfg.get("logistic_regression", {}),
        seeds=seeds,
        pred_dir=pred_dir,
    )
    all_results["primary_split"]["logistic_regression"] = lr_prim
    print(f"  LogReg    -> PR-AUC: {lr_prim['pr_auc_mean']:.4f} ± {lr_prim['pr_auc_std']:.4f} | "
          f"F1: {lr_prim['f1_mean']:.4f} ± {lr_prim['f1_std']:.4f} | "
          f"Rec@80%Prec: {lr_prim['recall_at_prec_80_mean']:.4f} ± {lr_prim['recall_at_prec_80_std']:.4f} | Rec@90%Prec: {lr_prim['recall_at_prec_90_mean']:.4f} ± {lr_prim['recall_at_prec_90_std']:.4f}")

    print("\n--- Model: XGBoost (Tier 1) [5 seeds, tuned on Val] ---")
    xgb_prim = train_and_eval_xgboost(
        split=primary_split,
        graph=graph,
        cfg=models_cfg.get("xgboost", {}),
        seeds=seeds,
        pred_dir=pred_dir,
    )
    all_results["primary_split"]["xgboost"] = xgb_prim
    print(f"  XGBoost   -> PR-AUC: {xgb_prim['pr_auc_mean']:.4f} ± {xgb_prim['pr_auc_std']:.4f} | "
          f"F1: {xgb_prim['f1_mean']:.4f} ± {xgb_prim['f1_std']:.4f} | "
          f"Rec@80%Prec: {xgb_prim['recall_at_prec_80_mean']:.4f} ± {xgb_prim['recall_at_prec_80_std']:.4f} | Rec@90%Prec: {xgb_prim['recall_at_prec_90_mean']:.4f} ± {xgb_prim['recall_at_prec_90_std']:.4f}")
    print(f"  Best XGBoost Params: {xgb_prim.get('best_params', {})}")

    # ==========================================
    # 2. ROLLING ORIGIN WINDOWS (EXPERIMENT C)
    # ==========================================
    print("\n" + "=" * 80)
    print("2. EVALUATING ON ROLLING-ORIGIN WINDOWS (Experiment C)")
    print("=" * 80)

    for r_split in rolling_splits:
        w_name = r_split.name
        print(f"\n>>> Window: {w_name} (Steps {r_split.test_steps[0]}-{r_split.test_steps[-1]}) <<<")
        all_results["rolling_origin"][w_name] = {}

        # Heuristic
        h_res = train_and_eval_heuristic(
            split=r_split,
            graph=graph,
            cfg=models_cfg.get("heuristic", {}),
            is_primary=False,
            frozen_rule_path=frozen_rule_path,
            pred_dir=pred_dir,
        )
        all_results["rolling_origin"][w_name]["heuristic"] = h_res
        print(f"  Heuristic -> PR-AUC: {h_res['pr_auc']:.4f} | F1: {h_res['f1']:.4f} | "
              f"Rec@80%Prec: {h_res['recall_at_prec_80']:.4f}")

        # Logistic Regression
        lr_res = train_and_eval_logistic_regression(
            split=r_split,
            graph=graph,
            cfg=models_cfg.get("logistic_regression", {}),
            seeds=seeds,
            pred_dir=pred_dir,
        )
        all_results["rolling_origin"][w_name]["logistic_regression"] = lr_res
        print(f"  LogReg    -> PR-AUC: {lr_res['pr_auc_mean']:.4f} ± {lr_res['pr_auc_std']:.4f} | "
              f"F1: {lr_res['f1_mean']:.4f} ± {lr_res['f1_std']:.4f} | "
              f"Rec@80%Prec: {lr_res['recall_at_prec_80_mean']:.4f} ± {lr_res['recall_at_prec_80_std']:.4f}")

        # XGBoost
        xgb_res = train_and_eval_xgboost(
            split=r_split,
            graph=graph,
            cfg=models_cfg.get("xgboost", {}),
            seeds=seeds,
            pred_dir=pred_dir,
        )
        all_results["rolling_origin"][w_name]["xgboost"] = xgb_res
        print(f"  XGBoost   -> PR-AUC: {xgb_res['pr_auc_mean']:.4f} ± {xgb_res['pr_auc_std']:.4f} | "
              f"F1: {xgb_res['f1_mean']:.4f} ± {xgb_res['f1_std']:.4f} | "
              f"Rec@80%Prec: {xgb_res['recall_at_prec_80_mean']:.4f} ± {xgb_res['recall_at_prec_80_std']:.4f}")

    # ==========================================
    # 3. SAVE RESULTS & PRINT SUMMARY TABLE
    # ==========================================
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll baseline results saved to: {results_file}")

    print_summary_table(all_results)
    return all_results


def print_summary_table(results: dict):
    """Print markdown formatted summary tables."""
    print("\n" + "=" * 90)
    print("PRIMARY TEST SET EVALUATION SUMMARY (Steps 40-49)")
    print("=" * 90)
    print(f"{'Model':<22} | {'PR-AUC (Illicit)':<20} | {'F1 (Illicit)':<18} | {'Rec@80%Prec':<14} | {'Rec@90%Prec':<14}")
    print("-" * 90)

    p = results["primary_split"]
    # Heuristic
    h = p["heuristic"]
    print(f"{'Heuristic (Tier 0)':<22} | {h['pr_auc']:<20.4f} | {h['f1']:<18.4f} | {h['recall_at_prec_80']:<14.4f} | {h['recall_at_prec_90']:<14.4f}")

    # LogReg
    lr = p["logistic_regression"]
    lr_prauc = f"{lr['pr_auc_mean']:.4f} ± {lr['pr_auc_std']:.4f}"
    lr_f1 = f"{lr['f1_mean']:.4f} ± {lr['f1_std']:.4f}"
    lr_r80 = f"{lr['recall_at_prec_80_mean']:.4f} ± {lr['recall_at_prec_80_std']:.4f}"
    lr_r90 = f"{lr['recall_at_prec_90_mean']:.4f} ± {lr['recall_at_prec_90_std']:.4f}"
    print(f"{'Logistic Regression':<22} | {lr_prauc:<20} | {lr_f1:<18} | {lr_r80:<14} | {lr_r90:<14}")

    # XGBoost
    xg = p["xgboost"]
    xg_prauc = f"{xg['pr_auc_mean']:.4f} ± {xg['pr_auc_std']:.4f}"
    xg_f1 = f"{xg['f1_mean']:.4f} ± {xg['f1_std']:.4f}"
    xg_r80 = f"{xg['recall_at_prec_80_mean']:.4f} ± {xg['recall_at_prec_80_std']:.4f}"
    xg_r90 = f"{xg['recall_at_prec_90_mean']:.4f} ± {xg['recall_at_prec_90_std']:.4f}"
    print(f"{'XGBoost (Tier 1)':<22} | {xg_prauc:<20} | {xg_f1:<18} | {xg_r80:<20} | {xg_r90:<20}")
    print("=" * 90)


if __name__ == "__main__":
    run_all_baselines()
