"""
Experiment A: Graph vs Tabular Representation & Architectural Ablations (Phase 3)

Compares:
  1. XGBoost (Local Only - 94 features)
  2. XGBoost (All Features - 165 features)
  3. GraphSAGE (All Features)
  4. GraphSAGE (Local Features Only)
  5. GAT (All Features)
  6. GraphSAGE + XGBoost (Hybrid)

Ablations:
  - GNN Depth (1-layer, 2-layer, 3-layer GraphSAGE)
  - Unknown Nodes in Message Passing (Full graph vs Known-only subgraph)
"""
import json
from pathlib import Path
import sys
from typing import Any, Dict

import numpy as np
import yaml

ML_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import aggregate_seed_metrics, compute_metrics
from src.eval.temporal_split import TemporalSplitter, load_graph
from src.models.gnn import GAT, GraphSAGE
from src.models.hybrid import GraphSAGE_XGBoost_Hybrid
from src.train.train_gnn import train_and_eval_gnn
from src.train.train_tabular import train_and_eval_xgboost


def run_experiment_a(config_path: Path = REPO_ROOT / "ml/experiments/configs/baseline_config.yaml") -> Dict[str, Any]:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    graph_path = REPO_ROOT / config.get("graph_path", "ml/data/processed/elliptic_graph.pkl")
    split_config_path = REPO_ROOT / config.get("split_config_path", "ml/experiments/configs/split_config.yaml")
    pred_dir = REPO_ROOT / config.get("predictions_dir", "ml/experiments/results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading graph from {graph_path}...")
    graph = load_graph(graph_path)
    splitter = TemporalSplitter.from_config(split_config_path)
    primary_split = splitter.get_primary_split(graph)
    seeds = config.get("seeds", [42, 123, 456, 789, 1337])

    results = {}
    print("\n=======================================================")
    print("      CHAINGUARD EXPERIMENT A: GRAPH VS TABULAR        ")
    print("=======================================================")

    # 1. XGBoost (Local Features Only - 94 dim)
    print("\n[1/7] XGBoost (Local Features Only)...")
    local_graph = graph.copy()
    local_graph["node_features"] = graph["node_features"][:, :94]
    xgb_cfg = config.get("models", {}).get("xgboost", {})
    res_xgb_local = train_and_eval_xgboost(primary_split, local_graph, xgb_cfg, seeds, pred_dir=None)
    results["xgboost_local"] = res_xgb_local
    print(f"  -> PR-AUC: {res_xgb_local['pr_auc_mean']:.4f} ± {res_xgb_local['pr_auc_std']:.4f} | "
          f"F1: {res_xgb_local['f1_mean']:.4f} | Rec@80%: {res_xgb_local['recall_at_prec_80_mean']:.4f}")

    # 2. XGBoost (All Features - 165 dim)
    print("\n[2/7] XGBoost (All Features)...")
    res_xgb_all = train_and_eval_xgboost(primary_split, graph, xgb_cfg, seeds, pred_dir=pred_dir)
    results["xgboost_all"] = res_xgb_all
    print(f"  -> PR-AUC: {res_xgb_all['pr_auc_mean']:.4f} ± {res_xgb_all['pr_auc_std']:.4f} | "
          f"F1: {res_xgb_all['f1_mean']:.4f} | Rec@80%: {res_xgb_all['recall_at_prec_80_mean']:.4f}")

    # 3. GraphSAGE (All Features)
    print("\n[3/7] GraphSAGE (All Features)...")
    res_sage_all = train_and_eval_gnn(
        GraphSAGE, primary_split, graph, seeds,
        param_grid={"lr": [0.01], "hidden": [128], "wd": [1e-5]},
        local_only=False, pred_dir=pred_dir, epochs=100
    )
    results["graphsage_all"] = res_sage_all
    print(f"  -> PR-AUC: {res_sage_all['pr_auc_mean']:.4f} ± {res_sage_all['pr_auc_std']:.4f} | "
          f"F1: {res_sage_all['f1_mean']:.4f} | Rec@80%: {res_sage_all['recall_at_prec_80_mean']:.4f}")

    # 4. GraphSAGE (Local Features Only)
    print("\n[4/7] GraphSAGE (Local Features Only)...")
    res_sage_local = train_and_eval_gnn(
        GraphSAGE, primary_split, graph, seeds,
        param_grid={"lr": [0.01], "hidden": [128], "wd": [1e-5]},
        local_only=True, pred_dir=pred_dir, epochs=100
    )
    results["graphsage_local"] = res_sage_local
    print(f"  -> PR-AUC: {res_sage_local['pr_auc_mean']:.4f} ± {res_sage_local['pr_auc_std']:.4f} | "
          f"F1: {res_sage_local['f1_mean']:.4f} | Rec@80%: {res_sage_local['recall_at_prec_80_mean']:.4f}")

    # 5. GAT (All Features)
    print("\n[5/7] GAT (All Features)...")
    res_gat_all = train_and_eval_gnn(
        GAT, primary_split, graph, seeds,
        param_grid={"lr": [0.01], "hidden": [64], "wd": [1e-5]},
        local_only=False, pred_dir=pred_dir, epochs=100, heads=4
    )
    results["gat_all"] = res_gat_all
    print(f"  -> PR-AUC: {res_gat_all['pr_auc_mean']:.4f} ± {res_gat_all['pr_auc_std']:.4f} | "
          f"F1: {res_gat_all['f1_mean']:.4f} | Rec@80%: {res_gat_all['recall_at_prec_80_mean']:.4f}")

    # 6. GraphSAGE + XGBoost (Hybrid)
    print("\n[6/7] GraphSAGE + XGBoost (Hybrid)...")
    hybrid_metrics = []
    for seed in seeds:
        hybrid = GraphSAGE_XGBoost_Hybrid(
            gnn_kwargs={"hidden_channels": 128},
            xgb_kwargs={"max_depth": 5, "learning_rate": 0.1, "n_estimators": 150, "tree_method": "hist"},
            random_state=seed,
        )
        hybrid.fit(primary_split, graph, gnn_epochs=80)
        probs = hybrid.predict_proba(primary_split, graph, mode="test")

        eval_mask = np.zeros(len(graph["labels"]), dtype=bool)
        eval_mask[primary_split.test_node_idx] = primary_split.test_supervised_mask[primary_split.test_node_idx]

        m = compute_metrics(
            y_true=graph["labels"],
            y_prob=probs,
            threshold=0.5,
            mask=eval_mask,
        )
        m["seed"] = seed
        hybrid_metrics.append(m)

        np.savez_compressed(
            pred_dir / f"hybrid_graphsage_xgboost_{primary_split.name}_seed{seed}.npz",
            node_idx=primary_split.test_node_idx,
            y_prob=probs[primary_split.test_node_idx],
            y_true=graph["labels"][primary_split.test_node_idx],
            supervised_mask=primary_split.test_supervised_mask[primary_split.test_node_idx],
        )

    res_hybrid = aggregate_seed_metrics(hybrid_metrics)
    results["hybrid"] = res_hybrid
    print(f"  -> PR-AUC: {res_hybrid['pr_auc_mean']:.4f} ± {res_hybrid['pr_auc_std']:.4f} | "
          f"F1: {res_hybrid['f1_mean']:.4f} | Rec@80%: {res_hybrid['recall_at_prec_80_mean']:.4f}")

    # 7. Ablations
    print("\n[7/7] Running Architectural Ablations...")
    
    # 7a. Depth Ablations
    for depth in [1, 3]:
        print(f"  -> Ablation: GraphSAGE Depth = {depth}...")
        class DepthSAGE(GraphSAGE):
            def __init__(self, in_channels: int, hidden_channels: int):
                super().__init__(in_channels, hidden_channels, num_layers=depth)

        res_depth = train_and_eval_gnn(
            DepthSAGE, primary_split, graph, seeds[:3],
            param_grid=None, local_only=False, epochs=80
        )
        results[f"graphsage_depth_{depth}"] = res_depth
        print(f"     Depth {depth} PR-AUC: {res_depth['pr_auc_mean']:.4f} ± {res_depth['pr_auc_std']:.4f}")

    # 7b. Unknown Node Message Passing Ablation
    print("  -> Ablation: Without Unknown Nodes in Message Passing...")
    graph_no_unknown = graph.copy()
    labels = graph["labels"]
    edge_index = graph["edge_index"]
    valid_edges = (labels[edge_index[0]] != -1) & (labels[edge_index[1]] != -1)
    graph_no_unknown["edge_index"] = edge_index[:, valid_edges]

    split_no_unknown = splitter.get_primary_split(graph_no_unknown)
    res_no_unknown = train_and_eval_gnn(
        GraphSAGE, split_no_unknown, graph_no_unknown, seeds[:3],
        param_grid=None, local_only=False, epochs=80
    )
    results["graphsage_no_unknown"] = res_no_unknown
    print(f"     Without Unknown Nodes PR-AUC: {res_no_unknown['pr_auc_mean']:.4f} ± {res_no_unknown['pr_auc_std']:.4f}")

    # Save complete results
    out_file = REPO_ROOT / "ml/experiments/results/experiment_a.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nExperiment A completed! Results saved to {out_file}")

    return results


if __name__ == "__main__":
    run_experiment_a()
