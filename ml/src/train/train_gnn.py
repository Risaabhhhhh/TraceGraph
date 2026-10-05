"""
Graph Neural Network Training & Evaluation (Phase 3)
Evaluates GraphSAGE and GAT architectures under strict zero-temporal-leakage conditions.
"""
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score
from torch_geometric.data import Data

ML_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import aggregate_seed_metrics, compute_metrics
from src.eval.temporal_split import SplitResult, TemporalSplitter, load_graph
from src.models.gnn import GAT, GraphSAGE

DEFAULT_CONFIG_PATH = REPO_ROOT / "ml/experiments/configs/baseline_config.yaml"


def create_pyg_data(
    split: SplitResult,
    graph: dict,
    mode: str,
    local_only: bool = False,
) -> Data:
    """
    Construct a PyG Data object for the specified temporal split mode.

    Zero-temporal-leakage guarantees:
      - 'train': edge_index includes ONLY edges within train time steps.
                 mask selects labeled nodes in train steps.
      - 'val': edge_index includes edges up to val time steps (train + val).
               mask selects labeled nodes in val steps.
      - 'test': edge_index includes edges up to test time steps (train + val + test).
                mask selects labeled nodes in test steps.
    """
    x = torch.tensor(graph["node_features"], dtype=torch.float32)
    if local_only:
        x = x[:, :94]
    
    y = torch.tensor(graph["labels"], dtype=torch.long)
    y[y == -1] = 0  # Replace unknown label -1 with 0 for tensor compatibility; masked during loss/eval
    edge_index = torch.tensor(graph["edge_index"], dtype=torch.long)

    if mode == "train":
        e_idx = split.train_edge_idx
        mask = split.train_supervised_mask.copy()
        eval_nodes = split.train_node_idx
    elif mode == "val":
        e_idx = split.val_edge_idx
        mask = split.val_supervised_mask.copy()
        eval_nodes = split.val_node_idx
    elif mode == "test":
        e_idx = split.test_edge_idx
        mask = split.test_supervised_mask.copy()
        eval_nodes = split.test_node_idx
    else:
        raise ValueError(f"Unknown mode: {mode}")

    data = Data(x=x, edge_index=edge_index[:, e_idx], y=y)
    data.mask = torch.tensor(mask, dtype=torch.bool)
    
    # eval_mask specifically isolates labeled nodes strictly within the requested split window
    eval_mask = np.zeros(len(y), dtype=bool)
    eval_mask[eval_nodes] = mask[eval_nodes]
    data.eval_mask = torch.tensor(eval_mask, dtype=torch.bool)

    return data


def train_single_gnn(
    model: nn.Module,
    train_data: Data,
    val_data: Optional[Data] = None,
    lr: float = 0.01,
    weight_decay: float = 1e-5,
    epochs: int = 100,
) -> nn.Module:
    """Train a single GNN model instance with optional validation tracking."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_val_ap = -1.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(train_data.x, train_data.edge_index).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(
            out[train_data.mask], train_data.y[train_data.mask].float()
        )
        loss.backward()
        optimizer.step()

        if val_data is not None and (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(val_data.x, val_data.edge_index).squeeze(-1)
                val_probs = torch.sigmoid(val_out[val_data.eval_mask]).numpy()
                y_val = val_data.y[val_data.eval_mask].numpy()
                if len(y_val) > 0 and np.sum(y_val) > 0:
                    val_ap = average_precision_score(y_val, val_probs)
                    if val_ap > best_val_ap:
                        best_val_ap = val_ap
                        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_and_eval_gnn(
    model_cls: Type[nn.Module],
    split: SplitResult,
    graph: dict,
    seeds: List[int],
    param_grid: Optional[Dict[str, List[Any]]] = None,
    local_only: bool = False,
    pred_dir: Optional[Path] = None,
    epochs: int = 100,
    **model_kwargs: Any,
) -> Dict[str, Any]:
    """
    Train and evaluate GNN across multiple random seeds with validation tuning.
    """
    train_data = create_pyg_data(split, graph, "train", local_only=local_only)
    val_data = create_pyg_data(split, graph, "val", local_only=local_only)
    test_data = create_pyg_data(split, graph, "test", local_only=local_only)

    best_params = {"lr": 0.01, "hidden": model_kwargs.get("hidden_channels", 128), "wd": 1e-5}
    
    # Tune hyperparameters strictly on validation set using seed 42
    if param_grid and len(split.val_node_idx) > 0:
        best_val_score = -1.0
        for lr in param_grid.get("lr", [0.01]):
            for hidden in param_grid.get("hidden", [128]):
                for wd in param_grid.get("wd", [1e-5]):
                    torch.manual_seed(42)
                    np.random.seed(42)
                    cfg_kwargs = dict(model_kwargs)
                    cfg_kwargs["hidden_channels"] = hidden
                    
                    candidate = model_cls(in_channels=train_data.x.shape[1], **cfg_kwargs)
                    candidate = train_single_gnn(
                        candidate, train_data, val_data=None, lr=lr, weight_decay=wd, epochs=60
                    )
                    
                    candidate.eval()
                    with torch.no_grad():
                        val_out = candidate(val_data.x, val_data.edge_index).squeeze(-1)
                        val_probs = torch.sigmoid(val_out[val_data.eval_mask]).numpy()
                        y_val = val_data.y[val_data.eval_mask].numpy()
                        if len(y_val) > 0 and np.sum(y_val) > 0:
                            score = average_precision_score(y_val, val_probs)
                            if score > best_val_score:
                                best_val_score = score
                                best_params = {"lr": lr, "hidden": hidden, "wd": wd}

    seed_metrics = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        run_kwargs = dict(model_kwargs)
        run_kwargs["hidden_channels"] = best_params["hidden"]
        
        model = model_cls(in_channels=train_data.x.shape[1], **run_kwargs)
        model = train_single_gnn(
            model,
            train_data,
            val_data=val_data,
            lr=best_params["lr"],
            weight_decay=best_params["wd"],
            epochs=epochs,
        )

        model.eval()
        with torch.no_grad():
            out = model(test_data.x, test_data.edge_index).squeeze(-1)
            probs = torch.sigmoid(out).numpy()

        test_sup_mask = test_data.eval_mask.numpy()
        m = compute_metrics(
            y_true=graph["labels"],
            y_prob=probs,
            threshold=0.5,
            mask=test_sup_mask,
        )
        m["seed"] = seed
        m["best_params"] = best_params
        seed_metrics.append(m)

        if pred_dir is not None:
            pred_dir.mkdir(parents=True, exist_ok=True)
            model_name = model_cls.__name__.lower()
            feat_suf = "local" if local_only else "all"
            np.savez_compressed(
                pred_dir / f"{model_name}_{feat_suf}_{split.name}_seed{seed}.npz",
                node_idx=split.test_node_idx,
                y_prob=probs[split.test_node_idx],
                y_true=graph["labels"][split.test_node_idx],
                supervised_mask=split.test_supervised_mask[split.test_node_idx],
            )

    return aggregate_seed_metrics(seed_metrics)


def run_gnn_experiments(config_path: Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Run all Phase 3 GNN experiments."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    graph_path = REPO_ROOT / config.get("graph_path", "ml/data/processed/elliptic_graph.pkl")
    split_config_path = REPO_ROOT / config.get("split_config_path", "ml/experiments/configs/split_config.yaml")
    
    print(f"Loading graph from {graph_path}...")
    graph = load_graph(graph_path)
    splitter = TemporalSplitter.from_config(split_config_path)
    primary_split = splitter.get_primary_split(graph)

    seeds = config.get("seeds", [42, 123, 456, 789, 1337])
    pred_dir = REPO_ROOT / config.get("predictions_dir", "ml/experiments/results/predictions")
    results = {"primary_split": {}, "rolling_origin": {}}

    print("\n--- Model: GraphSAGE (All Features) [5 seeds] ---")
    res_sage_all = train_and_eval_gnn(
        GraphSAGE,
        primary_split,
        graph,
        seeds,
        param_grid={"lr": [0.01], "hidden": [128], "wd": [1e-5]},
        local_only=False,
        pred_dir=pred_dir,
        epochs=100,
    )
    results["primary_split"]["graphsage_all"] = res_sage_all
    print(f"  GraphSAGE (All)   -> PR-AUC: {res_sage_all['pr_auc_mean']:.4f} ± {res_sage_all['pr_auc_std']:.4f} | "
          f"F1: {res_sage_all['f1_mean']:.4f} ± {res_sage_all['f1_std']:.4f} | "
          f"Rec@80%Prec: {res_sage_all['recall_at_prec_80_mean']:.4f} ± {res_sage_all['recall_at_prec_80_std']:.4f}")

    print("\n--- Model: GraphSAGE (Local Features Only) [5 seeds] ---")
    res_sage_local = train_and_eval_gnn(
        GraphSAGE,
        primary_split,
        graph,
        seeds,
        param_grid={"lr": [0.01], "hidden": [128], "wd": [1e-5]},
        local_only=True,
        pred_dir=pred_dir,
        epochs=100,
    )
    results["primary_split"]["graphsage_local"] = res_sage_local
    print(f"  GraphSAGE (Local) -> PR-AUC: {res_sage_local['pr_auc_mean']:.4f} ± {res_sage_local['pr_auc_std']:.4f} | "
          f"F1: {res_sage_local['f1_mean']:.4f} ± {res_sage_local['f1_std']:.4f} | "
          f"Rec@80%Prec: {res_sage_local['recall_at_prec_80_mean']:.4f} ± {res_sage_local['recall_at_prec_80_std']:.4f}")

    out_file = REPO_ROOT / "ml/experiments/results/gnn_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved GNN results to {out_file}")

    return results


if __name__ == "__main__":
    run_gnn_experiments()
