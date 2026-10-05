"""
Verification of metric consistency on canonical primary test split (11,184 labeled nodes).
"""

import sys
from pathlib import Path
import numpy as np

ML_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.metrics import compute_metrics, aggregate_seed_metrics
from src.eval.temporal_split import TemporalSplitter, load_graph

graph = load_graph(REPO_ROOT / "ml" / "data" / "processed" / "elliptic_graph.pkl")
splitter = TemporalSplitter.from_config(REPO_ROOT / "ml" / "experiments" / "configs" / "split_config.yaml")
split = splitter.get_primary_split(graph)

test_mask = split.test_supervised_mask[split.test_node_idx]
print(f"Canonical test labeled nodes: {np.sum(test_mask)}")

pred_dir = REPO_ROOT / "ml" / "experiments" / "results" / "predictions"
seeds = [42, 123, 456, 789, 1337]

models = [
    ("Heuristic", ["heuristic_primary_seed42.npz"]),
    ("Logistic Regression", [f"logistic_regression_primary_seed{s}.npz" for s in seeds]),
    ("XGBoost (Local)", []), # from baseline results
    ("XGBoost (All 165 Feat)", [f"xgboost_primary_seed{s}.npz" for s in seeds]),
    ("GraphSAGE (Local 94 Feat)", [f"graphsage_local_primary_seed{s}.npz" for s in seeds]),
    ("GraphSAGE (All 165 Feat)", [f"graphsage_all_primary_seed{s}.npz" for s in seeds]),
    ("GAT (All 165 Feat)", [f"gat_all_primary_seed{s}.npz" for s in seeds]),
    ("Hybrid (GraphSAGE->XGBoost)", [f"hybrid_graphsage_xgboost_primary_seed{s}.npz" for s in seeds]),
]

print("\n" + "=" * 95)
print(f"{'Model':<28} | {'PR-AUC':<18} | {'F1 Score':<18} | {'Rec@80%Prec':<14} | {'Rec@90%Prec':<14}")
print("-" * 95)

for name, fnames in models:
    if not fnames:
        continue
    ms = []
    for fn in fnames:
        fp = pred_dir / fn
        if not fp.exists():
            continue
        d = np.load(fp)
        y_true = d["y_true"][test_mask]
        y_prob = d["y_prob"][test_mask]
        m = compute_metrics(y_true, y_prob)
        ms.append(m)
    if not ms:
        continue
    if len(ms) == 1:
        m0 = ms[0]
        print(f"{name:<28} | {m0['pr_auc']:<18.4f} | {m0['f1']:<18.4f} | {m0['recall_at_prec_80']:<14.4f} | {m0['recall_at_prec_90']:<14.4f}")
    else:
        agg = aggregate_seed_metrics(ms)
        prauc_str = f"{agg['pr_auc_mean']:.4f} ± {agg['pr_auc_std']:.4f}"
        f1_str = f"{agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}"
        r80_str = f"{agg['recall_at_prec_80_mean']:.4f} ± {agg['recall_at_prec_80_std']:.4f}"
        r90_str = f"{agg['recall_at_prec_90_mean']:.4f} ± {agg['recall_at_prec_90_std']:.4f}"
        print(f"{name:<28} | {prauc_str:<18} | {f1_str:<18} | {r80_str:<14} | {r90_str:<14}")
