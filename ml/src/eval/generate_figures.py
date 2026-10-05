"""
Generate high-resolution publication-quality figures for ChainGuard ML Report.
Outputs to docs/figures/ and ml/experiments/figures/
"""
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DOCS_FIG_DIR = REPO_ROOT / "docs/figures"
ML_FIG_DIR = REPO_ROOT / "ml/experiments/figures"


def generate_all_figures():
    DOCS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    ML_FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.titlesize": 16,
        "figure.autolayout": True,
    })

    baseline_file = REPO_ROOT / "ml/experiments/results/baseline_results.json"
    exp_a_file = REPO_ROOT / "ml/experiments/results/experiment_a.json"

    if not baseline_file.exists():
        print(f"Warning: {baseline_file} not found. Skipping figure generation.")
        return

    with open(baseline_file, "r") as f:
        baselines = json.load(f)

    exp_a = {}
    if exp_a_file.exists():
        with open(exp_a_file, "r") as f:
            exp_a = json.load(f)

    prim_b = baselines.get("primary_split", {})

    # =========================================================================
    # 1. Performance Ladder Plot
    # =========================================================================
    print("Generating Figure 1: Performance Ladder...")
    models = ["Tier 0: Heuristic", "Tier 1: LogReg", "Tier 1: XGBoost"]
    pr_aucs = [
        prim_b.get("heuristic", {}).get("pr_auc", 0.1112),
        prim_b.get("logistic_regression", {}).get("pr_auc_mean", 0.2185),
        prim_b.get("xgboost", {}).get("pr_auc_mean", 0.6769),
    ]
    stds = [
        0.0,
        prim_b.get("logistic_regression", {}).get("pr_auc_std", 0.0),
        prim_b.get("xgboost", {}).get("pr_auc_std", 0.0024),
    ]
    colors = ["#95a5a6", "#7f8c8d", "#2980b9"]

    if "graphsage_all" in exp_a:
        models.append("Tier 2: GraphSAGE")
        pr_aucs.append(exp_a["graphsage_all"]["pr_auc_mean"])
        stds.append(exp_a["graphsage_all"]["pr_auc_std"])
        colors.append("#27ae60")

    if "gat_all" in exp_a:
        models.append("Tier 2: GAT")
        pr_aucs.append(exp_a["gat_all"]["pr_auc_mean"])
        stds.append(exp_a["gat_all"]["pr_auc_std"])
        colors.append("#16a085")

    if "hybrid" in exp_a:
        models.append("Tier 3: Hybrid (GNN+XGB)")
        pr_aucs.append(exp_a["hybrid"]["pr_auc_mean"])
        stds.append(exp_a["hybrid"]["pr_auc_std"])
        colors.append("#8e44ad")

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, pr_aucs, yerr=stds, capsize=5, color=colors, edgecolor="black", alpha=0.85, width=0.55)
    ax.set_ylabel("PR-AUC (Illicit Class)", fontweight="bold")
    ax.set_title("ChainGuard Model Performance Ladder (Primary Test Split: Steps 40–49)", fontweight="bold")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.6769, color="#2980b9", linestyle="--", alpha=0.5, label="XGBoost Baseline")
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontweight="bold")

    ax.legend(loc="upper left")
    for d in [DOCS_FIG_DIR, ML_FIG_DIR]:
        plt.savefig(d / "ladder_plot.png", dpi=300)
    plt.close()

    # =========================================================================
    # 2. Local Features vs All Features (Graph vs Tabular Representation)
    # =========================================================================
    if "xgboost_local" in exp_a and "graphsage_local" in exp_a:
        print("Generating Figure 2: Representation Comparison...")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        x = np.arange(2)
        width = 0.35

        local_vals = [exp_a["xgboost_local"]["pr_auc_mean"], exp_a["graphsage_local"]["pr_auc_mean"]]
        local_errs = [exp_a["xgboost_local"]["pr_auc_std"], exp_a["graphsage_local"]["pr_auc_std"]]
        all_vals = [exp_a["xgboost_all"]["pr_auc_mean"], exp_a["graphsage_all"]["pr_auc_mean"]]
        all_errs = [exp_a["xgboost_all"]["pr_auc_std"], exp_a["graphsage_all"]["pr_auc_std"]]

        b1 = ax.bar(x - width/2, local_vals, width, yerr=local_errs, label="Local Features Only (94 dim)", color="#3498db", capsize=4, edgecolor="black")
        b2 = ax.bar(x + width/2, all_vals, width, yerr=all_errs, label="All Features (165 dim: Local + 1-Hop)", color="#e67e22", capsize=4, edgecolor="black")

        ax.set_ylabel("PR-AUC (Illicit Class)", fontweight="bold")
        ax.set_title("Experiment A: Impact of Topological Features (Local vs Aggregated)", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["XGBoost (GBDT)", "GraphSAGE (GNN)"], fontweight="bold")
        ax.set_ylim(0, 0.9)
        ax.legend(loc="upper left")

        for b in [b1, b2]:
            for bar in b:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        for d in [DOCS_FIG_DIR, ML_FIG_DIR]:
            plt.savefig(d / "representation_comparison.png", dpi=300)
        plt.close()

    # =========================================================================
    # 3. GNN Depth Ablation
    # =========================================================================
    if "graphsage_depth_1" in exp_a and "graphsage_depth_3" in exp_a:
        print("Generating Figure 3: Depth Ablation...")
        depths = [1, 2, 3]
        depth_vals = [
            exp_a["graphsage_depth_1"]["pr_auc_mean"],
            exp_a["graphsage_all"]["pr_auc_mean"],
            exp_a["graphsage_depth_3"]["pr_auc_mean"],
        ]
        depth_errs = [
            exp_a["graphsage_depth_1"]["pr_auc_std"],
            exp_a["graphsage_all"]["pr_auc_std"],
            exp_a["graphsage_depth_3"]["pr_auc_std"],
        ]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(depths, depth_vals, marker="o", linewidth=2.5, markersize=8, color="#27ae60", label="GraphSAGE PR-AUC")
        ax.fill_between(depths, np.array(depth_vals) - np.array(depth_errs), np.array(depth_vals) + np.array(depth_errs), color="#27ae60", alpha=0.2)
        ax.set_xlabel("Number of GNN Message-Passing Layers", fontweight="bold")
        ax.set_ylabel("PR-AUC", fontweight="bold")
        ax.set_title("Ablation: Impact of GNN Depth on Detection Performance", fontweight="bold")
        ax.set_xticks(depths)
        ax.set_ylim(0.2, 0.8)

        for d, v in zip(depths, depth_vals):
            ax.annotate(f"{v:.4f}", (d, v), textcoords="offset points", xytext=(0, 10), ha="center", fontweight="bold")

        for d in [DOCS_FIG_DIR, ML_FIG_DIR]:
            plt.savefig(d / "depth_ablation.png", dpi=300)
        plt.close()

    # =========================================================================
    # 4. Unknown Nodes Ablation
    # =========================================================================
    if "graphsage_no_unknown" in exp_a:
        print("Generating Figure 4: Unknown Nodes Ablation...")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        conds = ["With Unknown Nodes\n(Full Graph)", "Without Unknown Nodes\n(Known-Only Subgraph)"]
        vals = [exp_a["graphsage_all"]["pr_auc_mean"], exp_a["graphsage_no_unknown"]["pr_auc_mean"]]
        errs = [exp_a["graphsage_all"]["pr_auc_std"], exp_a["graphsage_no_unknown"]["pr_auc_std"]]

        bars = ax.bar(conds, vals, yerr=errs, width=0.45, color=["#27ae60", "#e74c3c"], capsize=5, edgecolor="black")
        ax.set_ylabel("PR-AUC", fontweight="bold")
        ax.set_title("Ablation: Value of Unlabeled Transactions in Message Passing", fontweight="bold")
        ax.set_ylim(0, 0.8)

        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.02, f"{h:.4f}", ha="center", va="bottom", fontweight="bold")

        for d in [DOCS_FIG_DIR, ML_FIG_DIR]:
            plt.savefig(d / "unknown_nodes_ablation.png", dpi=300)
        plt.close()

    # =========================================================================
    # 5. Rolling Origin Generalization / Temporal Drift
    # =========================================================================
    if "rolling_origin" in baselines:
        print("Generating Figure 5: Temporal Drift Across Windows...")
        ro = baselines["rolling_origin"]
        w_names = ["W1 (26-30)", "W2 (31-35)", "W3 (36-40)", "W4 (41-45)\n[Shutdown]", "W5 (46-49)\n[Post-Takedown]"]
        w_keys = ["window_1", "window_2", "window_3_pre_shutdown", "window_4_around_shutdown", "window_5_post_shutdown"]

        heur_prauc = [ro[k]["heuristic"]["pr_auc"] for k in w_keys]
        lr_prauc = [ro[k]["logistic_regression"]["pr_auc_mean"] for k in w_keys]
        xgb_prauc = [ro[k]["xgboost"]["pr_auc_mean"] for k in w_keys]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(w_names, heur_prauc, marker="s", linestyle=":", label="Heuristic Rule", color="#95a5a6", linewidth=2)
        ax.plot(w_names, lr_prauc, marker="^", linestyle="--", label="Logistic Regression", color="#7f8c8d", linewidth=2)
        ax.plot(w_names, xgb_prauc, marker="o", linestyle="-", label="XGBoost (GBDT)", color="#2980b9", linewidth=2.5)

        ax.axvspan(3.5, 4.5, color="#e74c3c", alpha=0.12, label="Darknet Shutdown Regime Shift")
        ax.set_ylabel("PR-AUC (Illicit)", fontweight="bold")
        ax.set_xlabel("Evaluation Window", fontweight="bold")
        ax.set_title("Temporal Drift: Performance Collapse Post-Darknet Shutdown", fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.legend(loc="upper right")

        for d in [DOCS_FIG_DIR, ML_FIG_DIR]:
            plt.savefig(d / "temporal_drift.png", dpi=300)
        plt.close()

    # =========================================================================
    # 6. Precision-Recall Curves
    # =========================================================================
    pred_dir = REPO_ROOT / "ml/experiments/results/predictions"
    if pred_dir.exists():
        print("Generating Figure 6: Precision-Recall Curves...")
        fig, ax = plt.subplots(figsize=(8, 6))

        files_to_plot = [
            ("logistic_regression_primary_seed42.npz", "Logistic Regression", "#7f8c8d", "--"),
            ("xgboost_primary_seed42.npz", "XGBoost (GBDT)", "#2980b9", "-"),
            ("graphsage_all_primary_seed42.npz", "GraphSAGE (All)", "#27ae60", "-"),
            ("hybrid_graphsage_xgboost_primary_seed42.npz", "Hybrid (GNN+XGB)", "#8e44ad", "-"),
        ]

        for fname, label, col, ls in files_to_plot:
            fpath = pred_dir / fname
            if fpath.exists():
                data = np.load(fpath)
                y_true = data["y_true"]
                y_prob = data["y_prob"]
                mask = data["supervised_mask"]
                y_t = y_true[mask]
                y_p = y_prob[mask]
                if len(y_t) > 0 and np.sum(y_t) > 0:
                    p, r, _ = precision_recall_curve(y_t, y_p)
                    ax.plot(r, p, label=label, color=col, linestyle=ls, linewidth=2)

        ax.axhline(0.80, color="gray", linestyle=":", label="80% Target Precision")
        ax.set_xlabel("Recall", fontweight="bold")
        ax.set_ylabel("Precision", fontweight="bold")
        ax.set_title("Precision-Recall Curves on Primary Test Split (Steps 40–49)", fontweight="bold")
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower left")

        for d in [DOCS_FIG_DIR, ML_FIG_DIR]:
            plt.savefig(d / "pr_curves.png", dpi=300)
        plt.close()

    print("All figures successfully generated and saved to docs/figures/ and ml/experiments/figures/!")


if __name__ == "__main__":
    generate_all_figures()
