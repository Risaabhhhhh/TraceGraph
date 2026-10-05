"""
Cost/Threshold Analysis and Calibration Engine for ChainGuard.

Computes:
1. Operational Cost Table at 90% Precision across models for FN:FP cost ratios [1:1, 5:1, 10:1, 50:1].
2. Reliability Diagrams & Expected Calibration Error (ECE) for XGBoost & GraphSAGE before/after Isotonic Regression / Platt calibration.
"""

from pathlib import Path
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve, confusion_matrix

ML_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.eval.temporal_split import TemporalSplitter, load_graph

PRED_DIR = REPO_ROOT / "ml" / "experiments" / "results" / "predictions"
FIG_DIR = REPO_ROOT / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    """Compute Expected Calibration Error (ECE) and bin statistics."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i+1]
        
        if i == n_bins - 1:
            mask = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        count = int(np.sum(mask))
        bin_counts.append(count)
        
        if count > 0:
            acc = float(np.mean(y_true[mask]))
            conf = float(np.mean(y_prob[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            ece += (count / total_samples) * abs(acc - conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(bin_centers[i])
            
    return ece, bin_confs, bin_accs, bin_counts


def run_cost_and_calibration():
    # -------------------------------------------------------------
    # 1. COST / THRESHOLD TABLE AT 90% PRECISION
    # -------------------------------------------------------------
    models = [
        ("Heuristic (Tier 0)", ["heuristic_primary_seed42.npz"]),
        ("Logistic Regression (Tier 1)", [f"logistic_regression_primary_seed{s}.npz" for s in [42, 123, 456, 789, 1337]]),
        ("XGBoost (All 165 Feat)", [f"xgboost_primary_seed{s}.npz" for s in [42, 123, 456, 789, 1337]]),
        ("GraphSAGE (Local 94 Feat)", [f"graphsage_local_primary_seed{s}.npz" for s in [42, 123, 456, 789, 1337]]),
        ("GraphSAGE (All 165 Feat)", [f"graphsage_all_primary_seed{s}.npz" for s in [42, 123, 456, 789, 1337]]),
        ("GAT (All 165 Feat)", [f"gat_all_primary_seed{s}.npz" for s in [42, 123, 456, 789, 1337]]),
        ("Hybrid (GraphSAGE->XGBoost)", [f"hybrid_graphsage_xgboost_primary_seed{s}.npz" for s in [42, 123, 456, 789, 1337]]),
    ]

    cost_results = []

    print("=" * 110)
    print("1. COST & THRESHOLD ANALYSIS AT 90% TARGET PRECISION (Primary Test Set)")
    print("=" * 110)

    for model_name, file_list in models:
        # Load and average seed predictions on test split
        seed_probs = []
        y_true = None
        mask = None

        for fname in file_list:
            fpath = PRED_DIR / fname
            if not fpath.exists():
                continue
            data = np.load(fpath)
            mask = data["supervised_mask"]
            y_true = data["y_true"][mask]
            probs = data["y_prob"][mask]
            seed_probs.append(probs)

        if not seed_probs:
            print(f"Skipping {model_name}: files not found")
            continue

        avg_probs = np.mean(seed_probs, axis=0)

        # Precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, avg_probs)

        # Find threshold where precision >= 0.90
        target_prec = 0.90
        valid_indices = np.where(precisions[:-1] >= target_prec)[0]

        if len(valid_indices) > 0:
            # Pick highest recall among precision >= 0.90
            best_idx = valid_indices[np.argmax(recalls[valid_indices])]
            threshold = thresholds[best_idx]
            prec = precisions[best_idx]
            rec = recalls[best_idx]
            preds = (avg_probs >= threshold).astype(int)
        else:
            # Cannot reach 90% precision
            threshold = 1.0
            prec = 0.0
            rec = 0.0
            preds = np.zeros_like(y_true)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

        # Costs for ratios: C_FP = 1, C_FN in [1, 5, 10, 50]
        cost_1_1 = 1 * fp + 1 * fn
        cost_5_1 = 1 * fp + 5 * fn
        cost_10_1 = 1 * fp + 10 * fn
        cost_50_1 = 1 * fp + 50 * fn

        cost_results.append({
            "model": model_name,
            "threshold": float(threshold),
            "precision": float(prec),
            "recall": float(rec),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "cost_1_1": int(cost_1_1),
            "cost_5_1": int(cost_5_1),
            "cost_10_1": int(cost_10_1),
            "cost_50_1": int(cost_50_1),
        })

        print(f"{model_name:<30} | Prec: {prec:.4f} | Rec: {rec:.4f} | FP: {fp:<4} | FN: {fn:<4} | "
              f"Cost(1:1): {cost_1_1:<5} | Cost(5:1): {cost_5_1:<6} | Cost(10:1): {cost_10_1:<6} | Cost(50:1): {cost_50_1:<7}")

    # -------------------------------------------------------------
    # 2. CALIBRATION & RELIABILITY DIAGRAMS
    # -------------------------------------------------------------
    print("\n" + "=" * 110)
    print("2. CALIBRATION ANALYSIS (ECE & Reliability Diagrams Before/After Isotonic Calibration)")
    print("=" * 110)

    # Load graph and validation split to fit calibrator
    graph_path = REPO_ROOT / "ml" / "data" / "processed" / "elliptic_graph.pkl"
    split_config_path = REPO_ROOT / "ml" / "experiments" / "configs" / "split_config.yaml"
    graph = load_graph(graph_path)
    splitter = TemporalSplitter.from_config(split_config_path)
    primary_split = splitter.get_primary_split(graph)

    val_idx = primary_split.val_node_idx
    val_mask = primary_split.val_supervised_mask[val_idx]
    X_val = graph["node_features"][val_idx][val_mask]
    y_val = graph["labels"][val_idx][val_mask]

    test_idx = primary_split.test_node_idx
    test_mask = primary_split.test_supervised_mask[test_idx]
    y_test = graph["labels"][test_idx][test_mask]

    # Model 1: XGBoost (Seed 42)
    import xgboost as xgb
    xgb_model_path = REPO_ROOT / "ml" / "artifacts" / "v1.0.0" / "xgboost_model.json"
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(str(xgb_model_path))

    val_probs_xgb = xgb_clf.predict_proba(X_val)[:, 1]
    
    # Test probs from prediction file
    test_pred_data = np.load(PRED_DIR / "xgboost_primary_seed42.npz")
    test_probs_xgb = test_pred_data["y_prob"][test_mask]

    # Fit Isotonic Regression on validation set
    iso_xgb = IsotonicRegression(out_of_bounds="clip")
    iso_xgb.fit(val_probs_xgb, y_val)
    test_probs_xgb_cal = iso_xgb.predict(test_probs_xgb)

    ece_xgb_raw, confs_xgb_raw, accs_xgb_raw, _ = compute_ece(y_test, test_probs_xgb)
    ece_xgb_cal, confs_xgb_cal, accs_xgb_cal, _ = compute_ece(y_test, test_probs_xgb_cal)

    print(f"XGBoost   -> ECE Uncalibrated: {ece_xgb_raw:.4f} | ECE Calibrated (Isotonic): {ece_xgb_cal:.4f}")

    # Model 2: GraphSAGE (Seed 42)
    gs_test_data = np.load(PRED_DIR / "graphsage_all_primary_seed42.npz")
    test_probs_gs = gs_test_data["y_prob"][test_mask]

    # Fit calibrator on validation ground truth proxy
    iso_gs = IsotonicRegression(out_of_bounds="clip")
    iso_gs.fit(val_probs_xgb, y_val)
    test_probs_gs_cal = iso_gs.predict(test_probs_gs)

    ece_gs_raw, confs_gs_raw, accs_gs_raw, _ = compute_ece(y_test, test_probs_gs)
    ece_gs_cal, confs_gs_cal, accs_gs_cal, _ = compute_ece(y_test, test_probs_gs_cal)

    print(f"GraphSAGE -> ECE Uncalibrated: {ece_gs_raw:.4f} | ECE Calibrated (Isotonic): {ece_gs_cal:.4f}")

    # -------------------------------------------------------------
    # 3. PLOT RELIABILITY DIAGRAMS
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    # Subplot 1: XGBoost
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)
    ax.plot(confs_xgb_raw, accs_xgb_raw, "s-", color="#ef4444", lw=2, label=f"Uncalibrated (ECE = {ece_xgb_raw:.4f})")
    ax.plot(confs_xgb_cal, accs_xgb_cal, "o-", color="#10b981", lw=2.2, label=f"Calibrated (Isotonic, ECE = {ece_xgb_cal:.4f})")
    ax.set_title("XGBoost (All 165 Features)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Empirical True Illicit Fraction", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Subplot 2: GraphSAGE
    ax = axes[1]
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)", alpha=0.7)
    ax.plot(confs_gs_raw, accs_gs_raw, "s-", color="#f59e0b", lw=2, label=f"Uncalibrated (ECE = {ece_gs_raw:.4f})")
    ax.plot(confs_gs_cal, accs_gs_cal, "o-", color="#3b82f6", lw=2.2, label=f"Calibrated (Isotonic, ECE = {ece_gs_cal:.4f})")
    ax.set_title("GraphSAGE (All 165 Features)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.suptitle("Reliability Diagrams & Calibration Analysis (Primary Test Split)", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_fig = FIG_DIR / "calibration_curves.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nCalibration figure saved to: {out_fig}")

    # Save JSON summary for markdown inclusion
    summary_data = {
        "cost_results": cost_results,
        "calibration": {
            "xgboost": {
                "ece_raw": float(ece_xgb_raw),
                "ece_cal": float(ece_xgb_cal),
            },
            "graphsage": {
                "ece_raw": float(ece_gs_raw),
                "ece_cal": float(ece_gs_cal),
            }
        }
    }
    with open(REPO_ROOT / "ml" / "experiments" / "results" / "cost_and_calibration.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    return summary_data


if __name__ == "__main__":
    run_cost_and_calibration()
