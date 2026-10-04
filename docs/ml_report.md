# ChainGuard Machine Learning Report

## Phase 2: Tabular and Heuristic Baselines Evaluation

### 1. Executive Summary

This report documents the baseline models for ChainGuard's illicit cryptocurrency transaction detection pipeline on the Elliptic Bitcoin dataset (49 time steps, 203,769 transactions, 234,355 payment flows).

We evaluate three baseline model tiers under strict zero-temporal-leakage conditions:
1. **Tier 0: Interpretable Heuristic Rule** — A 2-level decision rule fitted strictly on training data and frozen to JSON (`ml/artifacts/heuristic/frozen_rule.json`).
2. **Tier 1: Logistic Regression** — Linear baseline with feature standardization computed strictly from training-set statistics. Evaluated across 5 fixed random seeds.
3. **Tier 1: XGBoost (GBDT)** — Gradient boosted tree baseline with class imbalance weighting (`scale_pos_weight`) and hyperparameter tuning (`max_depth`, `learning_rate`, `n_estimators`) performed strictly on the validation window (never touching test). Evaluated across 5 fixed random seeds.

---

### 2. Primary Test Set Results (Time Steps 40–49)

The primary split trains on Steps 1–34 (137,295 nodes, 29,894 labeled), validates hyperparameter choices on Steps 35–39 (20,857 nodes, 4,885 labeled), and evaluates on Steps 40–49 (45,617 nodes, 11,769 labeled).

Unknown nodes are preserved in the graph for topology and message passing but strictly excluded from loss computation and evaluation metrics.

| Model Tier | Model | PR-AUC (Illicit) | F1 Score (Illicit) | Recall @ 80% Precision | Recall @ 90% Precision |
|---|---|:---:|:---:|:---:|:---:|
| **Tier 0** | **Heuristic Rule (Frozen)** | 0.1112 | 0.2326 | 0.0000 | 0.0000 |
| **Tier 1** | **Logistic Regression** | 0.2185 ± 0.0000 | 0.2524 ± 0.0000 | 0.0000 | 0.0000 |
| **Tier 1** | **XGBoost (GBDT)** | **0.6769 ± 0.0024** | **0.6472 ± 0.0075** | **0.6000 ± 0.0000** | **0.5915 ± 0.0000** |

> [!NOTE]
> All learned model metrics report `mean ± std` across 5 independent random seeds (`[42, 123, 456, 789, 1337]`). The Heuristic rule is deterministic and run once.

---

### 3. Rolling-Origin Temporal Generalization (Experiment C)

To quantify performance degradation over time and across structural market events without data leakage, models are trained on the initial window (Steps 1–20, validated on Steps 21–25) and tested on 5 successive rolling test windows spanning Steps 26 to 49.

| Test Window | Time Steps | Regime Context | Heuristic PR-AUC | LogReg PR-AUC | XGBoost PR-AUC (mean ± std) | XGBoost Rec@80%Prec |
|---|:---:|---|:---:|:---:|:---:|:---:|
| **Window 1** | 26–30 | Stable Pre-Shutdown | 0.5722 | 0.5550 | 0.7736 ± 0.0054 | 0.6451 |
| **Window 2** | 31–35 | High Illicit Volume | 0.4363 | 0.4301 | 0.8124 ± 0.0066 | 0.7588 |
| **Window 3** | 36–40 | Pre-Takedown | 0.2505 | 0.3808 | 0.8491 ± 0.0077 | 0.7793 |
| **Window 4** | 41–45 | Shutdown Transition (Step 43) | 0.1235 | 0.1523 | 0.7608 ± 0.0134 | 0.6985 |
| **Window 5** | 46–49 | Post-Shutdown Regime Shift | 0.0463 | 0.1038 | 0.0671 ± 0.0095 | 0.0000 |

---

### 4. Key Findings & Analysis

#### A. The Heuristic vs. ML Performance Gap
- **Heuristic Baseline Limitations**: The Tier 0 heuristic rule achieves a PR-AUC of 0.1112 on the primary test set. While it isolates a high-probability subset of illicit nodes in training (`feat_52 <= -0.4661 AND feat_102 > -1.1586`), it fails to achieve 80% or 90% precision on unseen temporal test sets, resulting in 0% recall at production precision thresholds.
- **Linear Model Ceiling**: Logistic Regression improves PR-AUC to 0.2185 (+96.5% relative over Heuristic) but still suffers heavily from linear non-separability in the 165-dimensional transaction space, yielding 0% recall at $\ge 80\%$ precision.
- **Tree-Based Superiority on Tabular Data**: XGBoost dramatically outperforms simpler baselines, achieving **0.6769 PR-AUC** and **0.6472 F1 score**, capturing **60.0% of illicit transactions at $\ge 80\%$ precision** and **59.15% at $\ge 90\%$ precision**. Non-linear feature interactions and tree splits are critical for extracting signal from aggregated neighborhood features.

#### B. Impact of the Step 43 Darknet Shutdown & Concept Drift
- In Windows 1–3, XGBoost maintains high PR-AUC (0.77 to 0.85).
- At Window 4 (Steps 41–45), which encompasses the historical darknet marketplace takedown (~Step 43), performance drops to 0.7608.
- At Window 5 (Steps 46–49), illicit transaction prevalence plummets from ~10.6% down to <1.8% (and as low as 0.28% at Step 46). Models trained only on early data (Steps 1–20) experience severe performance degradation (PR-AUC drops to 0.0671).
- **Takeaway**: Static models degrade drastically when major macroeconomic or law enforcement structural breaks occur on public blockchains. This underscores the necessity of continuous monitoring, drift detection (PSI), and periodic/drift-triggered model retraining.
