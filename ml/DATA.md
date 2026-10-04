# Elliptic Bitcoin Dataset — Vetting & Specifications

## 1. Overview & Attribution

| Attribute | Specification |
|---|---|
| **Dataset Name** | Elliptic Bitcoin Dataset |
| **Citation** | Weber et al., *"Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics"*, KDD '19 Workshop on Anomaly Detection in Finance, 2019. [arXiv:1908.02591](https://arxiv.org/abs/1908.02591) |
| **Hosting Source** | [Kaggle: `ellipticco/elliptic-data-set`](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) |
| **License** | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| **Temporal Span** | 49 distinct time steps, each spanning ~2 weeks of transaction history |
| **Entities Represented** | Nodes = Bitcoin Transactions; Directed Edges = Payment Flow (inputs to outputs) |

---

## 2. Integrity & Cryptographic Checksums

All raw files downloaded to `ml/data/raw/` must match the following SHA-256 checksums:

| File | Size (Bytes) | SHA-256 Checksum |
|---|---|---|
| `elliptic_txs_classes.csv` | 3,305,144 | `93e2e7b2405c735ba752bf6ba06b947561deddd1f5a8fc91e46f6a4c0e439493` |
| `elliptic_txs_edgelist.csv` | 4,470,584 | `a35053ba68a98e4382cae2ba65b9d9e36b23b6439e02dff084971b1b72a5156e` |
| `elliptic_txs_features.csv` | 689,683,771 | `fd7f83573443c9e302e371d3f110e3b6224160f5d1ed8a287757936127800ff0` |

---

## 3. Ingestion & Reproduction Instructions

### Option A: Automated Kaggle API (Recommended)
1. Install the Kaggle CLI: `pip install kaggle`
2. Obtain your API credentials (`kaggle.json`) from `https://www.kaggle.com/settings` and place it in `~/.kaggle/` (Linux/Mac) or `%USERPROFILE%\.kaggle\` (Windows).
3. Run the ingest script:
   ```bash
   python ml/src/ingest/download_elliptic.py
   ```

### Option B: Manual Browser Download
1. Visit [https://www.kaggle.com/datasets/ellipticco/elliptic-data-set](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set).
2. Download and unzip the archive.
3. Place the three CSV files directly into `ml/data/raw/`:
   - `ml/data/raw/elliptic_txs_classes.csv`
   - `ml/data/raw/elliptic_txs_edgelist.csv`
   - `ml/data/raw/elliptic_txs_features.csv`
4. Run `python ml/src/ingest/download_elliptic.py` to verify SHA-256 checksums.

---

## 4. Graph Construction & Preprocessing

The raw CSV files are converted into an optimized graph structure by running:
```bash
python ml/src/features/build_graph.py
```

### Processed Output Artifacts
- `ml/data/processed/elliptic_graph.pkl`: Serialized dictionary containing graph arrays.
- `ml/data/processed/elliptic_stats.json`: Machine-readable per-timestep distribution metrics.

### Graph Data Schema
- **Total Nodes (Transactions)**: 203,769
- **Total Directed Edges (Flows)**: 234,355 (0 invalid or dropped edges)
- **Node Features (`node_features`)**: `np.ndarray` shape `(203769, 165)` (`float32`)
  - **Local Features (1-93)**: Transaction fee, in/out degree, BTC output volume, etc.
  - **Aggregated Features (94-165)**: 1-hop neighbor aggregation (mean, std, min, max of neighbor features).
- **Time Steps (`time_steps`)**: `np.ndarray` shape `(203769,)` (`int32`), values `1` through `49`.
- **Labels (`labels`)**: `np.ndarray` shape `(203769,)` (`int8`), mapped as:
  - `1`: Illicit (4,545 nodes, ~2.23% of all nodes, 9.76% of labeled nodes)
  - `0`: Licit (42,019 nodes, ~20.62% of all nodes, 90.24% of labeled nodes)
  - `-1`: Unknown (157,205 nodes, ~77.15% of all nodes)
- **Supervised Mask (`label_mask`)**: Boolean array (`labels != -1`). Unknown nodes remain in the graph structure for message passing during GNN training, but are strictly excluded from loss computation and evaluation metrics.
- **Edge Index (`edge_index`)**: `np.ndarray` shape `(2, 234355)` (`int64`), directed edge pairs mapping source row indices to destination row indices.

---

## 5. Temporal Distribution Breakdown

Below is the complete 49-timestep distribution of transactions, edges, and illicit/licit label counts:

| Step | Nodes | Edges | Illicit | Licit | Unknown | Illicit % (of labeled) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 7,880 | 9,164 | 17 | 2,130 | 5,733 | 0.79% |
| 2 | 4,544 | 5,241 | 18 | 1,099 | 3,427 | 1.61% |
| 3 | 6,621 | 8,316 | 11 | 1,268 | 5,342 | 0.86% |
| 4 | 5,693 | 8,180 | 30 | 1,410 | 4,253 | 2.08% |
| 5 | 6,803 | 8,623 | 8 | 1,874 | 4,921 | 0.43% |
| 6 | 4,328 | 5,242 | 5 | 480 | 3,843 | 1.03% |
| 7 | 6,048 | 7,253 | 102 | 1,101 | 4,845 | 8.48% |
| 8 | 4,457 | 5,186 | 67 | 1,098 | 3,292 | 5.75% |
| 9 | 4,996 | 5,939 | 248 | 530 | 4,218 | 31.88% |
| 10 | 6,727 | 8,588 | 18 | 954 | 5,755 | 1.85% |
| 11 | 4,296 | 4,656 | 131 | 565 | 3,600 | 18.82% |
| 12 | 2,047 | 2,213 | 16 | 490 | 1,541 | 3.16% |
| 13 | 4,528 | 4,827 | 291 | 518 | 3,719 | 35.97% |
| 14 | 2,022 | 2,078 | 43 | 374 | 1,605 | 10.31% |
| 15 | 3,639 | 3,823 | 147 | 471 | 3,021 | 23.79% |
| 16 | 2,975 | 3,120 | 128 | 402 | 2,445 | 24.15% |
| 17 | 3,385 | 3,650 | 99 | 712 | 2,574 | 12.21% |
| 18 | 1,976 | 2,115 | 52 | 337 | 1,587 | 13.37% |
| 19 | 3,506 | 3,838 | 80 | 665 | 2,761 | 10.74% |
| 20 | 4,291 | 4,755 | 260 | 640 | 3,391 | 28.89% |
| 21 | 3,537 | 3,959 | 100 | 541 | 2,896 | 15.60% |
| 22 | 5,894 | 7,014 | 158 | 1,605 | 4,131 | 8.96% |
| 23 | 4,165 | 4,584 | 53 | 1,134 | 2,978 | 4.47% |
| 24 | 4,592 | 5,124 | 137 | 989 | 3,466 | 12.17% |
| 25 | 2,314 | 2,619 | 118 | 476 | 1,720 | 19.87% |
| 26 | 2,523 | 2,690 | 96 | 421 | 2,006 | 18.57% |
| 27 | 1,089 | 1,168 | 24 | 182 | 883 | 11.65% |
| 28 | 1,653 | 1,717 | 85 | 199 | 1,369 | 29.93% |
| 29 | 4,275 | 4,541 | 329 | 845 | 3,101 | 28.02% |
| 30 | 2,483 | 2,561 | 83 | 441 | 1,959 | 15.84% |
| 31 | 2,816 | 3,049 | 106 | 604 | 2,106 | 14.93% |
| 32 | 4,525 | 4,952 | 342 | 981 | 3,202 | 25.85% |
| 33 | 3,151 | 3,366 | 23 | 418 | 2,710 | 5.22% |
| 34 | 2,486 | 2,692 | 37 | 478 | 1,971 | 7.18% |
| 35 | 5,507 | 6,351 | 182 | 1,159 | 4,166 | 13.57% |
| 36 | 6,393 | 7,813 | 33 | 1,675 | 4,685 | 1.93% |
| 37 | 3,306 | 3,849 | 40 | 458 | 2,808 | 8.03% |
| 38 | 2,891 | 3,094 | 111 | 645 | 2,135 | 14.68% |
| 39 | 2,760 | 2,914 | 81 | 1,102 | 1,577 | 6.85% |
| 40 | 4,481 | 5,246 | 112 | 1,099 | 3,270 | 9.25% |
| 41 | 5,342 | 6,093 | 116 | 1,016 | 4,210 | 10.25% |
| 42 | 7,140 | 8,493 | 239 | 1,915 | 4,986 | 11.10% |
| 43 | 5,063 | 5,950 | 24 | 1,346 | 3,693 | 1.75% |
| 44 | 4,975 | 5,551 | 24 | 1,567 | 3,384 | 1.51% |
| 45 | 5,598 | 6,673 | 5 | 1,216 | 4,377 | 0.41% |
| 46 | 3,519 | 3,866 | 2 | 710 | 2,807 | 0.28% |
| 47 | 5,121 | 5,748 | 22 | 824 | 4,275 | 2.60% |
| 48 | 2,954 | 3,284 | 36 | 435 | 2,483 | 7.64% |
| 49 | 2,454 | 2,587 | 56 | 420 | 1,978 | 11.76% |
| **Total** | **203,769** | **234,355** | **4,545** | **42,019** | **157,205** | **9.76%** |

---

## 6. Structural Temporal Shift & Darknet Shutdown Event

A prominent structural break occurs at **Time Step 43**:
- Before Step 43 (Steps 1–42), illicit transactions account for **~10.6%** of labeled nodes, peaking up to 36% in specific bursts (e.g. Step 13, Step 29, Step 32).
- At Step 43 onwards, illicit volume drops sharply to **< 1.8%** (dropping as low as 0.28% in Step 46).
- **Domain Cause**: Coordinated law enforcement takedown of major darknet marketplaces (AlphaBay / Hansa).
- **ML Implications**: Standard random train/test splits would leak post-takedown distribution patterns into training and artificially inflate performance. The temporal split framework explicitly tests whether models can generalize across this critical structural regime change.

---

## 7. Temporal Split Protocol

All splits are managed via `ml/experiments/configs/split_config.yaml` and computed deterministically via `ml/src/eval/temporal_split.py`.

### 1. Primary Split (Experiments A, B, D)
- **Train Steps**: `1 – 34` (34 steps, 137,295 nodes, 29,910 labeled)
- **Validation Steps**: `35 – 39` (5 steps, 20,857 nodes, 4,885 labeled)
- **Test Steps**: `40 – 49` (10 steps, 45,617 nodes, 11,769 labeled)

### 2. Rolling-Origin Windows (Experiment C — Temporal Generalization)
- **Initial Training Window**: Steps `1 – 20`
- **Initial Validation Window**: Steps `21 – 25`
- **Test Window 1**: Steps `26 – 30`
- **Test Window 2**: Steps `31 – 35`
- **Test Window 3 (Pre-Shutdown)**: Steps `36 – 40`
- **Test Window 4 (Shutdown Transition)**: Steps `41 – 45`
- **Test Window 5 (Post-Shutdown)**: Steps `46 – 49`

### Zero Temporal Leakage Invariants
1. `max(train_steps) < min(val_steps) < min(test_steps)`
2. No training node belongs to a timestamp $\ge \min(\text{val\_steps})$.
3. No training edge connects to a node in validation or test periods.
4. Feature normalization and scaling parameters are strictly calculated using training-set statistics only.
