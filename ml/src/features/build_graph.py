"""
Build the Elliptic transaction graph from raw CSV files.

Loads:
  - elliptic_txs_features.csv  (txId, time_step, 164 features)
  - elliptic_txs_classes.csv   (txId, class: 1=illicit, 2=licit, "unknown")
  - elliptic_txs_edgelist.csv  (txId1, txId2)

Outputs a single processed pickle file containing:
  - node_features : np.ndarray  (N, 166)  — time_step + 164 features + label_encoded
  - node_ids      : list[int]             — original txIds in row order
  - time_steps    : np.ndarray  (N,)      — time step per node
  - labels        : np.ndarray  (N,)      — 0=licit, 1=illicit, -1=unknown
  - edge_index    : np.ndarray  (2, E)    — directed edges as (src_row, dst_row)
  - label_mask    : np.ndarray  (N,)      — True for nodes with known labels
  - stats         : dict                  — per-timestep counts for DATA.md

Unknown-label nodes remain in the graph (for message passing) but are
flagged via label_mask so they can be excluded from supervised loss & metrics.
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "elliptic_graph.pkl"
STATS_FILE = PROCESSED_DIR / "elliptic_stats.json"


def load_features(raw_dir: Path) -> pd.DataFrame:
    """Load features CSV. Columns: txId, time_step, feat_0 ... feat_164 (167 total cols)."""
    path = raw_dir / "elliptic_txs_features.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run download_elliptic.py first.")
        sys.exit(1)
    # No header in original file — first col is txId, second is time_step,
    # remaining 165 are features (total 167 columns).
    cols = ["txId", "time_step"] + [f"feat_{i}" for i in range(165)]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def load_classes(raw_dir: Path) -> pd.DataFrame:
    """Load class labels. Columns: txId, class (1=illicit, 2=licit, 'unknown')."""
    path = raw_dir / "elliptic_txs_classes.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run download_elliptic.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    # Rename columns to standard names
    df.columns = ["txId", "class"]
    return df


def load_edges(raw_dir: Path) -> pd.DataFrame:
    """Load edge list. Columns: txId1, txId2."""
    path = raw_dir / "elliptic_txs_edgelist.csv"
    if not path.exists():
        print(f"ERROR: {path} not found. Run download_elliptic.py first.")
        sys.exit(1)
    df = pd.read_csv(path)
    df.columns = ["txId1", "txId2"]
    return df


def encode_labels(class_series: pd.Series) -> np.ndarray:
    """Map class labels: 1 -> 1 (illicit), 2 -> 0 (licit), 'unknown' -> -1."""
    mapping = {"1": 1, "2": 0, "unknown": -1}
    return class_series.astype(str).map(mapping).values.astype(np.int8)


def build_graph(raw_dir: Path = RAW_DIR) -> dict:
    """
    Build the processed graph dictionary from raw Elliptic CSV files.

    Returns a dict with keys:
      node_features, node_ids, time_steps, labels, edge_index,
      label_mask, stats
    """
    print("Loading features...")
    feat_df = load_features(raw_dir)

    print("Loading classes...")
    class_df = load_classes(raw_dir)

    print("Loading edges...")
    edge_df = load_edges(raw_dir)

    # Merge features with labels
    merged = feat_df.merge(class_df, on="txId", how="left")
    # Fill any missing class (shouldn't happen but be safe)
    merged["class"] = merged["class"].fillna("unknown")

    # Build node ID -> row index mapping
    node_ids = merged["txId"].values
    id_to_idx = {int(nid): idx for idx, nid in enumerate(node_ids)}

    # Features: 165 numeric features (exclude txId, time_step, and class)
    feature_cols = [c for c in merged.columns if c.startswith("feat_")]
    node_features = merged[feature_cols].values.astype(np.float32)

    # Time steps
    time_steps = merged["time_step"].values.astype(np.int32)

    # Labels
    labels = encode_labels(merged["class"])
    label_mask = labels != -1

    # Edge index: map txIds to row indices
    src_idx = edge_df["txId1"].map(id_to_idx)
    dst_idx = edge_df["txId2"].map(id_to_idx)
    valid_mask = src_idx.notna() & dst_idx.notna()
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"  Warning: {dropped} edges reference nodes not in features file (dropped).")

    edge_index = np.vstack([
        src_idx[valid_mask].astype(np.int64).values,
        dst_idx[valid_mask].astype(np.int64).values,
    ])

    # Compute per-timestep statistics
    stats = compute_stats(time_steps, labels, edge_index, id_to_idx, merged, edge_df)

    graph = {
        "node_features": node_features,
        "node_ids": node_ids.tolist(),
        "time_steps": time_steps,
        "labels": labels,
        "edge_index": edge_index,
        "label_mask": label_mask,
        "stats": stats,
    }

    print(f"\nGraph built:")
    print(f"  Nodes: {len(node_ids)}")
    print(f"  Edges: {edge_index.shape[1]}")
    print(f"  Illicit: {(labels == 1).sum()}")
    print(f"  Licit:   {(labels == 0).sum()}")
    print(f"  Unknown: {(labels == -1).sum()}")
    print(f"  Time steps: {time_steps.min()} to {time_steps.max()}")

    return graph


def compute_stats(
    time_steps: np.ndarray,
    labels: np.ndarray,
    edge_index: np.ndarray,
    id_to_idx: dict,
    merged_df: pd.DataFrame,
    edge_df: pd.DataFrame,
) -> dict:
    """Compute per-timestep statistics for DATA.md."""
    unique_steps = sorted(np.unique(time_steps))

    # Pre-compute node timestep lookup for edges
    node_ts = time_steps

    per_step = {}
    for ts in unique_steps:
        mask = time_steps == ts
        step_labels = labels[mask]

        # Count edges where source node is in this timestep
        src_in_step = node_ts[edge_index[0]] == ts
        edge_count = int(src_in_step.sum())

        illicit_count = int((step_labels == 1).sum())
        licit_count = int((step_labels == 0).sum())
        unknown_count = int((step_labels == -1).sum())
        total = int(mask.sum())

        illicit_ratio = illicit_count / max(illicit_count + licit_count, 1)

        per_step[int(ts)] = {
            "nodes": total,
            "edges": edge_count,
            "illicit": illicit_count,
            "licit": licit_count,
            "unknown": unknown_count,
            "illicit_ratio": round(illicit_ratio, 4),
        }

    return {"per_timestep": per_step, "total_steps": len(unique_steps)}


def save_graph(graph: dict, output_path: Path = OUTPUT_FILE):
    """Save processed graph to pickle."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(graph, f)
    print(f"\nProcessed graph saved to {output_path}")


def save_stats(stats: dict, stats_path: Path = STATS_FILE):
    """Save statistics to JSON for DATA.md generation."""
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Statistics saved to {stats_path}")


def main():
    graph = build_graph()
    save_graph(graph)
    save_stats(graph["stats"])

    # Print summary table
    print("\n" + "=" * 80)
    print("PER-TIMESTEP STATISTICS")
    print("=" * 80)
    print(f"{'Step':>4} | {'Nodes':>6} | {'Edges':>6} | {'Illicit':>7} | {'Licit':>6} | {'Unknown':>7} | {'Illicit%':>8}")
    print("-" * 80)
    for ts in sorted(graph["stats"]["per_timestep"].keys(), key=int):
        s = graph["stats"]["per_timestep"][ts]
        print(
            f"{ts:>4} | {s['nodes']:>6} | {s['edges']:>6} | "
            f"{s['illicit']:>7} | {s['licit']:>6} | {s['unknown']:>7} | "
            f"{s['illicit_ratio']:>7.2%}"
        )


if __name__ == "__main__":
    main()
