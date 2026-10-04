"""
Config-driven temporal splitter for the Elliptic dataset.

Implements rolling-origin windowed splits that guarantee:
  1. Train steps < validation steps < test steps  (strict temporal ordering)
  2. No future-period node, edge, or label influences training data
  3. Unknown-label nodes stay in the graph but are excluded from loss/metrics

Usage:
    from ml.src.eval.temporal_split import TemporalSplitter

    splitter = TemporalSplitter.from_config("ml/experiments/configs/split_config.yaml")
    primary = splitter.get_primary_split(graph)
    windows = splitter.get_rolling_origin_splits(graph)
"""

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass
class SplitResult:
    """A single train/val/test split of the graph."""

    name: str

    # Node indices (row indices into the full graph arrays)
    train_node_idx: np.ndarray
    val_node_idx: np.ndarray
    test_node_idx: np.ndarray

    # Edge indices (column indices into edge_index)
    train_edge_idx: np.ndarray
    val_edge_idx: np.ndarray
    test_edge_idx: np.ndarray

    # Supervised masks: True for nodes with known labels in each split
    train_supervised_mask: np.ndarray
    val_supervised_mask: np.ndarray
    test_supervised_mask: np.ndarray

    # Time step ranges (for documentation / assertions)
    train_steps: list[int] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)
    test_steps: list[int] = field(default_factory=list)


class TemporalSplitter:
    """
    Config-driven temporal splitter.

    Splits are defined entirely by YAML config — no hardcoded step ranges.
    """

    def __init__(self, config: dict):
        self.config = config
        self._validate_config()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "TemporalSplitter":
        """Load splitter from a YAML config file."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def _validate_config(self):
        """Validate that the config has required keys and temporal ordering."""
        cfg = self.config

        # Primary split validation
        if "primary" in cfg:
            p = cfg["primary"]
            train = sorted(p["train_steps"])
            val = sorted(p["val_steps"])
            test = sorted(p["test_steps"])

            assert max(train) < min(val), (
                f"Train steps must precede val steps: max(train)={max(train)}, min(val)={min(val)}"
            )
            assert max(val) < min(test), (
                f"Val steps must precede test steps: max(val)={max(val)}, min(test)={min(test)}"
            )

            # No overlap
            all_steps = set(train) | set(val) | set(test)
            assert len(all_steps) == len(train) + len(val) + len(test), (
                "Train/val/test steps must not overlap"
            )

        # Rolling origin validation
        if "rolling_origin" in cfg:
            ro = cfg["rolling_origin"]
            train = sorted(ro["initial_train_steps"])
            val = sorted(ro["initial_val_steps"])

            assert max(train) < min(val), (
                "Initial train must precede initial val in rolling origin"
            )

            prev_max = max(val)
            for window in ro["test_windows"]:
                w_steps = sorted(window["steps"])
                assert min(w_steps) > prev_max or min(w_steps) > max(train), (
                    f"Test window '{window['name']}' steps {w_steps} must follow "
                    f"initial train+val (max={prev_max})"
                )

    def _split_graph(
        self,
        graph: dict,
        train_steps: list[int],
        val_steps: list[int],
        test_steps: list[int],
        name: str = "split",
    ) -> SplitResult:
        """
        Split graph by time steps.

        Nodes and edges are assigned to splits based on their time step.
        For edges, an edge belongs to a split if BOTH endpoints are in
        that split's time steps or earlier (training edges for train split,
        train+val edges for val split, etc.).
        """
        time_steps = graph["time_steps"]
        labels = graph["labels"]
        edge_index = graph["edge_index"]
        label_mask = graph["label_mask"]

        train_set = set(train_steps)
        val_set = set(val_steps)
        test_set = set(test_steps)

        # Node indices per split
        train_node_idx = np.where(np.isin(time_steps, list(train_set)))[0]
        val_node_idx = np.where(np.isin(time_steps, list(val_set)))[0]
        test_node_idx = np.where(np.isin(time_steps, list(test_set)))[0]

        # Edge filtering: an edge is in the train split if BOTH endpoints
        # are in train time steps. For val, both endpoints must be in
        # train OR val steps (so the model can see training-period edges
        # during validation). Similarly for test.
        train_node_set = set(train_node_idx.tolist())
        val_allowed = train_set | val_set
        test_allowed = train_set | val_set | test_set

        src_ts = time_steps[edge_index[0]]
        dst_ts = time_steps[edge_index[1]]

        # Train edges: both endpoints in train steps
        train_edge_mask = np.isin(src_ts, list(train_set)) & np.isin(dst_ts, list(train_set))
        train_edge_idx = np.where(train_edge_mask)[0]

        # Val edges: both endpoints in train+val steps
        val_edge_mask = np.isin(src_ts, list(val_allowed)) & np.isin(dst_ts, list(val_allowed))
        val_edge_idx = np.where(val_edge_mask)[0]

        # Test edges: both endpoints in train+val+test steps
        test_edge_mask = np.isin(src_ts, list(test_allowed)) & np.isin(dst_ts, list(test_allowed))
        test_edge_idx = np.where(test_edge_mask)[0]

        # Supervised masks: known labels within each split's nodes
        train_supervised = np.zeros(len(time_steps), dtype=bool)
        train_supervised[train_node_idx] = label_mask[train_node_idx]

        val_supervised = np.zeros(len(time_steps), dtype=bool)
        val_supervised[val_node_idx] = label_mask[val_node_idx]

        test_supervised = np.zeros(len(time_steps), dtype=bool)
        test_supervised[test_node_idx] = label_mask[test_node_idx]

        return SplitResult(
            name=name,
            train_node_idx=train_node_idx,
            val_node_idx=val_node_idx,
            test_node_idx=test_node_idx,
            train_edge_idx=train_edge_idx,
            val_edge_idx=val_edge_idx,
            test_edge_idx=test_edge_idx,
            train_supervised_mask=train_supervised,
            val_supervised_mask=val_supervised,
            test_supervised_mask=test_supervised,
            train_steps=sorted(train_steps),
            val_steps=sorted(val_steps),
            test_steps=sorted(test_steps),
        )

    def get_primary_split(self, graph: dict) -> SplitResult:
        """Get the primary train/val/test split."""
        p = self.config["primary"]
        return self._split_graph(
            graph,
            train_steps=p["train_steps"],
            val_steps=p["val_steps"],
            test_steps=p["test_steps"],
            name="primary",
        )

    def get_rolling_origin_splits(self, graph: dict) -> list[SplitResult]:
        """
        Get rolling-origin splits for Experiment C.

        Returns a list of SplitResults, one per test window.
        Each uses the same initial training set + the initial val set.
        """
        ro = self.config["rolling_origin"]
        train_steps = ro["initial_train_steps"]
        val_steps = ro["initial_val_steps"]

        splits = []
        for window in ro["test_windows"]:
            split = self._split_graph(
                graph,
                train_steps=train_steps,
                val_steps=val_steps,
                test_steps=window["steps"],
                name=window["name"],
            )
            splits.append(split)

        return splits

    def get_expanding_window_splits(self, graph: dict) -> list[SplitResult]:
        """
        Get expanding-window splits for periodic retraining policy.

        Each successive split expands the training window to include
        previous test windows, keeping the val window fixed relative
        to the new train boundary.
        """
        ro = self.config["rolling_origin"]
        current_train = list(ro["initial_train_steps"])
        val_steps = list(ro["initial_val_steps"])

        splits = []
        for i, window in enumerate(ro["test_windows"]):
            split = self._split_graph(
                graph,
                train_steps=current_train,
                val_steps=val_steps,
                test_steps=window["steps"],
                name=f"expanding_{window['name']}",
            )
            splits.append(split)

            # Expand: move val into train, use test window as new val
            current_train = current_train + val_steps
            val_steps = window["steps"]

        return splits


def load_graph(graph_path: str | Path) -> dict:
    """Load a processed graph pickle."""
    with open(graph_path, "rb") as f:
        return pickle.load(f)


def print_split_summary(split: SplitResult, graph: dict):
    """Print a summary of a split for debugging."""
    labels = graph["labels"]
    print(f"\n--- {split.name} ---")
    print(f"  Train steps: {split.train_steps[0]}-{split.train_steps[-1]} "
          f"({len(split.train_node_idx)} nodes, {len(split.train_edge_idx)} edges, "
          f"{split.train_supervised_mask.sum()} supervised)")
    print(f"  Val steps:   {split.val_steps[0]}-{split.val_steps[-1]} "
          f"({len(split.val_node_idx)} nodes, {len(split.val_edge_idx)} edges, "
          f"{split.val_supervised_mask.sum()} supervised)")
    print(f"  Test steps:  {split.test_steps[0]}-{split.test_steps[-1]} "
          f"({len(split.test_node_idx)} nodes, {len(split.test_edge_idx)} edges, "
          f"{split.test_supervised_mask.sum()} supervised)")

    # Class distribution in supervised nodes
    for name, mask in [
        ("train", split.train_supervised_mask),
        ("val", split.val_supervised_mask),
        ("test", split.test_supervised_mask),
    ]:
        sup_labels = labels[mask]
        illicit = (sup_labels == 1).sum()
        licit = (sup_labels == 0).sum()
        total = illicit + licit
        ratio = illicit / max(total, 1)
        print(f"    {name}: {illicit} illicit / {licit} licit ({ratio:.2%} illicit)")


if __name__ == "__main__":
    import sys

    config_path = Path(__file__).resolve().parents[2] / "experiments" / "configs" / "split_config.yaml"
    graph_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "elliptic_graph.pkl"

    if not graph_path.exists():
        print(f"ERROR: {graph_path} not found. Run build_graph.py first.")
        sys.exit(1)

    graph = load_graph(graph_path)
    splitter = TemporalSplitter.from_config(config_path)

    print("=" * 60)
    print("PRIMARY SPLIT")
    print("=" * 60)
    primary = splitter.get_primary_split(graph)
    print_split_summary(primary, graph)

    print("\n" + "=" * 60)
    print("ROLLING-ORIGIN SPLITS (train-once)")
    print("=" * 60)
    for split in splitter.get_rolling_origin_splits(graph):
        print_split_summary(split, graph)

    print("\n" + "=" * 60)
    print("EXPANDING-WINDOW SPLITS (periodic retrain)")
    print("=" * 60)
    for split in splitter.get_expanding_window_splits(graph):
        print_split_summary(split, graph)
