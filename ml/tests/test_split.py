"""
Test: temporal split correctness.

Asserts that:
  1. Split windows are ordered (train < val < test).
  2. Split windows are non-overlapping.
  3. All configured steps are covered by the union of splits.
  4. Node counts per split match expectations from the graph.
  5. Edge assignments respect temporal boundaries.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.temporal_split import TemporalSplitter, load_graph

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic_graph.pkl"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "split_config.yaml"


@pytest.fixture(scope="module")
def graph():
    if not GRAPH_PATH.exists():
        pytest.skip(f"Processed graph not found at {GRAPH_PATH}.")
    return load_graph(GRAPH_PATH)


@pytest.fixture(scope="module")
def config():
    if not CONFIG_PATH.exists():
        pytest.skip(f"Split config not found at {CONFIG_PATH}.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def splitter(config):
    return TemporalSplitter(config)


@pytest.fixture(scope="module")
def primary_split(splitter, graph):
    return splitter.get_primary_split(graph)


@pytest.fixture(scope="module")
def rolling_splits(splitter, graph):
    return splitter.get_rolling_origin_splits(graph)


class TestPrimarySplitOrdering:
    """Tests for the primary split temporal ordering."""

    def test_train_before_val(self, primary_split):
        """Train steps must all be before val steps."""
        assert max(primary_split.train_steps) < min(primary_split.val_steps)

    def test_val_before_test(self, primary_split):
        """Val steps must all be before test steps."""
        assert max(primary_split.val_steps) < min(primary_split.test_steps)

    def test_no_overlap(self, primary_split):
        """Train, val, test must not overlap."""
        train = set(primary_split.train_steps)
        val = set(primary_split.val_steps)
        test = set(primary_split.test_steps)

        assert train & val == set(), "Train and val overlap"
        assert train & test == set(), "Train and test overlap"
        assert val & test == set(), "Val and test overlap"

    def test_coverage(self, primary_split, config):
        """Union of splits should cover all configured steps."""
        all_configured = (
            set(config["primary"]["train_steps"])
            | set(config["primary"]["val_steps"])
            | set(config["primary"]["test_steps"])
        )
        all_split = (
            set(primary_split.train_steps)
            | set(primary_split.val_steps)
            | set(primary_split.test_steps)
        )
        assert all_configured == all_split, (
            f"Missing steps: {all_configured - all_split}"
        )


class TestPrimarySplitNodeCounts:
    """Verify node/edge counts are sensible."""

    def test_all_nodes_assigned(self, graph, primary_split):
        """Every graph node in configured steps should appear in exactly one split."""
        time_steps = graph["time_steps"]
        all_steps = (
            set(primary_split.train_steps)
            | set(primary_split.val_steps)
            | set(primary_split.test_steps)
        )

        # Nodes in configured steps
        configured_nodes = set(np.where(np.isin(time_steps, list(all_steps)))[0].tolist())

        # Nodes in splits
        split_nodes = (
            set(primary_split.train_node_idx.tolist())
            | set(primary_split.val_node_idx.tolist())
            | set(primary_split.test_node_idx.tolist())
        )

        assert configured_nodes == split_nodes, (
            f"Mismatch: {len(configured_nodes - split_nodes)} missing, "
            f"{len(split_nodes - configured_nodes)} extra"
        )

    def test_no_node_in_multiple_splits(self, primary_split):
        """No node should appear in more than one split."""
        train = set(primary_split.train_node_idx.tolist())
        val = set(primary_split.val_node_idx.tolist())
        test = set(primary_split.test_node_idx.tolist())

        assert train & val == set(), "Nodes in both train and val"
        assert train & test == set(), "Nodes in both train and test"
        assert val & test == set(), "Nodes in both val and test"

    def test_supervised_mask_subset(self, graph, primary_split):
        """Supervised nodes must be a subset of split nodes with known labels."""
        label_mask = graph["label_mask"]

        for name, node_idx, sup_mask in [
            ("train", primary_split.train_node_idx, primary_split.train_supervised_mask),
            ("val", primary_split.val_node_idx, primary_split.val_supervised_mask),
            ("test", primary_split.test_node_idx, primary_split.test_supervised_mask),
        ]:
            sup_indices = set(np.where(sup_mask)[0].tolist())
            split_indices = set(node_idx.tolist())
            labeled_indices = set(np.where(label_mask)[0].tolist())

            # Supervised must be within split
            assert sup_indices <= split_indices, (
                f"{name}: supervised nodes outside split boundaries"
            )
            # Supervised must have known labels
            assert sup_indices <= labeled_indices, (
                f"{name}: supervised nodes without known labels"
            )


class TestRollingOriginWindows:
    """Tests for rolling-origin split windows."""

    def test_windows_are_ordered(self, rolling_splits):
        """Test windows should be in temporal order."""
        prev_max = 0
        for split in rolling_splits:
            min_test = min(split.test_steps)
            assert min_test > prev_max, (
                f"Window '{split.name}' starts at {min_test} but previous ended at {prev_max}"
            )
            prev_max = max(split.test_steps)

    def test_windows_non_overlapping(self, rolling_splits):
        """Test windows should not overlap."""
        seen_steps = set()
        for split in rolling_splits:
            current = set(split.test_steps)
            overlap = seen_steps & current
            assert overlap == set(), (
                f"Window '{split.name}' overlaps with previous windows: {overlap}"
            )
            seen_steps |= current

    def test_all_windows_share_same_train(self, rolling_splits, config):
        """All rolling-origin windows should use the same initial train set."""
        expected_train = sorted(config["rolling_origin"]["initial_train_steps"])
        for split in rolling_splits:
            assert split.train_steps == expected_train, (
                f"Window '{split.name}' has different train steps"
            )

    def test_rolling_coverage(self, rolling_splits, config):
        """All test windows in config should be covered."""
        expected_windows = {w["name"] for w in config["rolling_origin"]["test_windows"]}
        actual_windows = {s.name for s in rolling_splits}
        assert expected_windows == actual_windows, (
            f"Missing windows: {expected_windows - actual_windows}"
        )


class TestEdgeTemporalConsistency:
    """Test that edges respect temporal boundaries."""

    def test_train_edges_within_train_period(self, graph, primary_split):
        """All train edges should connect nodes within train time steps."""
        time_steps = graph["time_steps"]
        edge_index = graph["edge_index"]
        train_edges = edge_index[:, primary_split.train_edge_idx]
        train_set = set(primary_split.train_steps)

        src_ok = np.isin(time_steps[train_edges[0]], list(train_set))
        dst_ok = np.isin(time_steps[train_edges[1]], list(train_set))

        assert src_ok.all(), "Some train edge sources outside train period"
        assert dst_ok.all(), "Some train edge destinations outside train period"

    def test_edge_counts_monotonic(self, graph, primary_split):
        """Val edges >= train edges (val includes train period)."""
        assert len(primary_split.val_edge_idx) >= len(primary_split.train_edge_idx), (
            "Val should have at least as many edges as train "
            "(val includes train-period edges)"
        )

    def test_test_edges_superset(self, graph, primary_split):
        """Test edges should be superset of val edges."""
        assert len(primary_split.test_edge_idx) >= len(primary_split.val_edge_idx), (
            "Test should have at least as many edges as val"
        )
