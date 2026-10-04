"""
Test: no future-period data leakage in temporal splits.

Asserts that for every split produced by the temporal splitter:
  1. No training node belongs to a time step >= min(val_steps).
  2. No training edge connects to a node in val/test time steps.
  3. No val node belongs to a time step >= min(test_steps).
  4. No val edge connects to a node in test time steps.
  5. Labels used for training supervision come only from train-period nodes.
  6. Features of train nodes are from train time steps only.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.temporal_split import TemporalSplitter, load_graph

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic_graph.pkl"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "split_config.yaml"


@pytest.fixture(scope="module")
def graph():
    """Load the processed graph. Skip all tests if not built yet."""
    if not GRAPH_PATH.exists():
        pytest.skip(f"Processed graph not found at {GRAPH_PATH}. Run build_graph.py first.")
    return load_graph(GRAPH_PATH)


@pytest.fixture(scope="module")
def splitter():
    """Load the temporal splitter config."""
    if not CONFIG_PATH.exists():
        pytest.skip(f"Split config not found at {CONFIG_PATH}.")
    return TemporalSplitter.from_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def primary_split(splitter, graph):
    """Get the primary split."""
    return splitter.get_primary_split(graph)


@pytest.fixture(scope="module")
def rolling_splits(splitter, graph):
    """Get rolling-origin splits."""
    return splitter.get_rolling_origin_splits(graph)


class TestNoLeakagePrimary:
    """Leakage tests for the primary split."""

    def test_train_nodes_before_val(self, graph, primary_split):
        """No train node should be in a val or test time step."""
        time_steps = graph["time_steps"]
        train_ts = time_steps[primary_split.train_node_idx]
        min_val = min(primary_split.val_steps)

        violators = train_ts[train_ts >= min_val]
        assert len(violators) == 0, (
            f"Found {len(violators)} train nodes with time_step >= {min_val} (val start)"
        )

    def test_val_nodes_before_test(self, graph, primary_split):
        """No val node should be in a test time step."""
        time_steps = graph["time_steps"]
        val_ts = time_steps[primary_split.val_node_idx]
        min_test = min(primary_split.test_steps)

        violators = val_ts[val_ts >= min_test]
        assert len(violators) == 0, (
            f"Found {len(violators)} val nodes with time_step >= {min_test} (test start)"
        )

    def test_train_edges_no_future_nodes(self, graph, primary_split):
        """Train edges must not connect to val/test nodes."""
        time_steps = graph["time_steps"]
        edge_index = graph["edge_index"]
        train_edges = edge_index[:, primary_split.train_edge_idx]
        max_train_step = max(primary_split.train_steps)

        src_ts = time_steps[train_edges[0]]
        dst_ts = time_steps[train_edges[1]]

        future_src = src_ts[src_ts > max_train_step]
        future_dst = dst_ts[dst_ts > max_train_step]

        assert len(future_src) == 0, (
            f"{len(future_src)} train edge sources have future time steps"
        )
        assert len(future_dst) == 0, (
            f"{len(future_dst)} train edge destinations have future time steps"
        )

    def test_val_edges_no_test_nodes(self, graph, primary_split):
        """Val edges must not connect to test-only nodes."""
        time_steps = graph["time_steps"]
        edge_index = graph["edge_index"]
        val_edges = edge_index[:, primary_split.val_edge_idx]
        max_val_step = max(primary_split.val_steps)

        src_ts = time_steps[val_edges[0]]
        dst_ts = time_steps[val_edges[1]]

        future_src = src_ts[src_ts > max_val_step]
        future_dst = dst_ts[dst_ts > max_val_step]

        assert len(future_src) == 0, (
            f"{len(future_src)} val edge sources have test time steps"
        )
        assert len(future_dst) == 0, (
            f"{len(future_dst)} val edge destinations have test time steps"
        )

    def test_supervised_labels_only_from_own_period(self, graph, primary_split):
        """Supervised training labels come only from train-period nodes."""
        time_steps = graph["time_steps"]
        train_sup = primary_split.train_supervised_mask
        sup_ts = time_steps[train_sup]
        max_train = max(primary_split.train_steps)

        violators = sup_ts[sup_ts > max_train]
        assert len(violators) == 0, (
            f"{len(violators)} supervised train labels from future periods"
        )

    def test_no_test_labels_in_train_or_val(self, graph, primary_split):
        """Test-period labels should not appear in train or val supervised masks."""
        time_steps = graph["time_steps"]
        test_set = set(primary_split.test_steps)

        for name, mask in [
            ("train", primary_split.train_supervised_mask),
            ("val", primary_split.val_supervised_mask),
        ]:
            sup_ts = time_steps[mask]
            leaked = np.isin(sup_ts, list(test_set))
            assert leaked.sum() == 0, (
                f"{leaked.sum()} test-period labels leaked into {name} supervised mask"
            )


class TestNoLeakageRolling:
    """Leakage tests for rolling-origin splits."""

    def test_all_windows_have_no_leakage(self, graph, rolling_splits):
        """Each rolling-origin window must have train before val before test."""
        time_steps = graph["time_steps"]

        for split in rolling_splits:
            # Train nodes before val
            train_ts = time_steps[split.train_node_idx]
            if len(split.val_node_idx) > 0:
                min_val = min(split.val_steps)
                future_train = train_ts[train_ts >= min_val]
                assert len(future_train) == 0, (
                    f"[{split.name}] {len(future_train)} train nodes in val period"
                )

            # Train+val before test
            min_test = min(split.test_steps)
            if len(split.val_node_idx) > 0:
                val_ts = time_steps[split.val_node_idx]
                future_val = val_ts[val_ts >= min_test]
                assert len(future_val) == 0, (
                    f"[{split.name}] {len(future_val)} val nodes in test period"
                )

    def test_train_edges_isolated_from_future(self, graph, rolling_splits):
        """Train edges in each window connect only to train-period nodes."""
        time_steps = graph["time_steps"]
        edge_index = graph["edge_index"]

        for split in rolling_splits:
            train_edges = edge_index[:, split.train_edge_idx]
            max_train = max(split.train_steps)

            src_ts = time_steps[train_edges[0]]
            dst_ts = time_steps[train_edges[1]]

            assert (src_ts <= max_train).all(), (
                f"[{split.name}] train edge sources leak into future"
            )
            assert (dst_ts <= max_train).all(), (
                f"[{split.name}] train edge destinations leak into future"
            )


class TestFeatureLeakage:
    """Test that features don't encode future information."""

    def test_feature_dimensions_consistent(self, graph):
        """All nodes should have the same feature dimension."""
        features = graph["node_features"]
        assert features.ndim == 2, "Features should be 2D"
        assert features.shape[0] == len(graph["time_steps"]), (
            "Feature rows must match node count"
        )

    def test_no_nan_in_features(self, graph):
        """Features should not contain NaN values."""
        features = graph["node_features"]
        nan_count = np.isnan(features).sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in features"

    def test_train_features_only_from_train_steps(self, graph, primary_split):
        """
        Verify that features used for training come only from train-step nodes.

        This is inherently true by construction (features are indexed by node),
        but we verify the indexing is correct.
        """
        time_steps = graph["time_steps"]
        train_ts = time_steps[primary_split.train_node_idx]
        train_set = set(primary_split.train_steps)

        assert all(t in train_set for t in train_ts), (
            "Some train node features come from non-train time steps"
        )
