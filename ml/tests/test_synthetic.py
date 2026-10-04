"""
Synthetic unit tests for graph integrity and temporal splitting.

Runs completely independently of raw dataset files so CI and tests
can execute deterministically in milliseconds.
"""

import numpy as np
import pytest

from src.eval.temporal_split import TemporalSplitter, SplitResult


def make_synthetic_graph(num_steps: int = 10, nodes_per_step: int = 20, feat_dim: int = 165):
    """Create a synthetic graph dictionary matching the Elliptic schema."""
    total_nodes = num_steps * nodes_per_step
    node_ids = np.arange(1000, 1000 + total_nodes)
    time_steps = np.repeat(np.arange(1, num_steps + 1), nodes_per_step)

    # Random features
    rng = np.random.RandomState(42)
    node_features = rng.randn(total_nodes, feat_dim).astype(np.float32)

    # Labels: 10% illicit (1), 40% licit (0), 50% unknown (-1)
    raw_choices = rng.choice([1, 0, -1], size=total_nodes, p=[0.1, 0.4, 0.5])
    labels = raw_choices.astype(np.int8)
    label_mask = labels != -1

    # Edges: mostly within the same step or from earlier to later step
    edges = []
    for i in range(total_nodes):
        # 2 edges per node
        targets = rng.choice(total_nodes, size=2, replace=False)
        for t in targets:
            # only allow intra-step or forward-in-time edges
            if time_steps[i] <= time_steps[t]:
                edges.append((i, t))

    edge_index = np.array(edges, dtype=np.int64).T

    return {
        "node_features": node_features,
        "node_ids": node_ids.tolist(),
        "time_steps": time_steps,
        "labels": labels,
        "edge_index": edge_index,
        "label_mask": label_mask,
    }


@pytest.fixture
def synthetic_graph():
    return make_synthetic_graph(num_steps=10, nodes_per_step=20)


@pytest.fixture
def synthetic_config():
    return {
        "split_name": "synthetic_test_split",
        "total_steps": 10,
        "primary": {
            "train_steps": [1, 2, 3, 4, 5, 6],
            "val_steps": [7, 8],
            "test_steps": [9, 10],
        },
        "rolling_origin": {
            "initial_train_steps": [1, 2, 3, 4],
            "initial_val_steps": [5, 6],
            "test_windows": [
                {"name": "window_1", "steps": [7, 8]},
                {"name": "window_2", "steps": [9, 10]},
            ],
        },
    }


def test_synthetic_graph_shapes(synthetic_graph):
    """Test synthetic graph shapes and label encoding."""
    g = synthetic_graph
    assert g["node_features"].shape == (200, 165)
    assert len(g["node_ids"]) == 200
    assert len(g["time_steps"]) == 200
    assert len(g["labels"]) == 200
    assert g["edge_index"].shape[0] == 2
    assert set(np.unique(g["labels"])).issubset({-1, 0, 1})
    assert np.all(g["label_mask"] == (g["labels"] != -1))


def test_synthetic_primary_split_ordering(synthetic_graph, synthetic_config):
    """Test strict temporal ordering in primary split."""
    splitter = TemporalSplitter(synthetic_config)
    split = splitter.get_primary_split(synthetic_graph)

    assert max(split.train_steps) < min(split.val_steps)
    assert max(split.val_steps) < min(split.test_steps)

    time_steps = synthetic_graph["time_steps"]
    train_ts = time_steps[split.train_node_idx]
    val_ts = time_steps[split.val_node_idx]
    test_ts = time_steps[split.test_node_idx]

    assert max(train_ts) <= 6
    assert min(val_ts) >= 7 and max(val_ts) <= 8
    assert min(test_ts) >= 9


def test_synthetic_zero_temporal_leakage(synthetic_graph, synthetic_config):
    """Test zero temporal leakage on edges and node masks."""
    splitter = TemporalSplitter(synthetic_config)
    split = splitter.get_primary_split(synthetic_graph)

    time_steps = synthetic_graph["time_steps"]
    edge_index = synthetic_graph["edge_index"]
    train_edges = edge_index[:, split.train_edge_idx]

    # All train edges must connect only nodes from train period
    assert np.all(time_steps[train_edges[0]] <= 6)
    assert np.all(time_steps[train_edges[1]] <= 6)

    # Supervised mask checks
    labels = synthetic_graph["labels"]
    train_sup = split.train_supervised_mask
    assert np.all(time_steps[train_sup] <= 6)
    assert np.all(labels[train_sup] != -1)


def test_synthetic_rolling_origin(synthetic_graph, synthetic_config):
    """Test rolling origin windows sequence."""
    splitter = TemporalSplitter(synthetic_config)
    splits = splitter.get_rolling_origin_splits(synthetic_graph)

    assert len(splits) == 2
    assert splits[0].name == "window_1"
    assert splits[1].name == "window_2"

    assert splits[0].test_steps == [7, 8]
    assert splits[1].test_steps == [9, 10]

    # Both share initial training
    assert splits[0].train_steps == [1, 2, 3, 4]
    assert splits[1].train_steps == [1, 2, 3, 4]


def test_deterministic_splits(synthetic_graph, synthetic_config):
    """Test that creating splitter and getting splits is strictly deterministic."""
    splitter1 = TemporalSplitter(synthetic_config)
    split1 = splitter1.get_primary_split(synthetic_graph)

    splitter2 = TemporalSplitter(synthetic_config)
    split2 = splitter2.get_primary_split(synthetic_graph)

    np.testing.assert_array_equal(split1.train_node_idx, split2.train_node_idx)
    np.testing.assert_array_equal(split1.val_node_idx, split2.val_node_idx)
    np.testing.assert_array_equal(split1.test_node_idx, split2.test_node_idx)
    np.testing.assert_array_equal(split1.train_edge_idx, split2.train_edge_idx)
    np.testing.assert_array_equal(split1.train_supervised_mask, split2.train_supervised_mask)
