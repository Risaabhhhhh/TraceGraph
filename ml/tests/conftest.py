import sys
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.temporal_split import TemporalSplitter, load_graph

GRAPH_PATH = PROJECT_ROOT / "data" / "processed" / "elliptic_graph.pkl"
CONFIG_PATH = PROJECT_ROOT / "experiments" / "configs" / "split_config.yaml"


@pytest.fixture(scope="session")
def graph():
    """Load the processed graph once per test session."""
    if not GRAPH_PATH.exists():
        pytest.skip(f"Processed graph not found at {GRAPH_PATH}.")
    return load_graph(GRAPH_PATH)


@pytest.fixture(scope="session")
def config():
    """Load the split config."""
    if not CONFIG_PATH.exists():
        pytest.skip(f"Split config not found at {CONFIG_PATH}.")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def splitter(config):
    """Create temporal splitter from config."""
    return TemporalSplitter(config)


@pytest.fixture(scope="session")
def primary_split(splitter, graph):
    """Primary temporal train/val/test split."""
    return splitter.get_primary_split(graph)


@pytest.fixture(scope="session")
def rolling_splits(splitter, graph):
    """Rolling-origin temporal splits."""
    return splitter.get_rolling_origin_splits(graph)
