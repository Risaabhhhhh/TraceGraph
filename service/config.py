"""
Configuration settings for ChainGuard API service & publisher.
"""

import json
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from service/ or repo root
SERVICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_DIR.parent
load_dotenv(SERVICE_DIR / ".env")
load_dotenv(REPO_ROOT / ".env")

# Network & RPC
RPC_URL = os.getenv("RPC_URL", "http://127.0.0.1:8545")
# Default Hardhat Account #0 private key for local dev only:
DEFAULT_DEV_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"
PRIVATE_KEY = os.getenv("PRIVATE_KEY", DEFAULT_DEV_KEY)

# Artifacts & Paths
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts" / "v1.0.0"
MANIFEST_PATH = ARTIFACTS_DIR / "MANIFEST.json"
MODEL_PATH = ARTIFACTS_DIR / "xgboost_model.json"
DATA_PROCESSED_PATH = REPO_ROOT / "ml" / "data" / "processed" / "elliptic_graph.pkl"

# Deployments
DEPLOYMENT_PATH = REPO_ROOT / "deployments" / "localhost.json"


def get_contract_addresses():
    """Retrieve deployed contract addresses from env or deployments JSON."""
    model_registry = os.getenv("MODEL_REGISTRY_ADDRESS")
    risk_registry = os.getenv("RISK_REGISTRY_ADDRESS")

    if not model_registry or not risk_registry:
        if DEPLOYMENT_PATH.exists():
            try:
                with open(DEPLOYMENT_PATH, "r") as f:
                    data = json.load(f)
                    contracts = data.get("contracts", {})
                    model_registry = model_registry or contracts.get("ModelRegistry")
                    risk_registry = risk_registry or contracts.get("RiskRegistry")
            except Exception:
                pass

    return {
        "ModelRegistry": model_registry or "0x5FbDB2315678afecb367f032d93F642f64180aa3",
        "RiskRegistry": risk_registry or "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512",
    }
