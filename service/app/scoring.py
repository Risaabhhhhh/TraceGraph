"""
ChainGuard Inference Engine & Demo EVM Address Mapper.

Loads the exported production XGBoost model artifact and generates
deterministic demo EVM addresses mapped to Elliptic transaction graph nodes.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import xgboost as xgb

from service.config import (
    DATA_PROCESSED_PATH,
    MANIFEST_PATH,
    MODEL_PATH,
)


class ScoringEngine:
    _instance = None

    def __init__(self):
        self.manifest = self._load_manifest()
        self.model = self._load_model()
        self.graph = self._load_graph()
        self._address_to_node_map: Dict[str, int] = {}
        self._node_to_address_map: Dict[int, str] = {}
        self._init_demo_address_mappings()

    @classmethod
    def get_instance(cls) -> "ScoringEngine":
        if cls._instance is None:
            cls._instance = ScoringEngine()
        return cls._instance

    def _load_manifest(self) -> Dict[str, Any]:
        if not MANIFEST_PATH.exists():
            raise FileNotFoundError(f"Model manifest not found at: {MANIFEST_PATH}")
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)

    def _load_model(self) -> xgb.XGBClassifier:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model binary not found at: {MODEL_PATH}")
        clf = xgb.XGBClassifier()
        clf.load_model(str(MODEL_PATH))
        return clf

    def _load_graph(self) -> Dict[str, Any]:
        if DATA_PROCESSED_PATH.exists():
            import pickle
            with open(DATA_PROCESSED_PATH, "rb") as f:
                return pickle.load(f)
        return {"node_features": np.zeros((100, 165), dtype=np.float32), "time_steps": np.zeros(100)}

    def node_id_to_demo_address(self, node_id: int) -> str:
        """
        Deterministically map an Elliptic transaction index to a demo EVM address.
        Clearly labeled as a demo address in metadata.
        """
        if node_id in self._node_to_address_map:
            return self._node_to_address_map[node_id]

        # Generate deterministic 20-byte address from node_id
        seed_str = f"chainguard_demo_node_{node_id}"
        h = hashlib.sha256(seed_str.encode("utf-8")).hexdigest()
        addr = f"0x{h[-40:]}"
        # Checksum-style lowercase for consistency
        self._node_to_address_map[node_id] = addr
        self._address_to_node_map[addr.lower()] = node_id
        return addr

    def demo_address_to_node_id(self, address: str) -> Optional[int]:
        """Look up corresponding node index for a demo address if mapped."""
        return self._address_to_node_map.get(address.lower().strip())

    def _init_demo_address_mappings(self):
        """Pre-map test window nodes (steps 40..49) to demo addresses."""
        if "time_steps" in self.graph:
            test_mask = self.graph["time_steps"] >= 40
            test_indices = np.where(test_mask)[0]
            for idx in test_indices[:1000]:  # pre-map 1,000 demo test nodes
                self.node_id_to_demo_address(int(idx))

    def compute_risk_level(self, score_basis_points: int) -> str:
        """Categorize risk score into operational risk tiers."""
        if score_basis_points < 2000:
            return "LOW"
        elif score_basis_points < 5000:
            return "MEDIUM"
        elif score_basis_points < 8000:
            return "HIGH"
        else:
            return "CRITICAL"

    def predict_features(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run inference on feature matrix.
        Returns: (probabilities, integer_scores_in_basis_points)
        """
        X = np.asarray(features, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Ensure 165 features
        if X.shape[1] < 165:
            padded = np.zeros((X.shape[0], 165), dtype=np.float32)
            padded[:, :X.shape[1]] = X
            X = padded

        probs = self.model.predict_proba(X)[:, 1]
        scores = np.clip(np.round(probs * 10000), 0, 10000).astype(int)
        return probs, scores

    def score_single(
        self,
        address: Optional[str] = None,
        node_id: Optional[int] = None,
        features: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Score an individual query entity."""
        actual_node_id = None
        demo_address = address

        if features is not None:
            feat_arr = np.array(features, dtype=np.float32)
            demo_address = demo_address or "0x0000000000000000000000000000000000000000"
        elif node_id is not None:
            actual_node_id = node_id
            feat_arr = self.graph["node_features"][node_id]
            demo_address = self.node_id_to_demo_address(node_id)
        elif address is not None:
            mapped = self.demo_address_to_node_id(address)
            if mapped is not None:
                actual_node_id = mapped
                feat_arr = self.graph["node_features"][mapped]
            else:
                # Deterministic synthetic features from address hash for demonstration
                h = int(hashlib.md5(address.encode("utf-8")).hexdigest()[:8], 16)
                np.random.seed(h % 100000)
                feat_arr = np.random.randn(165).astype(np.float32)
        else:
            # Default sample node 0
            actual_node_id = 0
            feat_arr = self.graph["node_features"][0]
            demo_address = self.node_id_to_demo_address(0)

        probs, scores = self.predict_features(feat_arr)
        prob = float(probs[0])
        score = int(scores[0])
        risk_level = self.compute_risk_level(score)

        return {
            "address": demo_address,
            "is_demo_address": True,
            "node_id": actual_node_id,
            "probability": prob,
            "score": score,
            "risk_level": risk_level,
            "model_hash": self.manifest["model_hash"],
            "model_version": self.manifest["version"],
        }

    def sample_demo_batch(self, count: int = 20) -> List[Dict[str, Any]]:
        """Sample test-window nodes for batch publishing."""
        if "time_steps" in self.graph:
            test_mask = self.graph["time_steps"] >= 40
            test_indices = np.where(test_mask)[0]
            # Pick a representative sample including both licit and illicit if labeled
            sample_indices = test_indices[:min(count, len(test_indices))]
        else:
            sample_indices = list(range(count))

        results = []
        for idx in sample_indices:
            node_id = int(idx)
            res = self.score_single(node_id=node_id)
            results.append(res)
        return results
