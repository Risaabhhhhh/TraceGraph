"""
ChainGuard Data & Snapshot Hashing Utilities.

Computes deterministic SHA-256 hashes of input transaction batches,
features, and address score snapshots for tamper-evident on-chain recording.
"""

import hashlib
import json
from typing import Any, Dict, List, Tuple, Union
import numpy as np


def compute_sha256_bytes(data: bytes) -> str:
    """Compute 0x-prefixed 32-byte hex SHA-256 hash."""
    h = hashlib.sha256(data).hexdigest()
    return f"0x{h}"


def compute_sha256_file(file_path: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return f"0x{hasher.hexdigest()}"


def compute_snapshot_hash(records: List[Dict[str, Any]]) -> str:
    """
    Compute deterministic SHA-256 hash of address-score snapshot records.
    
    Each record must have 'address' (or 'account') and 'score' (int 0..10000).
    Sorted canonically by address before hashing.
    """
    canonical_items = []
    for r in records:
        addr = str(r.get("address", r.get("account", ""))).lower().strip()
        score = int(r.get("score", 0))
        canonical_items.append({"address": addr, "score": score})
    
    # Sort deterministically by lowercase address
    canonical_items.sort(key=lambda x: x["address"])
    canonical_json = json.dumps(canonical_items, separators=(",", ":"), sort_keys=True)
    return compute_sha256_bytes(canonical_json.encode("utf-8"))


def compute_array_hash(arr: np.ndarray) -> str:
    """Compute SHA-256 hash of a numpy array in C-contiguous byte order."""
    contiguous = np.ascontiguousarray(arr)
    return compute_sha256_bytes(contiguous.tobytes())
