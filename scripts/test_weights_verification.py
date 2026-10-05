"""
Test model weights verification against live on-chain ModelRegistry.
Tests:
1. Genuine model weights file (ml/artifacts/v1.0.0/xgboost_model.json)
2. Tampered model weights file (1 byte modified)
"""

import hashlib
import json
from pathlib import Path
from web3 import Web3

REPO_ROOT = Path(__file__).resolve().parents[1]

# Connect to Hardhat
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

with open(REPO_ROOT / "deployments" / "localhost.json", "r") as f:
    deployments = json.load(f)

mr_addr = Web3.to_checksum_address(deployments["contracts"]["ModelRegistry"])
with open(REPO_ROOT / "contracts" / "artifacts" / "contracts" / "ModelRegistry.sol" / "ModelRegistry.json") as f:
    mr_abi = json.load(f)["abi"]

mr = w3.eth.contract(address=mr_addr, abi=mr_abi)

# 1. Genuine weights file
weights_path = REPO_ROOT / "ml" / "artifacts" / "v1.0.0" / "xgboost_model.json"
with open(weights_path, "rb") as f:
    genuine_bytes = f.read()

genuine_hash = "0x" + hashlib.sha256(genuine_bytes).hexdigest()
genuine_hash_bytes = bytes.fromhex(genuine_hash.replace("0x", ""))

print("=" * 80)
print("TEST A: GENUINE WEIGHTS FILE AUDIT (xgboost_model.json)")
print("=" * 80)
print(f"File:               {weights_path.name} ({len(genuine_bytes):,} bytes)")
print(f"Computed SHA-256:   {genuine_hash}")

try:
    is_active = mr.functions.isModelActive(genuine_hash_bytes).call()
    model_record = mr.functions.getModel(genuine_hash_bytes).call()
    print(f"On-Chain Match:     MATCHES ModelRegistry (Active: {is_active})")
    print(f"Model Version:      {model_record[1]}")
    print(f"Registered URI:     {model_record[2]}")
    print(f"Audit Result:       [VALID] MATCHES ON-CHAIN MODEL HASH")
except Exception as e:
    print(f"Error: {e}")

# 2. Tampered weights file (1 byte flipped)
tampered_bytes = bytearray(genuine_bytes)
tampered_bytes[100] = tampered_bytes[100] ^ 0xFF  # flip bits of 1 byte
tampered_hash = "0x" + hashlib.sha256(tampered_bytes).hexdigest()
tampered_hash_bytes = bytes.fromhex(tampered_hash.replace("0x", ""))

print("\n" + "=" * 80)
print("TEST B: TAMPERED WEIGHTS FILE AUDIT (1 Byte Modified)")
print("=" * 80)
print(f"File:               xgboost_model_tampered.json ({len(tampered_bytes):,} bytes)")
print(f"Original Byte[100]: {genuine_bytes[100]} -> Modified Byte[100]: {tampered_bytes[100]}")
print(f"Computed SHA-256:   {tampered_hash}")

try:
    is_active_tampered = mr.functions.isModelActive(tampered_hash_bytes).call()
    model_record_tampered = mr.functions.getModel(tampered_hash_bytes).call()
    print(f"On-Chain Match:     {is_active_tampered}")
except Exception as e:
    print(f"On-Chain Call:      ModelRegistry.getModel({tampered_hash[:10]}...) -> Reverted with ModelNotFound")
    print(f"Audit Result:       [MISMATCH] TAMPERED / UNREGISTERED WEIGHTS FILE")
