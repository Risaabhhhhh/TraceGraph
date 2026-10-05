"""
ChainGuard Risk Score On-Chain Publisher.

Reads scored demo addresses from the inference engine, computes the
cryptographic snapshot hash, interacts with ModelRegistry.sol to verify
model activation, and executes createBatch + publishScores on RiskRegistry.sol.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Ensure repository root is on sys.path
SERVICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVICE_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from web3 import Web3
from eth_account import Account

from ml.src.registry.snapshot_hash import compute_snapshot_hash
from service.config import (
    DEPLOYMENT_PATH,
    MANIFEST_PATH,
    PRIVATE_KEY,
    RPC_URL,
    get_contract_addresses,
)
from service.app.scoring import ScoringEngine


def get_contract_abis():
    """Load compiled contract ABIs from contracts/artifacts."""
    contracts_dir = REPO_ROOT / "contracts" / "artifacts" / "contracts"
    mr_abi_path = contracts_dir / "ModelRegistry.sol" / "ModelRegistry.json"
    rr_abi_path = contracts_dir / "RiskRegistry.sol" / "RiskRegistry.json"

    mr_abi, rr_abi = [], []
    if mr_abi_path.exists():
        with open(mr_abi_path, "r") as f:
            mr_abi = json.load(f).get("abi", [])
    if rr_abi_path.exists():
        with open(rr_abi_path, "r") as f:
            rr_abi = json.load(f).get("abi", [])

    return mr_abi, rr_abi


class ChainGuardPublisher:
    def __init__(
        self,
        rpc_url: Optional[str] = None,
        private_key: Optional[str] = None,
    ):
        self.rpc_url = rpc_url or RPC_URL
        self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
        self.private_key = private_key or PRIVATE_KEY
        
        if self.private_key and not self.private_key.startswith("0x"):
            self.private_key = f"0x{self.private_key}"

        if self.private_key:
            self.account = Account.from_key(self.private_key)
            self.publisher_address = self.account.address
        else:
            self.account = None
            self.publisher_address = "0x0000000000000000000000000000000000000000"

        self.addresses = get_contract_addresses()
        self.mr_abi, self.rr_abi = get_contract_abis()
        self.engine = ScoringEngine.get_instance()

    def get_model_registry_contract(self):
        if not self.mr_abi:
            raise RuntimeError("ModelRegistry ABI not found. Compile contracts first.")
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(self.addresses["ModelRegistry"]),
            abi=self.mr_abi,
        )

    def get_risk_registry_contract(self):
        if not self.rr_abi:
            raise RuntimeError("RiskRegistry ABI not found. Compile contracts first.")
        return self.w3.eth.contract(
            address=Web3.to_checksum_address(self.addresses["RiskRegistry"]),
            abi=self.rr_abi,
        )

    def ensure_model_registered(self, model_hash: str, version: str, uri: str):
        """Ensure the production model is registered and active in ModelRegistry."""
        if not self.w3.is_connected() or not self.account:
            return

        mr = self.get_model_registry_contract()
        model_hash_bytes = bytes.fromhex(model_hash.replace("0x", ""))

        try:
            is_active = mr.functions.isModelActive(model_hash_bytes).call()
            if is_active:
                return
        except Exception:
            pass

        print(f"Registering model {version} on-chain ({model_hash})...")
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        tx = mr.functions.registerModel(model_hash_bytes, version, uri).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 300000,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed = self.w3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"Model registered successfully in tx: {tx_hash.hex()}")

    def publish_demo_batch(
        self,
        sample_size: int = 20,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Score a demo batch of addresses and publish to RiskRegistry.
        """
        # 1. Sample and score demo addresses
        demo_scores = self.engine.sample_demo_batch(count=sample_size)
        manifest = self.engine.manifest
        model_hash = manifest["model_hash"]
        version = manifest["version"]
        uri = f"ipfs://chainguard/models/{version}/MANIFEST.json"

        # 2. Compute canonical snapshot hash
        snapshot_records = [
            {"address": r["address"], "score": r["score"]}
            for r in demo_scores
        ]
        snapshot_hash = compute_snapshot_hash(snapshot_records)

        result_payload = {
            "success": True,
            "dry_run": dry_run,
            "model_hash": model_hash,
            "version": version,
            "snapshot_hash": snapshot_hash,
            "count": len(demo_scores),
            "scores_sample": demo_scores[:5],
        }

        if dry_run:
            print(f"[DRY-RUN] Scored {len(demo_scores)} demo addresses.")
            print(f"  Model Hash:    {model_hash}")
            print(f"  Snapshot Hash: {snapshot_hash}")
            return result_payload

        # 3. Check connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Cannot connect to EVM RPC at: {self.rpc_url}")

        if not self.account:
            raise ValueError("No private key configured for publisher.")

        # Ensure model is registered
        self.ensure_model_registered(model_hash, version, uri)

        rr = self.get_risk_registry_contract()
        model_hash_bytes = bytes.fromhex(model_hash.replace("0x", ""))
        snapshot_hash_bytes = bytes.fromhex(snapshot_hash.replace("0x", ""))

        # 4. Create Batch
        print(f"Creating batch for model {model_hash[:10]}... and snapshot {snapshot_hash[:10]}...")
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        create_tx = rr.functions.createBatch(model_hash_bytes, snapshot_hash_bytes).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 200000,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed_create = self.w3.eth.account.sign_transaction(create_tx, self.private_key)
        create_tx_hash = self.w3.eth.send_raw_transaction(signed_create.raw_transaction)
        receipt_create = self.w3.eth.wait_for_transaction_receipt(create_tx_hash)

        # Parse batchId from event or nextBatchId - 1
        batch_id = rr.functions.nextBatchId().call() - 1

        # 5. Publish Scores
        accounts = [Web3.to_checksum_address(r["address"]) for r in demo_scores]
        scores = [int(r["score"]) for r in demo_scores]

        print(f"Publishing {len(accounts)} scores for batch {batch_id}...")
        nonce = self.w3.eth.get_transaction_count(self.account.address)
        pub_tx = rr.functions.publishScores(batch_id, accounts, scores).build_transaction({
            "from": self.account.address,
            "nonce": nonce,
            "gas": 3000000,
            "gasPrice": self.w3.eth.gas_price,
        })
        signed_pub = self.w3.eth.account.sign_transaction(pub_tx, self.private_key)
        pub_tx_hash = self.w3.eth.send_raw_transaction(signed_pub.raw_transaction)
        receipt_pub = self.w3.eth.wait_for_transaction_receipt(pub_tx_hash)

        print(f"Batch {batch_id} published successfully!")
        print(f"  Create Tx:  {create_tx_hash.hex()}")
        print(f"  Publish Tx: {pub_tx_hash.hex()}")
        print(f"  Gas Used:   {receipt_pub.gasUsed}")

        result_payload["batch_id"] = batch_id
        result_payload["create_tx_hash"] = create_tx_hash.hex()
        result_payload["tx_hash"] = pub_tx_hash.hex()
        result_payload["gas_used"] = receipt_pub.gasUsed
        return result_payload


def main():
    parser = argparse.ArgumentParser(description="ChainGuard Risk Score Publisher")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of demo transactions to publish")
    parser.add_argument("--dry-run", action="store_true", help="Score and compute hashes without sending tx")
    parser.add_argument("--rpc-url", type=str, default=None, help="EVM RPC URL")
    parser.add_argument("--private-key", type=str, default=None, help="Publisher private key")
    args = parser.parse_args()

    publisher = ChainGuardPublisher(rpc_url=args.rpc_url, private_key=args.private_key)
    res = publisher.publish_demo_batch(sample_size=args.sample_size, dry_run=args.dry_run)
    print("\nResult:")
    print(json.dumps({k: v for k, v in res.items() if k != "scores_sample"}, indent=2))


if __name__ == "__main__":
    main()
