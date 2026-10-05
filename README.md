# ChainGuard: Cryptographic ML Provenance & On-Chain Risk Registry

ChainGuard is an end-to-end framework for illicit cryptocurrency transaction detection that couples high-performance machine learning (XGBoost & Graph Neural Networks on the Elliptic Bitcoin dataset) with tamper-evident smart contracts on the Ethereum Virtual Machine (EVM).

> **Core Philosophy**: *"Verified origin, not verified accuracy."*  
> On-chain registries prove the complete cryptographic provenance and audit trail of inferences (which model binary and data snapshot generated a score), while acknowledging that ML inferences remain probabilistic.

---

## 1. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Machine Learning (ML)                    │
│  - Elliptic Bitcoin Graph (203k txs, 49 time-steps)         │
│  - Strict Zero-Temporal-Leakage Splits                      │
│  - Production Model: XGBoost (165 Feat, Seed 42)            │
│  - Artifacts: xgboost_model.json, MANIFEST.json             │
└──────────────────────────────┬──────────────────────────────┘
                               │ SHA-256 Hashes
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   Smart Contracts (EVM)                     │
│  - ModelRegistry.sol: Versioned model hash registry         │
│  - RiskRegistry.sol: Batch score recording (basis points)   │
│  - createBatch(modelHash, snapshotHash)                     │
│  - publishScores(batchId, accounts[], scores[])             │
└──────────────────────────────┬──────────────────────────────┘
                               │ Web3 RPC
                               ▼
┌──────────────────────────────┴──────────────────────────────┐
│                  Service & DApp Layer                       │
│  - FastAPI (/health, /model-info, /score, /publish)         │
│  - publisher.py: Automated on-chain batch score publisher   │
│  - Minimal DApp: Address lookup, client-side verifier       │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Layout

```
├── contracts/               # EVM Smart Contracts (Hardhat + TypeScript + OpenZeppelin)
│   ├── contracts/           # ModelRegistry.sol, RiskRegistry.sol, IModelRegistry.sol
│   ├── scripts/             # deploy.ts (Localhost & Sepolia)
│   ├── test/                # Hardhat test suite & gas benchmarks
│   └── deployments/         # localhost.json deployment addresses
├── ml/                      # Machine Learning Pipeline
│   ├── artifacts/v1.0.0/    # Production model binary, config, metrics, MANIFEST.json
│   ├── data/                # Elliptic raw & processed transaction graph
│   ├── experiments/         # Configs, benchmark results, generated figures
│   ├── src/                 # Ingest, temporal splits, models (XGBoost, GNN), registry tools
│   └── tests/               # 39 pytest unit tests (metrics, splits, leakage checks)
├── service/                 # Inference Service & Publisher
│   ├── app/                 # FastAPI application (main, routes, schemas, scoring)
│   ├── publisher.py         # Web3 on-chain publisher script
│   └── tests/               # Service integration tests
├── dapp/                    # Minimal Explorer & Verifier Frontend (HTML/CSS/JS)
│   ├── index.html           # Address risk lookup & manifest verifier
│   ├── style.css            # Dark mode glassmorphic styling
│   └── app.js               # Client-side WebCrypto SHA-256 verifier
└── docs/                    # Technical Reports
    ├── bc_report.md         # Blockchain integration, gas analysis & provenance
    ├── ml_report.md         # Full ML benchmark results & ablation studies
    └── figures/             # Evaluation charts & ladder plots
```

---

## 3. Quickstart & End-to-End Demo

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### Step 1: Install Dependencies
```bash
# Python dependencies (ML & Service)
pip install -r ml/requirements.txt
pip install web3 fastapi uvicorn pydantic python-dotenv requests

# Smart Contracts dependencies
cd contracts
npm install
cd ..
```

### Step 2: Run Tests
```bash
# Run ML test suite (39 tests: zero-leakage, temporal splits, metrics)
python -m pytest ml/tests -v

# Run Smart Contract test suite (13 tests: access control, batch creation, gas reports)
cd contracts
npx hardhat test
cd ..

# Run Service test suite
python -m pytest service/tests -v
```

### Step 3: Launch Local Blockchain Node & Deploy Contracts
In Terminal 1:
```bash
cd contracts
npx hardhat node
```

In Terminal 2 (Deploy to local node):
```bash
cd contracts
npx hardhat run scripts/deploy.ts --network localhost
cd ..
```

### Step 4: Start FastAPI Inference Service
In Terminal 3:
```bash
uvicorn service.app.main:app --host 127.0.0.1 --port 8000
```

### Step 5: Publish a Demo Score Batch On-Chain
```bash
# Dry-run mode (scores & hashes without sending on-chain transaction):
python service/publisher.py --sample-size 20 --dry-run

# Live on-chain publish to local Hardhat node:
python service/publisher.py --sample-size 20
```

### Step 6: Launch DApp
```bash
cd dapp
npx serve -l 3000 .
```
Open [http://localhost:3000](http://localhost:3000) in your browser:
- **Address Risk Lookup**: Search any EVM address or click quick demo chips to inspect on-chain scores and cryptographic provenance.
- **Manifest Verifier**: Click "Load v1.0.0 Production Manifest" to compute SHA-256 hashes in-browser and audit against `ModelRegistry.sol`.
- **Demo Addresses**: View sample test window transactions mapped to demo EVM addresses.

---

## 4. Key Results Summary

| Model Tier | Model | Feature Set | PR-AUC (Illicit) | F1 Score | Recall @ 80% Prec |
|---|---|---|:---:|:---:|:---:|
| **Tier 0** | Heuristic Rule | All (165) | 0.1112 | 0.2326 | 0.0000 |
| **Tier 1** | Logistic Regression | All (165) | 0.2185 | 0.2524 | 0.0000 |
| **Tier 1** | **XGBoost (Reference Model v1.0.0)** | All (165) | **0.6769** | **0.6472** | **0.6000** |
| **Tier 2** | GraphSAGE | Local (94) | 0.5900 | 0.6241 | 0.5167 |
| **Tier 2** | GraphSAGE | All (165) | 0.4845 | 0.4522 | 0.2918 |
| **Tier 2** | GAT | All (165) | 0.3861 | 0.3237 | 0.0852 |
| **Tier 3** | Hybrid (GraphSAGE $\to$ XGBoost) | Local + GNN Emb. | 0.6277 | 0.6320 | 0.5434 |

### Gas Optimization Profile (`RiskRegistry.sol`)
- **Batch Size 1**: 115,301 gas (115,301 gas/addr)
- **Batch Size 10**: 548,194 gas (54,819 gas/addr — **52.5% reduction**)
- **Batch Size 50**: 2,548,128 gas (50,963 gas/addr — **55.8% reduction**)
