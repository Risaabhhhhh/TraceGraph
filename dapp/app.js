/**
 * ChainGuard DApp — On-Chain Direct Reader & Client-Side Manifest Verifier
 * 
 * Direct JSON-RPC interaction with ModelRegistry.sol and RiskRegistry.sol on Localhost Hardhat.
 */

const RPC_URL = "http://127.0.0.1:8545";
const API_BASE = "http://127.0.0.1:8000";

// Verified ABIs matching compiled contract artifacts in contracts/artifacts/contracts/
const RISK_REGISTRY_ABI = [
  "function getScore(address account) external view returns (uint16 score, uint256 batchId, uint256 timestamp, bytes32 modelHash, bytes32 snapshotHash)",
  "function getBatch(uint256 batchId) external view returns (bytes32 modelHash, bytes32 snapshotHash, uint256 timestamp, uint256 count)",
  "function nextBatchId() external view returns (uint256)",
  "function modelRegistry() external view returns (address)"
];

const MODEL_REGISTRY_ABI = [
  "function isModelActive(bytes32 modelHash) external view returns (bool)",
  "function getModel(bytes32 modelHash) external view returns (tuple(bytes32 modelHash, string version, string uri, bool active, uint256 registeredAt))",
  "function getModelCount() external view returns (uint256)",
  "function getAllModelHashes() external view returns (bytes32[])"
];

// Deployed contract addresses
let CONTRACT_ADDRESSES = {
  ModelRegistry: "0x5FbDB2315678afecb367f032d93F642f64180aa3",
  RiskRegistry: "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
};

// Load deployed addresses from deployments/localhost.json if available
async function loadDeploymentAddresses() {
  try {
    const res = await fetch("/deployments/localhost.json");
    if (res.ok) {
      const data = await res.json();
      if (data.contracts) {
        CONTRACT_ADDRESSES.ModelRegistry = data.contracts.ModelRegistry || CONTRACT_ADDRESSES.ModelRegistry;
        CONTRACT_ADDRESSES.RiskRegistry = data.contracts.RiskRegistry || CONTRACT_ADDRESSES.RiskRegistry;
        console.log("Loaded on-chain deployment addresses:", CONTRACT_ADDRESSES);
      }
    }
  } catch (e) {
    console.log("Using default contract addresses:", CONTRACT_ADDRESSES);
  }
}

function getProvider() {
  if (typeof ethers !== "undefined") {
    return new ethers.JsonRpcProvider(RPC_URL);
  }
  return null;
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-content").forEach(el => el.style.display = "none");
  document.querySelectorAll(".tab-btn").forEach(el => el.classList.remove("active"));

  const targetTab = document.getElementById(`tab-${tabId}`);
  if (targetTab) targetTab.style.display = "flex";

  const targetBtn = document.getElementById(`tab-${tabId}-btn`);
  if (targetBtn) targetBtn.classList.add("active");

  if (tabId === "demo") {
    loadDemoBatchTable();
  }
}

async function sha256Client(text) {
  const enc = new TextEncoder();
  const data = enc.encode(text);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  const hashHex = hashArray.map(b => b.toString(16).padStart(2, "0")).join("");
  return "0x" + hashHex;
}

/**
 * DIRECT ON-CHAIN LOOKUP:
 * Queries RiskRegistry and ModelRegistry on localhost Hardhat node.
 */
async function lookupAddressScore() {
  const addrInput = document.getElementById("lookup-address");
  const address = addrInput.value.trim();

  const statusCard = document.getElementById("lookup-status-card");
  const statusContent = document.getElementById("lookup-status-content");
  const statusIcon = document.getElementById("lookup-status-icon");
  const statusMsg = document.getElementById("lookup-status-msg");
  const resultsCard = document.getElementById("lookup-results");

  // Reset display
  statusCard.style.display = "none";
  resultsCard.style.display = "none";

  if (!address || !ethers.isAddress(address)) {
    showStatusMessage("Invalid Address", "Please enter a valid checksummed EVM address (e.g. 0x...).", "warning");
    return;
  }

  const btnSearch = document.getElementById("btn-search");
  btnSearch.innerText = "Querying Chain...";
  btnSearch.disabled = true;

  try {
    const provider = getProvider();
    if (!provider) {
      throw new Error("ethers.js library not loaded in browser.");
    }

    // Direct JSON-RPC call to RiskRegistry.getScore(address)
    const riskContract = new ethers.Contract(CONTRACT_ADDRESSES.RiskRegistry, RISK_REGISTRY_ABI, provider);
    let onChainScoreResult;

    try {
      onChainScoreResult = await riskContract.getScore(address);
    } catch (rpcErr) {
      throw new Error(`RPC Error querying RiskRegistry at ${CONTRACT_ADDRESSES.RiskRegistry}: ${rpcErr.message}. Ensure local Hardhat node is running at ${RPC_URL}.`);
    }

    const [score, batchId, timestamp, modelHash, snapshotHash] = onChainScoreResult;

    // Check if address has an on-chain record (timestamp > 0)
    if (Number(timestamp) === 0) {
      showStatusMessage(
        "No on-chain record for this address",
        `Address ${address} has not been published in any on-chain risk score batch.`,
        "info"
      );
      return;
    }

    // Query ModelRegistry.getModel(modelHash) on-chain
    let modelVersionStr;
    try {
      const modelContract = new ethers.Contract(CONTRACT_ADDRESSES.ModelRegistry, MODEL_REGISTRY_ABI, provider);
      const modelRecord = await modelContract.getModel(modelHash);
      modelVersionStr = `${modelRecord.version} (Active: ${modelRecord.active})`;
    } catch (modelErr) {
      modelVersionStr = `Model record not found on-chain (Error: ${modelErr.shortMessage || modelErr.message || "ModelNotFound"})`;
    }

    renderScoreResult({
      address: address,
      score: Number(score),
      batchId: Number(batchId),
      timestamp: Number(timestamp),
      modelHash: modelHash,
      snapshotHash: snapshotHash,
      modelVersion: modelVersionStr
    });

  } catch (err) {
    console.error("On-Chain Lookup Error:", err);
    showStatusMessage(
      "RPC / Network Error",
      err.message || "Failed to communicate with Ethereum JSON-RPC node at http://127.0.0.1:8545.",
      "error"
    );
  } finally {
    btnSearch.innerText = "Query Risk Score";
    btnSearch.disabled = false;
  }
}

function showStatusMessage(title, detail, type = "info") {
  const statusCard = document.getElementById("lookup-status-card");
  const statusContent = document.getElementById("lookup-status-content");
  const statusIcon = document.getElementById("lookup-status-icon");
  const statusMsg = document.getElementById("lookup-status-msg");
  const resultsCard = document.getElementById("lookup-results");

  resultsCard.style.display = "none";
  statusCard.style.display = "block";

  if (type === "error") {
    statusIcon.innerText = "⚠️";
    statusContent.style.background = "rgba(239, 68, 68, 0.15)";
    statusContent.style.borderColor = "rgba(239, 68, 68, 0.4)";
    statusContent.style.color = "#fca5a5";
  } else if (type === "warning") {
    statusIcon.innerText = "⚠️";
    statusContent.style.background = "rgba(245, 158, 11, 0.15)";
    statusContent.style.borderColor = "rgba(245, 158, 11, 0.4)";
    statusContent.style.color = "#fde68a";
  } else {
    statusIcon.innerText = "ℹ️";
    statusContent.style.background = "rgba(59, 130, 246, 0.15)";
    statusContent.style.borderColor = "rgba(59, 130, 246, 0.4)";
    statusContent.style.color = "#93c5fd";
  }

  statusMsg.innerHTML = `<strong>${title}:</strong> ${detail}`;
}

function renderScoreResult(result) {
  const resultsCard = document.getElementById("lookup-results");
  const statusCard = document.getElementById("lookup-status-card");
  statusCard.style.display = "none";
  resultsCard.style.display = "flex";

  const pct = (result.score / 100).toFixed(2);
  document.getElementById("score-percentage").innerText = `${pct}%`;
  document.getElementById("score-basis-points").innerText = `${result.score} / 10,000 basis points`;

  let riskTier = "LOW";
  if (result.score >= 8000) riskTier = "CRITICAL";
  else if (result.score >= 5000) riskTier = "HIGH";
  else if (result.score >= 2000) riskTier = "MEDIUM";

  const badge = document.getElementById("risk-badge");
  badge.innerText = `${riskTier} RISK`;
  badge.className = "risk-badge " + riskTier.toLowerCase();

  document.getElementById("res-address").innerText = result.address;
  document.getElementById("res-batch-id").innerText = `Batch #${result.batchId}`;
  document.getElementById("res-model-ver").innerText = result.modelVersion;
  document.getElementById("res-model-hash").innerText = result.modelHash;
  document.getElementById("res-snapshot-hash").innerText = result.snapshotHash;
}

function setAndQueryAddress(addr) {
  document.getElementById("lookup-address").value = addr;
  lookupAddressScore();
}

/**
 * CLIENT-SIDE MANIFEST VERIFIER & ON-CHAIN MODEL AUDITOR
 */
/**
 * AUDIT MODEL WEIGHTS BINARY CLIENT-SIDE VIA WEBCRYPTO
 */
async function handleWeightsFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  document.getElementById("weights-file-name").innerText = `${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  const resultDiv = document.getElementById("weights-audit-result");
  resultDiv.style.display = "block";
  resultDiv.style.background = "rgba(59, 130, 246, 0.15)";
  resultDiv.style.borderColor = "rgba(59, 130, 246, 0.4)";
  resultDiv.style.color = "#93c5fd";
  resultDiv.innerHTML = "Computing client-side SHA-256 hash via WebCrypto...";

  try {
    const arrayBuffer = await file.arrayBuffer();
    const hashBuffer = await crypto.subtle.digest("SHA-256", arrayBuffer);
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const computedHash = "0x" + hashArray.map(b => b.toString(16).padStart(2, "0")).join("");

    console.log(`Computed SHA-256 for ${file.name}: ${computedHash}`);

    const provider = getProvider();
    if (!provider) {
      throw new Error("Ethers.js provider not initialized");
    }

    const modelContract = new ethers.Contract(CONTRACT_ADDRESSES.ModelRegistry, MODEL_REGISTRY_ABI, provider);

    try {
      const isActive = await modelContract.isModelActive(computedHash);
      const record = await modelContract.getModel(computedHash);

      if (isActive) {
        resultDiv.style.background = "rgba(16, 185, 129, 0.15)";
        resultDiv.style.border = "1px solid rgba(16, 185, 129, 0.4)";
        resultDiv.style.color = "#34d399";
        resultDiv.innerHTML = `
          <div style="font-weight: bold; margin-bottom: 0.35rem;">✅ VALID WEIGHTS FILE — MATCHES ON-CHAIN MODEL HASH</div>
          <div><strong>Computed SHA-256:</strong> ${computedHash}</div>
          <div><strong>Model Version:</strong> ${record.version} | <strong>Status:</strong> ACTIVE</div>
          <div style="word-break: break-all;"><strong>Registered URI:</strong> ${record.uri}</div>
        `;
      } else {
        resultDiv.style.background = "rgba(245, 158, 11, 0.15)";
        resultDiv.style.border = "1px solid rgba(245, 158, 11, 0.4)";
        resultDiv.style.color = "#fbbf24";
        resultDiv.innerHTML = `
          <div style="font-weight: bold; margin-bottom: 0.35rem;">⚠️ MODEL WEIGHTS REGISTERED BUT INACTIVE / REVOKED</div>
          <div><strong>Computed SHA-256:</strong> ${computedHash}</div>
          <div><strong>Version:</strong> ${record.version} (Revoked on-chain)</div>
        `;
      }
    } catch (e) {
      // Reverted with ModelNotFound
      resultDiv.style.background = "rgba(239, 68, 68, 0.15)";
      resultDiv.style.border = "1px solid rgba(239, 68, 68, 0.4)";
      resultDiv.style.color = "#f87171";
      resultDiv.innerHTML = `
        <div style="font-weight: bold; margin-bottom: 0.35rem;">❌ TAMPERED / UNREGISTERED WEIGHTS FILE (MISMATCH)</div>
        <div><strong>Computed SHA-256:</strong> ${computedHash}</div>
        <div><strong>On-Chain Status:</strong> ModelNotFound in ModelRegistry (${CONTRACT_ADDRESSES.ModelRegistry})</div>
      `;
    }
  } catch (err) {
    resultDiv.style.background = "rgba(239, 68, 68, 0.15)";
    resultDiv.style.border = "1px solid rgba(239, 68, 68, 0.4)";
    resultDiv.style.color = "#f87171";
    resultDiv.innerHTML = `<strong>Error auditing weights file:</strong> ${err.message}`;
  }
}

async function loadProductionManifest() {
  const sampleManifest = {
    "version": "v1.0.0",
    "model_type": "xgboost_all_features",
    "model_hash": "0x69082f2df0e866391ad2400ea6042f458ac0011925eda7630e58fe7d65736f5f",
    "config_hash": "0xc10edd52cb5cecec712a8eb2dbdb3d5010c887914a96888ed897892c2cfff275",
    "metrics_hash": "0xa0c59676c88a07c276757e52a7e8c49b00b0079b83582a4af4ff7dd52f0700a9",
    "data_split_id": "primary_steps_1_34_val_35_39_test_40_49",
    "trained_at": "2026-10-05T15:26:08.536459+00:00",
    "seed": 42,
    "metrics": {
      "pr_auc": 0.67868,
      "f1": 0.65588,
      "recall_at_80_precision": 0.60063
    }
  };

  document.getElementById("manifest-json-input").value = JSON.stringify(sampleManifest, null, 2);
}

async function verifyClientManifest() {
  const input = document.getElementById("manifest-json-input").value.trim();
  if (!input) {
    alert("Please paste MANIFEST.json content or click 'Load v1.0.0 Production Manifest'.");
    return;
  }

  try {
    const parsed = JSON.parse(input);
    const resultsPanel = document.getElementById("manifest-verify-results");
    resultsPanel.style.display = "block";

    const modelHash = parsed.model_hash || "";
    document.getElementById("audit-model-val").innerText = modelHash || "N/A";
    document.getElementById("audit-config-val").innerText = parsed.config_hash || "N/A";
    document.getElementById("audit-metrics-val").innerText = parsed.metrics_hash || "N/A";

    const provider = getProvider();
    const statusEl = document.getElementById("audit-chain-status");
    const contractValEl = document.getElementById("audit-contract-val");

    if (provider && modelHash && ethers.isHexString(modelHash, 32)) {
      try {
        const modelContract = new ethers.Contract(CONTRACT_ADDRESSES.ModelRegistry, MODEL_REGISTRY_ABI, provider);
        const isActive = await modelContract.isModelActive(modelHash);
        const record = await modelContract.getModel(modelHash);

        if (isActive) {
          statusEl.innerText = `ON-CHAIN REGISTERED & ACTIVE (${record.version})`;
          statusEl.className = "audit-status match";
          contractValEl.innerText = `Contract: ${CONTRACT_ADDRESSES.ModelRegistry} | Registered At: ${new Date(Number(record.registeredAt) * 1000).toISOString()}`;
        } else {
          statusEl.innerText = "MODEL REVOKED / INACTIVE ON-CHAIN";
          statusEl.className = "audit-status error";
          contractValEl.innerText = `Contract: ${CONTRACT_ADDRESSES.ModelRegistry}`;
        }
      } catch (e) {
        statusEl.innerText = "TAMPERED / UNREGISTERED MODEL HASH (Not Found in ModelRegistry)";
        statusEl.className = "audit-status error";
        contractValEl.innerText = `ModelRegistry queried at ${CONTRACT_ADDRESSES.ModelRegistry}: ModelNotFound`;
      }
    } else {
      statusEl.innerText = "INVALID MODEL HASH FORMAT (Must be 32-byte hex)";
      statusEl.className = "audit-status error";
      contractValEl.innerText = "Invalid bytes32 identifier";
    }
  } catch (e) {
    alert("Invalid JSON format in manifest input: " + e.message);
  }
}

// Demo batch table
async function loadDemoBatchTable() {
  const tbody = document.getElementById("demo-table-body");
  tbody.innerHTML = `<tr><td colspan="5" style="text-align: center;">Fetching demo transactions...</td></tr>`;

  try {
    const res = await fetch(`${API_BASE}/score/demo-batch?count=10`);
    if (!res.ok) throw new Error("HTTP error");
    const data = await res.json();
    
    tbody.innerHTML = "";
    data.items.forEach(item => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="mono" style="color: #93c5fd;">${item.address}</td>
        <td>Step 40+ (#${item.node_id})</td>
        <td class="mono font-semibold">${item.score} / 10000</td>
        <td><span class="badge risk-badge ${item.risk_level.toLowerCase()}">${item.risk_level}</span></td>
        <td><button class="btn btn-secondary btn-sm" onclick="setAndQueryAddress('${item.address}')">Query On-Chain</button></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">Inference service offline. Use manual address query.</td></tr>`;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadDeploymentAddresses();
  console.log("ChainGuard DApp Initialized. Direct EVM JSON-RPC provider configured.");
});
