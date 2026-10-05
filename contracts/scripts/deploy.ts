import { ethers } from "hardhat";
import * as fs from "fs";
import * as path from "path";

async function main() {
  const [deployer] = await ethers.getSigners();
  console.log(`Deploying ChainGuard contracts with deployer: ${deployer.address}`);

  // 1. Deploy ModelRegistry
  const ModelRegistryFactory = await ethers.getContractFactory("ModelRegistry");
  const modelRegistry = await ModelRegistryFactory.deploy(deployer.address);
  await modelRegistry.waitForDeployment();
  const modelRegistryAddress = await modelRegistry.getAddress();
  console.log(`ModelRegistry deployed at: ${modelRegistryAddress}`);

  // 2. Deploy RiskRegistry
  const RiskRegistryFactory = await ethers.getContractFactory("RiskRegistry");
  const riskRegistry = await RiskRegistryFactory.deploy(deployer.address, modelRegistryAddress);
  await riskRegistry.waitForDeployment();
  const riskRegistryAddress = await riskRegistry.getAddress();
  console.log(`RiskRegistry deployed at: ${riskRegistryAddress}`);

  // 3. Check for MANIFEST.json and auto-register initial model
  const manifestPath = path.resolve(__dirname, "../../ml/artifacts/v1.0.0/MANIFEST.json");
  let modelHash = "0x69082f2df0e866391ad2400ea6042f458ac0011925eda7630e58fe7d65736f5f";
  let version = "v1.0.0";
  let uri = "ipfs://chainguard/models/v1.0.0/MANIFEST.json";

  if (fs.existsSync(manifestPath)) {
    try {
      const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf-8"));
      modelHash = manifest.model_hash || modelHash;
      version = manifest.version || version;
      uri = `ipfs://chainguard/models/${version}/MANIFEST.json`;
    } catch (e) {
      console.warn("Could not parse MANIFEST.json, using defaults.");
    }
  }

  console.log(`Registering production model (${version}) with hash: ${modelHash}...`);
  const regTx = await modelRegistry.registerModel(modelHash, version, uri);
  await regTx.wait();
  console.log(`Model registered successfully!`);

  // 4. Save deployment addresses to JSON files
  const deploymentInfo = {
    network: (await ethers.provider.getNetwork()).name,
    chainId: Number((await ethers.provider.getNetwork()).chainId),
    deployer: deployer.address,
    contracts: {
      ModelRegistry: modelRegistryAddress,
      RiskRegistry: riskRegistryAddress,
    },
    initialModel: {
      modelHash,
      version,
      uri,
    },
    deployedAt: new Date().toISOString(),
  };

  const outputPaths = [
    path.resolve(__dirname, "../deployments/localhost.json"),
    path.resolve(__dirname, "../../deployments/localhost.json"),
  ];

  for (const outPath of outputPaths) {
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, JSON.stringify(deploymentInfo, null, 2));
    console.log(`Saved deployment info to: ${outPath}`);
  }
}

main().catch((error) => {
  console.error("Deployment failed:", error);
  process.exitCode = 1;
});
