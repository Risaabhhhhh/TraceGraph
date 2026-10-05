import { expect } from "chai";
import { ethers } from "hardhat";
import { ModelRegistry, RiskRegistry } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("ChainGuard On-Chain Registries", function () {
  let modelRegistry: ModelRegistry;
  let riskRegistry: RiskRegistry;
  let admin: SignerWithAddress;
  let publisher: SignerWithAddress;
  let user: SignerWithAddress;

  const MODEL_HASH = "0x69082f2df0e866391ad2400ea6042f458ac0011925eda7630e58fe7d65736f5f";
  const SNAPSHOT_HASH = "0xa1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0";
  const VERSION = "v1.0.0";
  const URI = "ipfs://chainguard/models/v1.0.0/MANIFEST.json";

  beforeEach(async function () {
    [admin, publisher, user] = await ethers.getSigners();

    // Deploy ModelRegistry
    const ModelRegistryFactory = await ethers.getContractFactory("ModelRegistry");
    modelRegistry = await ModelRegistryFactory.deploy(admin.address);
    await modelRegistry.waitForDeployment();

    // Deploy RiskRegistry
    const RiskRegistryFactory = await ethers.getContractFactory("RiskRegistry");
    riskRegistry = await RiskRegistryFactory.deploy(admin.address, await modelRegistry.getAddress());
    await riskRegistry.waitForDeployment();

    // Grant PUBLISHER_ROLE to publisher
    const PUBLISHER_ROLE = await riskRegistry.PUBLISHER_ROLE();
    await riskRegistry.grantRole(PUBLISHER_ROLE, publisher.address);
  });

  describe("1. ModelRegistry Management & Access Control", function () {
    it("should allow MODEL_MANAGER_ROLE to register a model and emit ModelRegistered event", async function () {
      await expect(modelRegistry.registerModel(MODEL_HASH, VERSION, URI))
        .to.emit(modelRegistry, "ModelRegistered")
        .withArgs(MODEL_HASH, VERSION, URI, (val: any) => val > 0);

      expect(await modelRegistry.isModelActive(MODEL_HASH)).to.be.true;

      const record = await modelRegistry.getModel(MODEL_HASH);
      expect(record.modelHash).to.equal(MODEL_HASH);
      expect(record.version).to.equal(VERSION);
      expect(record.uri).to.equal(URI);
      expect(record.active).to.be.true;
    });

    it("should revert if non-manager tries to register a model", async function () {
      await expect(
        modelRegistry.connect(user).registerModel(MODEL_HASH, VERSION, URI)
      ).to.be.revertedWithCustomError(modelRegistry, "AccessControlUnauthorizedAccount");
    });

    it("should revert if model already exists or hash is zero", async function () {
      await modelRegistry.registerModel(MODEL_HASH, VERSION, URI);
      await expect(
        modelRegistry.registerModel(MODEL_HASH, VERSION, URI)
      ).to.be.revertedWithCustomError(modelRegistry, "ModelAlreadyExists");

      await expect(
        modelRegistry.registerModel(ethers.ZeroHash, VERSION, URI)
      ).to.be.revertedWithCustomError(modelRegistry, "InvalidModelHash");
    });

    it("should allow activating and deactivating models", async function () {
      await modelRegistry.registerModel(MODEL_HASH, VERSION, URI);
      expect(await modelRegistry.isModelActive(MODEL_HASH)).to.be.true;

      await expect(modelRegistry.setModelStatus(MODEL_HASH, false))
        .to.emit(modelRegistry, "ModelStatusUpdated")
        .withArgs(MODEL_HASH, false, (val: any) => val > 0);

      expect(await modelRegistry.isModelActive(MODEL_HASH)).to.be.false;

      await modelRegistry.setModelStatus(MODEL_HASH, true);
      expect(await modelRegistry.isModelActive(MODEL_HASH)).to.be.true;
    });
  });

  describe("2. RiskRegistry Batch Creation & Validation", function () {
    beforeEach(async function () {
      await modelRegistry.registerModel(MODEL_HASH, VERSION, URI);
    });

    it("should create a batch for an active model and emit BatchCreated", async function () {
      await expect(riskRegistry.connect(publisher).createBatch(MODEL_HASH, SNAPSHOT_HASH))
        .to.emit(riskRegistry, "BatchCreated")
        .withArgs(0, MODEL_HASH, SNAPSHOT_HASH, (val: any) => val > 0);

      const batch = await riskRegistry.getBatch(0);
      expect(batch.modelHash).to.equal(MODEL_HASH);
      expect(batch.snapshotHash).to.equal(SNAPSHOT_HASH);
      expect(batch.count).to.equal(0);
    });

    it("should revert createBatch if model is inactive or unregistered", async function () {
      const UNREGISTERED = "0x1111111111111111111111111111111111111111111111111111111111111111";
      await expect(
        riskRegistry.connect(publisher).createBatch(UNREGISTERED, SNAPSHOT_HASH)
      ).to.be.revertedWithCustomError(riskRegistry, "InactiveModel");

      // Deactivate model
      await modelRegistry.setModelStatus(MODEL_HASH, false);
      await expect(
        riskRegistry.connect(publisher).createBatch(MODEL_HASH, SNAPSHOT_HASH)
      ).to.be.revertedWithCustomError(riskRegistry, "InactiveModel");
    });

    it("should revert if non-publisher attempts createBatch", async function () {
      await expect(
        riskRegistry.connect(user).createBatch(MODEL_HASH, SNAPSHOT_HASH)
      ).to.be.revertedWithCustomError(riskRegistry, "AccessControlUnauthorizedAccount");
    });
  });

  describe("3. Score Publishing & Constraints", function () {
    beforeEach(async function () {
      await modelRegistry.registerModel(MODEL_HASH, VERSION, URI);
      await riskRegistry.connect(publisher).createBatch(MODEL_HASH, SNAPSHOT_HASH);
    });

    it("should publish scores, emit events, and update account risk records", async function () {
      const demoAccount1 = ethers.Wallet.createRandom().address;
      const demoAccount2 = ethers.Wallet.createRandom().address;
      const accounts = [demoAccount1, demoAccount2];
      const scores = [8540, 320]; // 85.40% and 3.20%

      const tx = await riskRegistry.connect(publisher).publishScores(0, accounts, scores);
      await expect(tx)
        .to.emit(riskRegistry, "ScoreUpdated")
        .withArgs(demoAccount1, 8540, 0, (val: any) => val > 0);
      await expect(tx)
        .to.emit(riskRegistry, "ScoresPublished")
        .withArgs(0, 2, (val: any) => val > 0);

      const res1 = await riskRegistry.getScore(demoAccount1);
      expect(res1.score).to.equal(8540);
      expect(res1.batchId).to.equal(0);
      expect(res1.modelHash).to.equal(MODEL_HASH);
      expect(res1.snapshotHash).to.equal(SNAPSHOT_HASH);

      const batch = await riskRegistry.getBatch(0);
      expect(batch.count).to.equal(2);
    });

    it("should revert if score exceeds 10000 (ScoreOutOfRange)", async function () {
      const demoAccount = ethers.Wallet.createRandom().address;
      await expect(
        riskRegistry.connect(publisher).publishScores(0, [demoAccount], [10001])
      ).to.be.revertedWithCustomError(riskRegistry, "ScoreOutOfRange");
    });

    it("should revert if accounts and scores lengths mismatch", async function () {
      const demo1 = ethers.Wallet.createRandom().address;
      const demo2 = ethers.Wallet.createRandom().address;
      await expect(
        riskRegistry.connect(publisher).publishScores(0, [demo1, demo2], [5000])
      ).to.be.revertedWithCustomError(riskRegistry, "ArrayLengthMismatch");
    });

    it("should revert if batch is empty", async function () {
      await expect(
        riskRegistry.connect(publisher).publishScores(0, [], [])
      ).to.be.revertedWithCustomError(riskRegistry, "EmptyBatch");
    });

    it("should revert if model is deactivated before publishing", async function () {
      await modelRegistry.setModelStatus(MODEL_HASH, false);
      const demo = ethers.Wallet.createRandom().address;
      await expect(
        riskRegistry.connect(publisher).publishScores(0, [demo], [5000])
      ).to.be.revertedWithCustomError(riskRegistry, "InactiveModel");
    });
  });

  describe("4. Gas Profiling (Batch sizes 1, 10, 50)", function () {
    beforeEach(async function () {
      await modelRegistry.registerModel(MODEL_HASH, VERSION, URI);
      await riskRegistry.connect(publisher).createBatch(MODEL_HASH, SNAPSHOT_HASH);
    });

    async function measureGas(batchSize: number): Promise<{ totalGas: bigint; gasPerAddress: number }> {
      const accounts: string[] = [];
      const scores: number[] = [];
      for (let i = 0; i < batchSize; i++) {
        accounts.push(ethers.Wallet.createRandom().address);
        scores.push(Math.floor(Math.random() * 10000));
      }

      const tx = await riskRegistry.connect(publisher).publishScores(0, accounts, scores);
      const receipt = await tx.wait();
      const gasUsed = receipt!.gasUsed;
      const gasPerAddress = Number(gasUsed) / batchSize;
      return { totalGas: gasUsed, gasPerAddress };
    }

    it("should measure gas consumption for batch sizes 1, 10, and 50", async function () {
      const gas1 = await measureGas(1);
      const gas10 = await measureGas(10);
      const gas50 = await measureGas(50);

      console.log("\n========================================================");
      console.log("GAS REPORT: RiskRegistry.publishScores");
      console.log("========================================================");
      console.log(`Batch Size  1: Total Gas = ${gas1.totalGas.toString()} (Gas/Addr = ${gas1.gasPerAddress.toFixed(0)})`);
      console.log(`Batch Size 10: Total Gas = ${gas10.totalGas.toString()} (Gas/Addr = ${gas10.gasPerAddress.toFixed(0)})`);
      console.log(`Batch Size 50: Total Gas = ${gas50.totalGas.toString()} (Gas/Addr = ${gas50.gasPerAddress.toFixed(0)})`);
      console.log("========================================================\n");

      expect(gas1.totalGas).to.be.greaterThan(0);
      expect(gas10.totalGas).to.be.greaterThan(gas1.totalGas);
      expect(gas50.totalGas).to.be.greaterThan(gas10.totalGas);
    });
  });
});
