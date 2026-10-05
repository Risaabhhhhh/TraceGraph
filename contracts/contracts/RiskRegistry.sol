// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "./IModelRegistry.sol";

/**
 * @title RiskRegistry
 * @dev On-chain registry for batch-published illicit transaction risk scores.
 * All score batches must link to an active model registered in ModelRegistry.
 *
 * Scores are integers from 0 to 10000 representing basis points (0.00% to 100.00% risk).
 */
contract RiskRegistry is AccessControl {
    bytes32 public constant PUBLISHER_ROLE = keccak256("PUBLISHER_ROLE");

    struct BatchRecord {
        bytes32 modelHash;
        bytes32 snapshotHash;
        uint256 timestamp;
        uint256 count;
    }

    struct RiskScore {
        uint16 score; // 0 to 10000 (basis points)
        uint256 batchId;
        uint256 timestamp;
    }

    IModelRegistry public modelRegistry;
    uint256 public nextBatchId;

    mapping(uint256 => BatchRecord) public batches;
    mapping(address => RiskScore) public scores;

    event ModelRegistryUpdated(address indexed newModelRegistry);
    event BatchCreated(
        uint256 indexed batchId,
        bytes32 indexed modelHash,
        bytes32 snapshotHash,
        uint256 timestamp
    );
    event ScoreUpdated(
        address indexed account,
        uint16 score,
        uint256 indexed batchId,
        uint256 timestamp
    );
    event ScoresPublished(
        uint256 indexed batchId,
        uint256 count,
        uint256 timestamp
    );

    error InactiveModel(bytes32 modelHash);
    error InvalidBatchId(uint256 batchId);
    error ArrayLengthMismatch();
    error EmptyBatch();
    error ScoreOutOfRange(uint16 score);
    error ZeroAddress();

    constructor(address admin, address _modelRegistry) {
        if (admin == address(0) || _modelRegistry == address(0)) {
            revert ZeroAddress();
        }
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(PUBLISHER_ROLE, admin);

        modelRegistry = IModelRegistry(_modelRegistry);
        emit ModelRegistryUpdated(_modelRegistry);
    }

    /**
     * @notice Update the reference to ModelRegistry.
     */
    function setModelRegistry(address _modelRegistry) external onlyRole(DEFAULT_ADMIN_ROLE) {
        if (_modelRegistry == address(0)) {
            revert ZeroAddress();
        }
        modelRegistry = IModelRegistry(_modelRegistry);
        emit ModelRegistryUpdated(_modelRegistry);
    }

    /**
     * @notice Initialize a new risk score publishing batch.
     * @param modelHash Cryptographic hash of the registered model used to compute scores.
     * @param snapshotHash Cryptographic hash of the input dataset/transactions snapshot.
     * @return batchId Unique identifier for the created batch.
     */
    function createBatch(
        bytes32 modelHash,
        bytes32 snapshotHash
    ) external onlyRole(PUBLISHER_ROLE) returns (uint256 batchId) {
        if (!modelRegistry.isModelActive(modelHash)) {
            revert InactiveModel(modelHash);
        }

        batchId = nextBatchId;
        nextBatchId++;

        batches[batchId] = BatchRecord({
            modelHash: modelHash,
            snapshotHash: snapshotHash,
            timestamp: block.timestamp,
            count: 0
        });

        emit BatchCreated(batchId, modelHash, snapshotHash, block.timestamp);
    }

    /**
     * @notice Publish scores for a batch of EVM addresses.
     * @param batchId Identifier of the batch created via createBatch.
     * @param accounts Array of addresses to score.
     * @param newScores Array of risk scores (0..10000).
     */
    function publishScores(
        uint256 batchId,
        address[] calldata accounts,
        uint16[] calldata newScores
    ) external onlyRole(PUBLISHER_ROLE) {
        if (batchId >= nextBatchId || batches[batchId].timestamp == 0) {
            revert InvalidBatchId(batchId);
        }
        if (accounts.length != newScores.length) {
            revert ArrayLengthMismatch();
        }
        if (accounts.length == 0) {
            revert EmptyBatch();
        }

        bytes32 modelHash = batches[batchId].modelHash;
        if (!modelRegistry.isModelActive(modelHash)) {
            revert InactiveModel(modelHash);
        }

        uint256 len = accounts.length;
        for (uint256 i = 0; i < len; ) {
            uint16 score = newScores[i];
            if (score > 10000) {
                revert ScoreOutOfRange(score);
            }

            scores[accounts[i]] = RiskScore({
                score: score,
                batchId: batchId,
                timestamp: block.timestamp
            });

            emit ScoreUpdated(accounts[i], score, batchId, block.timestamp);

            unchecked {
                ++i;
            }
        }

        batches[batchId].count += len;

        emit ScoresPublished(batchId, len, block.timestamp);
    }

    /**
     * @notice Query the latest risk score for an address along with provenance data.
     */
    function getScore(
        address account
    )
        external
        view
        returns (
            uint16 score,
            uint256 batchId,
            uint256 timestamp,
            bytes32 modelHash,
            bytes32 snapshotHash
        )
    {
        RiskScore memory r = scores[account];
        if (r.timestamp == 0) {
            return (0, 0, 0, bytes32(0), bytes32(0));
        }

        BatchRecord memory b = batches[r.batchId];
        return (r.score, r.batchId, r.timestamp, b.modelHash, b.snapshotHash);
    }

    /**
     * @notice Get metadata for a specific batch.
     */
    function getBatch(
        uint256 batchId
    )
        external
        view
        returns (
            bytes32 modelHash,
            bytes32 snapshotHash,
            uint256 timestamp,
            uint256 count
        )
    {
        if (batchId >= nextBatchId || batches[batchId].timestamp == 0) {
            revert InvalidBatchId(batchId);
        }
        BatchRecord memory b = batches[batchId];
        return (b.modelHash, b.snapshotHash, b.timestamp, b.count);
    }
}
