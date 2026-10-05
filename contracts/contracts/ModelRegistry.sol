// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/access/AccessControl.sol";
import "./IModelRegistry.sol";

/**
 * @title ModelRegistry
 * @dev On-chain registry for verified ML model artifacts and metadata.
 * Models are uniquely identified by their cryptographic SHA-256 hash (bytes32).
 */
contract ModelRegistry is AccessControl, IModelRegistry {
    bytes32 public constant MODEL_MANAGER_ROLE = keccak256("MODEL_MANAGER_ROLE");

    mapping(bytes32 => ModelRecord) private _models;
    bytes32[] private _modelHashes;

    event ModelRegistered(
        bytes32 indexed modelHash,
        string version,
        string uri,
        uint256 timestamp
    );

    event ModelStatusUpdated(
        bytes32 indexed modelHash,
        bool active,
        uint256 timestamp
    );

    error ModelAlreadyExists(bytes32 modelHash);
    error ModelNotFound(bytes32 modelHash);
    error InvalidModelHash();

    constructor(address admin) {
        _grantRole(DEFAULT_ADMIN_ROLE, admin);
        _grantRole(MODEL_MANAGER_ROLE, admin);
    }

    /**
     * @notice Register a newly verified model artifact.
     * @param modelHash SHA-256 hash of the model artifact file.
     * @param version Semantic version string (e.g. "v1.0.0").
     * @param uri URI linking to MANIFEST.json or model documentation.
     */
    function registerModel(
        bytes32 modelHash,
        string calldata version,
        string calldata uri
    ) external onlyRole(MODEL_MANAGER_ROLE) {
        if (modelHash == bytes32(0)) {
            revert InvalidModelHash();
        }
        if (_models[modelHash].registeredAt != 0) {
            revert ModelAlreadyExists(modelHash);
        }

        _models[modelHash] = ModelRecord({
            modelHash: modelHash,
            version: version,
            uri: uri,
            active: true,
            registeredAt: block.timestamp
        });

        _modelHashes.push(modelHash);

        emit ModelRegistered(modelHash, version, uri, block.timestamp);
    }

    /**
     * @notice Activate or deactivate a registered model.
     * @param modelHash Model identifier.
     * @param active True if model is approved for score publishing.
     */
    function setModelStatus(
        bytes32 modelHash,
        bool active
    ) external onlyRole(MODEL_MANAGER_ROLE) {
        if (_models[modelHash].registeredAt == 0) {
            revert ModelNotFound(modelHash);
        }

        _models[modelHash].active = active;

        emit ModelStatusUpdated(modelHash, active, block.timestamp);
    }

    /**
     * @notice Check whether a model is active and approved.
     */
    function isModelActive(bytes32 modelHash) external view returns (bool) {
        return _models[modelHash].active && _models[modelHash].registeredAt != 0;
    }

    /**
     * @notice Get full details of a registered model.
     */
    function getModel(bytes32 modelHash) external view returns (ModelRecord memory) {
        if (_models[modelHash].registeredAt == 0) {
            revert ModelNotFound(modelHash);
        }
        return _models[modelHash];
    }

    /**
     * @notice Return total count of registered models.
     */
    function getModelCount() external view returns (uint256) {
        return _modelHashes.length;
    }

    /**
     * @notice Return list of all registered model hashes.
     */
    function getAllModelHashes() external view returns (bytes32[] memory) {
        return _modelHashes;
    }
}
