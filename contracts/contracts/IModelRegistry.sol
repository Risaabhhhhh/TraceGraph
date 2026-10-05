// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

interface IModelRegistry {
    struct ModelRecord {
        bytes32 modelHash;
        string version;
        string uri;
        bool active;
        uint256 registeredAt;
    }

    function registerModel(bytes32 modelHash, string calldata version, string calldata uri) external;
    function setModelStatus(bytes32 modelHash, bool active) external;
    function isModelActive(bytes32 modelHash) external view returns (bool);
    function getModel(bytes32 modelHash) external view returns (ModelRecord memory);
    function getAllModelHashes() external view returns (bytes32[] memory);
}
