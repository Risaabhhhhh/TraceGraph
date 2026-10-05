"""
Pydantic schemas for ChainGuard API service.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "chainguard-service"
    version: str = "v1.0.0"
    timestamp: str


class ModelInfoResponse(BaseModel):
    version: str
    model_type: str
    model_hash: str
    config_hash: str
    metrics_hash: str
    manifest_hash: str
    trained_at: str
    seed: int
    data_split_id: str
    metrics: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    active_on_chain: Optional[bool] = None


class ScoreRequest(BaseModel):
    # Support direct EVM demo address OR raw feature vector
    address: Optional[str] = Field(None, description="EVM demo address (e.g., 0x...)")
    node_id: Optional[int] = Field(None, description="Elliptic node index (0..203768)")
    features: Optional[List[float]] = Field(None, description="165-dim feature vector")


class ScoreResult(BaseModel):
    address: str
    is_demo_address: bool = True
    node_id: Optional[int] = None
    probability: float
    score: int = Field(..., description="Integer score 0..10000 (basis points)")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    model_hash: str
    model_version: str


class ScoreResponse(BaseModel):
    results: List[ScoreResult]
    total_scored: int


class PublishRequest(BaseModel):
    sample_size: int = Field(20, description="Number of demo test transactions to score & publish")
    dry_run: bool = Field(False, description="If true, computes scores and hashes without on-chain tx")


class PublishResponse(BaseModel):
    success: bool
    dry_run: bool
    batch_id: Optional[int] = None
    model_hash: str
    snapshot_hash: str
    count: int
    tx_hash: Optional[str] = None
    scores_sample: List[Dict[str, Any]]
