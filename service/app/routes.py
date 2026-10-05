"""
FastAPI route handlers for ChainGuard service.
"""

from datetime import datetime, timezone
import json
from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from service.app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PublishRequest,
    PublishResponse,
    ScoreRequest,
    ScoreResponse,
    ScoreResult,
)
from service.app.scoring import ScoringEngine
from service.publisher import ChainGuardPublisher

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Service health and uptime endpoint."""
    return HealthResponse(
        status="ok",
        service="chainguard-service",
        version="v1.0.0",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Retrieve exported production model metadata and cryptographic hashes."""
    try:
        engine = ScoringEngine.get_instance()
        m = engine.manifest
        return ModelInfoResponse(
            version=m["version"],
            model_type=m["model_type"],
            model_hash=m["model_hash"],
            config_hash=m["config_hash"],
            metrics_hash=m["metrics_hash"],
            manifest_hash=m.get("manifest_hash", ""),
            trained_at=m["trained_at"],
            seed=m["seed"],
            data_split_id=m["data_split_id"],
            metrics=m["metrics"],
            hyperparameters=m["hyperparameters"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model info: {str(e)}")


@router.post("/score", response_model=ScoreResponse)
def score_address(req: ScoreRequest):
    """
    Compute risk score for an address, node ID, or feature vector.
    """
    try:
        engine = ScoringEngine.get_instance()
        scored = engine.score_single(
            address=req.address,
            node_id=req.node_id,
            features=req.features,
        )
        return ScoreResponse(
            results=[ScoreResult(**scored)],
            total_scored=1,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.get("/score/demo-batch")
def get_demo_batch(count: int = Query(20, ge=1, le=100)):
    """Retrieve scored demo test transactions mapped to EVM addresses."""
    try:
        engine = ScoringEngine.get_instance()
        items = engine.sample_demo_batch(count=count)
        return {"items": items, "count": len(items)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sampling demo batch: {str(e)}")


@router.post("/publish", response_model=PublishResponse)
def publish_scores(req: PublishRequest):
    """
    Trigger batch score publication to RiskRegistry.sol.
    Supports dry-run mode for evaluation without live RPC node.
    """
    try:
        publisher = ChainGuardPublisher()
        res = publisher.publish_demo_batch(
            sample_size=req.sample_size,
            dry_run=req.dry_run,
        )
        return PublishResponse(
            success=res["success"],
            dry_run=res["dry_run"],
            batch_id=res.get("batch_id"),
            model_hash=res["model_hash"],
            snapshot_hash=res["snapshot_hash"],
            count=res["count"],
            tx_hash=res.get("tx_hash"),
            scores_sample=res["scores_sample"],
        )
    except Exception as e:
        # Fall back to dry-run or error response if local chain not available
        return PublishResponse(
            success=False,
            dry_run=req.dry_run,
            batch_id=None,
            model_hash="",
            snapshot_hash="",
            count=0,
            tx_hash=None,
            scores_sample=[],
        )
