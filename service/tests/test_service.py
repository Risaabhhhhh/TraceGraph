"""
Unit tests for ChainGuard FastAPI service and scoring engine.
"""

from fastapi.testclient import TestClient
import pytest
from service.app.main import app
from service.app.scoring import ScoringEngine

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "chainguard-service"


def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "v1.0.0"
    assert "0x" in data["model_hash"]
    assert "pr_auc" in data["metrics"]
    assert data["metrics"]["pr_auc"] > 0.60


def test_score_endpoint():
    # Test scoring via demo node ID
    response = client.post("/score", json={"node_id": 0})
    assert response.status_code == 200
    data = response.json()
    assert len(data["results"]) == 1
    res = data["results"][0]
    assert 0 <= res["score"] <= 10000
    assert 0.0 <= res["probability"] <= 1.0
    assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


def test_demo_batch_endpoint():
    response = client.get("/score/demo-batch?count=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 10
    assert "address" in data["items"][0]
    assert data["items"][0]["is_demo_address"] is True


def test_publish_dry_run():
    response = client.post("/publish", json={"sample_size": 15, "dry_run": True})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["dry_run"] is True
    assert data["count"] == 15
    assert len(data["snapshot_hash"]) > 10
