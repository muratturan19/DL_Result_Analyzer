"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


class TestAPI:
    """Test suite for API endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["message"] == "DL_Result_Analyzer API"

    def test_upload_results_missing_csv(self, client):
        """Test upload endpoint with missing CSV file."""
        response = client.post("/api/upload/results")
        # Should return 422 (Unprocessable Entity) for missing required field
        assert response.status_code == 422

    def test_analyze_metrics_valid_payload(self, client):
        """Test analyze metrics endpoint with valid payload."""
        payload = {
            "precision": 0.79,
            "recall": 0.82,
            "map50": 0.86,
            "map50_95": 0.40,
            "loss": 0.73,
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.01,
            "iou_threshold": 0.5,
            "conf_threshold": 0.5,
        }

        response = client.post("/api/analyze/metrics", json=payload)
        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "summary" in data
        assert "strengths" in data
        assert "weaknesses" in data
        assert "action_items" in data
        assert "risk_level" in data

    def test_analyze_metrics_invalid_payload(self, client):
        """Test analyze metrics endpoint with invalid payload."""
        payload = {
            "precision": "not_a_number",  # Invalid type
        }

        response = client.post("/api/analyze/metrics", json=payload)
        # Should return 422 for validation error
        assert response.status_code == 422

    def test_compare_endpoint(self, client):
        """Test compare endpoint (currently TODO)."""
        response = client.post("/api/compare", json=["run1", "run2"])
        assert response.status_code == 200
        data = response.json()
        assert "message" in data

    def test_history_endpoint(self, client):
        """Test history endpoint (currently TODO)."""
        response = client.get("/api/history")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)
