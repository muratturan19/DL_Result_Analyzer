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
        assert "actions" in data
        assert "risk" in data
        assert "deploy_profile" in data

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

    def test_report_qa_requires_existing_report(self, client):
        """QA endpoint should return 404 for unknown report IDs."""
        response = client.post(
            "/api/report/nonexistent/qa",
            json={"question": "Test"},
        )
        assert response.status_code == 404

    def test_report_qa_success(
        self,
        client,
        sample_csv_path,
        sample_yaml_path,
    ):
        """Uploading results should allow follow-up Q/A queries."""

        with sample_csv_path.open("rb") as csv_file, sample_yaml_path.open("rb") as yaml_file:
            files = [
                ("results_csv", ("sample_results.csv", csv_file, "text/csv")),
                ("config_yaml", ("sample_args.yaml", yaml_file, "application/x-yaml")),
            ]
            upload_response = client.post("/api/upload/results", files=files)

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        report_id = upload_data.get("report_id")
        assert report_id

        dataset_info = upload_data.get("config", {}).get("dataset", {})
        assert dataset_info.get("train_images") == 280

        qa_response = client.post(
            f"/api/report/{report_id}/qa",
            json={"question": "Eğitim setindeki görsel sayısı nedir?"},
        )
        assert qa_response.status_code == 200

        qa_data = qa_response.json()
        assert qa_data.get("report_id") == report_id
        assert qa_data.get("qa", {}).get("question") == "Eğitim setindeki görsel sayısı nedir?"
        answer_text = qa_data.get("qa", {}).get("answer", "")
        assert "280" in answer_text
        references = qa_data.get("qa", {}).get("references", [])
        assert isinstance(references, list)
        assert references
