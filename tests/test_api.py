"""Tests for the FastAPI toxicity prediction API."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app


@pytest.fixture(scope="module")
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoints:
    """Test health and info endpoints."""

    def test_root(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_health(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] > 0

    def test_endpoints_list(self, client):
        """Test endpoint listing."""
        response = client.get("/endpoints")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        for endpoint in data:
            assert "name" in endpoint
            assert "task" in endpoint
            assert "available" in endpoint


class TestSMILESValidation:
    """Test SMILES validation endpoints."""

    def test_validate_valid_smiles(self, client):
        """Test validation of valid SMILES."""
        response = client.post("/validate", json={"smiles": "CCO"})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["canonical_smiles"] is not None

    def test_validate_invalid_smiles(self, client):
        """Test validation of invalid SMILES."""
        response = client.post("/validate", json={"smiles": "invalid_smiles_xyz"})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False

    def test_validate_get(self, client):
        """Test GET validation endpoint."""
        response = client.get("/validate?smiles=CCO")
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True


class TestSinglePrediction:
    """Test single compound prediction."""

    @pytest.mark.parametrize("endpoint", ["herg", "hepatotox", "nephrotox", "ames", "skin_sens", "cytotox"])
    def test_predict_all_endpoints(self, client, endpoint):
        """Test prediction for each available endpoint."""
        response = client.post("/predict", json={
            "smiles": "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
            "endpoint": endpoint
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["endpoint"] == endpoint
        assert "prediction" in data
        assert "applicability_domain" in data

    def test_predict_invalid_smiles(self, client):
        """Test prediction with invalid SMILES."""
        response = client.post("/predict", json={
            "smiles": "invalid_smiles",
            "endpoint": "hepatotox"
        })
        assert response.status_code == 400

    def test_predict_invalid_endpoint(self, client):
        """Test prediction with invalid endpoint."""
        response = client.post("/predict", json={
            "smiles": "CCO",
            "endpoint": "invalid_endpoint"
        })
        assert response.status_code == 400

    def test_predict_get(self, client):
        """Test GET prediction endpoint."""
        response = client.get("/predict/hepatotox?smiles=CCO")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestMultiEndpointPrediction:
    """Test multi-endpoint prediction."""

    def test_predict_multi_all(self, client):
        """Test prediction across all endpoints."""
        response = client.post("/predict/multi", json={
            "smiles": "CC(=O)Nc1ccc(O)cc1"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data
        assert len(data["predictions"]) > 0

    def test_predict_multi_selected(self, client):
        """Test prediction for selected endpoints."""
        response = client.post("/predict/multi", json={
            "smiles": "CCO",
            "endpoints": ["hepatotox", "nephrotox"]
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "hepatotox" in data["predictions"]
        assert "nephrotox" in data["predictions"]


class TestBatchPrediction:
    """Test batch prediction."""

    def test_batch_predict(self, client):
        """Test batch prediction for multiple compounds."""
        response = client.post("/predict/batch", json={
            "smiles_list": [
                "CCO",  # Ethanol
                "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"  # Caffeine
            ],
            "endpoint": "hepatotox"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["successful"] == 3
        assert len(data["results"]) == 3

    def test_batch_with_invalid_smiles(self, client):
        """Test batch with some invalid SMILES."""
        response = client.post("/predict/batch", json={
            "smiles_list": ["CCO", "invalid_smiles", "CC"],
            "endpoint": "hepatotox"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        # Should have 2 successful, 1 failed
        assert data["successful"] == 2


class TestInterpretability:
    """Test SHAP interpretability endpoints."""

    def test_shap_status(self, client):
        """Test SHAP status endpoint."""
        response = client.get("/shap/status")
        assert response.status_code == 200
        data = response.json()
        assert "shap_available" in data

    def test_predict_with_interpretation(self, client):
        """Test prediction with SHAP interpretation."""
        response = client.post("/predict/interpret", json={
            "smiles": "CC(=O)Nc1ccc(O)cc1",
            "endpoint": "hepatotox",
            "top_k": 5
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        # Check for SHAP explanation in response
        assert "shap_explanation" in data or "interpretation" in data

    def test_integrated_prediction(self, client):
        """Test integrated multi-endpoint prediction with interpretation."""
        response = client.post("/predict/integrated", json={
            "smiles": "CC(=O)Nc1ccc(O)cc1",
            "include_interpretation": True,
            "top_k": 5
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "predictions" in data
        # Check for integrated assessment
        assert "integrated_assessment" in data or "risk_assessment" in data

    def test_integrated_without_interpretation(self, client):
        """Test integrated prediction without interpretation."""
        response = client.post("/predict/integrated", json={
            "smiles": "CCO",
            "include_interpretation": False
        })
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_interpret_get(self, client):
        """Test GET interpretation endpoint."""
        response = client.get("/interpret/hepatotox?smiles=CCO&top_k=5")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_smiles(self, client):
        """Test with empty SMILES."""
        response = client.post("/predict", json={
            "smiles": "",
            "endpoint": "hepatotox"
        })
        # Empty SMILES - API may process it and return failure in response
        if response.status_code == 200:
            data = response.json()
            # If 200, should indicate failure
            assert data.get("success") is False or "error" in data
        else:
            assert response.status_code in [400, 422]

    def test_very_long_smiles(self, client):
        """Test with a long but valid SMILES."""
        # Taxol - complex molecule
        taxol = "CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
        response = client.post("/predict", json={
            "smiles": taxol,
            "endpoint": "hepatotox"
        })
        # May succeed or fail based on AD, but shouldn't crash
        assert response.status_code in [200, 400]

    def test_special_characters_in_smiles(self, client):
        """Test SMILES with special characters."""
        # Valid SMILES with brackets and charges
        response = client.post("/predict", json={
            "smiles": "[Na+].[Cl-]",  # NaCl
            "endpoint": "hepatotox"
        })
        # May fail for salts, but shouldn't crash
        assert response.status_code in [200, 400]
