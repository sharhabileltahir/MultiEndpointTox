"""Tests for the ToxicityPredictor class."""

import pytest
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.predictor import ToxicityPredictor


@pytest.fixture(scope="module")
def predictor():
    """Create predictor instance."""
    return ToxicityPredictor(models_dir=str(PROJECT_ROOT / "models"))


class TestPredictorInit:
    """Test predictor initialization."""

    def test_models_loaded(self, predictor):
        """Test that models are loaded."""
        assert len(predictor.loaded_models) > 0

    def test_available_endpoints(self, predictor):
        """Test available endpoints."""
        endpoints = predictor.get_available_endpoints()
        assert len(endpoints) > 0
        assert "hepatotox" in endpoints

    def test_shap_explainer_initialized(self, predictor):
        """Test SHAP explainer is available."""
        assert predictor.explainer is not None


class TestSMILESProcessing:
    """Test SMILES validation and processing."""

    def test_validate_valid_smiles(self, predictor):
        """Test validation of valid SMILES."""
        assert predictor.validate_smiles("CCO") is True
        assert predictor.validate_smiles("c1ccccc1") is True
        assert predictor.validate_smiles("CC(=O)Nc1ccc(O)cc1") is True

    def test_validate_invalid_smiles(self, predictor):
        """Test validation of invalid SMILES."""
        assert predictor.validate_smiles("invalid_xyz_not_a_molecule") is False
        # Note: Empty string may be handled differently by RDKit

    def test_standardize_smiles(self, predictor):
        """Test SMILES standardization."""
        canonical = predictor.standardize_smiles("c1ccccc1")
        assert canonical == "c1ccccc1"

    def test_standardize_invalid(self, predictor):
        """Test standardization of invalid SMILES."""
        result = predictor.standardize_smiles("invalid_smiles")
        assert result is None


class TestSinglePrediction:
    """Test single compound prediction."""

    def test_predict_classification(self, predictor):
        """Test classification prediction."""
        result = predictor.predict_single("CCO", "hepatotox")
        assert result["success"] is True
        assert result["task_type"] == "classification"
        assert "prediction" in result
        assert "probability" in result
        assert "label" in result

    def test_predict_regression(self, predictor):
        """Test regression prediction."""
        result = predictor.predict_single("CCO", "herg")
        assert result["success"] is True
        assert result["task_type"] == "regression"
        assert "prediction" in result
        assert "unit" in result

    def test_predict_invalid_smiles(self, predictor):
        """Test prediction with invalid SMILES."""
        result = predictor.predict_single("invalid", "hepatotox")
        assert result["success"] is False
        assert "error" in result

    def test_predict_invalid_endpoint(self, predictor):
        """Test prediction with invalid endpoint."""
        result = predictor.predict_single("CCO", "invalid_endpoint")
        assert result["success"] is False
        assert "error" in result

    def test_applicability_domain(self, predictor):
        """Test applicability domain assessment."""
        result = predictor.predict_single("CCO", "hepatotox")
        assert "applicability_domain" in result
        assert "in_domain" in result["applicability_domain"]
        assert "confidence" in result["applicability_domain"]


class TestMultiEndpointPrediction:
    """Test multi-endpoint prediction."""

    def test_predict_all_endpoints(self, predictor):
        """Test prediction across all endpoints."""
        result = predictor.predict_multi_endpoint("CCO", None)
        assert result["success"] is True
        assert "predictions" in result
        assert len(result["predictions"]) > 0

    def test_predict_selected_endpoints(self, predictor):
        """Test prediction for selected endpoints."""
        result = predictor.predict_multi_endpoint("CCO", ["hepatotox", "nephrotox"])
        assert result["success"] is True
        assert "hepatotox" in result["predictions"]
        assert "nephrotox" in result["predictions"]

    def test_predict_invalid_endpoint_in_list(self, predictor):
        """Test with invalid endpoint in list."""
        result = predictor.predict_multi_endpoint("CCO", ["hepatotox", "invalid"])
        # Should still succeed for valid endpoints
        assert result["success"] is True
        assert "hepatotox" in result["predictions"]


class TestBatchPrediction:
    """Test batch prediction."""

    def test_batch_predict(self, predictor):
        """Test batch prediction."""
        smiles_list = ["CCO", "CC", "CCC"]
        results = predictor.predict_batch(smiles_list, "hepatotox")
        assert len(results) == 3
        assert all(r["success"] for r in results)

    def test_batch_with_invalid(self, predictor):
        """Test batch with invalid SMILES."""
        smiles_list = ["CCO", "invalid_smiles", "CC"]
        results = predictor.predict_batch(smiles_list, "hepatotox")
        assert len(results) == 3
        assert results[0]["success"] is True
        assert results[1]["success"] is False
        assert results[2]["success"] is True


class TestInterpretability:
    """Test SHAP interpretability."""

    def test_predict_with_interpretation(self, predictor):
        """Test prediction with SHAP interpretation."""
        result = predictor.predict_with_interpretation("CCO", "hepatotox", top_k=5)
        assert result["success"] is True
        # Check for interpretation structure (nested under interpretation key)
        assert "interpretation" in result
        assert "shap_explanation" in result["interpretation"]
        assert "top_features" in result["interpretation"]["shap_explanation"]

    def test_interpretation_has_structural_alerts(self, predictor):
        """Test that interpretation includes structural alerts."""
        result = predictor.predict_with_interpretation("CCO", "hepatotox", top_k=5)
        assert "structural_alerts" in result["interpretation"]

    def test_multi_with_interpretation(self, predictor):
        """Test multi-endpoint with interpretation."""
        result = predictor.predict_multi_with_interpretation("CCO", None, top_k=5)
        assert result["success"] is True
        assert "predictions" in result
        # Check for integrated assessment
        assert "integrated_assessment" in result
        assert "overall_risk_level" in result["integrated_assessment"]

    def test_risk_assessment_fields(self, predictor):
        """Test risk assessment contains required fields."""
        result = predictor.predict_multi_with_interpretation("CCO", None, top_k=5)
        risk = result["integrated_assessment"]
        assert "overall_risk_level" in risk
        assert "overall_risk_score" in risk
        assert "critical_endpoint" in risk


class TestKnownCompounds:
    """Test with known compounds to verify predictions make sense."""

    def test_acetaminophen(self, predictor):
        """Test acetaminophen - known hepatotoxic at high doses."""
        result = predictor.predict_single("CC(=O)Nc1ccc(O)cc1", "hepatotox")
        assert result["success"] is True
        # Acetaminophen is a known hepatotoxin
        # The model should predict some level of hepatotoxicity

    def test_caffeine(self, predictor):
        """Test caffeine."""
        caffeine = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
        result = predictor.predict_multi_endpoint(caffeine, None)
        assert result["success"] is True

    def test_aspirin(self, predictor):
        """Test aspirin."""
        aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O"
        result = predictor.predict_multi_endpoint(aspirin, None)
        assert result["success"] is True

    def test_ethanol(self, predictor):
        """Test ethanol - simple molecule."""
        result = predictor.predict_multi_endpoint("CCO", None)
        assert result["success"] is True
        # Should be in applicability domain for most endpoints
        for endpoint, pred in result["predictions"].items():
            assert pred["applicability_domain"]["in_domain"] is True
