"""FastAPI application for toxicity prediction."""

import sys
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.predictor import ToxicityPredictor


# Pydantic models for request/response
class SinglePredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    endpoint: str = Field(..., description="Toxicity endpoint (herg, hepatotox, nephrotox, ames, skin_sens, cytotox)")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "endpoint": "hepatotox"
            }
        }


class MultiEndpointRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    endpoints: Optional[List[str]] = Field(None, description="List of endpoints (None = all available)")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "endpoints": ["hepatotox", "nephrotox", "cytotox"]
            }
        }


class BatchPredictionRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings")
    endpoint: str = Field(..., description="Toxicity endpoint")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles_list": [
                    "CC(=O)Nc1ccc(O)cc1",
                    "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                ],
                "endpoint": "hepatotox"
            }
        }


class SMILESValidationRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string to validate")


class ApplicabilityDomainResult(BaseModel):
    in_domain: bool
    confidence: float


class PredictionResult(BaseModel):
    success: bool
    endpoint: Optional[str] = None
    smiles: Optional[str] = None
    task_type: Optional[str] = None
    prediction: Optional[float] = None
    label: Optional[str] = None
    probability: Optional[float] = None
    unit: Optional[str] = None
    applicability_domain: Optional[ApplicabilityDomainResult] = None
    model: Optional[str] = None
    error: Optional[str] = None


class EndpointInfo(BaseModel):
    name: str
    task: str
    description: str
    available: bool


# Global predictor instance
predictor: Optional[ToxicityPredictor] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize predictor on startup."""
    global predictor
    logger.info("Loading toxicity prediction models...")
    predictor = ToxicityPredictor(models_dir=str(PROJECT_ROOT / "models"))
    available = predictor.get_available_endpoints()
    logger.info(f"Loaded models for endpoints: {available}")
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="MultiEndpointTox API",
    description="""
    Multi-endpoint toxicity prediction API for drug safety assessment.

    ## Available Endpoints

    - **hERG**: Cardiotoxicity (hERG channel inhibition) - Regression
    - **Hepatotox**: Drug-induced liver injury - Classification
    - **Nephrotox**: Drug-induced kidney injury - Classification
    - **Ames**: Ames mutagenicity - Classification
    - **Skin Sensitization**: Skin sensitization potential - Classification
    - **Cytotoxicity**: General cytotoxicity - Classification

    ## Features

    - Single compound prediction
    - Multi-endpoint prediction (all endpoints at once)
    - Batch prediction
    - SMILES validation
    - Applicability domain assessment
    - **SHAP interpretability** (feature importance)
    - **Structural alerts** detection
    - **Design recommendations**
    - **Integrated risk assessment**
    """,
    version="0.3.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """API root - returns basic info."""
    return {
        "name": "MultiEndpointTox API",
        "version": "0.3.0",
        "description": "Multi-endpoint toxicity prediction for drug safety",
        "endpoints": predictor.get_available_endpoints() if predictor else [],
        "features": [
            "Multi-endpoint prediction",
            "SHAP interpretability",
            "Integrated risk assessment",
            "Structural alerts",
            "Design recommendations"
        ],
        "shap_available": predictor.explainer is not None if predictor else False,
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": len(predictor.loaded_models) if predictor else 0,
        "available_endpoints": predictor.get_available_endpoints() if predictor else [],
    }


@app.get("/endpoints", tags=["Endpoints"], response_model=List[EndpointInfo])
async def list_endpoints():
    """List all toxicity endpoints and their availability."""
    all_endpoints = ToxicityPredictor.ENDPOINTS
    available = predictor.get_available_endpoints() if predictor else []

    return [
        EndpointInfo(
            name=name,
            task=info["task"],
            description=info["description"],
            available=name in available,
        )
        for name, info in all_endpoints.items()
    ]


@app.post("/predict", tags=["Prediction"], response_model=PredictionResult)
async def predict_single(request: SinglePredictionRequest):
    """
    Predict toxicity for a single compound on a single endpoint.

    - **smiles**: SMILES string of the compound
    - **endpoint**: Toxicity endpoint name
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = predictor.predict_single(request.smiles, request.endpoint)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.post("/predict/multi", tags=["Prediction"])
async def predict_multi_endpoint(request: MultiEndpointRequest):
    """
    Predict toxicity for a single compound across multiple endpoints.

    - **smiles**: SMILES string of the compound
    - **endpoints**: List of endpoints (optional, defaults to all available)
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = predictor.predict_multi_endpoint(request.smiles, request.endpoints)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict toxicity for multiple compounds on a single endpoint.

    - **smiles_list**: List of SMILES strings
    - **endpoint**: Toxicity endpoint name
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    if len(request.smiles_list) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 compounds per batch")

    results = predictor.predict_batch(request.smiles_list, request.endpoint)

    return {
        "endpoint": request.endpoint,
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success")),
        "results": results,
    }


@app.get("/predict/{endpoint}", tags=["Prediction"])
async def predict_get(
    endpoint: str,
    smiles: str = Query(..., description="SMILES string of the compound")
):
    """
    Predict toxicity using GET request.

    Convenient for simple queries via browser or curl.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = predictor.predict_single(smiles, endpoint)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.post("/validate", tags=["Utilities"])
async def validate_smiles(request: SMILESValidationRequest):
    """
    Validate a SMILES string.

    Returns whether the SMILES is valid and the standardized canonical form.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    is_valid = predictor.validate_smiles(request.smiles)
    std_smiles = predictor.standardize_smiles(request.smiles) if is_valid else None

    return {
        "input": request.smiles,
        "valid": is_valid,
        "canonical_smiles": std_smiles,
    }


@app.get("/validate", tags=["Utilities"])
async def validate_smiles_get(
    smiles: str = Query(..., description="SMILES string to validate")
):
    """Validate SMILES via GET request."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    is_valid = predictor.validate_smiles(smiles)
    std_smiles = predictor.standardize_smiles(smiles) if is_valid else None

    return {
        "input": smiles,
        "valid": is_valid,
        "canonical_smiles": std_smiles,
    }


# ============================================================================
# INTERPRETABILITY ENDPOINTS
# ============================================================================

class InterpretationRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    endpoint: str = Field(..., description="Toxicity endpoint")
    top_k: int = Field(10, description="Number of top features to explain")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "endpoint": "hepatotox",
                "top_k": 10
            }
        }


class IntegratedPredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    endpoints: Optional[List[str]] = Field(None, description="Endpoints (None = all)")
    top_k: int = Field(10, description="Number of top features per endpoint")
    include_interpretation: bool = Field(True, description="Include SHAP analysis")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "endpoints": None,
                "top_k": 10,
                "include_interpretation": True
            }
        }


@app.post("/predict/interpret", tags=["Interpretability"])
async def predict_with_interpretation(request: InterpretationRequest):
    """
    Predict toxicity with SHAP feature importance explanation.

    Returns:
    - Prediction results
    - Top contributing features
    - Feature categories (physicochemical, fingerprints, substructures)
    - Human-readable interpretation
    - Structural alerts
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = predictor.predict_with_interpretation(
        request.smiles,
        request.endpoint,
        request.top_k
    )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.post("/predict/integrated", tags=["Interpretability"])
async def predict_integrated(request: IntegratedPredictionRequest):
    """
    Comprehensive multi-endpoint prediction with full interpretability.

    Returns:
    - Predictions for all endpoints
    - Integrated risk assessment (overall risk score, critical endpoint)
    - SHAP explanations per endpoint
    - Cross-endpoint feature analysis
    - Structural alerts
    - Design recommendations
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    if request.include_interpretation:
        result = predictor.predict_multi_with_interpretation(
            request.smiles,
            request.endpoints,
            request.top_k
        )
    else:
        result = predictor.predict_multi_endpoint(
            request.smiles,
            request.endpoints
        )

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.get("/predict/integrated/{smiles}", tags=["Interpretability"])
async def predict_integrated_get(
    smiles: str,
    top_k: int = Query(10, description="Number of top features")
):
    """
    Quick integrated prediction via GET request.

    Returns all endpoints with interpretation.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = predictor.predict_multi_with_interpretation(smiles, None, top_k)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.get("/interpret/{endpoint}", tags=["Interpretability"])
async def interpret_get(
    endpoint: str,
    smiles: str = Query(..., description="SMILES string"),
    top_k: int = Query(10, description="Number of top features")
):
    """
    Get SHAP interpretation via GET request.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    result = predictor.predict_with_interpretation(smiles, endpoint, top_k)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Prediction failed"))

    return result


@app.get("/shap/status", tags=["Interpretability"])
async def shap_status():
    """Check SHAP availability and explainer status."""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    return {
        "shap_available": predictor.explainer is not None,
        "explainers_initialized": list(predictor.explainer.explainers.keys()) if predictor.explainer else [],
        "endpoints_available": predictor.get_available_endpoints(),
    }


# Run with: uvicorn src.api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
