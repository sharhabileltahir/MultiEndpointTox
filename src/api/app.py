"""FastAPI application for toxicity prediction."""

import sys
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.predictor import ToxicityPredictor

# Import docking module (optional)
try:
    from src.docking import DockingManager, check_docking_dependencies, get_installation_instructions
    DOCKING_AVAILABLE = True
except ImportError:
    DOCKING_AVAILABLE = False
    DockingManager = None
    check_docking_dependencies = None
    get_installation_instructions = None
    logger.warning("Docking module not available")


# Pydantic models for request/response
class SinglePredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    endpoint: str = Field(..., description="Toxicity endpoint (herg, hepatotox, nephrotox, ames, skin_sens, cytotox, reproductive_tox)")

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


# Global predictor and docking instances
predictor: Optional[ToxicityPredictor] = None
docking_manager: Optional[Any] = None


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize predictor and docking on startup."""
    global predictor, docking_manager

    # Load config
    config = load_config()

    # Initialize ML predictor
    logger.info("Loading toxicity prediction models...")
    predictor = ToxicityPredictor(models_dir=str(PROJECT_ROOT / "models"))
    available = predictor.get_available_endpoints()
    logger.info(f"Loaded models for endpoints: {available}")

    # Initialize docking manager (if available and enabled)
    if DOCKING_AVAILABLE:
        docking_config = config.get("docking", {})
        if docking_config.get("enabled", False):
            try:
                docking_manager = DockingManager(
                    config=config,
                    structures_dir=str(PROJECT_ROOT / docking_config.get("structures_dir", "data/structures")),
                    engine=docking_config.get("engine", "vina")
                )
                logger.info(f"Docking manager initialized (engine: {docking_config.get('engine', 'vina')})")
            except Exception as e:
                logger.warning(f"Failed to initialize docking: {e}")
                docking_manager = None
        else:
            logger.info("Docking is disabled in config")
    else:
        logger.info("Docking module not available")

    yield

    # Cleanup
    if docking_manager:
        docking_manager.cleanup()
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
    - **Reproductive Tox**: Reproductive/developmental toxicity - Classification

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
    - **Molecular docking** (AutoDock Vina integration)
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


# ============================================================================
# MOLECULAR DOCKING ENDPOINTS
# ============================================================================

class DockingRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    target: str = Field(..., description="Protein target (herg, hepatotox, cyp3a4, etc.)")
    exhaustiveness: Optional[int] = Field(8, description="Docking exhaustiveness (1-32)")
    n_poses: Optional[int] = Field(9, description="Number of poses to generate")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "target": "herg",
                "exhaustiveness": 8,
                "n_poses": 9
            }
        }


class BatchDockingRequest(BaseModel):
    smiles_list: List[str] = Field(..., description="List of SMILES strings")
    target: str = Field(..., description="Protein target")
    exhaustiveness: Optional[int] = Field(8, description="Docking exhaustiveness")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles_list": [
                    "CC(=O)Nc1ccc(O)cc1",
                    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
                ],
                "target": "herg"
            }
        }


class EnsemblePredictionRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string")
    endpoint: str = Field(..., description="Toxicity endpoint")
    include_docking: bool = Field(True, description="Include docking in prediction")
    ml_weight: Optional[float] = Field(0.7, description="Weight for ML prediction (0-1)")
    docking_weight: Optional[float] = Field(0.3, description="Weight for docking score (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "endpoint": "herg",
                "include_docking": True,
                "ml_weight": 0.7,
                "docking_weight": 0.3
            }
        }


@app.get("/docking/status", tags=["Docking"])
async def docking_status():
    """Check docking availability and configuration."""
    if check_docking_dependencies:
        deps = check_docking_dependencies()
    else:
        deps = {"ready": False}

    if not docking_manager:
        return {
            "available": False,
            "dependencies": deps,
            "reason": "Docking not enabled or dependencies missing",
            "install_instructions": get_installation_instructions() if get_installation_instructions else (
                "Install docking dependencies:\n"
                "1. Download Vina from https://vina.scripps.edu/downloads/\n"
                "2. pip install meeko\n"
                "3. Set docking.enabled=true in config/config.yaml"
            )
        }

    status = docking_manager.get_status()
    status["dependencies"] = deps
    return status


@app.get("/docking/targets", tags=["Docking"])
async def list_docking_targets():
    """List available protein targets for docking."""
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking not available")

    return {
        "targets": docking_manager.structure_manager.get_available_targets(),
        "endpoint_mappings": docking_manager.ENDPOINT_TARGETS,
    }


@app.post("/dock", tags=["Docking"])
async def dock_compound(request: DockingRequest):
    """
    Dock a compound against a protein target.

    Returns binding affinity (kcal/mol) and pose information.
    More negative affinity = stronger binding = higher toxicity risk.

    Typical interpretation:
    - < -8 kcal/mol: Strong binder (high risk)
    - -6 to -8: Moderate binder
    - > -6: Weak binder (low risk)
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking not available")

    if not docking_manager.is_available():
        raise HTTPException(
            status_code=503,
            detail="Docking engine not available. Install AutoDock Vina."
        )

    result = docking_manager.dock(
        smiles=request.smiles,
        target=request.target,
        exhaustiveness=request.exhaustiveness,
        n_poses=request.n_poses,
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Docking failed")

    return result.to_dict()


@app.post("/dock/batch", tags=["Docking"])
async def dock_batch(request: BatchDockingRequest):
    """
    Dock multiple compounds against a protein target.

    Processes compounds in parallel for efficiency.
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking not available")

    if len(request.smiles_list) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 compounds per batch")

    results = docking_manager.dock_batch(
        smiles_list=request.smiles_list,
        target=request.target,
        exhaustiveness=request.exhaustiveness,
    )

    return {
        "target": request.target,
        "total": len(results),
        "successful": sum(1 for r in results if r.success),
        "results": [r.to_dict() for r in results],
    }


@app.get("/dock/{target}", tags=["Docking"])
async def dock_get(
    target: str,
    smiles: str = Query(..., description="SMILES string"),
    exhaustiveness: int = Query(8, description="Docking exhaustiveness")
):
    """Dock compound via GET request."""
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking not available")

    result = docking_manager.dock(
        smiles=smiles,
        target=target,
        exhaustiveness=exhaustiveness,
    )

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Docking failed")

    return result.to_dict()


@app.post("/predict/ensemble", tags=["Docking"])
async def predict_ensemble(request: EnsemblePredictionRequest):
    """
    Ensemble prediction combining ML model and molecular docking.

    Provides more robust toxicity assessment by combining:
    - Machine learning prediction (trained on experimental data)
    - Molecular docking score (physics-based binding affinity)

    The ensemble score is a weighted combination of both approaches.
    """
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")

    # Get ML prediction
    ml_result = predictor.predict_single(request.smiles, request.endpoint)

    if not ml_result["success"]:
        raise HTTPException(status_code=400, detail=ml_result.get("error", "ML prediction failed"))

    response = {
        "smiles": ml_result.get("smiles", request.smiles),
        "endpoint": request.endpoint,
        "ml_prediction": {
            k: v for k, v in ml_result.items()
            if k not in ["success", "smiles", "endpoint"]
        },
    }

    # Add docking if requested and available
    if request.include_docking and docking_manager and docking_manager.is_available():
        docking_results = docking_manager.dock_for_endpoint(
            request.smiles,
            request.endpoint
        )

        if docking_results:
            # Use primary target for ensemble
            primary_target = list(docking_results.keys())[0]
            primary_docking = docking_results[primary_target]

            response["docking"] = {
                target: result.to_dict()
                for target, result in docking_results.items()
            }

            # Compute ensemble score
            ensemble = docking_manager.compute_ensemble_score(
                ml_prediction=ml_result,
                docking_result=primary_docking,
                ml_weight=request.ml_weight or 0.7,
                docking_weight=request.docking_weight or 0.3,
            )
            response["ensemble"] = ensemble
        else:
            response["docking"] = {"message": "No docking targets configured for this endpoint"}
            response["ensemble"] = None
    else:
        response["docking"] = None
        response["ensemble"] = None

    response["success"] = True
    return response


@app.post("/predict/with-docking", tags=["Docking"])
async def predict_with_docking(request: EnsemblePredictionRequest):
    """
    Alias for /predict/ensemble - comprehensive prediction with docking.
    """
    return await predict_ensemble(request)


# ============================================================================
# 3D DESCRIPTORS & PHARMACOPHORE ENDPOINTS
# ============================================================================

class Descriptors3DRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1"
            }
        }


class EnhancedDockingRequest(BaseModel):
    smiles: str = Field(..., description="SMILES string of the compound")
    target: str = Field(..., description="Protein target")
    exhaustiveness: Optional[int] = Field(8, description="Docking exhaustiveness")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles": "CC(=O)Nc1ccc(O)cc1",
                "target": "herg",
                "exhaustiveness": 8
            }
        }


class PharmacophoreCompareRequest(BaseModel):
    smiles1: str = Field(..., description="First SMILES string")
    smiles2: str = Field(..., description="Second SMILES string")

    class Config:
        json_schema_extra = {
            "example": {
                "smiles1": "CC(=O)Nc1ccc(O)cc1",
                "smiles2": "CC(=O)OC1=CC=CC=C1C(=O)O"
            }
        }


@app.post("/descriptors/3d", tags=["3D Descriptors"])
async def calculate_3d_descriptors(request: Descriptors3DRequest):
    """
    Calculate 3D molecular descriptors and pharmacophore features.

    Returns:
    - Shape descriptors (asphericity, eccentricity, PMI ratios)
    - Surface descriptors (TPSA, SASA)
    - Volume descriptors
    - Pharmacophore feature counts and positions
    - Conformer energy
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking module not available")

    result = docking_manager.calculate_3d_descriptors(request.smiles)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Failed to calculate 3D descriptors")

    return result.to_dict()


@app.get("/descriptors/3d", tags=["3D Descriptors"])
async def get_3d_descriptors(
    smiles: str = Query(..., description="SMILES string")
):
    """Get 3D descriptors via GET request."""
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking module not available")

    result = docking_manager.calculate_3d_descriptors(smiles)

    if not result.success:
        raise HTTPException(status_code=400, detail=result.error or "Failed to calculate 3D descriptors")

    return result.to_dict()


@app.post("/pharmacophore/features", tags=["3D Descriptors"])
async def get_pharmacophore_features(request: Descriptors3DRequest):
    """
    Extract pharmacophore features from a compound.

    Returns:
    - Feature type counts (HBA, HBD, aromatic, hydrophobic, ionizable)
    - 3D positions of each feature
    - Feature atom indices
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking module not available")

    result = docking_manager.get_pharmacophore_features(request.smiles)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to extract pharmacophore features"))

    return result


@app.post("/pharmacophore/compare", tags=["3D Descriptors"])
async def compare_pharmacophores(request: PharmacophoreCompareRequest):
    """
    Compare pharmacophore profiles of two compounds.

    Returns:
    - Pharmacophore fingerprint similarity (Tanimoto)
    - Feature count comparison
    - Shape comparison
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking module not available")

    result = docking_manager.compare_pharmacophores(request.smiles1, request.smiles2)

    if not result.get("success"):
        raise HTTPException(status_code=400, detail=result.get("error", "Failed to compare pharmacophores"))

    return result


@app.post("/dock/enhanced", tags=["3D Descriptors"])
async def dock_enhanced(request: EnhancedDockingRequest):
    """
    Enhanced docking with 3D descriptors and binding compatibility analysis.

    Returns:
    - Standard docking results (affinity, poses)
    - 3D molecular descriptors
    - Binding compatibility analysis (shape, size, pharmacophore match)
    - Enhanced docking score incorporating all factors
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking not available")

    if not docking_manager.is_available():
        raise HTTPException(
            status_code=503,
            detail="Docking engine not available. Install AutoDock Vina."
        )

    result = docking_manager.dock_with_descriptors(
        smiles=request.smiles,
        target=request.target,
        exhaustiveness=request.exhaustiveness,
    )

    if not result.get("docking", {}).get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("docking", {}).get("error", "Enhanced docking failed")
        )

    return result


@app.post("/dock/endpoint/enhanced", tags=["3D Descriptors"])
async def dock_endpoint_enhanced(request: EnsemblePredictionRequest):
    """
    Enhanced docking for all targets relevant to a toxicity endpoint.

    Includes 3D descriptors and binding compatibility for each target.
    """
    if not docking_manager:
        raise HTTPException(status_code=503, detail="Docking not available")

    result = docking_manager.dock_for_endpoint_enhanced(
        smiles=request.smiles,
        endpoint=request.endpoint,
    )

    return result


# Run with: uvicorn src.api.app:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
