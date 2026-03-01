"""Core prediction engine for toxicity endpoints."""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from loguru import logger

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

# Import explainer (optional SHAP support)
try:
    from src.api.explainer import ToxicityExplainer, SHAP_AVAILABLE
except ImportError:
    SHAP_AVAILABLE = False
    ToxicityExplainer = None

# RDKit descriptor list
RDKIT_DESCRIPTORS = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumHeteroatoms', 'NumAliphaticRings', 'NumAromaticRings',
    'NumSaturatedRings', 'NumAliphaticHeterocycles', 'NumAromaticHeterocycles',
    'NumSaturatedHeterocycles', 'NumAliphaticCarbocycles', 'NumAromaticCarbocycles',
    'NumSaturatedCarbocycles', 'RingCount', 'FractionCSP3',
    'HeavyAtomCount', 'NHOHCount', 'NOCount',
    'NumHeavyAtoms', 'NumValenceElectrons',
]


class ToxicityPredictor:
    """
    Multi-endpoint toxicity predictor.

    Supports:
    - hERG (cardiotoxicity) - Regression
    - Hepatotoxicity - Classification
    - Nephrotoxicity - Classification
    - Ames mutagenicity - Classification
    - Skin sensitization - Classification
    - Cytotoxicity - Classification
    """

    ENDPOINTS = {
        "herg": {"task": "regression", "description": "hERG channel inhibition (cardiotoxicity)"},
        "hepatotox": {"task": "classification", "description": "Drug-induced liver injury"},
        "nephrotox": {"task": "classification", "description": "Drug-induced kidney injury"},
        "ames": {"task": "classification", "description": "Ames mutagenicity"},
        "skin_sens": {"task": "classification", "description": "Skin sensitization"},
        "cytotox": {"task": "classification", "description": "Cytotoxicity"},
    }

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.loaded_models: Dict[str, Dict] = {}
        self._load_available_models()

        # Initialize explainer for SHAP interpretability
        self.explainer = None
        if ToxicityExplainer is not None and SHAP_AVAILABLE:
            try:
                self.explainer = ToxicityExplainer(models_dir=models_dir)
                logger.info("SHAP explainer initialized")
            except Exception as e:
                logger.warning(f"Could not initialize SHAP explainer: {e}")

    def _load_available_models(self):
        """Load all available trained models."""
        for endpoint in self.ENDPOINTS:
            endpoint_dir = self.models_dir / endpoint
            if endpoint_dir.exists():
                try:
                    self._load_endpoint_model(endpoint)
                except Exception as e:
                    logger.warning(f"Could not load {endpoint} model: {e}")

    def _load_endpoint_model(self, endpoint: str):
        """Load model and associated files for an endpoint."""
        endpoint_dir = self.models_dir / endpoint

        # Find best model (optimized > baseline)
        model_path = None
        for suffix in ["_optimized.pkl", "_baseline.pkl"]:
            candidates = list(endpoint_dir.glob(f"*{suffix}"))
            if candidates:
                # Prefer certain model types
                for pref in ["xgboost", "randomforest", "lightgbm", "svc", "svr"]:
                    for c in candidates:
                        if pref in c.stem.lower():
                            model_path = c
                            break
                    if model_path:
                        break
                if not model_path:
                    model_path = candidates[0]
                break

        if not model_path:
            raise FileNotFoundError(f"No model found for {endpoint}")

        # Load model using joblib (more robust for sklearn models)
        model = joblib.load(model_path)

        # Load scaler
        scaler = None
        scaler_path = endpoint_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        # Load feature selector
        feature_selector = None
        fs_path = endpoint_dir / "feature_selection.pkl"
        if fs_path.exists():
            feature_selector = joblib.load(fs_path)

        # Load applicability domain
        ad = None
        ad_path = endpoint_dir / "applicability_domain.pkl"
        if ad_path.exists():
            ad = joblib.load(ad_path)

        # Determine task type
        task_type_path = endpoint_dir / "task_type.txt"
        if task_type_path.exists():
            task_type = task_type_path.read_text().strip()
        else:
            task_type = self.ENDPOINTS.get(endpoint, {}).get("task", "classification")

        self.loaded_models[endpoint] = {
            "model": model,
            "scaler": scaler,
            "feature_selector": feature_selector,
            "applicability_domain": ad,
            "task_type": task_type,
            "model_name": model_path.stem,
        }

        logger.info(f"Loaded {endpoint} model: {model_path.stem}")

    def get_available_endpoints(self) -> List[str]:
        """Get list of available endpoints with trained models."""
        return list(self.loaded_models.keys())

    def validate_smiles(self, smiles: str) -> bool:
        """Check if SMILES is valid."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False

    def standardize_smiles(self, smiles: str) -> Optional[str]:
        """Standardize SMILES to canonical form."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None

    def _compute_features(self, smiles: str) -> Optional[np.ndarray]:
        """Compute molecular features for a single SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

            # Compute descriptors
            descriptors = []
            for desc_name in RDKIT_DESCRIPTORS:
                try:
                    func = getattr(Descriptors, desc_name, None)
                    if func:
                        descriptors.append(func(mol))
                    else:
                        descriptors.append(0)
                except:
                    descriptors.append(0)

            # Compute Morgan fingerprint
            morgan_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            morgan_bits = list(morgan_fp)

            # Compute MACCS keys
            maccs_fp = MACCSkeys.GenMACCSKeys(mol)
            maccs_bits = list(maccs_fp)

            # Combine all features
            all_features = descriptors + morgan_bits + maccs_bits

            return np.array(all_features).reshape(1, -1)

        except Exception as e:
            logger.error(f"Feature computation failed for {smiles}: {e}")
            return None

    def _apply_feature_selection(self, features: np.ndarray, endpoint: str) -> np.ndarray:
        """Apply feature selection if available."""
        model_info = self.loaded_models.get(endpoint)
        if not model_info:
            return features

        fs = model_info.get("feature_selector")
        if fs is None:
            return features

        try:
            # Handle different feature selector types
            if hasattr(fs, "selected_indices"):
                return features[:, fs.selected_indices]
            elif hasattr(fs, "transform"):
                return fs.transform(features)
            elif isinstance(fs, dict):
                if "selected_indices" in fs:
                    return features[:, fs["selected_indices"]]
                elif "selected_columns" in fs:
                    # selected_columns contains column names - we need indices
                    # The columns are named based on our feature order
                    n_selected = fs.get("n_selected_features", 500)
                    # Take first n_selected features (they should be in order)
                    return features[:, :n_selected]
            elif isinstance(fs, np.ndarray):
                return features[:, fs]
        except Exception as e:
            logger.warning(f"Feature selection failed for {endpoint}: {e}")

        return features

    def _apply_scaling(self, features: np.ndarray, endpoint: str) -> np.ndarray:
        """Apply feature scaling if available."""
        model_info = self.loaded_models.get(endpoint)
        if not model_info:
            return features

        scaler = model_info.get("scaler")
        if scaler is None:
            return features

        try:
            # Handle different scaler types
            if hasattr(scaler, "transform"):
                return scaler.transform(features)
            elif isinstance(scaler, dict):
                # Dict-based scaler with mean/std
                if "mean" in scaler and "std" in scaler:
                    mean = np.array(scaler["mean"])
                    std = np.array(scaler["std"])
                    std[std == 0] = 1  # Avoid division by zero
                    return (features - mean) / std
                return features
            else:
                return features
        except Exception as e:
            logger.warning(f"Scaling failed: {e}")
            return features

    def _check_applicability_domain(self, features: np.ndarray, endpoint: str) -> Dict[str, Any]:
        """Check if compound is within applicability domain."""
        model_info = self.loaded_models.get(endpoint)
        if not model_info:
            return {"in_domain": True, "confidence": 1.0}

        ad = model_info.get("applicability_domain")
        if ad is None:
            return {"in_domain": True, "confidence": 1.0}

        try:
            if hasattr(ad, "predict"):
                # sklearn-style AD
                in_domain = ad.predict(features)[0] == 1
                confidence = 1.0 if in_domain else 0.5
            elif hasattr(ad, "assess"):
                # Custom AD
                result = ad.assess(features)
                in_domain = result.get("in_domain", True)
                confidence = result.get("confidence", 1.0)
            else:
                in_domain = True
                confidence = 1.0

            return {"in_domain": bool(in_domain), "confidence": float(confidence)}
        except:
            return {"in_domain": True, "confidence": 1.0}

    def predict_single(self, smiles: str, endpoint: str) -> Dict[str, Any]:
        """
        Make prediction for a single compound on a single endpoint.

        Args:
            smiles: SMILES string
            endpoint: Endpoint name (herg, hepatotox, etc.)

        Returns:
            Dictionary with prediction results
        """
        # Validate endpoint
        if endpoint not in self.loaded_models:
            return {
                "success": False,
                "error": f"Endpoint '{endpoint}' not available. Available: {self.get_available_endpoints()}",
            }

        # Validate SMILES
        std_smiles = self.standardize_smiles(smiles)
        if std_smiles is None:
            return {
                "success": False,
                "error": f"Invalid SMILES: {smiles}",
            }

        # Compute features
        features = self._compute_features(std_smiles)
        if features is None:
            return {
                "success": False,
                "error": "Feature computation failed",
            }

        model_info = self.loaded_models[endpoint]

        # Apply feature selection
        features = self._apply_feature_selection(features, endpoint)

        # Apply scaling
        features = self._apply_scaling(features, endpoint)

        # Check applicability domain
        ad_result = self._check_applicability_domain(features, endpoint)

        # Make prediction
        model = model_info["model"]
        task_type = model_info["task_type"]

        try:
            if task_type == "regression":
                prediction = float(model.predict(features)[0])
                result = {
                    "success": True,
                    "endpoint": endpoint,
                    "smiles": std_smiles,
                    "task_type": "regression",
                    "prediction": prediction,
                    "unit": "pIC50" if endpoint == "herg" else "value",
                    "applicability_domain": ad_result,
                    "model": model_info["model_name"],
                }
            else:
                # Classification
                prediction = int(model.predict(features)[0])

                # Get probability if available
                probability = None
                if hasattr(model, "predict_proba"):
                    try:
                        proba = model.predict_proba(features)[0]
                        probability = float(proba[1]) if len(proba) > 1 else float(proba[0])
                    except:
                        pass

                # Interpret prediction
                if endpoint == "herg":
                    label = "Cardiotoxic" if prediction == 1 else "Non-cardiotoxic"
                elif endpoint == "hepatotox":
                    label = "Hepatotoxic" if prediction == 1 else "Non-hepatotoxic"
                elif endpoint == "nephrotox":
                    label = "Nephrotoxic" if prediction == 1 else "Non-nephrotoxic"
                elif endpoint == "ames":
                    label = "Mutagenic" if prediction == 1 else "Non-mutagenic"
                elif endpoint == "skin_sens":
                    label = "Sensitizer" if prediction == 1 else "Non-sensitizer"
                elif endpoint == "cytotox":
                    label = "Cytotoxic" if prediction == 1 else "Non-cytotoxic"
                else:
                    label = "Positive" if prediction == 1 else "Negative"

                result = {
                    "success": True,
                    "endpoint": endpoint,
                    "smiles": std_smiles,
                    "task_type": "classification",
                    "prediction": prediction,
                    "label": label,
                    "probability": probability,
                    "applicability_domain": ad_result,
                    "model": model_info["model_name"],
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction failed: {str(e)}",
            }

    def predict_multi_endpoint(self, smiles: str, endpoints: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Make predictions for a single compound across multiple endpoints.

        Args:
            smiles: SMILES string
            endpoints: List of endpoints (None = all available)

        Returns:
            Dictionary with all predictions
        """
        if endpoints is None:
            endpoints = self.get_available_endpoints()

        std_smiles = self.standardize_smiles(smiles)
        if std_smiles is None:
            return {
                "success": False,
                "error": f"Invalid SMILES: {smiles}",
            }

        results = {
            "success": True,
            "smiles": std_smiles,
            "predictions": {},
        }

        for endpoint in endpoints:
            pred = self.predict_single(std_smiles, endpoint)
            if pred["success"]:
                results["predictions"][endpoint] = {
                    k: v for k, v in pred.items()
                    if k not in ["success", "smiles", "endpoint"]
                }
            else:
                results["predictions"][endpoint] = {"error": pred.get("error", "Unknown error")}

        return results

    def predict_batch(self, smiles_list: List[str], endpoint: str) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple compounds on a single endpoint.

        Args:
            smiles_list: List of SMILES strings
            endpoint: Endpoint name

        Returns:
            List of prediction results
        """
        return [self.predict_single(smiles, endpoint) for smiles in smiles_list]

    def predict_with_interpretation(
        self,
        smiles: str,
        endpoint: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Make prediction with SHAP interpretation.

        Args:
            smiles: SMILES string
            endpoint: Endpoint name
            top_k: Number of top features to explain

        Returns:
            Prediction with feature importance explanation
        """
        # Get base prediction
        prediction = self.predict_single(smiles, endpoint)

        if not prediction["success"]:
            return prediction

        # Add interpretation if available
        if self.explainer is None:
            prediction["interpretation"] = {
                "available": False,
                "reason": "SHAP not available"
            }
            return prediction

        try:
            # Compute features for interpretation
            std_smiles = self.standardize_smiles(smiles)
            features = self._compute_features(std_smiles)

            if features is None:
                prediction["interpretation"] = {
                    "available": False,
                    "reason": "Feature computation failed"
                }
                return prediction

            # Apply feature selection
            features_selected = self._apply_feature_selection(features, endpoint)

            # Apply scaling
            features_scaled = self._apply_scaling(features_selected, endpoint)

            # Get model
            model = self.loaded_models[endpoint]["model"]

            # Get SHAP explanation
            explanation = self.explainer.explain_prediction(
                endpoint=endpoint,
                features=features_scaled,
                model=model,
                top_k=top_k
            )

            # Get structural alerts
            structural_alerts = self.explainer.get_structural_alerts(
                std_smiles,
                {endpoint: prediction}
            )

            prediction["interpretation"] = {
                "available": True,
                "shap_explanation": explanation,
                "structural_alerts": structural_alerts,
            }

        except Exception as e:
            logger.error(f"Interpretation failed: {e}")
            prediction["interpretation"] = {
                "available": False,
                "reason": str(e)
            }

        return prediction

    def predict_multi_with_interpretation(
        self,
        smiles: str,
        endpoints: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Multi-endpoint prediction with full interpretation.

        Args:
            smiles: SMILES string
            endpoints: List of endpoints (None = all)
            top_k: Number of top features per endpoint

        Returns:
            Complete prediction with cross-endpoint analysis
        """
        if endpoints is None:
            endpoints = self.get_available_endpoints()

        std_smiles = self.standardize_smiles(smiles)
        if std_smiles is None:
            return {
                "success": False,
                "error": f"Invalid SMILES: {smiles}"
            }

        # Compute features once
        features = self._compute_features(std_smiles)
        if features is None:
            return {
                "success": False,
                "error": "Feature computation failed"
            }

        results = {
            "success": True,
            "smiles": std_smiles,
            "predictions": {},
            "interpretations": {},
            "integrated_assessment": {},
        }

        # Get predictions and interpretations for each endpoint
        for endpoint in endpoints:
            pred = self.predict_with_interpretation(std_smiles, endpoint, top_k)
            if pred["success"]:
                results["predictions"][endpoint] = {
                    k: v for k, v in pred.items()
                    if k not in ["success", "smiles", "endpoint", "interpretation"]
                }
                if "interpretation" in pred:
                    results["interpretations"][endpoint] = pred["interpretation"]
            else:
                results["predictions"][endpoint] = {"error": pred.get("error")}

        # Generate integrated assessment
        results["integrated_assessment"] = self._generate_integrated_assessment(
            results["predictions"],
            std_smiles
        )

        # Cross-endpoint analysis (simplified to avoid crashes)
        if self.explainer is not None:
            try:
                # Get structural alerts (lightweight, no SHAP needed)
                results["structural_alerts"] = self.explainer.get_structural_alerts(
                    std_smiles,
                    results["predictions"]
                )

                # Simple cross-endpoint analysis from existing interpretations
                results["cross_endpoint_analysis"] = {
                    "common_features": [],
                    "insights": []
                }

                # Generate design recommendations based on structural alerts
                results["design_recommendations"] = self.explainer.generate_design_recommendations(
                    {"endpoint_explanations": results.get("interpretations", {})},
                    results["structural_alerts"]
                )

            except Exception as e:
                logger.error(f"Cross-endpoint analysis failed: {e}")
                results["cross_endpoint_analysis"] = {"error": str(e)}
                results["structural_alerts"] = []
                results["design_recommendations"] = []

        return results

    def _generate_integrated_assessment(
        self,
        predictions: Dict[str, Any],
        smiles: str
    ) -> Dict[str, Any]:
        """
        Generate integrated risk assessment across all endpoints.

        Args:
            predictions: Dictionary of endpoint predictions
            smiles: SMILES string

        Returns:
            Integrated risk assessment
        """
        risk_scores = {}
        risk_levels = {}

        for endpoint, pred in predictions.items():
            if not isinstance(pred, dict):
                continue
            if "error" in pred:
                continue

            task_type = pred.get("task_type", "classification")

            if task_type == "classification":
                # Use probability for risk score, fallback to prediction
                prob = pred.get("probability")
                if prob is not None:
                    risk_scores[endpoint] = float(prob)
                else:
                    # Use prediction (0 or 1) as fallback
                    prediction = pred.get("prediction", 0)
                    risk_scores[endpoint] = float(prediction) if prediction is not None else 0.5
            else:
                # Regression - normalize based on typical range
                value = pred.get("prediction", 0)
                # For hERG pIC50: higher = more potent inhibitor = higher risk
                if endpoint == "herg":
                    # pIC50 > 6 is concerning, > 7 is high risk
                    risk_scores[endpoint] = min(1.0, max(0.0, (value - 5) / 3))
                else:
                    risk_scores[endpoint] = min(1.0, max(0.0, value))

            # Determine risk level
            score = risk_scores[endpoint]
            if score < 0.3:
                risk_levels[endpoint] = "low"
            elif score < 0.7:
                risk_levels[endpoint] = "moderate"
            else:
                risk_levels[endpoint] = "high"

        if not risk_scores:
            return {"error": "No valid predictions"}

        # Calculate overall risk
        overall_risk = max(risk_scores.values()) if risk_scores else 0.0

        # Rank endpoints by risk
        sorted_endpoints = sorted(
            risk_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Determine critical endpoint
        if sorted_endpoints:
            critical_endpoint = sorted_endpoints[0][0]
            critical_score = sorted_endpoints[0][1]
        else:
            critical_endpoint = None
            critical_score = 0.0

        # Overall risk level
        if overall_risk < 0.3:
            overall_level = "low"
            recommendation = "Compound shows low toxicity risk across endpoints"
        elif overall_risk < 0.7:
            overall_level = "moderate"
            recommendation = f"Monitor {critical_endpoint} - moderate risk detected"
        else:
            overall_level = "high"
            recommendation = f"Caution: High {critical_endpoint} risk - consider structural optimization"

        return {
            "overall_risk_score": round(overall_risk, 4),
            "overall_risk_level": overall_level,
            "critical_endpoint": critical_endpoint,
            "critical_endpoint_score": round(critical_score, 4) if critical_score else None,
            "endpoint_risk_scores": {k: round(v, 4) for k, v in risk_scores.items()},
            "endpoint_risk_levels": risk_levels,
            "endpoint_ranking": [ep for ep, _ in sorted_endpoints],
            "recommendation": recommendation
        }
