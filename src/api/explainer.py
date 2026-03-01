"""SHAP-based model interpretability for toxicity predictions."""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors


# Feature names for interpretability
RDKIT_DESCRIPTOR_NAMES = [
    'MolWt', 'MolLogP', 'MolMR', 'TPSA', 'LabuteASA',
    'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds',
    'NumHeteroatoms', 'NumAliphaticRings', 'NumAromaticRings',
    'NumSaturatedRings', 'NumAliphaticHeterocycles', 'NumAromaticHeterocycles',
    'NumSaturatedHeterocycles', 'NumAliphaticCarbocycles', 'NumAromaticCarbocycles',
    'NumSaturatedCarbocycles', 'RingCount', 'FractionCSP3',
    'HeavyAtomCount', 'NHOHCount', 'NOCount',
    'NumHeavyAtoms', 'NumValenceElectrons',
]


class ToxicityExplainer:
    """
    SHAP-based explainer for toxicity predictions.

    Provides:
    - Feature importance (global and local)
    - SHAP values for individual predictions
    - Feature contribution analysis
    - Structural alert identification
    """

    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.explainers: Dict[str, Any] = {}
        self.background_data: Dict[str, np.ndarray] = {}
        self.feature_names: List[str] = self._generate_feature_names()

    def _generate_feature_names(self) -> List[str]:
        """Generate feature names for all features."""
        names = []

        # RDKit descriptors
        names.extend(RDKIT_DESCRIPTOR_NAMES)

        # Morgan fingerprints (2048 bits)
        names.extend([f"Morgan_{i}" for i in range(2048)])

        # MACCS keys (167 bits)
        names.extend([f"MACCS_{i}" for i in range(167)])

        return names

    def _get_selected_feature_names(self, endpoint: str) -> List[str]:
        """Get feature names after feature selection."""
        fs_path = self.models_dir / endpoint / "feature_selection.pkl"

        if fs_path.exists():
            fs = joblib.load(fs_path)
            if hasattr(fs, "selected_indices"):
                indices = fs.selected_indices
            elif isinstance(fs, dict) and "selected_indices" in fs:
                indices = fs["selected_indices"]
            elif isinstance(fs, np.ndarray):
                indices = fs
            else:
                return self.feature_names[:500]  # Default

            return [self.feature_names[i] for i in indices if i < len(self.feature_names)]

        return self.feature_names[:500]  # Default to first 500

    def initialize_explainer(self, endpoint: str, model: Any, X_background: Optional[np.ndarray] = None) -> bool:
        """
        Initialize SHAP explainer for a specific endpoint.

        Args:
            endpoint: Endpoint name
            model: Trained model
            X_background: Background data for SHAP (optional)

        Returns:
            True if successful
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, skipping explainer initialization")
            return False

        try:
            model_type = type(model).__name__.lower()

            # Create appropriate explainer based on model type
            if "xgb" in model_type:
                # XGBoost models use TreeExplainer
                self.explainers[endpoint] = shap.TreeExplainer(model)
                logger.info(f"Created TreeExplainer for {endpoint} (XGBoost)")

            elif "lgbm" in model_type or "lightgbm" in model_type:
                # LightGBM models use TreeExplainer
                self.explainers[endpoint] = shap.TreeExplainer(model)
                logger.info(f"Created TreeExplainer for {endpoint} (LightGBM)")

            elif "forest" in model_type:
                # RandomForest - try TreeExplainer, fall back to simple approach
                try:
                    self.explainers[endpoint] = shap.TreeExplainer(model)
                    logger.info(f"Created TreeExplainer for {endpoint} (RandomForest)")
                except Exception as e:
                    logger.warning(f"TreeExplainer failed for {endpoint}: {e}, using feature_importances_")
                    # Use a simple wrapper that returns feature importances
                    self.explainers[endpoint] = None  # Will use fallback
                    return False

            else:
                # For other models, skip SHAP (too slow for real-time)
                logger.info(f"Skipping SHAP for {endpoint} ({model_type}) - using feature importance fallback")
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to initialize explainer for {endpoint}: {e}")
            return False

    def explain_prediction(
        self,
        endpoint: str,
        features: np.ndarray,
        model: Any,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single prediction.

        Args:
            endpoint: Endpoint name
            features: Feature array (1, n_features)
            model: Trained model
            top_k: Number of top features to return

        Returns:
            Dictionary with SHAP values and feature contributions
        """
        if not SHAP_AVAILABLE:
            return self._fallback_feature_importance(model, features, top_k, endpoint)

        # Initialize explainer if not already done
        if endpoint not in self.explainers:
            success = self.initialize_explainer(endpoint, model, features)
            if not success:
                # Use fallback feature importance
                return self._fallback_feature_importance(model, features, top_k, endpoint)

        try:
            explainer = self.explainers[endpoint]

            # Calculate SHAP values
            shap_values = explainer.shap_values(features)

            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Classification: use class 1 (positive class)
                shap_vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values

            # Flatten if needed
            if shap_vals.ndim > 1:
                shap_vals = shap_vals.flatten()

            # Get feature names
            feature_names = self._get_selected_feature_names(endpoint)

            # Ensure lengths match
            n_features = min(len(shap_vals), len(feature_names))
            shap_vals = shap_vals[:n_features]
            feature_names = feature_names[:n_features]

            # Get top contributing features
            abs_shap = np.abs(shap_vals)
            top_indices = np.argsort(abs_shap)[::-1][:top_k]

            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_names[idx],
                    "shap_value": float(shap_vals[idx]),
                    "contribution": "increases" if shap_vals[idx] > 0 else "decreases",
                    "importance": float(abs_shap[idx])
                })

            # Get base value (expected value)
            if hasattr(explainer, "expected_value"):
                expected = explainer.expected_value
                if isinstance(expected, (list, np.ndarray)):
                    base_value = float(expected[1]) if len(expected) > 1 else float(expected[0])
                else:
                    base_value = float(expected)
            else:
                base_value = 0.0

            # Categorize features by type
            feature_categories = self._categorize_features(top_features)

            return {
                "success": True,
                "endpoint": endpoint,
                "base_value": base_value,
                "prediction_contribution": float(np.sum(shap_vals)),
                "top_features": top_features,
                "feature_categories": feature_categories,
                "n_features_analyzed": n_features,
                "interpretation": self._generate_interpretation(top_features, endpoint)
            }

        except Exception as e:
            logger.error(f"SHAP explanation failed for {endpoint}: {e}")
            # Fallback to feature importance
            return self._fallback_feature_importance(model, features, top_k, endpoint)

    def _fallback_feature_importance(
        self,
        model: Any,
        features: np.ndarray,
        top_k: int,
        endpoint: str
    ) -> Dict[str, Any]:
        """Fallback to model's built-in feature importance when SHAP fails."""
        try:
            feature_names = self._get_selected_feature_names(endpoint)

            # Try to get feature importances from model
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                # No feature importance available
                return {
                    "success": True,
                    "method": "none",
                    "message": "Feature importance not available for this model type",
                    "top_features": [],
                    "endpoint": endpoint
                }

            # Ensure lengths match
            n_features = min(len(importances), len(feature_names))
            importances = importances[:n_features]
            feature_names = feature_names[:n_features]

            # Get top features
            top_indices = np.argsort(importances)[::-1][:top_k]

            top_features = []
            for idx in top_indices:
                top_features.append({
                    "feature": feature_names[idx],
                    "shap_value": float(importances[idx]),  # Use importance as proxy
                    "contribution": "contributes",
                    "importance": float(importances[idx])
                })

            return {
                "success": True,
                "endpoint": endpoint,
                "method": "feature_importance",
                "base_value": 0.0,
                "prediction_contribution": float(np.sum(importances)),
                "top_features": top_features,
                "feature_categories": self._categorize_features(top_features),
                "n_features_analyzed": n_features,
                "interpretation": self._generate_interpretation(top_features, endpoint)
            }

        except Exception as e:
            logger.error(f"Fallback feature importance failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "method": "failed"
            }

    def _categorize_features(self, top_features: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize features by type."""
        categories = {
            "physicochemical": [],
            "structural_fingerprints": [],
            "substructure_keys": []
        }

        for feat in top_features:
            name = feat["feature"]
            if name in RDKIT_DESCRIPTOR_NAMES:
                categories["physicochemical"].append(feat)
            elif name.startswith("Morgan_"):
                categories["structural_fingerprints"].append(feat)
            elif name.startswith("MACCS_"):
                categories["substructure_keys"].append(feat)

        return categories

    def _generate_interpretation(self, top_features: List[Dict], endpoint: str) -> str:
        """Generate human-readable interpretation."""
        if not top_features:
            return "No significant features identified."

        # Get top 3 contributing features
        positive_features = [f for f in top_features if f["shap_value"] > 0][:3]
        negative_features = [f for f in top_features if f["shap_value"] < 0][:3]

        interpretation_parts = []

        # Endpoint-specific interpretations
        endpoint_labels = {
            "herg": "cardiotoxicity risk",
            "hepatotox": "hepatotoxicity risk",
            "nephrotox": "nephrotoxicity risk",
            "ames": "mutagenicity risk",
            "skin_sens": "skin sensitization risk",
            "cytotox": "cytotoxicity risk"
        }
        risk_label = endpoint_labels.get(endpoint, "toxicity risk")

        if positive_features:
            pos_names = [f["feature"] for f in positive_features]
            interpretation_parts.append(
                f"Features increasing {risk_label}: {', '.join(pos_names)}"
            )

        if negative_features:
            neg_names = [f["feature"] for f in negative_features]
            interpretation_parts.append(
                f"Features decreasing {risk_label}: {', '.join(neg_names)}"
            )

        # Add physicochemical insights
        for feat in top_features[:5]:
            name = feat["feature"]
            direction = "increases" if feat["shap_value"] > 0 else "decreases"

            if name == "MolWt":
                interpretation_parts.append(f"Molecular weight {direction} {risk_label}")
            elif name == "MolLogP":
                interpretation_parts.append(f"Lipophilicity (LogP) {direction} {risk_label}")
            elif name == "TPSA":
                interpretation_parts.append(f"Polar surface area {direction} {risk_label}")
            elif name == "NumHDonors":
                interpretation_parts.append(f"H-bond donors {direction} {risk_label}")
            elif name == "NumHAcceptors":
                interpretation_parts.append(f"H-bond acceptors {direction} {risk_label}")
            elif name == "NumAromaticRings":
                interpretation_parts.append(f"Aromatic ring count {direction} {risk_label}")

        return "; ".join(interpretation_parts) if interpretation_parts else "Analysis complete."

    def explain_multi_endpoint(
        self,
        features: np.ndarray,
        models: Dict[str, Any],
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations for all endpoints.

        Args:
            features: Feature array
            models: Dictionary of {endpoint: model}
            top_k: Number of top features per endpoint

        Returns:
            Combined explanation for all endpoints
        """
        explanations = {}

        for endpoint, model in models.items():
            explanations[endpoint] = self.explain_prediction(
                endpoint, features, model, top_k
            )

        # Find common important features across endpoints
        common_features = self._find_common_features(explanations)

        # Generate cross-endpoint insights
        cross_insights = self._generate_cross_endpoint_insights(explanations)

        return {
            "endpoint_explanations": explanations,
            "common_important_features": common_features,
            "cross_endpoint_insights": cross_insights
        }

    def _find_common_features(self, explanations: Dict[str, Dict]) -> List[Dict]:
        """Find features that are important across multiple endpoints."""
        feature_counts = {}
        feature_impacts = {}

        for endpoint, expl in explanations.items():
            if "top_features" not in expl:
                continue

            for feat in expl["top_features"]:
                name = feat["feature"]
                if name not in feature_counts:
                    feature_counts[name] = 0
                    feature_impacts[name] = []

                feature_counts[name] += 1
                feature_impacts[name].append({
                    "endpoint": endpoint,
                    "shap_value": feat["shap_value"],
                    "contribution": feat["contribution"]
                })

        # Features appearing in 2+ endpoints
        common = []
        for name, count in feature_counts.items():
            if count >= 2:
                common.append({
                    "feature": name,
                    "endpoints_affected": count,
                    "impacts": feature_impacts[name]
                })

        # Sort by number of endpoints affected
        common.sort(key=lambda x: x["endpoints_affected"], reverse=True)

        return common[:10]

    def _generate_cross_endpoint_insights(self, explanations: Dict[str, Dict]) -> List[str]:
        """Generate insights comparing feature importance across endpoints."""
        insights = []

        # Check for features with opposite effects
        feature_directions = {}
        for endpoint, expl in explanations.items():
            if "top_features" not in expl:
                continue
            for feat in expl["top_features"][:5]:
                name = feat["feature"]
                direction = 1 if feat["shap_value"] > 0 else -1

                if name not in feature_directions:
                    feature_directions[name] = {}
                feature_directions[name][endpoint] = direction

        for name, directions in feature_directions.items():
            if len(directions) >= 2:
                values = list(directions.values())
                if not all(v == values[0] for v in values):
                    endpoints = list(directions.keys())
                    insights.append(
                        f"'{name}' has opposite effects on {endpoints[0]} vs {endpoints[1]}"
                    )

        return insights[:5]

    def get_structural_alerts(self, smiles: str, predictions: Dict[str, Any]) -> List[Dict]:
        """
        Identify structural alerts based on predictions and known toxicophores.

        Args:
            smiles: SMILES string
            predictions: Prediction results

        Returns:
            List of structural alerts found
        """
        alerts = []

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return alerts

        # Common toxicophores (SMARTS patterns)
        toxicophores = {
            "nitro_aromatic": {
                "smarts": "[N+](=O)[O-]c",
                "concern": "Mutagenicity, hepatotoxicity",
                "severity": "high"
            },
            "aromatic_amine": {
                "smarts": "c-[NH2]",
                "concern": "Mutagenicity, carcinogenicity",
                "severity": "high"
            },
            "quinone": {
                "smarts": "O=C1C=CC(=O)C=C1",
                "concern": "Cytotoxicity, oxidative stress",
                "severity": "moderate"
            },
            "epoxide": {
                "smarts": "C1OC1",
                "concern": "Reactive metabolite, hepatotoxicity",
                "severity": "moderate"
            },
            "michael_acceptor": {
                "smarts": "C=CC=O",
                "concern": "Protein reactivity, hepatotoxicity",
                "severity": "moderate"
            },
            "thiophene": {
                "smarts": "c1ccsc1",
                "concern": "Hepatotoxicity (metabolic activation)",
                "severity": "low"
            },
            "furan": {
                "smarts": "c1ccoc1",
                "concern": "Hepatotoxicity (metabolic activation)",
                "severity": "moderate"
            },
            "hydrazine": {
                "smarts": "NN",
                "concern": "Hepatotoxicity, mutagenicity",
                "severity": "high"
            },
            "halogenated_aromatic": {
                "smarts": "c[F,Cl,Br,I]",
                "concern": "Potential metabolic issues",
                "severity": "low"
            },
            "aldehyde": {
                "smarts": "[CH]=O",
                "concern": "Protein reactivity",
                "severity": "moderate"
            }
        }

        for alert_name, alert_info in toxicophores.items():
            pattern = Chem.MolFromSmarts(alert_info["smarts"])
            if pattern and mol.HasSubstructMatch(pattern):
                matches = mol.GetSubstructMatches(pattern)
                alerts.append({
                    "alert": alert_name,
                    "smarts": alert_info["smarts"],
                    "concern": alert_info["concern"],
                    "severity": alert_info["severity"],
                    "occurrences": len(matches)
                })

        return alerts

    def generate_design_recommendations(
        self,
        explanations: Dict[str, Any],
        structural_alerts: List[Dict]
    ) -> List[Dict]:
        """
        Generate medicinal chemistry recommendations.

        Args:
            explanations: SHAP explanations
            structural_alerts: Identified structural alerts

        Returns:
            List of design recommendations
        """
        recommendations = []

        # Based on structural alerts
        for alert in structural_alerts:
            if alert["severity"] == "high":
                recommendations.append({
                    "priority": "high",
                    "type": "structural_modification",
                    "recommendation": f"Consider removing or replacing {alert['alert']} moiety",
                    "reason": alert["concern"]
                })

        # Based on SHAP values
        for endpoint, expl in explanations.get("endpoint_explanations", {}).items():
            if "top_features" not in expl:
                continue

            for feat in expl["top_features"][:3]:
                if feat["shap_value"] > 0:  # Feature increasing toxicity
                    name = feat["feature"]

                    if name == "MolLogP" or name == "LogP":
                        recommendations.append({
                            "priority": "medium",
                            "type": "property_optimization",
                            "recommendation": "Consider reducing lipophilicity (LogP)",
                            "reason": f"High LogP contributes to {endpoint} risk"
                        })
                    elif name == "MolWt":
                        recommendations.append({
                            "priority": "low",
                            "type": "property_optimization",
                            "recommendation": "Consider reducing molecular weight",
                            "reason": f"High MW contributes to {endpoint} risk"
                        })
                    elif name == "NumAromaticRings":
                        recommendations.append({
                            "priority": "medium",
                            "type": "structural_modification",
                            "recommendation": "Consider reducing aromatic ring count",
                            "reason": f"Aromatic rings contribute to {endpoint} risk"
                        })

        # Remove duplicates and sort by priority
        seen = set()
        unique_recs = []
        for rec in recommendations:
            key = rec["recommendation"]
            if key not in seen:
                seen.add(key)
                unique_recs.append(rec)

        priority_order = {"high": 0, "medium": 1, "low": 2}
        unique_recs.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return unique_recs[:10]
