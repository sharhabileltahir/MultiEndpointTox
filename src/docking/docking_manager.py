"""High-level docking manager for toxicity prediction."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from src.docking.docking_engine import (
    DockingEngine,
    VinaDockingEngine,
    SminaDockingEngine,
    DockingResult,
)
from src.docking.structure_manager import ProteinStructureManager
from src.docking.descriptors_3d import (
    Descriptors3DCalculator,
    Descriptors3DResult,
    analyze_binding_compatibility,
    get_pharmacophore_similarity,
)


class DockingManager:
    """
    High-level interface for molecular docking in toxicity prediction.

    Coordinates:
    - Protein structure preparation
    - Docking engine selection
    - Batch processing
    - Result caching
    - Ensemble score computation
    """

    # Mapping of toxicity endpoints to relevant protein targets
    ENDPOINT_TARGETS = {
        "herg": ["herg"],
        "hepatotox": ["hepatotox", "cyp2d6", "cyp2c9"],
        "nephrotox": ["herg"],  # hERG also relevant for kidney
        "cytotox": ["herg"],
        "ames": [],  # Ames typically doesn't use docking
        "skin_sens": [],
        "reproductive_tox": ["ar", "er_alpha"],  # Hormone receptors
    }

    def __init__(
        self,
        config: Optional[Dict] = None,
        structures_dir: str = "data/structures",
        engine: str = "vina",
    ):
        """
        Initialize docking manager.

        Args:
            config: Configuration dictionary
            structures_dir: Directory for protein structures
            engine: Docking engine ("vina" or "smina")
        """
        self.config = config or {}
        self.structures_dir = structures_dir

        # Initialize structure manager
        self.structure_manager = ProteinStructureManager(
            structures_dir=structures_dir,
            config=config
        )

        # Initialize docking engine
        self.engine_name = engine
        self.engine = self._create_engine(engine)

        # Results cache
        self._cache: Dict[str, DockingResult] = {}
        self._descriptor_cache: Dict[str, Descriptors3DResult] = {}

        # Docking parameters from config
        docking_config = self.config.get("docking", {})
        self.default_params = docking_config.get("docking_params", {
            "exhaustiveness": 8,
            "n_poses": 9,
        })

        # Initialize 3D descriptor calculator
        self.descriptor_calculator = Descriptors3DCalculator(
            n_conformers=docking_config.get("n_conformers", 10),
            random_seed=42
        )

        logger.info(f"Docking manager initialized with {engine} engine")

    def _create_engine(self, engine_name: str) -> DockingEngine:
        """Create docking engine instance."""
        if engine_name.lower() == "smina":
            engine = SminaDockingEngine()
        else:
            engine = VinaDockingEngine()

        if not engine.is_available():
            logger.warning(f"{engine_name} not available. "
                          "Install Vina or use Python bindings.")

        return engine

    def is_available(self) -> bool:
        """Check if docking is available."""
        return self.engine.is_available()

    def get_status(self) -> Dict[str, Any]:
        """Get docking system status."""
        return {
            "available": self.is_available(),
            "engine": self.engine_name,
            "targets": self.structure_manager.get_available_targets(),
            "endpoint_mappings": self.ENDPOINT_TARGETS,
            "cached_results": len(self._cache),
        }

    def dock(
        self,
        smiles: str,
        target: str,
        exhaustiveness: Optional[int] = None,
        n_poses: Optional[int] = None,
        use_cache: bool = True,
    ) -> DockingResult:
        """
        Dock a compound against a protein target.

        Args:
            smiles: SMILES string of compound
            target: Protein target ID (e.g., "herg", "hepatotox")
            exhaustiveness: Sampling exhaustiveness (default from config)
            n_poses: Number of poses to generate
            use_cache: Whether to use cached results

        Returns:
            DockingResult with binding affinity and poses
        """
        # Check cache
        cache_key = f"{smiles}:{target}"
        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached docking result for {target}")
            return self._cache[cache_key]

        if not self.engine.is_available():
            return DockingResult(
                success=False,
                smiles=smiles,
                protein_id=target,
                error="Docking engine not available"
            )

        try:
            # Prepare protein
            protein_path = self.structure_manager.prepare_protein(target)
            center, box_size = self.structure_manager.get_binding_site_params(target)

            # Get parameters
            exh = exhaustiveness or self.default_params.get("exhaustiveness", 8)
            n_pos = n_poses or self.default_params.get("n_poses", 9)

            # Run docking
            logger.info(f"Docking {smiles[:30]}... against {target}")
            result = self.engine.dock(
                smiles=smiles,
                protein_path=protein_path,
                center=center,
                box_size=box_size,
                exhaustiveness=exh,
                n_poses=n_pos,
            )

            # Cache result
            if use_cache and result.success:
                self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.error(f"Docking failed for {smiles} against {target}: {e}")
            return DockingResult(
                success=False,
                smiles=smiles,
                protein_id=target,
                error=str(e)
            )

    def dock_for_endpoint(
        self,
        smiles: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, DockingResult]:
        """
        Dock a compound against all relevant targets for a toxicity endpoint.

        Args:
            smiles: SMILES string
            endpoint: Toxicity endpoint (e.g., "herg", "hepatotox")
            **kwargs: Additional arguments for dock()

        Returns:
            Dictionary of target -> DockingResult
        """
        targets = self.ENDPOINT_TARGETS.get(endpoint, [])

        if not targets:
            return {}

        results = {}
        for target in targets:
            results[target] = self.dock(smiles, target, **kwargs)

        return results

    def dock_batch(
        self,
        smiles_list: List[str],
        target: str,
        max_workers: int = 4,
        **kwargs
    ) -> List[DockingResult]:
        """
        Dock multiple compounds against a target.

        Args:
            smiles_list: List of SMILES strings
            target: Protein target ID
            max_workers: Number of parallel workers
            **kwargs: Additional arguments for dock()

        Returns:
            List of DockingResult objects
        """
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.dock, smiles, target, **kwargs): smiles
                for smiles in smiles_list
            }

            for future in as_completed(futures):
                smiles = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch docking failed for {smiles}: {e}")
                    results.append(DockingResult(
                        success=False,
                        smiles=smiles,
                        protein_id=target,
                        error=str(e)
                    ))

        return results

    def compute_ensemble_score(
        self,
        ml_prediction: Dict[str, Any],
        docking_result: DockingResult,
        ml_weight: float = 0.7,
        docking_weight: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Compute ensemble prediction combining ML and docking scores.

        Args:
            ml_prediction: ML prediction result dict
            docking_result: Docking result object
            ml_weight: Weight for ML prediction (0-1)
            docking_weight: Weight for docking score (0-1)

        Returns:
            Ensemble prediction with combined score
        """
        # Normalize weights
        total = ml_weight + docking_weight
        ml_weight = ml_weight / total
        docking_weight = docking_weight / total

        # Get ML risk score
        ml_score = 0.5
        if ml_prediction.get("success"):
            task_type = ml_prediction.get("task_type", "classification")
            if task_type == "classification":
                prob = ml_prediction.get("probability")
                if prob is not None:
                    ml_score = float(prob)
                else:
                    ml_score = float(ml_prediction.get("prediction", 0))
            else:
                # Regression - normalize
                value = ml_prediction.get("prediction", 0)
                # For hERG pIC50: higher = more potent = higher risk
                ml_score = min(1.0, max(0.0, (value - 5) / 3))

        # Get docking risk score
        docking_score = docking_result.normalized_score if docking_result.success else 0.5

        # Compute ensemble score
        ensemble_score = (ml_weight * ml_score) + (docking_weight * docking_score)

        # Determine risk level
        if ensemble_score < 0.3:
            risk_level = "low"
        elif ensemble_score < 0.7:
            risk_level = "moderate"
        else:
            risk_level = "high"

        return {
            "ensemble_score": round(ensemble_score, 4),
            "risk_level": risk_level,
            "ml_contribution": {
                "score": round(ml_score, 4),
                "weight": round(ml_weight, 2),
            },
            "docking_contribution": {
                "score": round(docking_score, 4),
                "weight": round(docking_weight, 2),
                "affinity": docking_result.affinity,
            },
            "confidence": self._compute_confidence(ml_prediction, docking_result),
        }

    def _compute_confidence(
        self,
        ml_prediction: Dict[str, Any],
        docking_result: DockingResult
    ) -> float:
        """Compute confidence in ensemble prediction."""
        confidence_factors = []

        # ML model confidence
        if ml_prediction.get("success"):
            ad = ml_prediction.get("applicability_domain", {})
            if ad.get("in_domain", True):
                confidence_factors.append(ad.get("confidence", 0.8))
            else:
                confidence_factors.append(0.5)

        # Docking success
        if docking_result.success:
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)

        # Agreement between ML and docking
        if ml_prediction.get("success") and docking_result.success:
            ml_score = ml_prediction.get("probability", 0.5)
            dock_score = docking_result.normalized_score
            agreement = 1 - abs(ml_score - dock_score)
            confidence_factors.append(agreement)

        return round(sum(confidence_factors) / len(confidence_factors), 3) if confidence_factors else 0.5

    def get_docking_features(
        self,
        smiles: str,
        endpoint: str
    ) -> Dict[str, float]:
        """
        Compute docking-derived features for ML integration.

        Args:
            smiles: SMILES string
            endpoint: Toxicity endpoint

        Returns:
            Dictionary of feature name -> value
        """
        results = self.dock_for_endpoint(smiles, endpoint)

        features = {}
        for target, result in results.items():
            prefix = f"dock_{target}"
            features[f"{prefix}_affinity"] = result.affinity if result.success else 0.0
            features[f"{prefix}_score"] = result.normalized_score if result.success else 0.5
            features[f"{prefix}_success"] = 1.0 if result.success else 0.0

        return features

    def clear_cache(self):
        """Clear the results cache."""
        self._cache.clear()
        logger.info("Docking cache cleared")

    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.engine, 'cleanup'):
            self.engine.cleanup()

    def calculate_3d_descriptors(
        self,
        smiles: str,
        use_cache: bool = True
    ) -> Descriptors3DResult:
        """
        Calculate 3D molecular descriptors and pharmacophore features.

        Args:
            smiles: SMILES string
            use_cache: Whether to use cached results

        Returns:
            Descriptors3DResult with all 3D descriptors
        """
        if use_cache and smiles in self._descriptor_cache:
            logger.debug(f"Using cached 3D descriptors")
            return self._descriptor_cache[smiles]

        result = self.descriptor_calculator.calculate_all(smiles)

        if use_cache and result.success:
            self._descriptor_cache[smiles] = result

        return result

    def dock_with_descriptors(
        self,
        smiles: str,
        target: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform docking with integrated 3D descriptors and pharmacophore analysis.

        Args:
            smiles: SMILES string
            target: Protein target ID
            **kwargs: Additional arguments for dock()

        Returns:
            Enhanced docking result with 3D descriptors
        """
        # Run docking
        docking_result = self.dock(smiles, target, **kwargs)

        # Calculate 3D descriptors
        descriptors = self.calculate_3d_descriptors(smiles)

        # Analyze binding compatibility
        compatibility = analyze_binding_compatibility(descriptors, target)

        # Build enhanced result
        result = {
            "docking": docking_result.to_dict() if hasattr(docking_result, 'to_dict') else {
                "success": docking_result.success,
                "affinity": docking_result.affinity,
                "normalized_score": docking_result.normalized_score,
                "num_poses": docking_result.num_poses,
                "error": docking_result.error,
            },
            "descriptors_3d": descriptors.to_dict(),
            "binding_compatibility": compatibility,
            "enhanced_score": self._compute_enhanced_score(
                docking_result, descriptors, compatibility
            ),
        }

        return result

    def _compute_enhanced_score(
        self,
        docking_result: DockingResult,
        descriptors: Descriptors3DResult,
        compatibility: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute enhanced docking score incorporating 3D descriptors.

        Combines:
        - Raw docking affinity
        - Shape compatibility
        - Pharmacophore match quality
        - Size fitness
        """
        if not docking_result.success or not descriptors.success:
            return {
                "score": 0.5,
                "confidence": "low",
                "risk_level": "unknown"
            }

        # Base score from docking
        base_score = docking_result.normalized_score

        # Shape modifier
        shape_modifier = 1.0
        if compatibility.get("shape_compatibility") == "good":
            shape_modifier = 1.1
        elif compatibility.get("shape_compatibility") == "poor":
            shape_modifier = 0.8

        # Pharmacophore modifier
        pharm_modifier = 1.0
        pharm_match = compatibility.get("pharmacophore_match", "unknown")
        if pharm_match == "high":
            pharm_modifier = 1.15
        elif pharm_match == "moderate":
            pharm_modifier = 1.0
        elif pharm_match == "low":
            pharm_modifier = 0.85

        # Size modifier
        size_modifier = 1.0
        if compatibility.get("size_compatibility") == "poor":
            size_modifier = 0.9

        # Compute enhanced score
        enhanced = base_score * shape_modifier * pharm_modifier * size_modifier
        enhanced = min(1.0, max(0.0, enhanced))

        # Determine risk level
        if enhanced >= 0.7:
            risk_level = "high"
        elif enhanced >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"

        # Confidence based on descriptor quality
        confidence = "high" if descriptors.success else "low"
        if compatibility.get("binding_risk") == "high":
            confidence = "high"

        return {
            "score": round(enhanced, 4),
            "base_docking_score": round(base_score, 4),
            "shape_modifier": round(shape_modifier, 2),
            "pharmacophore_modifier": round(pharm_modifier, 2),
            "size_modifier": round(size_modifier, 2),
            "confidence": confidence,
            "risk_level": risk_level,
        }

    def dock_for_endpoint_enhanced(
        self,
        smiles: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced docking for endpoint with 3D descriptors.

        Args:
            smiles: SMILES string
            endpoint: Toxicity endpoint
            **kwargs: Additional arguments

        Returns:
            Enhanced results for all targets
        """
        targets = self.ENDPOINT_TARGETS.get(endpoint, [])

        if not targets:
            return {"targets": {}, "summary": {}}

        # Calculate 3D descriptors once
        descriptors = self.calculate_3d_descriptors(smiles)

        results = {"targets": {}, "descriptors_3d": descriptors.to_dict()}

        for target in targets:
            docking_result = self.dock(smiles, target, **kwargs)
            compatibility = analyze_binding_compatibility(descriptors, target)

            results["targets"][target] = {
                "docking": {
                    "success": docking_result.success,
                    "affinity": docking_result.affinity,
                    "normalized_score": docking_result.normalized_score,
                    "num_poses": docking_result.num_poses,
                    "error": docking_result.error,
                },
                "binding_compatibility": compatibility,
                "enhanced_score": self._compute_enhanced_score(
                    docking_result, descriptors, compatibility
                ),
            }

        # Compute summary across targets
        results["summary"] = self._compute_endpoint_summary(results["targets"])

        return results

    def _compute_endpoint_summary(
        self,
        target_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compute summary statistics across targets."""
        if not target_results:
            return {}

        affinities = []
        enhanced_scores = []
        risk_factors = []

        for target, result in target_results.items():
            docking = result.get("docking", {})
            if docking.get("success"):
                aff = docking.get("affinity")
                if aff is not None:
                    affinities.append(aff)

            enhanced = result.get("enhanced_score", {})
            if enhanced.get("score") is not None:
                enhanced_scores.append(enhanced["score"])

            if enhanced.get("risk_level") == "high":
                risk_factors.append(target)

        summary = {
            "num_targets": len(target_results),
            "successful_docks": len(affinities),
        }

        if affinities:
            summary["best_affinity"] = round(min(affinities), 2)
            summary["avg_affinity"] = round(sum(affinities) / len(affinities), 2)

        if enhanced_scores:
            summary["max_enhanced_score"] = round(max(enhanced_scores), 4)
            summary["avg_enhanced_score"] = round(
                sum(enhanced_scores) / len(enhanced_scores), 4
            )

        if risk_factors:
            summary["high_risk_targets"] = risk_factors

        # Overall risk assessment
        if summary.get("max_enhanced_score", 0) >= 0.7:
            summary["overall_risk"] = "high"
        elif summary.get("max_enhanced_score", 0) >= 0.4:
            summary["overall_risk"] = "moderate"
        else:
            summary["overall_risk"] = "low"

        return summary

    def get_pharmacophore_features(
        self,
        smiles: str
    ) -> Dict[str, Any]:
        """
        Get pharmacophore features for a compound.

        Args:
            smiles: SMILES string

        Returns:
            Pharmacophore feature summary
        """
        descriptors = self.calculate_3d_descriptors(smiles)

        if not descriptors.success:
            return {"success": False, "error": descriptors.error}

        return {
            "success": True,
            "feature_counts": {
                "h_bond_acceptors": descriptors.n_hba,
                "h_bond_donors": descriptors.n_hbd,
                "aromatic_rings": descriptors.n_aromatic,
                "hydrophobic_centers": descriptors.n_hydrophobic,
                "positive_ionizable": descriptors.n_pos_ionizable,
                "negative_ionizable": descriptors.n_neg_ionizable,
            },
            "features": [f.to_dict() for f in descriptors.pharmacophore_features],
            "has_pharmacophore_fp": descriptors.pharmacophore_fp is not None,
        }

    def compare_pharmacophores(
        self,
        smiles1: str,
        smiles2: str
    ) -> Dict[str, Any]:
        """
        Compare pharmacophore profiles of two compounds.

        Args:
            smiles1: First SMILES string
            smiles2: Second SMILES string

        Returns:
            Comparison results with similarity score
        """
        desc1 = self.calculate_3d_descriptors(smiles1)
        desc2 = self.calculate_3d_descriptors(smiles2)

        if not desc1.success or not desc2.success:
            return {
                "success": False,
                "error": "Failed to calculate descriptors for one or both compounds"
            }

        # Calculate pharmacophore fingerprint similarity
        fp_similarity = get_pharmacophore_similarity(
            desc1.pharmacophore_fp,
            desc2.pharmacophore_fp
        )

        # Compare feature counts
        feature_comparison = {
            "h_bond_acceptors": (desc1.n_hba, desc2.n_hba),
            "h_bond_donors": (desc1.n_hbd, desc2.n_hbd),
            "aromatic_rings": (desc1.n_aromatic, desc2.n_aromatic),
            "hydrophobic_centers": (desc1.n_hydrophobic, desc2.n_hydrophobic),
            "positive_ionizable": (desc1.n_pos_ionizable, desc2.n_pos_ionizable),
            "negative_ionizable": (desc1.n_neg_ionizable, desc2.n_neg_ionizable),
        }

        # Compare shape
        shape_comparison = {
            "asphericity": (round(desc1.asphericity, 3), round(desc2.asphericity, 3)),
            "eccentricity": (round(desc1.eccentricity, 3), round(desc2.eccentricity, 3)),
            "radius_of_gyration": (
                round(desc1.radius_of_gyration, 2),
                round(desc2.radius_of_gyration, 2)
            ),
            "molecular_volume": (
                round(desc1.molecular_volume, 1),
                round(desc2.molecular_volume, 1)
            ),
        }

        return {
            "success": True,
            "pharmacophore_similarity": round(fp_similarity, 4),
            "feature_comparison": feature_comparison,
            "shape_comparison": shape_comparison,
        }
