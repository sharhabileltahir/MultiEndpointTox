"""
3D Molecular Descriptors and Pharmacophore Feature Extraction

This module provides:
- 3D conformer generation
- 3D molecular descriptors (shape, electrostatics, surface)
- Pharmacophore feature identification
- Pharmacophore fingerprints for similarity analysis
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
    from rdkit.Chem import rdMolTransforms
    from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate
    from rdkit.Chem.Pharm2D.SigFactory import SigFactory
    from rdkit import DataStructs
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - 3D descriptors disabled")

try:
    from rdkit.Chem import rdFreeSASA
    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False

try:
    from rdkit.Chem.Pharm3D import Pharmacophore, EmbedLib
    from rdkit.Chem import ChemicalFeatures
    from rdkit import RDConfig
    import os
    FDEF_FILE = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    PHARM3D_AVAILABLE = True
except ImportError:
    PHARM3D_AVAILABLE = False


@dataclass
class PharmacophoreFeature:
    """Represents a single pharmacophore feature."""
    type: str  # HBA, HBD, Aromatic, Hydrophobic, PosIonizable, NegIonizable
    position: Tuple[float, float, float]
    atom_indices: List[int]
    family: str = ""

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "position": list(self.position),
            "atom_indices": self.atom_indices,
            "family": self.family
        }


@dataclass
class Descriptors3DResult:
    """Container for 3D descriptor results."""
    # Shape descriptors
    asphericity: float = 0.0
    eccentricity: float = 0.0
    inertial_shape_factor: float = 0.0
    npr1: float = 0.0  # Normalized principal moments ratio 1
    npr2: float = 0.0  # Normalized principal moments ratio 2
    pmi1: float = 0.0  # Principal moment of inertia 1
    pmi2: float = 0.0  # Principal moment of inertia 2
    pmi3: float = 0.0  # Principal moment of inertia 3
    radius_of_gyration: float = 0.0
    spherocity_index: float = 0.0

    # Surface descriptors
    tpsa_3d: float = 0.0
    sasa: float = 0.0  # Solvent accessible surface area

    # Volume descriptors
    molecular_volume: float = 0.0

    # Electrostatic descriptors
    dipole_moment: float = 0.0

    # Conformational descriptors
    num_rotatable_bonds: int = 0
    conformer_energy: float = 0.0

    # Pharmacophore counts
    n_hba: int = 0  # H-bond acceptors (3D)
    n_hbd: int = 0  # H-bond donors (3D)
    n_aromatic: int = 0
    n_hydrophobic: int = 0
    n_pos_ionizable: int = 0
    n_neg_ionizable: int = 0

    # Pharmacophore features list
    pharmacophore_features: List[PharmacophoreFeature] = field(default_factory=list)

    # Pharmacophore fingerprint
    pharmacophore_fp: Optional[Any] = None

    # Success flag
    success: bool = False
    error: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            "shape_descriptors": {
                "asphericity": round(self.asphericity, 4),
                "eccentricity": round(self.eccentricity, 4),
                "inertial_shape_factor": round(self.inertial_shape_factor, 4),
                "npr1": round(self.npr1, 4),
                "npr2": round(self.npr2, 4),
                "pmi1": round(self.pmi1, 2),
                "pmi2": round(self.pmi2, 2),
                "pmi3": round(self.pmi3, 2),
                "radius_of_gyration": round(self.radius_of_gyration, 3),
                "spherocity_index": round(self.spherocity_index, 4),
            },
            "surface_descriptors": {
                "tpsa_3d": round(self.tpsa_3d, 2),
                "sasa": round(self.sasa, 2),
            },
            "volume_descriptors": {
                "molecular_volume": round(self.molecular_volume, 2),
            },
            "electrostatic_descriptors": {
                "dipole_moment": round(self.dipole_moment, 3),
            },
            "conformational_descriptors": {
                "num_rotatable_bonds": self.num_rotatable_bonds,
                "conformer_energy": round(self.conformer_energy, 2),
            },
            "pharmacophore_counts": {
                "h_bond_acceptors": self.n_hba,
                "h_bond_donors": self.n_hbd,
                "aromatic_rings": self.n_aromatic,
                "hydrophobic_centers": self.n_hydrophobic,
                "positive_ionizable": self.n_pos_ionizable,
                "negative_ionizable": self.n_neg_ionizable,
            },
            "pharmacophore_features": [f.to_dict() for f in self.pharmacophore_features],
            "success": self.success,
        }
        if self.error:
            result["error"] = self.error
        return result


class Descriptors3DCalculator:
    """Calculate 3D molecular descriptors and pharmacophore features."""

    # Pharmacophore feature definitions
    PHARM_FEATURE_TYPES = {
        "Donor": "HBD",
        "Acceptor": "HBA",
        "Aromatic": "Aromatic",
        "Hydrophobe": "Hydrophobic",
        "PosIonizable": "PosIonizable",
        "NegIonizable": "NegIonizable",
        "LumpedHydrophobe": "Hydrophobic",
    }

    def __init__(self, n_conformers: int = 10, random_seed: int = 42):
        """
        Initialize 3D descriptor calculator.

        Args:
            n_conformers: Number of conformers to generate for ensemble
            random_seed: Random seed for reproducibility
        """
        self.n_conformers = n_conformers
        self.random_seed = random_seed
        self.feature_factory = None

        if RDKIT_AVAILABLE and PHARM3D_AVAILABLE:
            try:
                self.feature_factory = ChemicalFeatures.BuildFeatureFactory(FDEF_FILE)
                logger.debug("Initialized pharmacophore feature factory")
            except Exception as e:
                logger.warning(f"Could not initialize feature factory: {e}")

    def generate_conformer(self, mol: 'Chem.Mol', optimize: bool = True) -> Optional['Chem.Mol']:
        """
        Generate 3D conformer for molecule.

        Args:
            mol: RDKit molecule
            optimize: Whether to optimize geometry with force field

        Returns:
            Molecule with 3D coordinates or None if failed
        """
        if not RDKIT_AVAILABLE:
            return None

        try:
            mol_3d = Chem.AddHs(mol)

            # Generate conformer using ETKDG (Experimental-Torsion Knowledge Distance Geometry)
            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.numThreads = 0  # Use all available threads

            result = AllChem.EmbedMolecule(mol_3d, params)

            if result == -1:
                # Try with random coordinates as fallback
                params.useRandomCoords = True
                result = AllChem.EmbedMolecule(mol_3d, params)

            if result == -1:
                logger.warning("Failed to generate 3D conformer")
                return None

            # Optimize geometry
            if optimize:
                try:
                    # Try MMFF first (more accurate)
                    mmff_result = AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=500)
                    if mmff_result != 0:
                        # Fall back to UFF
                        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=500)
                except:
                    try:
                        AllChem.UFFOptimizeMolecule(mol_3d, maxIters=500)
                    except:
                        pass  # Keep unoptimized conformer

            return mol_3d

        except Exception as e:
            logger.error(f"Error generating conformer: {e}")
            return None

    def generate_conformer_ensemble(self, mol: 'Chem.Mol') -> Optional['Chem.Mol']:
        """
        Generate ensemble of conformers.

        Args:
            mol: RDKit molecule

        Returns:
            Molecule with multiple conformers
        """
        if not RDKIT_AVAILABLE:
            return None

        try:
            mol_3d = Chem.AddHs(mol)

            params = AllChem.ETKDGv3()
            params.randomSeed = self.random_seed
            params.numThreads = 0
            params.pruneRmsThresh = 0.5  # Remove similar conformers

            conf_ids = AllChem.EmbedMultipleConfs(
                mol_3d,
                numConfs=self.n_conformers,
                params=params
            )

            if len(conf_ids) == 0:
                return self.generate_conformer(mol)

            # Optimize all conformers
            try:
                AllChem.MMFFOptimizeMoleculeConfs(mol_3d, maxIters=500)
            except:
                try:
                    for conf_id in conf_ids:
                        AllChem.UFFOptimizeMolecule(mol_3d, confId=conf_id, maxIters=500)
                except:
                    pass

            return mol_3d

        except Exception as e:
            logger.error(f"Error generating conformer ensemble: {e}")
            return self.generate_conformer(mol)

    def calculate_shape_descriptors(self, mol: 'Chem.Mol', conf_id: int = -1) -> Dict[str, float]:
        """Calculate 3D shape descriptors."""
        descriptors = {}

        try:
            descriptors['asphericity'] = Descriptors3D.Asphericity(mol, confId=conf_id)
        except:
            descriptors['asphericity'] = 0.0

        try:
            descriptors['eccentricity'] = Descriptors3D.Eccentricity(mol, confId=conf_id)
        except:
            descriptors['eccentricity'] = 0.0

        try:
            descriptors['inertial_shape_factor'] = Descriptors3D.InertialShapeFactor(mol, confId=conf_id)
        except:
            descriptors['inertial_shape_factor'] = 0.0

        try:
            descriptors['npr1'] = Descriptors3D.NPR1(mol, confId=conf_id)
        except:
            descriptors['npr1'] = 0.0

        try:
            descriptors['npr2'] = Descriptors3D.NPR2(mol, confId=conf_id)
        except:
            descriptors['npr2'] = 0.0

        try:
            descriptors['pmi1'] = Descriptors3D.PMI1(mol, confId=conf_id)
        except:
            descriptors['pmi1'] = 0.0

        try:
            descriptors['pmi2'] = Descriptors3D.PMI2(mol, confId=conf_id)
        except:
            descriptors['pmi2'] = 0.0

        try:
            descriptors['pmi3'] = Descriptors3D.PMI3(mol, confId=conf_id)
        except:
            descriptors['pmi3'] = 0.0

        try:
            descriptors['radius_of_gyration'] = Descriptors3D.RadiusOfGyration(mol, confId=conf_id)
        except:
            descriptors['radius_of_gyration'] = 0.0

        try:
            descriptors['spherocity_index'] = Descriptors3D.SpherocityIndex(mol, confId=conf_id)
        except:
            descriptors['spherocity_index'] = 0.0

        return descriptors

    def calculate_surface_descriptors(self, mol: 'Chem.Mol', conf_id: int = -1) -> Dict[str, float]:
        """Calculate surface-related descriptors."""
        descriptors = {}

        # TPSA (topological, but included for comparison)
        try:
            descriptors['tpsa'] = rdMolDescriptors.CalcTPSA(mol)
        except:
            descriptors['tpsa'] = 0.0

        # Solvent Accessible Surface Area
        if FREESASA_AVAILABLE:
            try:
                radii = rdFreeSASA.classifyAtoms(mol)
                sasa = rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id)
                descriptors['sasa'] = sasa
            except:
                descriptors['sasa'] = 0.0
        else:
            # Estimate SASA from Van der Waals radii
            try:
                conf = mol.GetConformer(conf_id)
                # Approximate SASA
                descriptors['sasa'] = self._estimate_sasa(mol, conf)
            except:
                descriptors['sasa'] = 0.0

        return descriptors

    def _estimate_sasa(self, mol: 'Chem.Mol', conf) -> float:
        """Estimate SASA using simple sphere model."""
        # Van der Waals radii (Angstroms)
        vdw_radii = {
            'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47,
            'P': 1.8, 'S': 1.8, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
        }
        probe_radius = 1.4  # Water probe radius

        total_sasa = 0.0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            r = vdw_radii.get(symbol, 1.7) + probe_radius
            # Approximate exposed surface (simplified)
            total_sasa += 4 * np.pi * r * r * 0.5  # Assume 50% exposed

        return total_sasa

    def calculate_volume(self, mol: 'Chem.Mol', conf_id: int = -1) -> float:
        """Calculate molecular volume."""
        try:
            return AllChem.ComputeMolVolume(mol, confId=conf_id)
        except:
            # Estimate from atom count
            return mol.GetNumHeavyAtoms() * 18.0  # ~18 Å³ per heavy atom

    def calculate_conformer_energy(self, mol: 'Chem.Mol', conf_id: int = -1) -> float:
        """Calculate force field energy of conformer."""
        try:
            # Try MMFF first
            ff = AllChem.MMFFGetMoleculeForceField(
                mol,
                AllChem.MMFFGetMoleculeProperties(mol),
                confId=conf_id
            )
            if ff:
                return ff.CalcEnergy()
        except:
            pass

        try:
            # Fall back to UFF
            ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
            if ff:
                return ff.CalcEnergy()
        except:
            pass

        return 0.0

    def extract_pharmacophore_features(self, mol: 'Chem.Mol', conf_id: int = -1) -> List[PharmacophoreFeature]:
        """
        Extract 3D pharmacophore features from molecule.

        Args:
            mol: RDKit molecule with 3D coordinates
            conf_id: Conformer ID to use

        Returns:
            List of pharmacophore features
        """
        features = []

        if not self.feature_factory:
            return features

        try:
            conf = mol.GetConformer(conf_id)
            raw_features = self.feature_factory.GetFeaturesForMol(mol, confId=conf_id)

            for feat in raw_features:
                family = feat.GetFamily()
                feat_type = self.PHARM_FEATURE_TYPES.get(family, family)

                # Get 3D position
                pos = feat.GetPos()
                position = (pos.x, pos.y, pos.z)

                # Get atom indices
                atom_indices = list(feat.GetAtomIds())

                pharm_feat = PharmacophoreFeature(
                    type=feat_type,
                    position=position,
                    atom_indices=atom_indices,
                    family=family
                )
                features.append(pharm_feat)

        except Exception as e:
            logger.warning(f"Error extracting pharmacophore features: {e}")

        return features

    def calculate_pharmacophore_fingerprint(self, mol: 'Chem.Mol') -> Optional[Any]:
        """
        Calculate 2D pharmacophore fingerprint (Gobbi).

        This fingerprint encodes pharmacophore feature pairs and their
        topological distances.
        """
        try:
            sig_factory = Gobbi_Pharm2D.factory
            fp = Generate.Gen2DFingerprint(mol, sig_factory)
            return fp
        except Exception as e:
            logger.warning(f"Error calculating pharmacophore fingerprint: {e}")
            return None

    def calculate_all(self, smiles: str) -> Descriptors3DResult:
        """
        Calculate all 3D descriptors and pharmacophore features.

        Args:
            smiles: SMILES string

        Returns:
            Descriptors3DResult with all calculated values
        """
        result = Descriptors3DResult()

        if not RDKIT_AVAILABLE:
            result.error = "RDKit not available"
            return result

        try:
            # Parse SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.error = "Invalid SMILES"
                return result

            # Generate 3D conformer
            mol_3d = self.generate_conformer(mol)
            if mol_3d is None:
                result.error = "Failed to generate 3D conformer"
                return result

            conf_id = mol_3d.GetConformer().GetId()

            # Calculate shape descriptors
            shape = self.calculate_shape_descriptors(mol_3d, conf_id)
            result.asphericity = shape.get('asphericity', 0.0)
            result.eccentricity = shape.get('eccentricity', 0.0)
            result.inertial_shape_factor = shape.get('inertial_shape_factor', 0.0)
            result.npr1 = shape.get('npr1', 0.0)
            result.npr2 = shape.get('npr2', 0.0)
            result.pmi1 = shape.get('pmi1', 0.0)
            result.pmi2 = shape.get('pmi2', 0.0)
            result.pmi3 = shape.get('pmi3', 0.0)
            result.radius_of_gyration = shape.get('radius_of_gyration', 0.0)
            result.spherocity_index = shape.get('spherocity_index', 0.0)

            # Calculate surface descriptors
            surface = self.calculate_surface_descriptors(mol_3d, conf_id)
            result.tpsa_3d = surface.get('tpsa', 0.0)
            result.sasa = surface.get('sasa', 0.0)

            # Calculate volume
            result.molecular_volume = self.calculate_volume(mol_3d, conf_id)

            # Calculate conformer energy
            result.conformer_energy = self.calculate_conformer_energy(mol_3d, conf_id)

            # Rotatable bonds
            result.num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

            # Extract pharmacophore features
            result.pharmacophore_features = self.extract_pharmacophore_features(mol_3d, conf_id)

            # Count pharmacophore feature types
            for feat in result.pharmacophore_features:
                if feat.type == "HBA":
                    result.n_hba += 1
                elif feat.type == "HBD":
                    result.n_hbd += 1
                elif feat.type == "Aromatic":
                    result.n_aromatic += 1
                elif feat.type == "Hydrophobic":
                    result.n_hydrophobic += 1
                elif feat.type == "PosIonizable":
                    result.n_pos_ionizable += 1
                elif feat.type == "NegIonizable":
                    result.n_neg_ionizable += 1

            # Calculate pharmacophore fingerprint
            result.pharmacophore_fp = self.calculate_pharmacophore_fingerprint(mol)

            result.success = True
            logger.debug(f"Calculated 3D descriptors for {smiles[:30]}...")

        except Exception as e:
            result.error = str(e)
            logger.error(f"Error calculating 3D descriptors: {e}")

        return result


def get_pharmacophore_similarity(fp1, fp2) -> float:
    """
    Calculate Tanimoto similarity between two pharmacophore fingerprints.

    Args:
        fp1: First pharmacophore fingerprint
        fp2: Second pharmacophore fingerprint

    Returns:
        Tanimoto similarity (0-1)
    """
    if fp1 is None or fp2 is None:
        return 0.0

    try:
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0


def analyze_binding_compatibility(
    descriptors: Descriptors3DResult,
    target_type: str
) -> Dict[str, Any]:
    """
    Analyze compound compatibility with target binding site.

    Args:
        descriptors: 3D descriptors result
        target_type: Type of target (herg, cyp3a4, etc.)

    Returns:
        Compatibility analysis
    """
    analysis = {
        "shape_compatibility": "unknown",
        "size_compatibility": "unknown",
        "pharmacophore_match": "unknown",
        "binding_risk": "unknown",
        "recommendations": []
    }

    if not descriptors.success:
        return analysis

    # Target-specific binding site characteristics
    target_profiles = {
        "herg": {
            "preferred_shape": "elongated",  # hERG prefers elongated molecules
            "max_volume": 600,
            "min_aromatic": 1,
            "key_features": ["Aromatic", "HBA", "Hydrophobic"],
            "risk_threshold": {"n_aromatic": 2, "n_hba": 2}
        },
        "cyp3a4": {
            "preferred_shape": "any",
            "max_volume": 800,
            "min_aromatic": 0,
            "key_features": ["HBA", "Hydrophobic"],
            "risk_threshold": {}
        },
        "cyp2d6": {
            "preferred_shape": "any",
            "max_volume": 500,
            "min_aromatic": 1,
            "key_features": ["HBD", "Aromatic", "PosIonizable"],
            "risk_threshold": {"n_pos_ionizable": 1}
        },
        "ar": {
            "preferred_shape": "flat",
            "max_volume": 500,
            "min_aromatic": 1,
            "key_features": ["Aromatic", "HBA", "Hydrophobic"],
            "risk_threshold": {}
        },
        "er_alpha": {
            "preferred_shape": "flat",
            "max_volume": 450,
            "min_aromatic": 2,
            "key_features": ["Aromatic", "HBD", "Hydrophobic"],
            "risk_threshold": {}
        }
    }

    profile = target_profiles.get(target_type.lower(), {})

    if not profile:
        return analysis

    # Shape analysis
    if descriptors.npr1 > 0 and descriptors.npr2 > 0:
        # PMI plot analysis (rod-disc-sphere)
        if descriptors.npr1 < 0.3 and descriptors.npr2 > 0.7:
            shape = "elongated"
        elif descriptors.npr1 > 0.6 and descriptors.npr2 < 0.4:
            shape = "flat"
        else:
            shape = "spherical"

        if profile["preferred_shape"] == "any" or shape == profile["preferred_shape"]:
            analysis["shape_compatibility"] = "good"
        else:
            analysis["shape_compatibility"] = "moderate"
            analysis["recommendations"].append(
                f"Shape ({shape}) may not be optimal for {target_type} binding"
            )

    # Size analysis
    if descriptors.molecular_volume > 0:
        if descriptors.molecular_volume <= profile.get("max_volume", 1000):
            analysis["size_compatibility"] = "good"
        else:
            analysis["size_compatibility"] = "poor"
            analysis["recommendations"].append(
                f"Molecule may be too large for {target_type} binding site"
            )

    # Pharmacophore feature analysis
    key_features = profile.get("key_features", [])
    matched_features = 0

    feature_counts = {
        "HBA": descriptors.n_hba,
        "HBD": descriptors.n_hbd,
        "Aromatic": descriptors.n_aromatic,
        "Hydrophobic": descriptors.n_hydrophobic,
        "PosIonizable": descriptors.n_pos_ionizable,
        "NegIonizable": descriptors.n_neg_ionizable,
    }

    for feat in key_features:
        if feature_counts.get(feat, 0) > 0:
            matched_features += 1

    if key_features:
        match_ratio = matched_features / len(key_features)
        if match_ratio >= 0.7:
            analysis["pharmacophore_match"] = "high"
        elif match_ratio >= 0.4:
            analysis["pharmacophore_match"] = "moderate"
        else:
            analysis["pharmacophore_match"] = "low"

    # Risk assessment for specific targets (e.g., hERG)
    risk_threshold = profile.get("risk_threshold", {})
    risk_factors = 0

    for feat, threshold in risk_threshold.items():
        if getattr(descriptors, feat, 0) >= threshold:
            risk_factors += 1

    if risk_factors >= 2:
        analysis["binding_risk"] = "high"
    elif risk_factors >= 1:
        analysis["binding_risk"] = "moderate"
    else:
        analysis["binding_risk"] = "low"

    return analysis
