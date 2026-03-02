"""Molecular docking engine implementations."""

import os
import tempfile
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from loguru import logger

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logger.warning("RDKit not available - docking functionality limited")

try:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    MEEKO_AVAILABLE = True
except ImportError:
    MEEKO_AVAILABLE = False
    logger.warning("Meeko not available - will use Open Babel for ligand preparation")

try:
    from vina import Vina
    VINA_PYTHON_AVAILABLE = True
except ImportError:
    VINA_PYTHON_AVAILABLE = False
    logger.info("Vina Python bindings not available - will use command-line Vina")


@dataclass
class DockingResult:
    """Container for docking results."""
    success: bool
    smiles: str
    protein_id: str
    affinity: Optional[float] = None  # kcal/mol (negative = better binding)
    poses: List[Dict[str, Any]] = field(default_factory=list)
    num_poses: int = 0
    best_pose_rmsd: Optional[float] = None
    interactions: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "success": self.success,
            "smiles": self.smiles,
            "protein_id": self.protein_id,
            "affinity": self.affinity,
            "num_poses": self.num_poses,
            "best_pose_rmsd": self.best_pose_rmsd,
            "poses": self.poses[:3] if self.poses else [],  # Top 3 poses
            "interactions": self.interactions,
            "error": self.error,
            "execution_time": round(self.execution_time, 2),
        }

    @property
    def normalized_score(self) -> float:
        """
        Normalize affinity to 0-1 range for ensemble prediction.

        Typical Vina scores range from -12 (excellent) to 0 (no binding).
        We map this to 0 (no risk) to 1 (high risk for toxicity).

        For toxicity endpoints like hERG:
        - Strong binding (< -8 kcal/mol) = high toxicity risk
        - Moderate binding (-6 to -8) = moderate risk
        - Weak binding (> -6) = low risk
        """
        if self.affinity is None:
            return 0.5  # Unknown

        # Map affinity to risk score
        # -12 or better -> 1.0 (very high risk)
        # -8 -> 0.7 (high risk)
        # -6 -> 0.3 (moderate risk)
        # -4 or weaker -> 0.0 (low risk)

        affinity = self.affinity
        if affinity <= -12:
            return 1.0
        elif affinity >= -4:
            return 0.0
        else:
            # Linear interpolation between -12 and -4
            return (affinity + 4) / (-8)  # Maps -12->1.0, -4->0.0


class DockingEngine(ABC):
    """Abstract base class for molecular docking engines."""

    @abstractmethod
    def dock(
        self,
        smiles: str,
        protein_path: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exhaustiveness: int = 8,
        n_poses: int = 9,
    ) -> DockingResult:
        """
        Perform molecular docking.

        Args:
            smiles: SMILES string of ligand
            protein_path: Path to prepared protein structure (PDBQT format)
            center: (x, y, z) coordinates of binding site center
            box_size: (x, y, z) dimensions of search box
            exhaustiveness: Sampling exhaustiveness (higher = more accurate but slower)
            n_poses: Number of poses to generate

        Returns:
            DockingResult with binding affinity and pose information
        """
        pass

    @abstractmethod
    def prepare_ligand(self, smiles: str) -> str:
        """
        Prepare ligand for docking.

        Args:
            smiles: SMILES string

        Returns:
            Path to prepared ligand file (PDBQT format)
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the docking engine is available."""
        pass


class VinaDockingEngine(DockingEngine):
    """AutoDock Vina docking engine implementation."""

    def __init__(self, vina_path: Optional[str] = None):
        """
        Initialize Vina docking engine.

        Args:
            vina_path: Path to Vina executable (if not using Python bindings)
        """
        self.vina_path = vina_path or self._find_vina_executable()
        self.use_python_api = VINA_PYTHON_AVAILABLE
        self._temp_dir = tempfile.mkdtemp(prefix="vina_docking_")

    def _find_vina_executable(self) -> Optional[str]:
        """Find Vina executable in PATH or common locations."""
        import shutil

        # Check PATH
        vina_cmd = shutil.which("vina")
        if vina_cmd:
            return vina_cmd

        # Check common locations
        common_paths = [
            "/usr/local/bin/vina",
            "/usr/bin/vina",
            "C:/Program Files/Vina/vina.exe",
            os.path.expanduser("~/bin/vina"),
        ]

        for path in common_paths:
            if os.path.isfile(path):
                return path

        return None

    def is_available(self) -> bool:
        """Check if Vina is available (Python API or executable)."""
        if self.use_python_api:
            return True
        return self.vina_path is not None

    def prepare_ligand(self, smiles: str) -> str:
        """
        Prepare ligand from SMILES for Vina docking.

        Uses Meeko (preferred) or RDKit for 3D generation and PDBQT conversion.
        """
        if not RDKIT_AVAILABLE:
            raise RuntimeError("RDKit required for ligand preparation")

        # Parse SMILES and generate 3D coordinates
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D coordinates
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result == -1:
            # Try with random coordinates
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        # Optimize geometry
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol, maxIters=200)

        # Generate unique filename using timestamp to avoid caching issues
        import time
        ligand_id = f"{abs(hash(smiles)) % (10 ** 8)}_{int(time.time() * 1000) % 100000}"

        if MEEKO_AVAILABLE:
            # Use Meeko for PDBQT generation (preferred)
            pdbqt_path = os.path.join(self._temp_dir, f"ligand_{ligand_id}.pdbqt")
            preparator = MoleculePreparation()
            mol_setup = preparator.prepare(mol)
            pdbqt_string = PDBQTWriterLegacy.write_string(mol_setup)[0]

            with open(pdbqt_path, 'w') as f:
                f.write(pdbqt_string)
        else:
            # Fallback: Save as PDB and convert using Open Babel
            pdb_path = os.path.join(self._temp_dir, f"ligand_{ligand_id}.pdb")
            pdbqt_path = os.path.join(self._temp_dir, f"ligand_{ligand_id}.pdbqt")

            Chem.MolToPDBFile(mol, pdb_path)

            # Try Open Babel conversion
            try:
                subprocess.run(
                    ["obabel", pdb_path, "-O", pdbqt_path, "-xh"],
                    check=True,
                    capture_output=True
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Manual PDBQT generation (basic)
                pdbqt_path = self._pdb_to_pdbqt_simple(pdb_path, pdbqt_path)

        return pdbqt_path

    def _pdb_to_pdbqt_simple(self, pdb_path: str, pdbqt_path: str) -> str:
        """Simple PDB to PDBQT conversion for ligands (adds charges, atom types, and torsion info)."""
        with open(pdb_path, 'r') as f:
            pdb_lines = f.readlines()

        pdbqt_lines = []

        # Add REMARK for ligand
        pdbqt_lines.append("REMARK  SMILES ligand prepared by MultiEndpointTox\n")
        pdbqt_lines.append("REMARK  Simple PDBQT conversion - install meeko for better results\n")

        # Add ROOT (treat entire molecule as rigid for simplicity)
        pdbqt_lines.append("ROOT\n")

        atom_count = 0
        for line in pdb_lines:
            if line.startswith(("ATOM", "HETATM")):
                atom_count += 1

                # Get atom info from PDB line
                atom_name = line[12:16].strip()

                # Get element - try from columns 76-78 first, then from atom name
                if len(line) > 76:
                    element = line[76:78].strip()
                else:
                    # Extract element from atom name (first 1-2 chars)
                    element = ''.join(c for c in atom_name if c.isalpha())[:2]
                    if len(element) > 1 and element[1].isupper():
                        element = element[0]

                if not element:
                    element = "C"

                # Determine AutoDock atom type
                ad_type = self._get_autodock_type(element)

                # Build PDBQT line with proper formatting
                # Columns 1-6: Record name
                # Columns 7-11: Atom serial number
                # Columns 13-16: Atom name
                # Columns 18-20: Residue name
                # Column 22: Chain
                # Columns 23-26: Residue number
                # Columns 31-38: X
                # Columns 39-46: Y
                # Columns 47-54: Z
                # Columns 55-60: Occupancy
                # Columns 61-66: Temp factor
                # Columns 67-76: Charge
                # Columns 77-78: Atom type

                # Ensure we have coordinates (columns 30-54)
                if len(line) < 54:
                    continue

                # Extract coordinates
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except ValueError:
                    continue

                # Format PDBQT line
                new_line = (
                    f"ATOM  {atom_count:5d} {atom_name:<4s} LIG     1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00    +0.000 {ad_type:>2s}\n"
                )
                pdbqt_lines.append(new_line)

        # Close ROOT
        pdbqt_lines.append("ENDROOT\n")

        # Add TORSDOF (number of rotatable bonds - 0 for rigid docking)
        pdbqt_lines.append("TORSDOF 0\n")

        with open(pdbqt_path, 'w') as f:
            f.writelines(pdbqt_lines)

        logger.info(f"Created ligand PDBQT with {atom_count} atoms")
        return pdbqt_path

    def _get_autodock_type(self, element: str) -> str:
        """Map element to AutoDock atom type."""
        type_map = {
            "C": "C",
            "N": "N",
            "O": "OA",
            "S": "SA",
            "H": "HD",
            "F": "F",
            "Cl": "Cl",
            "Br": "Br",
            "I": "I",
            "P": "P",
        }
        return type_map.get(element, "C")

    def dock(
        self,
        smiles: str,
        protein_path: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exhaustiveness: int = 8,
        n_poses: int = 9,
    ) -> DockingResult:
        """
        Perform docking using AutoDock Vina.
        """
        import time
        start_time = time.time()

        try:
            # Prepare ligand
            ligand_path = self.prepare_ligand(smiles)

            if self.use_python_api:
                result = self._dock_python_api(
                    ligand_path, protein_path, center, box_size,
                    exhaustiveness, n_poses
                )
            else:
                result = self._dock_command_line(
                    ligand_path, protein_path, center, box_size,
                    exhaustiveness, n_poses
                )

            result.smiles = smiles
            result.execution_time = time.time() - start_time
            return result

        except Exception as e:
            logger.error(f"Docking failed for {smiles}: {e}")
            return DockingResult(
                success=False,
                smiles=smiles,
                protein_id=Path(protein_path).stem,
                error=str(e),
                execution_time=time.time() - start_time
            )

    def _dock_python_api(
        self,
        ligand_path: str,
        protein_path: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exhaustiveness: int,
        n_poses: int,
    ) -> DockingResult:
        """Dock using Vina Python bindings."""
        v = Vina(sf_name='vina')

        v.set_receptor(protein_path)
        v.set_ligand_from_file(ligand_path)
        v.compute_vina_maps(
            center=list(center),
            box_size=list(box_size)
        )

        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

        # Get results
        energies = v.energies()

        poses = []
        for i, energy in enumerate(energies):
            poses.append({
                "pose_id": i + 1,
                "affinity": float(energy[0]),
                "rmsd_lb": float(energy[1]) if len(energy) > 1 else None,
                "rmsd_ub": float(energy[2]) if len(energy) > 2 else None,
            })

        return DockingResult(
            success=True,
            smiles="",
            protein_id=Path(protein_path).stem,
            affinity=float(energies[0][0]) if energies else None,
            poses=poses,
            num_poses=len(poses),
            best_pose_rmsd=float(energies[0][1]) if energies and len(energies[0]) > 1 else None,
        )

    def _dock_command_line(
        self,
        ligand_path: str,
        protein_path: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float],
        exhaustiveness: int,
        n_poses: int,
    ) -> DockingResult:
        """Dock using Vina command-line executable."""
        if not self.vina_path:
            raise RuntimeError("Vina executable not found")

        output_path = os.path.join(
            self._temp_dir,
            f"output_{abs(hash(ligand_path)) % 10**6}.pdbqt"
        )
        log_path = os.path.join(
            self._temp_dir,
            f"log_{abs(hash(ligand_path)) % 10**6}.txt"
        )

        cmd = [
            self.vina_path,
            "--receptor", protein_path,
            "--ligand", ligand_path,
            "--center_x", str(center[0]),
            "--center_y", str(center[1]),
            "--center_z", str(center[2]),
            "--size_x", str(box_size[0]),
            "--size_y", str(box_size[1]),
            "--size_z", str(box_size[2]),
            "--exhaustiveness", str(exhaustiveness),
            "--num_modes", str(n_poses),
            "--out", output_path,
            "--log", log_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            raise RuntimeError(f"Vina failed: {result.stderr}")

        # Parse log file for energies
        poses = self._parse_vina_log(log_path)

        return DockingResult(
            success=True,
            smiles="",
            protein_id=Path(protein_path).stem,
            affinity=poses[0]["affinity"] if poses else None,
            poses=poses,
            num_poses=len(poses),
            best_pose_rmsd=poses[0].get("rmsd_lb") if poses else None,
        )

    def _parse_vina_log(self, log_path: str) -> List[Dict[str, Any]]:
        """Parse Vina log file for docking results."""
        poses = []

        try:
            with open(log_path, 'r') as f:
                in_results = False
                for line in f:
                    if "mode |   affinity" in line:
                        in_results = True
                        continue
                    if in_results and line.strip():
                        parts = line.split()
                        if len(parts) >= 4 and parts[0].isdigit():
                            poses.append({
                                "pose_id": int(parts[0]),
                                "affinity": float(parts[1]),
                                "rmsd_lb": float(parts[2]),
                                "rmsd_ub": float(parts[3]),
                            })
        except Exception as e:
            logger.warning(f"Failed to parse Vina log: {e}")

        return poses

    def cleanup(self):
        """Clean up temporary files."""
        import shutil
        try:
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            pass

    def __del__(self):
        """Destructor to clean up temp files."""
        self.cleanup()


class SminaDockingEngine(VinaDockingEngine):
    """
    Smina docking engine - a fork of Vina with additional scoring functions.

    Smina supports custom scoring functions and is often faster than Vina.
    """

    def __init__(self, smina_path: Optional[str] = None):
        super().__init__()
        self.vina_path = smina_path or self._find_smina_executable()
        self.use_python_api = False  # Smina doesn't have Python bindings

    def _find_smina_executable(self) -> Optional[str]:
        """Find Smina executable."""
        import shutil

        smina_cmd = shutil.which("smina")
        if smina_cmd:
            return smina_cmd

        common_paths = [
            "/usr/local/bin/smina",
            "/usr/bin/smina",
            os.path.expanduser("~/bin/smina"),
        ]

        for path in common_paths:
            if os.path.isfile(path):
                return path

        return None

    def is_available(self) -> bool:
        """Check if Smina is available."""
        return self.vina_path is not None
