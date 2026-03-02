"""Protein structure management for molecular docking."""

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass

from loguru import logger

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class ProteinTarget:
    """Configuration for a protein target."""
    name: str
    pdb_id: Optional[str] = None
    local_path: Optional[str] = None
    pdbqt_path: Optional[str] = None
    center: Tuple[float, float, float] = (0, 0, 0)
    box_size: Tuple[float, float, float] = (20, 20, 20)
    description: str = ""
    chain: str = "A"


# Pre-configured protein targets for toxicity endpoints
DEFAULT_TARGETS = {
    "herg": ProteinTarget(
        name="hERG Potassium Channel",
        pdb_id="5VA1",  # Cryo-EM structure of hERG
        center=(145.0, 145.0, 170.0),
        box_size=(25, 25, 30),
        description="Human ether-a-go-go related gene (hERG) potassium channel - cardiotoxicity target",
        chain="A",
    ),
    "hepatotox": ProteinTarget(
        name="CYP3A4",
        pdb_id="1TQN",  # CYP3A4 with ketoconazole
        center=(25.0, 10.0, 45.0),
        box_size=(22, 22, 22),
        description="Cytochrome P450 3A4 - major drug-metabolizing enzyme, hepatotoxicity",
        chain="A",
    ),
    "cyp2d6": ProteinTarget(
        name="CYP2D6",
        pdb_id="4WNT",  # CYP2D6 structure
        center=(30.0, 15.0, 40.0),
        box_size=(20, 20, 20),
        description="Cytochrome P450 2D6 - drug metabolism enzyme",
        chain="A",
    ),
    "cyp2c9": ProteinTarget(
        name="CYP2C9",
        pdb_id="1OG5",  # CYP2C9 with warfarin
        center=(20.0, 85.0, 45.0),
        box_size=(22, 22, 22),
        description="Cytochrome P450 2C9 - warfarin metabolism",
        chain="A",
    ),
    "ppar_gamma": ProteinTarget(
        name="PPAR-gamma",
        pdb_id="2PRG",  # PPAR-gamma with rosiglitazone
        center=(25.0, 0.0, 15.0),
        box_size=(22, 22, 22),
        description="Peroxisome proliferator-activated receptor gamma - metabolic toxicity",
        chain="A",
    ),
    "ar": ProteinTarget(
        name="Androgen Receptor",
        pdb_id="2AM9",  # AR ligand binding domain
        center=(10.0, 25.0, 5.0),
        box_size=(20, 20, 20),
        description="Androgen receptor - reproductive/endocrine toxicity",
        chain="A",
    ),
    "er_alpha": ProteinTarget(
        name="Estrogen Receptor Alpha",
        pdb_id="1ERE",  # ER-alpha with estradiol
        center=(95.0, 15.0, 25.0),
        box_size=(20, 20, 20),
        description="Estrogen receptor alpha - reproductive toxicity",
        chain="A",
    ),
}


class ProteinStructureManager:
    """
    Manages protein structures for molecular docking.

    Handles:
    - Downloading structures from PDB
    - Converting PDB to PDBQT format
    - Caching prepared structures
    - Binding site configuration
    """

    def __init__(
        self,
        structures_dir: str = "data/structures",
        config: Optional[Dict] = None
    ):
        """
        Initialize protein structure manager.

        Args:
            structures_dir: Directory to store protein structures
            config: Optional configuration dict with custom targets
        """
        self.structures_dir = Path(structures_dir)
        self.structures_dir.mkdir(parents=True, exist_ok=True)

        self.targets: Dict[str, ProteinTarget] = DEFAULT_TARGETS.copy()

        # Load custom targets from config
        if config and "docking" in config:
            self._load_custom_targets(config["docking"])

        self._prepared_proteins: Dict[str, str] = {}

    def _load_custom_targets(self, docking_config: Dict):
        """Load custom protein targets from configuration."""
        if "protein_structures" not in docking_config:
            return

        for endpoint, target_config in docking_config["protein_structures"].items():
            self.targets[endpoint] = ProteinTarget(
                name=target_config.get("name", endpoint),
                pdb_id=target_config.get("pdb_id"),
                local_path=target_config.get("local_path"),
                pdbqt_path=target_config.get("pdbqt_path"),
                center=tuple(target_config.get("center", [0, 0, 0])),
                box_size=tuple(target_config.get("box_size", [20, 20, 20])),
                description=target_config.get("description", ""),
                chain=target_config.get("chain", "A"),
            )

    def get_available_targets(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available protein targets."""
        return {
            name: {
                "name": target.name,
                "pdb_id": target.pdb_id,
                "description": target.description,
                "center": target.center,
                "box_size": target.box_size,
                "prepared": name in self._prepared_proteins,
            }
            for name, target in self.targets.items()
        }

    def get_target(self, endpoint: str) -> Optional[ProteinTarget]:
        """Get protein target configuration for an endpoint."""
        return self.targets.get(endpoint)

    def prepare_protein(self, endpoint: str, force_download: bool = False) -> str:
        """
        Prepare protein structure for docking.

        Downloads PDB if needed, converts to PDBQT format.

        Args:
            endpoint: Endpoint name (e.g., "herg", "hepatotox")
            force_download: Force re-download even if file exists

        Returns:
            Path to prepared PDBQT file
        """
        if endpoint in self._prepared_proteins and not force_download:
            return self._prepared_proteins[endpoint]

        target = self.targets.get(endpoint)
        if not target:
            raise ValueError(f"No protein target configured for endpoint: {endpoint}")

        # Check for pre-prepared PDBQT
        if target.pdbqt_path and os.path.exists(target.pdbqt_path):
            self._prepared_proteins[endpoint] = target.pdbqt_path
            return target.pdbqt_path

        # Check for local PDB
        if target.local_path and os.path.exists(target.local_path):
            pdb_path = target.local_path
        elif target.pdb_id:
            pdb_path = self._download_pdb(target.pdb_id, force_download)
        else:
            raise ValueError(f"No PDB source configured for {endpoint}")

        # Convert to PDBQT
        pdbqt_path = self._convert_to_pdbqt(pdb_path, endpoint, target.chain)
        self._prepared_proteins[endpoint] = pdbqt_path

        logger.info(f"Prepared protein for {endpoint}: {pdbqt_path}")
        return pdbqt_path

    def _download_pdb(self, pdb_id: str, force: bool = False) -> str:
        """
        Download PDB structure from RCSB.

        Args:
            pdb_id: PDB identifier
            force: Force re-download

        Returns:
            Path to downloaded PDB file
        """
        if not REQUESTS_AVAILABLE:
            raise RuntimeError("requests library required for PDB download")

        pdb_path = self.structures_dir / f"{pdb_id.lower()}.pdb"

        if pdb_path.exists() and not force:
            logger.debug(f"Using cached PDB: {pdb_path}")
            return str(pdb_path)

        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"

        logger.info(f"Downloading PDB {pdb_id} from RCSB...")
        response = requests.get(url, timeout=60)

        if response.status_code != 200:
            raise RuntimeError(f"Failed to download PDB {pdb_id}: HTTP {response.status_code}")

        with open(pdb_path, 'w') as f:
            f.write(response.text)

        logger.info(f"Downloaded {pdb_id} to {pdb_path}")
        return str(pdb_path)

    def _convert_to_pdbqt(
        self,
        pdb_path: str,
        endpoint: str,
        chain: str = "A"
    ) -> str:
        """
        Convert PDB to PDBQT format for AutoDock Vina.

        Args:
            pdb_path: Path to input PDB file
            endpoint: Endpoint name (for output filename)
            chain: Chain identifier to extract

        Returns:
            Path to PDBQT file
        """
        pdbqt_path = self.structures_dir / f"{endpoint}_receptor.pdbqt"

        # Try using prepare_receptor from AutoDockTools (MGLTools)
        try:
            self._convert_with_mgltools(pdb_path, str(pdbqt_path), chain)
            return str(pdbqt_path)
        except Exception as e:
            logger.warning(f"MGLTools conversion failed: {e}")

        # Try Open Babel
        try:
            self._convert_with_openbabel(pdb_path, str(pdbqt_path), chain)
            return str(pdbqt_path)
        except Exception as e:
            logger.warning(f"Open Babel conversion failed: {e}")

        # Fallback: Simple conversion
        try:
            self._convert_simple(pdb_path, str(pdbqt_path), chain)
        except Exception as e:
            logger.error(f"Simple conversion failed: {e}")
            raise RuntimeError(f"All PDBQT conversion methods failed for {pdb_path}")

        if not pdbqt_path.exists():
            raise RuntimeError(f"PDBQT file was not created: {pdbqt_path}")

        return str(pdbqt_path)

    def _convert_with_mgltools(
        self,
        pdb_path: str,
        pdbqt_path: str,
        chain: str
    ):
        """Convert using MGLTools prepare_receptor."""
        # First, extract the specific chain if needed
        cleaned_pdb = self._extract_chain(pdb_path, chain)

        cmd = [
            "prepare_receptor",
            "-r", cleaned_pdb,
            "-o", pdbqt_path,
            "-A", "hydrogens",
            "-U", "nphs_lps_waters_nonstdres",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"prepare_receptor failed: {result.stderr}")

    def _convert_with_openbabel(
        self,
        pdb_path: str,
        pdbqt_path: str,
        chain: str
    ):
        """Convert using Open Babel."""
        cleaned_pdb = self._extract_chain(pdb_path, chain)

        cmd = [
            "obabel", cleaned_pdb,
            "-O", pdbqt_path,
            "-xr",  # Output as receptor
            "-h",   # Add hydrogens
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            raise RuntimeError(f"Open Babel failed: {result.stderr}")

    def _convert_simple(
        self,
        pdb_path: str,
        pdbqt_path: str,
        chain: str
    ):
        """
        Simple PDB to PDBQT conversion.

        This is a basic conversion that adds charges and atom types.
        For production use, MGLTools or Open Babel is recommended.
        """
        logger.warning("Using simple PDB->PDBQT conversion. "
                      "Install MGLTools or Open Babel for better results.")

        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        pdbqt_lines = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                # Filter by chain
                line_chain = line[21] if len(line) > 21 else "A"
                if line_chain != chain:
                    continue

                # Skip water, ions, and metal-containing residues
                residue = line[17:20].strip()
                if residue in ["HOH", "WAT", "NA", "CL", "MG", "CA", "ZN", "FE", "HEM", "HEC"]:
                    continue

                # Get element early to skip metals
                element = line[76:78].strip() if len(line) > 76 else line[12:14].strip()
                element = element.upper().strip()

                # Skip metal atoms - Vina doesn't support them
                if element in ["FE", "ZN", "MG", "MN", "CU", "CO", "NI", "CA"]:
                    continue

                # Get atom info
                atom_name = line[12:16].strip()
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]

                # Determine AutoDock atom type
                ad_type = self._get_protein_atom_type(element, atom_name, residue)

                # PDBQT format requires specific column positions:
                # Columns 1-6: Record name (ATOM/HETATM)
                # Columns 7-11: Atom serial number
                # Columns 13-16: Atom name
                # Columns 17: Alternate location indicator
                # Columns 18-20: Residue name
                # Column 22: Chain identifier
                # Columns 23-26: Residue sequence number
                # Columns 31-38: X coordinate
                # Columns 39-46: Y coordinate
                # Columns 47-54: Z coordinate
                # Columns 55-60: Occupancy (we use for partial charge)
                # Columns 61-66: Temperature factor (we use for partial charge)
                # Columns 77-78: Element symbol
                # Columns 79-80: Atom type (AutoDock)

                # Ensure line is long enough
                if len(line) < 54:
                    continue

                # Build proper PDBQT line
                # Keep first 54 characters (coordinates), add charge and type
                base = line[:54].rstrip()
                # Pad to exactly 54 characters
                base = base.ljust(54)
                # Add occupancy (partial charge placeholder) and temp factor
                new_line = f"{base}  1.00  0.00    +0.000 {ad_type:>2}\n"
                pdbqt_lines.append(new_line)

            elif line.startswith("TER"):
                pdbqt_lines.append("TER\n")
            # Skip END, ENDMDL, MASTER, etc. - Vina doesn't need them for receptor

        # Write the file
        with open(pdbqt_path, 'w') as f:
            f.writelines(pdbqt_lines)

        logger.info(f"Created PDBQT with {len(pdbqt_lines)} lines")

    def _extract_chain(self, pdb_path: str, chain: str) -> str:
        """Extract a specific chain from PDB file."""
        output_path = pdb_path.replace(".pdb", f"_chain{chain}.pdb")

        with open(pdb_path, 'r') as f:
            lines = f.readlines()

        filtered_lines = []
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                line_chain = line[21] if len(line) > 21 else "A"
                if line_chain == chain:
                    filtered_lines.append(line)
            elif line.startswith(("HEADER", "TITLE", "COMPND", "SOURCE",
                                 "REMARK", "SEQRES", "TER", "END")):
                filtered_lines.append(line)

        with open(output_path, 'w') as f:
            f.writelines(filtered_lines)

        return output_path

    def _get_protein_atom_type(
        self,
        element: str,
        atom_name: str,
        residue: str
    ) -> str:
        """Determine AutoDock atom type for protein atoms."""
        element = element.upper().strip()

        # Hydrogen types
        if element == "H":
            # Polar hydrogens
            if atom_name in ["HN", "H", "HE", "HE1", "HE2", "HD1", "HD2",
                           "HH", "HH11", "HH12", "HH21", "HH22", "HG", "HG1"]:
                return "HD"
            return "H"

        # Carbon types
        if element == "C":
            if atom_name in ["C", "CA", "CB"]:
                return "C"
            # Aromatic carbons (Phe, Tyr, Trp, His)
            if residue in ["PHE", "TYR", "TRP", "HIS"]:
                if atom_name.startswith("C") and atom_name not in ["C", "CA", "CB"]:
                    return "A"
            return "C"

        # Nitrogen types
        if element == "N":
            # Aromatic nitrogens
            if residue in ["HIS", "TRP"] and atom_name in ["NE2", "ND1", "NE1"]:
                return "NA"
            return "N"

        # Oxygen types
        if element == "O":
            return "OA"

        # Sulfur
        if element == "S":
            return "SA"

        # Metal atoms (common in metalloproteins like CYPs)
        if element in ["FE", "ZN", "MG", "CA", "MN", "CU", "CO", "NI"]:
            return element

        # Halogens
        if element in ["F", "CL", "BR", "I"]:
            return element

        # Phosphorus
        if element == "P":
            return "P"

        # Default - return element if 1-2 chars, otherwise C
        if len(element) <= 2:
            return element
        return "C"

    def get_binding_site_params(
        self,
        endpoint: str
    ) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """
        Get binding site center and box size for an endpoint.

        Returns:
            (center, box_size) tuples
        """
        target = self.targets.get(endpoint)
        if not target:
            raise ValueError(f"No target configured for {endpoint}")

        return target.center, target.box_size

    def set_binding_site(
        self,
        endpoint: str,
        center: Tuple[float, float, float],
        box_size: Tuple[float, float, float]
    ):
        """Update binding site parameters for an endpoint."""
        if endpoint not in self.targets:
            raise ValueError(f"Unknown endpoint: {endpoint}")

        self.targets[endpoint].center = center
        self.targets[endpoint].box_size = box_size
