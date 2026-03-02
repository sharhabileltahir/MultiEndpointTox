"""Utility functions for molecular docking setup and installation."""

import os
import sys
import shutil
import platform
import subprocess
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger


def check_docking_dependencies() -> dict:
    """
    Check availability of docking dependencies.

    Returns:
        Dictionary with status of each dependency
    """
    status = {
        "rdkit": False,
        "meeko": False,
        "openbabel": False,
        "vina_python": False,
        "vina_cli": False,
        "vina_path": None,
        "ready": False,
    }

    # Check RDKit
    try:
        from rdkit import Chem
        status["rdkit"] = True
    except ImportError:
        pass

    # Check Meeko
    try:
        from meeko import MoleculePreparation
        status["meeko"] = True
    except ImportError:
        pass

    # Check Open Babel
    try:
        result = subprocess.run(
            ["obabel", "-V"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            status["openbabel"] = True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Check Vina Python bindings
    try:
        from vina import Vina
        status["vina_python"] = True
    except ImportError:
        pass

    # Check Vina CLI
    vina_path = find_vina_executable()
    if vina_path:
        status["vina_cli"] = True
        status["vina_path"] = vina_path

    # Determine if docking is ready
    has_vina = status["vina_python"] or status["vina_cli"]
    has_ligand_prep = status["meeko"] or status["openbabel"] or status["rdkit"]
    status["ready"] = has_vina and has_ligand_prep and status["rdkit"]

    return status


def find_vina_executable() -> Optional[str]:
    """Find AutoDock Vina executable in system PATH or common locations."""
    # Check PATH first
    vina_cmd = shutil.which("vina")
    if vina_cmd:
        return vina_cmd

    # Check common locations based on OS
    if platform.system() == "Windows":
        common_paths = [
            r"C:\Program Files\Vina\vina.exe",
            r"C:\Program Files (x86)\Vina\vina.exe",
            r"C:\vina\vina.exe",
            os.path.expanduser(r"~\vina\vina.exe"),
            os.path.expanduser(r"~\Downloads\vina_1.2.5_win\vina.exe"),
        ]
    else:
        common_paths = [
            "/usr/local/bin/vina",
            "/usr/bin/vina",
            "/opt/vina/vina",
            os.path.expanduser("~/bin/vina"),
            os.path.expanduser("~/vina/vina"),
        ]

    for path in common_paths:
        if os.path.isfile(path):
            return path

    return None


def get_installation_instructions() -> str:
    """Get installation instructions for docking dependencies."""
    status = check_docking_dependencies()

    instructions = []
    instructions.append("=" * 60)
    instructions.append("MOLECULAR DOCKING SETUP INSTRUCTIONS")
    instructions.append("=" * 60)
    instructions.append("")

    # Current status
    instructions.append("Current Status:")
    instructions.append(f"  - RDKit: {'✓ Installed' if status['rdkit'] else '✗ Missing'}")
    instructions.append(f"  - Meeko: {'✓ Installed' if status['meeko'] else '✗ Missing (optional)'}")
    instructions.append(f"  - Open Babel: {'✓ Installed' if status['openbabel'] else '✗ Missing (optional)'}")
    instructions.append(f"  - Vina Python: {'✓ Installed' if status['vina_python'] else '✗ Missing'}")
    instructions.append(f"  - Vina CLI: {'✓ Found at ' + status['vina_path'] if status['vina_cli'] else '✗ Not found'}")
    instructions.append(f"  - Ready for docking: {'✓ Yes' if status['ready'] else '✗ No'}")
    instructions.append("")

    if status["ready"]:
        instructions.append("Docking is ready to use!")
        return "\n".join(instructions)

    instructions.append("Installation Steps:")
    instructions.append("")

    # RDKit
    if not status["rdkit"]:
        instructions.append("1. Install RDKit (required):")
        instructions.append("   conda install -c conda-forge rdkit")
        instructions.append("")

    # Vina options
    if not status["vina_python"] and not status["vina_cli"]:
        instructions.append("2. Install AutoDock Vina (choose one option):")
        instructions.append("")
        instructions.append("   OPTION A - Command-line Vina (Recommended for Windows):")
        instructions.append("   - Download from: https://vina.scripps.edu/downloads/")
        instructions.append("   - Extract to C:\\vina\\ or add to PATH")
        instructions.append("   - Windows: Download vina_1.2.5_windows_x86_64.zip")
        instructions.append("")
        instructions.append("   OPTION B - Conda install (Linux/Mac):")
        instructions.append("   conda install -c conda-forge autodock-vina")
        instructions.append("")
        instructions.append("   OPTION C - Python bindings (requires Boost):")
        instructions.append("   conda install -c conda-forge boost")
        instructions.append("   pip install vina")
        instructions.append("")

    # Ligand preparation
    if not status["meeko"] and not status["openbabel"]:
        instructions.append("3. Install ligand preparation tools (optional but recommended):")
        instructions.append("   pip install meeko")
        instructions.append("   # OR")
        instructions.append("   conda install -c conda-forge openbabel")
        instructions.append("")

    instructions.append("After installation, restart the API server.")
    instructions.append("")

    return "\n".join(instructions)


def setup_vina_path(vina_path: str) -> bool:
    """
    Set up Vina executable path.

    Args:
        vina_path: Path to Vina executable

    Returns:
        True if successful
    """
    if not os.path.isfile(vina_path):
        logger.error(f"Vina executable not found at: {vina_path}")
        return False

    # Test that it runs
    try:
        result = subprocess.run(
            [vina_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "AutoDock Vina" in result.stdout or result.returncode == 0:
            logger.info(f"Vina executable validated: {vina_path}")
            return True
    except Exception as e:
        logger.error(f"Failed to validate Vina: {e}")

    return False


def download_test_structure() -> Tuple[bool, str]:
    """
    Download a test protein structure to verify setup.

    Returns:
        (success, message)
    """
    try:
        import requests
    except ImportError:
        return False, "requests library not installed"

    test_pdb = "1ERE"  # Small protein for testing
    url = f"https://files.rcsb.org/download/{test_pdb}.pdb"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return True, f"Successfully downloaded {test_pdb} ({len(response.content)} bytes)"
        else:
            return False, f"Failed to download: HTTP {response.status_code}"
    except Exception as e:
        return False, f"Download error: {e}"


if __name__ == "__main__":
    # Print installation instructions when run directly
    print(get_installation_instructions())
