"""Basic tests for the pipeline modules."""

import pytest
from pathlib import Path


def test_project_structure():
    """Verify critical project directories exist."""
    root = Path(__file__).parent.parent
    required_dirs = ["src", "config", "data/raw", "data/processed", "models", "reports"]
    for d in required_dirs:
        assert (root / d).exists(), f"Missing directory: {d}"


def test_config_loads():
    """Verify config.yaml is valid."""
    import yaml
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    assert "project" in config
    assert "data_curation" in config
    assert "modeling" in config


def test_rdkit_import():
    """Verify RDKit is available."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles("CCO")
    assert mol is not None


def test_smiles_standardization():
    """Test SMILES standardization function."""
    from src.data_curation.data_cleaner import DataCleaner
    result = DataCleaner._standardize_smiles("c1ccccc1")
    assert result == "c1ccccc1"
    assert DataCleaner._standardize_smiles("invalid_smiles") is None
