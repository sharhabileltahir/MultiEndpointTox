"""Shared chemistry utility functions."""
from rdkit import Chem
from rdkit.Chem import Descriptors

def is_valid_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def get_molecular_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }
