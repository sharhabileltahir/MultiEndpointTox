"""Molecular Fingerprint Generator - Morgan, MACCS fingerprints."""

import pandas as pd
from pathlib import Path
from loguru import logger

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


class FingerprintGenerator:

    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.fp_config = config["features"]["morgan_fingerprints"]

    def generate(self, endpoint):
        data_path = self.processed_dir / f"{endpoint}_curated.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Generating fingerprints for {len(df)} compounds...")

        morgan_fps = self._generate_morgan(df["std_smiles"])
        logger.info(f"Morgan FP: {morgan_fps.shape}")

        maccs_fps = self._generate_maccs(df["std_smiles"])
        logger.info(f"MACCS keys: {maccs_fps.shape}")

        combined = pd.concat([morgan_fps, maccs_fps], axis=1)
        output_path = self.interim_dir / f"{endpoint}_fingerprints.csv"
        combined.to_csv(output_path, index=False)
        logger.info(f"Fingerprints saved: {combined.shape}")
        return combined

    def _generate_morgan(self, smiles_series):
        radius = self.fp_config["radius"]
        n_bits = self.fp_config["n_bits"]
        fps = []
        for smi in smiles_series:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
                    fps.append(list(fp))
                else:
                    fps.append([0] * n_bits)
            except Exception:
                fps.append([0] * n_bits)
        columns = [f"Morgan_{i}" for i in range(n_bits)]
        return pd.DataFrame(fps, columns=columns)

    def _generate_maccs(self, smiles_series):
        fps = []
        for smi in smiles_series:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = MACCSkeys.GenMACCSKeys(mol)
                    fps.append(list(fp))
                else:
                    fps.append([0] * 167)
            except Exception:
                fps.append([0] * 167)
        columns = [f"MACCS_{i}" for i in range(167)]
        return pd.DataFrame(fps, columns=columns)
