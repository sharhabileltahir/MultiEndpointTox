"""Molecular Descriptor Calculator - RDKit 2D descriptors for QSAR."""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


class DescriptorCalculator:

    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.interim_dir.mkdir(parents=True, exist_ok=True)

    def calculate(self, endpoint):
        data_path = self.processed_dir / f"{endpoint}_curated.csv"
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} compounds for descriptor calculation")

        descriptor_names = [desc[0] for desc in Descriptors._descList]
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
        logger.info(f"Calculating {len(descriptor_names)} RDKit descriptors...")

        results = []
        failed = 0
        for idx, row in df.iterrows():
            try:
                mol = Chem.MolFromSmiles(row["std_smiles"])
                if mol is not None:
                    descs = calc.CalcDescriptors(mol)
                    results.append(descs)
                else:
                    results.append([np.nan] * len(descriptor_names))
                    failed += 1
            except Exception:
                results.append([np.nan] * len(descriptor_names))
                failed += 1

        desc_df = pd.DataFrame(results, columns=descriptor_names, index=df.index)

        if failed > 0:
            logger.warning(f"{failed} compounds failed descriptor calculation")

        desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
        missing_pct = desc_df.isnull().sum() / len(desc_df)
        desc_df = desc_df.loc[:, missing_pct < 0.2]
        desc_df = desc_df.loc[:, desc_df.std() > 0]
        desc_df = desc_df.fillna(desc_df.median())

        logger.info(f"Final descriptor matrix: {desc_df.shape}")
        output_path = self.interim_dir / f"{endpoint}_descriptors.csv"
        desc_df.to_csv(output_path, index=False)
        return desc_df
