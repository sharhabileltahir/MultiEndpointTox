"""
Data Cleaner - Standardize, deduplicate, and validate chemical structures.
Supports both regression (pchembl_value) and classification (activity labels).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger

from rdkit import Chem
from rdkit.Chem import Descriptors, inchi
from rdkit.Chem.MolStandardize import rdMolStandardize


class DataCleaner:

    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, df, endpoint):
        """
        Clean data for regression or classification based on endpoint type.

        - herg: Regression (uses pchembl_value)
        - hepatotox/nephrotox: Classification (uses activity_comment)
        """
        logger.info(f"Cleaning pipeline started - {len(df)} input records")

        # Check if data already has activity_label (e.g., from DIRIL)
        if "activity_label" in df.columns and "std_smiles" in df.columns:
            logger.info("Data already has activity labels (pre-processed dataset)")
            return self._clean_preprocessed_classification(df, endpoint)

        # Determine task type based on endpoint
        if endpoint == "herg":
            return self._clean_regression(df, endpoint)
        else:
            # Try regression first, fall back to classification
            regression_df = self._try_regression_clean(df)
            if len(regression_df) >= 100:
                logger.info(f"Using regression mode for {endpoint}")
                return self._finalize_clean(regression_df, endpoint, task="regression")
            else:
                logger.info(f"Insufficient pchembl data ({len(regression_df)}), switching to classification mode")
                return self._clean_classification(df, endpoint)

    def _clean_regression(self, df, endpoint):
        """Standard regression cleaning using pchembl_value."""
        df = df.dropna(subset=["canonical_smiles", "pchembl_value"])
        logger.info(f"After removing missing values: {len(df)}")

        df["std_smiles"] = df["canonical_smiles"].apply(self._standardize_smiles)
        df = df.dropna(subset=["std_smiles"])
        logger.info(f"After SMILES standardization: {len(df)}")

        df["inchikey"] = df["std_smiles"].apply(self._smiles_to_inchikey)
        df = df.dropna(subset=["inchikey"])

        df["pchembl_value"] = pd.to_numeric(df["pchembl_value"], errors="coerce")
        df = (
            df.groupby("inchikey")
            .agg({
                "std_smiles": "first",
                "pchembl_value": "median",
                "molecule_chembl_id": "first",
                "canonical_smiles": "first",
            })
            .reset_index()
        )
        logger.info(f"After deduplication: {len(df)} unique compounds")

        return self._finalize_clean(df, endpoint, task="regression")

    def _try_regression_clean(self, df):
        """Try to clean for regression, return cleaned df."""
        df_copy = df.copy()
        df_copy = df_copy.dropna(subset=["canonical_smiles", "pchembl_value"])
        return df_copy

    def _clean_preprocessed_classification(self, df, endpoint):
        """
        Clean data that already has activity_label and std_smiles (e.g., DIRIL).
        """
        logger.info("Cleaning pre-processed classification data")

        # Ensure required columns exist
        df = df.dropna(subset=["std_smiles", "activity_label"])
        logger.info(f"After removing missing values: {len(df)}")

        # Generate InChIKey if not present
        if "inchikey" not in df.columns or df["inchikey"].isna().any():
            df["inchikey"] = df["std_smiles"].apply(self._smiles_to_inchikey)
            df = df.dropna(subset=["inchikey"])
            logger.info(f"After InChIKey generation: {len(df)}")

        # Ensure activity_label is int
        df["activity_label"] = df["activity_label"].astype(int)

        # Deduplicate by InChIKey (majority vote for label)
        agg_dict = {
            "std_smiles": "first",
            "activity_label": lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else int(x.median()),
        }

        # Add optional columns to aggregation
        if "molecule_chembl_id" in df.columns:
            agg_dict["molecule_chembl_id"] = "first"
        if "canonical_smiles" in df.columns:
            agg_dict["canonical_smiles"] = "first"
        if "activity_comment" in df.columns:
            agg_dict["activity_comment"] = "first"
        if "drug_name" in df.columns:
            agg_dict["drug_name"] = "first"

        df = df.groupby("inchikey").agg(agg_dict).reset_index()
        logger.info(f"After deduplication: {len(df)} unique compounds")

        # Log class distribution
        class_counts = df["activity_label"].value_counts()
        logger.info(f"Class distribution: Toxic={class_counts.get(1, 0)}, Non-toxic={class_counts.get(0, 0)}")

        return self._finalize_clean(df, endpoint, task="classification")

    def _clean_classification(self, df, endpoint):
        """
        Clean data for classification using activity_comment.

        Maps activity comments to binary labels:
        - Toxic/Active/Positive -> 1
        - Non-toxic/Inactive/Negative -> 0
        """
        logger.info("Using classification mode (activity_comment)")

        # Check for activity_comment column
        if "activity_comment" not in df.columns:
            logger.error("No activity_comment column found for classification")
            return pd.DataFrame()

        # Keep rows with canonical_smiles and activity_comment
        df = df.dropna(subset=["canonical_smiles"])
        df = df[df["activity_comment"].notna() & (df["activity_comment"] != "")]
        logger.info(f"After removing missing SMILES/comments: {len(df)}")

        if len(df) == 0:
            logger.error("No valid records with activity comments")
            return pd.DataFrame()

        # Map activity comments to binary labels
        df["activity_label"] = df["activity_comment"].apply(self._map_activity_to_label)
        df = df.dropna(subset=["activity_label"])
        logger.info(f"After mapping activity labels: {len(df)}")

        # Standardize SMILES
        df["std_smiles"] = df["canonical_smiles"].apply(self._standardize_smiles)
        df = df.dropna(subset=["std_smiles"])
        logger.info(f"After SMILES standardization: {len(df)}")

        # Generate InChIKey for deduplication
        df["inchikey"] = df["std_smiles"].apply(self._smiles_to_inchikey)
        df = df.dropna(subset=["inchikey"])

        # Deduplicate by InChIKey (majority vote for label)
        df["activity_label"] = df["activity_label"].astype(int)
        df = (
            df.groupby("inchikey")
            .agg({
                "std_smiles": "first",
                "activity_label": lambda x: int(x.mode().iloc[0]) if len(x.mode()) > 0 else int(x.median()),
                "molecule_chembl_id": "first",
                "canonical_smiles": "first",
                "activity_comment": "first",
            })
            .reset_index()
        )
        logger.info(f"After deduplication: {len(df)} unique compounds")

        # Log class distribution
        class_counts = df["activity_label"].value_counts()
        logger.info(f"Class distribution: Toxic={class_counts.get(1, 0)}, Non-toxic={class_counts.get(0, 0)}")

        return self._finalize_clean(df, endpoint, task="classification")

    def _finalize_clean(self, df, endpoint, task="regression"):
        """Apply MW filter and save."""
        df["mol_wt"] = df["std_smiles"].apply(self._get_mol_weight)
        df = df[(df["mol_wt"] >= 100) & (df["mol_wt"] <= 900)]
        logger.info(f"After MW filter (100-900 Da): {len(df)}")

        # Add task type metadata
        df["task_type"] = task

        output_path = self.processed_dir / f"{endpoint}_curated.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Curated data saved to {output_path} (task: {task})")
        return df

    @staticmethod
    def _map_activity_to_label(comment):
        """
        Map activity comment to binary label.

        Returns:
            1 for toxic/active
            0 for non-toxic/inactive
            None for ambiguous
        """
        if pd.isna(comment):
            return None

        comment_lower = str(comment).lower().strip()

        # Toxic/Active indicators
        toxic_keywords = [
            "active", "toxic", "positive", "hepatotoxic", "nephrotoxic",
            "cytotoxic", "genotoxic", "cardiotoxic", "dili", "liver injury",
            "inhibitor", "inhibition", "potent", "strong"
        ]

        # Non-toxic/Inactive indicators
        nontoxic_keywords = [
            "inactive", "non-toxic", "nontoxic", "negative", "not active",
            "no activity", "no effect", "safe", "non-hepatotoxic", "weak",
            "not determined to be", "no significant"
        ]

        # Check for non-toxic first (more specific)
        for keyword in nontoxic_keywords:
            if keyword in comment_lower:
                return 0

        # Then check for toxic
        for keyword in toxic_keywords:
            if keyword in comment_lower:
                return 1

        # Ambiguous - skip
        return None

    @staticmethod
    def _standardize_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            uncharger = rdMolStandardize.Uncharger()
            mol = uncharger.uncharge(mol)
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    @staticmethod
    def _smiles_to_inchikey(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return inchi.MolToInchiKey(mol)
        except Exception:
            return None

    @staticmethod
    def _get_mol_weight(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return 0.0
            return Descriptors.MolWt(mol)
        except Exception:
            return 0.0
