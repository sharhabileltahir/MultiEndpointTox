#!/usr/bin/env python3
"""
Prepare a Chemprop-ready hERG regression dataset with columns:
- smiles
- pchembl_value
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def infer_smiles_column(columns: List[str], preferred: str) -> str:
    if preferred in columns:
        return preferred

    lookup = {c.lower(): c for c in columns}
    for candidate in [preferred.lower(), "smiles", "std_smiles", "canonical_smiles"]:
        if candidate in lookup:
            return lookup[candidate]
    raise ValueError(
        f"Could not locate a SMILES column. Preferred='{preferred}', columns={columns}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Chemprop hERG regression CSV (smiles, pchembl_value)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="data/processed/herg_curated.csv",
        help="Input hERG CSV containing SMILES and pchembl_value.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="data/processed/herg_chemprop_regression.csv",
        help="Output Chemprop-ready CSV path.",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="std_smiles",
        help="Preferred SMILES column name.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="pchembl_value",
        help="Target column name.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("prepare_chemprop_herg")

    in_path = Path(args.input_csv)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    smiles_col = infer_smiles_column(df.columns.tolist(), args.smiles_col)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in {in_path}")

    work = df[[smiles_col, args.target_col]].copy()
    work.columns = ["smiles", "pchembl_value"]
    work["pchembl_value"] = pd.to_numeric(work["pchembl_value"], errors="coerce")
    work = work.dropna(subset=["smiles", "pchembl_value"])
    work["smiles"] = work["smiles"].astype(str).str.strip()
    work = work[work["smiles"] != ""]

    # Keep only valid SMILES to avoid runtime failures in Chemprop featurization.
    valid_mask = work["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    invalid_n = int((~valid_mask).sum())
    if invalid_n > 0:
        logger.warning("Dropping %d rows with invalid SMILES.", invalid_n)
    work = work.loc[valid_mask].reset_index(drop=True)

    work.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(work), out_path)


if __name__ == "__main__":
    main()
