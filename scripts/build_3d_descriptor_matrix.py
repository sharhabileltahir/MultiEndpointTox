#!/usr/bin/env python3
"""
Build a cleaned 3D descriptor matrix from a CSV file.

Pipeline:
1) Read CSV with SMILES and binary target label.
2) Generate 10 ETKDG conformers per molecule.
3) Optimize each conformer with UFF and keep the lowest-energy conformer.
4) Compute 3D descriptors (plus optional SASA and Mordred 3D descriptors).
5) Remove low-variance features.
6) Remove highly correlated features.
7) Standardize features.
8) Save X_3D.npy and y.npy.
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors

try:
    from rdkit.Chem import rdFreeSASA

    FREESASA_AVAILABLE = True
except ImportError:
    FREESASA_AVAILABLE = False

try:
    from mordred import Calculator, descriptors as mordred_descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False


CORE_DESCRIPTOR_NAMES = [
    "asphericity",
    "eccentricity",
    "spherocity_index",
    "radius_of_gyration",
    "pmi1",
    "pmi2",
    "pmi3",
    "psa_3d",
    "sasa",
    "uff_lowest_energy",
]


def setup_logging(log_file: Path, level: str) -> None:
    """Configure console + file logging."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def set_reproducibility(seed: int) -> None:
    """Set all relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def _numeric(value: object) -> float:
    """Safely coerce a value to float."""
    try:
        return float(value)
    except Exception:
        return np.nan


def _safe_descriptor(callable_obj) -> float:
    """Run a descriptor function and return NaN on failure."""
    try:
        return float(callable_obj())
    except Exception:
        return np.nan


def _sanitize_feature_name(name: str) -> str:
    """Convert a descriptor name to a filesystem/model-safe feature name."""
    sanitized = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_").lower()


def _infer_smiles_column(df: pd.DataFrame, preferred: str) -> str:
    """Resolve SMILES column robustly (case-insensitive + common aliases)."""
    if preferred in df.columns:
        return preferred

    lower_to_original = {col.lower(): col for col in df.columns}
    candidates = [preferred.lower(), "smiles", "std_smiles", "canonical_smiles"]
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    raise ValueError(
        f"Could not find SMILES column. Tried '{preferred}' and common aliases."
    )


def _encode_binary_target(values: Iterable[object]) -> np.ndarray:
    """Encode a binary target to {0,1}."""
    series = pd.Series(list(values))
    if series.isna().any():
        raise ValueError("Target column contains missing values after filtering.")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        unique_vals = sorted(numeric.unique().tolist())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Target must be binary. Found {len(unique_vals)} classes: {unique_vals}"
            )
        if set(unique_vals) == {0.0, 1.0}:
            return numeric.astype(np.int64).to_numpy()

        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        logging.getLogger(__name__).info(
            "Mapped numeric target classes to binary labels: %s", mapping
        )
        return numeric.map(mapping).astype(np.int64).to_numpy()

    unique_vals = series.unique().tolist()
    if len(unique_vals) != 2:
        raise ValueError(
            f"Target must be binary. Found {len(unique_vals)} classes: {unique_vals}"
        )

    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
    logging.getLogger(__name__).info(
        "Mapped string target classes to binary labels: %s", mapping
    )
    return series.map(mapping).astype(np.int64).to_numpy()


class Descriptor3DPipeline:
    """End-to-end 3D descriptor extraction and cleaning."""

    POLAR_ATOMIC_NUMBERS = {7, 8, 15, 16}

    def __init__(self, n_conformers: int = 10, random_seed: int = 42) -> None:
        self.n_conformers = n_conformers
        self.random_seed = random_seed
        self.logger = logging.getLogger(self.__class__.__name__)

        self.mordred_calculator: Optional[Calculator] = None
        self.mordred_3d_indices: List[int] = []
        self.mordred_3d_names: List[str] = []
        if MORDRED_AVAILABLE:
            try:
                # Full Mordred registry with 3D descriptors enabled.
                self.mordred_calculator = Calculator(mordred_descriptors, ignore_3D=False)
                for idx, descriptor_obj in enumerate(self.mordred_calculator.descriptors):
                    if getattr(descriptor_obj, "require_3D", False):
                        name = f"mordred3d_{_sanitize_feature_name(str(descriptor_obj))}"
                        self.mordred_3d_indices.append(idx)
                        self.mordred_3d_names.append(name)
                self.logger.info(
                    "Mordred enabled with %d total descriptors; %d require 3D.",
                    len(self.mordred_calculator.descriptors),
                    len(self.mordred_3d_indices),
                )
            except Exception as exc:
                self.logger.warning("Failed to initialize Mordred descriptors: %s", exc)
                self.mordred_calculator = None
                self.mordred_3d_indices = []
                self.mordred_3d_names = []
        else:
            self.logger.info("Mordred not installed. Mordred 3D descriptors will be skipped.")

        if not FREESASA_AVAILABLE:
            self.logger.info(
                "rdFreeSASA not available. 'sasa' will be NaN and 'psa_3d' falls back to TPSA."
            )

    def _build_etkdg_params(self, seed: int, random_coords: bool) -> object:
        """Create ETKDGv3 parameters."""
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed)
        params.numThreads = 0
        params.useRandomCoords = random_coords
        return params

    def _generate_lowest_energy_conformer(
        self, mol: Chem.Mol, row_index: int
    ) -> Tuple[Optional[Chem.Mol], Optional[int], Optional[float], Optional[str]]:
        """Generate conformers, optimize with UFF, return lowest-energy conformer."""
        mol_h = Chem.AddHs(Chem.Mol(mol))
        per_molecule_seed = self.random_seed

        conf_ids: List[int] = []
        for use_random_coords in (False, True):
            params = self._build_etkdg_params(per_molecule_seed, use_random_coords)
            try:
                ids = AllChem.EmbedMultipleConfs(
                    mol_h,
                    numConfs=self.n_conformers,
                    params=params,
                )
                conf_ids = list(ids)
            except Exception:
                conf_ids = []

            if conf_ids:
                break

        if not conf_ids:
            return None, None, None, "ETKDG embedding failed"

        energies: List[Tuple[int, float]] = []
        if not AllChem.UFFHasAllMoleculeParams(mol_h):
            return None, None, None, "UFF parameters unavailable for at least one atom"

        for conf_id in conf_ids:
            try:
                # UFF geometry optimization as requested.
                AllChem.UFFOptimizeMolecule(mol_h, confId=conf_id, maxIters=500)
                ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=conf_id)
                if ff is None:
                    continue
                energies.append((conf_id, float(ff.CalcEnergy())))
            except Exception:
                continue

        if not energies:
            return None, None, None, "UFF optimization/energy evaluation failed"

        best_conf_id, best_energy = min(energies, key=lambda x: x[1])

        best_mol = Chem.Mol(mol_h)
        for conf_id in sorted(
            [conf.GetId() for conf in best_mol.GetConformers() if conf.GetId() != best_conf_id],
            reverse=True,
        ):
            best_mol.RemoveConformer(conf_id)

        kept_conf_id = best_mol.GetConformer().GetId()
        return best_mol, kept_conf_id, best_energy, None

    def _compute_sasa_and_psa3d(self, mol_3d: Chem.Mol, conf_id: int) -> Tuple[float, float]:
        """Compute SASA and 3D PSA (polar SASA) when rdFreeSASA is available."""
        if FREESASA_AVAILABLE:
            try:
                radii = rdFreeSASA.classifyAtoms(mol_3d)
                sasa_total = rdFreeSASA.CalcSASA(mol_3d, radii, confIdx=conf_id)

                polar_sasa = 0.0
                for atom in mol_3d.GetAtoms():
                    if atom.GetAtomicNum() in self.POLAR_ATOMIC_NUMBERS and atom.HasProp("SASA"):
                        polar_sasa += _numeric(atom.GetProp("SASA"))
                return float(sasa_total), float(polar_sasa)
            except Exception:
                pass

        # Fallback when rdFreeSASA is unavailable or fails:
        # use TPSA as a robust approximation for polar surface area.
        tpsa = _safe_descriptor(lambda: rdMolDescriptors.CalcTPSA(Chem.RemoveHs(mol_3d)))
        return np.nan, tpsa

    def _compute_mordred_3d_descriptors(self, mol_3d: Chem.Mol) -> Dict[str, float]:
        """Compute Mordred descriptors that explicitly require 3D coordinates."""
        if self.mordred_calculator is None or not self.mordred_3d_indices:
            return {}

        try:
            mordred_result = self.mordred_calculator(mol_3d)
            descriptors: Dict[str, float] = {}
            for idx, name in zip(self.mordred_3d_indices, self.mordred_3d_names):
                descriptors[name] = _numeric(mordred_result[idx])
            return descriptors
        except Exception as exc:
            self.logger.warning("Mordred 3D calculation failed for one molecule: %s", exc)
            return {}

    def compute_descriptors(self, mol_3d: Chem.Mol, conf_id: int, energy: float) -> Dict[str, float]:
        """Compute all requested descriptors for one molecule."""
        sasa, psa_3d = self._compute_sasa_and_psa3d(mol_3d, conf_id)
        descriptors: Dict[str, float] = {
            "asphericity": _safe_descriptor(
                lambda: Descriptors3D.Asphericity(mol_3d, confId=conf_id)
            ),
            "eccentricity": _safe_descriptor(
                lambda: Descriptors3D.Eccentricity(mol_3d, confId=conf_id)
            ),
            "spherocity_index": _safe_descriptor(
                lambda: Descriptors3D.SpherocityIndex(mol_3d, confId=conf_id)
            ),
            "radius_of_gyration": _safe_descriptor(
                lambda: Descriptors3D.RadiusOfGyration(mol_3d, confId=conf_id)
            ),
            "pmi1": _safe_descriptor(lambda: Descriptors3D.PMI1(mol_3d, confId=conf_id)),
            "pmi2": _safe_descriptor(lambda: Descriptors3D.PMI2(mol_3d, confId=conf_id)),
            "pmi3": _safe_descriptor(lambda: Descriptors3D.PMI3(mol_3d, confId=conf_id)),
            "psa_3d": psa_3d,
            "sasa": sasa,
            "uff_lowest_energy": float(energy),
        }
        descriptors.update(self._compute_mordred_3d_descriptors(mol_3d))
        return descriptors

    def build_descriptor_table(
        self, df: pd.DataFrame, smiles_col: str, target_col: str
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Generate descriptor table and aligned binary targets."""
        descriptor_rows: List[Dict[str, float]] = []
        kept_targets: List[object] = []
        total = len(df)
        failed = 0

        for row_idx, row in df.iterrows():
            smiles = str(row[smiles_col]).strip()
            target = row[target_col]

            if not smiles or smiles.lower() == "nan":
                self.logger.warning("Row %d skipped: empty SMILES.", row_idx)
                failed += 1
                continue
            if pd.isna(target):
                self.logger.warning("Row %d skipped: missing target.", row_idx)
                failed += 1
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                self.logger.warning("Row %d skipped: invalid SMILES '%s'.", row_idx, smiles)
                failed += 1
                continue

            mol_3d, conf_id, energy, err = self._generate_lowest_energy_conformer(mol, row_idx)
            if mol_3d is None or conf_id is None or energy is None:
                self.logger.warning("Row %d skipped: %s.", row_idx, err or "conformer failure")
                failed += 1
                continue

            descriptors = self.compute_descriptors(mol_3d, conf_id, energy)
            descriptor_rows.append(descriptors)
            kept_targets.append(target)
            try:
                # Remove explicit hydrogens after descriptor calculation.
                Chem.RemoveHs(mol_3d)
            except Exception:
                pass

            if len(descriptor_rows) % 100 == 0:
                self.logger.info("Processed %d/%d molecules...", len(descriptor_rows), total)

        if not descriptor_rows:
            raise RuntimeError("No valid molecules remained after conformer/descriptor steps.")

        self.logger.info(
            "Descriptor extraction complete. Total=%d, Kept=%d, Failed=%d",
            total,
            len(descriptor_rows),
            failed,
        )

        descriptor_df = pd.DataFrame(descriptor_rows)
        for col in CORE_DESCRIPTOR_NAMES:
            if col not in descriptor_df.columns:
                descriptor_df[col] = np.nan

        descriptor_df = descriptor_df.replace([np.inf, -np.inf], np.nan)
        descriptor_df = descriptor_df.apply(pd.to_numeric, errors="coerce")
        y = _encode_binary_target(kept_targets)
        return descriptor_df, y


def remove_high_correlation(
    feature_df: pd.DataFrame, threshold: float, logger: logging.Logger
) -> pd.DataFrame:
    """Drop one feature from each highly correlated pair."""
    if feature_df.shape[1] <= 1 or feature_df.shape[0] <= 1:
        return feature_df

    corr = feature_df.corr(method="pearson").abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if (upper[col] > threshold).any()]
    if to_drop:
        logger.info(
            "Removing %d highly correlated features (|r| > %.2f).",
            len(to_drop),
            threshold,
        )
        feature_df = feature_df.drop(columns=to_drop)
    return feature_df


def run_pipeline(args: argparse.Namespace) -> None:
    logger = logging.getLogger("build_3d_descriptor_matrix")

    set_reproducibility(args.seed)
    RDLogger.DisableLog("rdApp.*")

    input_path = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading input CSV: %s", input_path)
    df = pd.read_csv(input_path)
    smiles_col = _infer_smiles_column(df, args.smiles_col)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found in input CSV.")

    logger.info("Using SMILES column '%s' and target column '%s'.", smiles_col, args.target_col)
    logger.info("Seed=%d, conformers per molecule=%d", args.seed, args.n_conformers)

    extractor = Descriptor3DPipeline(
        n_conformers=args.n_conformers,
        random_seed=args.seed,
    )
    descriptor_df, y = extractor.build_descriptor_table(df, smiles_col, args.target_col)

    X_raw = descriptor_df.to_numpy(dtype=np.float64, copy=True)
    print("Raw descriptor matrix shape:", X_raw.shape)
    print("Total NaNs:", np.isnan(X_raw).sum())
    logger.info("Raw descriptor matrix shape: %s", X_raw.shape)
    logger.info("Total NaNs: %d", int(np.isnan(X_raw).sum()))

    missing_fraction = descriptor_df.isna().mean()
    high_missing_cols = missing_fraction[missing_fraction > args.max_missing_fraction].index.tolist()
    if high_missing_cols:
        logger.info(
            "Dropping %d columns with > %.0f%% missing values.",
            len(high_missing_cols),
            args.max_missing_fraction * 100.0,
        )
        descriptor_df = descriptor_df.drop(columns=high_missing_cols)

    descriptor_df = descriptor_df.fillna(descriptor_df.median(numeric_only=True))
    remaining_nans = int(descriptor_df.isna().sum().sum())
    if remaining_nans > 0:
        logger.warning(
            "Median imputation left %d NaNs (likely all-NaN columns). Filling with 0.0.",
            remaining_nans,
        )
        descriptor_df = descriptor_df.fillna(0.0)

    logger.info("Descriptor matrix shape after missing-data processing: %s", descriptor_df.shape)

    variance_filter = VarianceThreshold(threshold=args.variance_threshold)
    X_var = variance_filter.fit_transform(descriptor_df.values)
    kept_cols = descriptor_df.columns[variance_filter.get_support()].tolist()
    filtered_df = pd.DataFrame(X_var, columns=kept_cols)
    print("After variance filtering:", filtered_df.shape)
    logger.info(
        "After low-variance filtering (threshold=%.4f): %s",
        args.variance_threshold,
        filtered_df.shape,
    )

    filtered_df = remove_high_correlation(filtered_df, args.corr_threshold, logger)
    print("After correlation filtering:", filtered_df.shape)
    if filtered_df.shape[1] == 0:
        raise RuntimeError("All features were removed by filtering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(filtered_df.values).astype(np.float32)

    np.save(output_dir / "X_3D.npy", X_scaled)
    np.save(output_dir / "y.npy", y.astype(np.int64))

    logger.info("Saved X: %s", output_dir / "X_3D.npy")
    logger.info("Saved y: %s", output_dir / "y.npy")
    logger.info("Final X shape: %s, y shape: %s", X_scaled.shape, y.shape)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate cleaned 3D descriptor features from SMILES CSV."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Input CSV containing SMILES and target columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for output files (X_3D.npy, y.npy).",
    )
    parser.add_argument(
        "--smiles-col",
        type=str,
        default="SMILES",
        help="SMILES column name (auto-detection is case-insensitive).",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Binary target column name.",
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=10,
        help="Number of ETKDG conformers to generate per molecule.",
    )
    parser.add_argument(
        "--variance-threshold",
        type=float,
        default=0.01,
        help="Low-variance feature removal threshold.",
    )
    parser.add_argument(
        "--corr-threshold",
        type=float,
        default=0.95,
        help="Absolute Pearson correlation threshold for dropping features.",
    )
    parser.add_argument(
        "--max-missing-fraction",
        type=float,
        default=0.30,
        help="Drop descriptor columns only if missing fraction exceeds this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="reports/3d_descriptor_pipeline.log",
        help="Path to log file.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(Path(args.log_file), args.log_level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
