#!/usr/bin/env python
"""
Batch Docking Workflow Script

Dock compound libraries against multiple protein targets for toxicity assessment.

Usage:
    python scripts/batch_docking.py --input compounds.csv --targets herg,hepatotox --output results/
    python scripts/batch_docking.py --input compounds.sdf --targets all --output results/ --exhaustiveness 16
    python scripts/batch_docking.py --smiles "CCO,CCC,CCCC" --targets herg --output results/

Input formats:
    - CSV file with 'smiles' or 'SMILES' column (and optional 'name' or 'id' column)
    - SDF file with molecules
    - Comma-separated SMILES strings via --smiles argument

Output:
    - CSV file with docking results for all compounds and targets
    - JSON file with detailed results including poses
    - Summary statistics
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")


def load_config():
    """Load configuration from YAML file."""
    import yaml
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def load_compounds_from_csv(filepath: str) -> List[Dict[str, str]]:
    """Load compounds from CSV file."""
    df = pd.read_csv(filepath)

    # Find SMILES column
    smiles_col = None
    for col in ['smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'std_smiles']:
        if col in df.columns:
            smiles_col = col
            break

    if smiles_col is None:
        raise ValueError(f"No SMILES column found in {filepath}. Expected one of: smiles, SMILES, canonical_smiles")

    # Find name/ID column
    name_col = None
    for col in ['name', 'Name', 'id', 'ID', 'compound_id', 'mol_id']:
        if col in df.columns:
            name_col = col
            break

    compounds = []
    for idx, row in df.iterrows():
        smiles = str(row[smiles_col]).strip()
        if pd.isna(smiles) or smiles == '' or smiles == 'nan':
            continue

        name = str(row[name_col]) if name_col else f"compound_{idx+1}"
        compounds.append({"smiles": smiles, "name": name})

    return compounds


def load_compounds_from_sdf(filepath: str) -> List[Dict[str, str]]:
    """Load compounds from SDF file."""
    try:
        from rdkit import Chem
    except ImportError:
        raise ImportError("RDKit required for SDF file reading. Install with: conda install -c conda-forge rdkit")

    compounds = []
    suppl = Chem.SDMolSupplier(filepath)

    for idx, mol in enumerate(suppl):
        if mol is None:
            continue

        smiles = Chem.MolToSmiles(mol)
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else f"compound_{idx+1}"
        compounds.append({"smiles": smiles, "name": name})

    return compounds


def load_compounds(source: str, smiles_list: Optional[str] = None) -> List[Dict[str, str]]:
    """Load compounds from file or SMILES string."""
    if smiles_list:
        # Parse comma-separated SMILES
        smiles = [s.strip() for s in smiles_list.split(',')]
        return [{"smiles": s, "name": f"compound_{i+1}"} for i, s in enumerate(smiles) if s]

    filepath = Path(source)
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {source}")

    ext = filepath.suffix.lower()
    if ext == '.csv':
        return load_compounds_from_csv(source)
    elif ext in ['.sdf', '.mol']:
        return load_compounds_from_sdf(source)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .sdf")


def dock_compound_all_targets(
    docking_manager,
    compound: Dict[str, str],
    targets: List[str],
    exhaustiveness: int = 8,
    n_poses: int = 3
) -> Dict[str, Any]:
    """Dock a single compound against all specified targets."""
    results = {
        "name": compound["name"],
        "smiles": compound["smiles"],
        "targets": {},
        "best_affinity": None,
        "best_target": None,
    }

    best_affinity = 0

    for target in targets:
        try:
            result = docking_manager.dock(
                smiles=compound["smiles"],
                target=target,
                exhaustiveness=exhaustiveness,
                n_poses=n_poses,
                use_cache=True
            )

            results["targets"][target] = {
                "success": result.success,
                "affinity": result.affinity,
                "num_poses": result.num_poses,
                "execution_time": result.execution_time,
                "error": result.error,
            }

            if result.success and result.affinity is not None:
                if result.affinity < best_affinity:
                    best_affinity = result.affinity
                    results["best_affinity"] = result.affinity
                    results["best_target"] = target

        except Exception as e:
            results["targets"][target] = {
                "success": False,
                "affinity": None,
                "error": str(e),
            }

    return results


def run_batch_docking(
    compounds: List[Dict[str, str]],
    targets: List[str],
    config: Dict,
    exhaustiveness: int = 8,
    n_poses: int = 3,
    max_workers: int = 4,
    progress_callback=None
) -> List[Dict[str, Any]]:
    """Run batch docking for all compounds against all targets."""
    from src.docking import DockingManager

    # Initialize docking manager
    docking_manager = DockingManager(
        config=config,
        structures_dir=str(PROJECT_ROOT / config.get("docking", {}).get("structures_dir", "data/structures")),
        engine=config.get("docking", {}).get("engine", "vina")
    )

    if not docking_manager.is_available():
        raise RuntimeError("Docking engine not available. Install AutoDock Vina.")

    results = []
    total = len(compounds)

    logger.info(f"Starting batch docking: {total} compounds × {len(targets)} targets")

    # Process compounds
    for idx, compound in enumerate(compounds):
        if progress_callback:
            progress_callback(idx + 1, total, compound["name"])
        else:
            logger.info(f"[{idx+1}/{total}] Docking {compound['name']}")

        result = dock_compound_all_targets(
            docking_manager,
            compound,
            targets,
            exhaustiveness,
            n_poses
        )
        results.append(result)

    # Cleanup
    docking_manager.cleanup()

    return results


def generate_results_csv(results: List[Dict], targets: List[str], output_path: str):
    """Generate CSV file with docking results."""
    rows = []

    for result in results:
        row = {
            "name": result["name"],
            "smiles": result["smiles"],
            "best_affinity": result["best_affinity"],
            "best_target": result["best_target"],
        }

        for target in targets:
            target_result = result["targets"].get(target, {})
            row[f"{target}_affinity"] = target_result.get("affinity")
            row[f"{target}_success"] = target_result.get("success", False)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    return df


def generate_results_json(results: List[Dict], output_path: str):
    """Generate JSON file with detailed results."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Detailed results saved to {output_path}")


def generate_summary(results: List[Dict], targets: List[str]) -> Dict[str, Any]:
    """Generate summary statistics."""
    summary = {
        "total_compounds": len(results),
        "targets_screened": targets,
        "num_targets": len(targets),
        "statistics": {},
        "risk_distribution": {},
    }

    for target in targets:
        affinities = []
        successes = 0
        failures = 0

        for result in results:
            target_result = result["targets"].get(target, {})
            if target_result.get("success"):
                successes += 1
                if target_result.get("affinity") is not None:
                    affinities.append(target_result["affinity"])
            else:
                failures += 1

        if affinities:
            summary["statistics"][target] = {
                "success_rate": successes / len(results) * 100,
                "mean_affinity": np.mean(affinities),
                "min_affinity": np.min(affinities),
                "max_affinity": np.max(affinities),
                "std_affinity": np.std(affinities),
                "strong_binders": sum(1 for a in affinities if a < -8),
                "moderate_binders": sum(1 for a in affinities if -8 <= a < -6),
                "weak_binders": sum(1 for a in affinities if a >= -6),
            }

    # Overall risk distribution based on best affinity
    high_risk = sum(1 for r in results if r["best_affinity"] and r["best_affinity"] < -8)
    moderate_risk = sum(1 for r in results if r["best_affinity"] and -8 <= r["best_affinity"] < -6)
    low_risk = sum(1 for r in results if r["best_affinity"] and r["best_affinity"] >= -6)
    no_binding = sum(1 for r in results if r["best_affinity"] is None)

    summary["risk_distribution"] = {
        "high_risk": high_risk,
        "moderate_risk": moderate_risk,
        "low_risk": low_risk,
        "no_binding_data": no_binding,
    }

    return summary


def print_summary(summary: Dict):
    """Print summary to console."""
    print("\n" + "=" * 60)
    print("BATCH DOCKING SUMMARY")
    print("=" * 60)
    print(f"Total compounds: {summary['total_compounds']}")
    print(f"Targets screened: {', '.join(summary['targets_screened'])}")
    print()

    print("Risk Distribution (based on best binding affinity):")
    rd = summary["risk_distribution"]
    print(f"  High risk (< -8 kcal/mol):    {rd['high_risk']:4d} compounds")
    print(f"  Moderate risk (-8 to -6):     {rd['moderate_risk']:4d} compounds")
    print(f"  Low risk (> -6 kcal/mol):     {rd['low_risk']:4d} compounds")
    print(f"  No binding data:              {rd['no_binding_data']:4d} compounds")
    print()

    print("Per-Target Statistics:")
    for target, stats in summary["statistics"].items():
        print(f"\n  {target}:")
        print(f"    Success rate: {stats['success_rate']:.1f}%")
        print(f"    Mean affinity: {stats['mean_affinity']:.2f} kcal/mol")
        print(f"    Best affinity: {stats['min_affinity']:.2f} kcal/mol")
        print(f"    Strong binders: {stats['strong_binders']}")
        print(f"    Moderate binders: {stats['moderate_binders']}")
        print(f"    Weak binders: {stats['weak_binders']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Batch molecular docking workflow for toxicity screening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/batch_docking.py --input data/compounds.csv --targets herg,hepatotox
  python scripts/batch_docking.py --input data/compounds.sdf --targets all --exhaustiveness 16
  python scripts/batch_docking.py --smiles "CCO,CCC,CCCC" --targets herg --output results/
        """
    )

    parser.add_argument(
        "--input", "-i",
        help="Input file (CSV or SDF format)"
    )
    parser.add_argument(
        "--smiles", "-s",
        help="Comma-separated SMILES strings (alternative to --input)"
    )
    parser.add_argument(
        "--targets", "-t",
        default="all",
        help="Comma-separated list of targets or 'all' (default: all)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/batch_docking",
        help="Output directory/prefix (default: results/batch_docking)"
    )
    parser.add_argument(
        "--exhaustiveness", "-e",
        type=int,
        default=8,
        help="Docking exhaustiveness (default: 8)"
    )
    parser.add_argument(
        "--n-poses", "-n",
        type=int,
        default=3,
        help="Number of poses per docking (default: 3)"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)"
    )

    args = parser.parse_args()

    # Validate input
    if not args.input and not args.smiles:
        parser.error("Either --input or --smiles must be provided")

    # Load config
    config = load_config()

    # Available targets
    available_targets = list(config.get("docking", {}).get("protein_structures", {}).keys())
    if not available_targets:
        available_targets = ["herg", "hepatotox", "cyp2d6", "cyp2c9", "er_alpha", "ar"]

    # Parse targets
    if args.targets.lower() == "all":
        targets = available_targets
    else:
        targets = [t.strip() for t in args.targets.split(",")]
        invalid = [t for t in targets if t not in available_targets]
        if invalid:
            logger.error(f"Invalid targets: {invalid}. Available: {available_targets}")
            sys.exit(1)

    # Load compounds
    try:
        compounds = load_compounds(args.input, args.smiles)
        logger.info(f"Loaded {len(compounds)} compounds")
    except Exception as e:
        logger.error(f"Failed to load compounds: {e}")
        sys.exit(1)

    if not compounds:
        logger.error("No compounds loaded")
        sys.exit(1)

    # Setup output directory
    output_path = Path(args.output)
    output_dir = output_path.parent if output_path.suffix else output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = output_path.stem if output_path.suffix else f"batch_docking_{timestamp}"

    # Run batch docking
    start_time = time.time()

    try:
        results = run_batch_docking(
            compounds=compounds,
            targets=targets,
            config=config,
            exhaustiveness=args.exhaustiveness,
            n_poses=args.n_poses,
            max_workers=args.workers
        )
    except Exception as e:
        logger.error(f"Batch docking failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.time() - start_time
    logger.info(f"Docking completed in {elapsed:.1f}s")

    # Generate output files
    csv_path = output_dir / f"{base_name}.csv"
    json_path = output_dir / f"{base_name}.json"
    summary_path = output_dir / f"{base_name}_summary.json"

    generate_results_csv(results, targets, str(csv_path))
    generate_results_json(results, str(json_path))

    # Generate and save summary
    summary = generate_summary(results, targets)
    summary["execution_time_seconds"] = elapsed
    summary["timestamp"] = timestamp

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print_summary(summary)

    logger.info(f"\nOutput files:")
    logger.info(f"  CSV:     {csv_path}")
    logger.info(f"  JSON:    {json_path}")
    logger.info(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
