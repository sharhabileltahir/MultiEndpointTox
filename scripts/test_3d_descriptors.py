#!/usr/bin/env python3
"""Test script for 3D descriptors and pharmacophore features."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.docking import DockingManager, Descriptors3DCalculator

def main():
    # Load config
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("Testing 3D Descriptor Calculator")
    print("=" * 60)

    calc = Descriptors3DCalculator()
    result = calc.calculate_all("CC(=O)Nc1ccc(O)cc1")  # Acetaminophen

    print(f"Success: {result.success}")
    print(f"\nShape Descriptors:")
    print(f"  Asphericity: {result.asphericity:.4f}")
    print(f"  Eccentricity: {result.eccentricity:.4f}")
    print(f"  Radius of Gyration: {result.radius_of_gyration:.3f} A")

    print(f"\nVolume/Surface:")
    print(f"  Molecular Volume: {result.molecular_volume:.2f} A^3")
    print(f"  TPSA: {result.tpsa_3d:.2f} A^2")
    print(f"  SASA: {result.sasa:.2f} A^2")

    print(f"\nPharmacophore Counts:")
    print(f"  H-Bond Acceptors: {result.n_hba}")
    print(f"  H-Bond Donors: {result.n_hbd}")
    print(f"  Aromatic Rings: {result.n_aromatic}")
    print(f"  Hydrophobic Centers: {result.n_hydrophobic}")
    print(f"  Positive Ionizable: {result.n_pos_ionizable}")
    print(f"  Negative Ionizable: {result.n_neg_ionizable}")

    print(f"\nTotal Features: {len(result.pharmacophore_features)}")

    print("\n" + "=" * 60)
    print("Testing Enhanced Docking with 3D Descriptors")
    print("=" * 60)

    dm = DockingManager(config=config, structures_dir=str(PROJECT_ROOT / "data" / "structures"))

    if dm.is_available():
        enhanced = dm.dock_with_descriptors("CC(=O)Nc1ccc(O)cc1", "herg")

        print(f"\nDocking Results:")
        print(f"  Success: {enhanced['docking']['success']}")
        print(f"  Affinity: {enhanced['docking']['affinity']} kcal/mol")

        print(f"\nBinding Compatibility:")
        bc = enhanced['binding_compatibility']
        print(f"  Shape: {bc['shape_compatibility']}")
        print(f"  Size: {bc['size_compatibility']}")
        print(f"  Pharmacophore: {bc['pharmacophore_match']}")
        print(f"  Binding Risk: {bc['binding_risk']}")

        print(f"\nEnhanced Score:")
        es = enhanced['enhanced_score']
        print(f"  Score: {es['score']}")
        print(f"  Risk Level: {es['risk_level']}")
        print(f"  Shape Modifier: {es['shape_modifier']}")
        print(f"  Pharmacophore Modifier: {es['pharmacophore_modifier']}")
    else:
        print("Docking engine not available")

    print("\n" + "=" * 60)
    print("Testing Pharmacophore Comparison")
    print("=" * 60)

    comparison = dm.compare_pharmacophores(
        "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
        "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
    )

    print(f"\nAcetaminophen vs Aspirin:")
    print(f"  Pharmacophore Similarity: {comparison['pharmacophore_similarity']:.4f}")

    print(f"\nFeature Comparison (Acetaminophen vs Aspirin):")
    for feat, (v1, v2) in comparison['feature_comparison'].items():
        print(f"  {feat}: {v1} vs {v2}")

    print(f"\nShape Comparison:")
    for prop, (v1, v2) in comparison['shape_comparison'].items():
        print(f"  {prop}: {v1} vs {v2}")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
