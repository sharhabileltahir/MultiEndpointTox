#!/usr/bin/env python3
"""Test script for the MultiEndpointTox API."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.predictor import ToxicityPredictor


def test_predictor():
    """Test the ToxicityPredictor directly."""
    print("=" * 60)
    print("Testing ToxicityPredictor")
    print("=" * 60)

    # Initialize predictor
    predictor = ToxicityPredictor(models_dir=str(PROJECT_ROOT / "models"))

    print(f"\nAvailable endpoints: {predictor.get_available_endpoints()}")

    # Test compounds
    test_compounds = [
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O"),
        ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
        ("Doxorubicin", "COc1cccc2C(=O)c3c(O)c4C[C@](O)(C[C@H](O[C@H]5C[C@H](N)[C@H](O)[C@H](C)O5)c4c(O)c3C(=O)c12)C(=O)CO"),
    ]

    print("\n" + "-" * 60)
    print("Single endpoint predictions")
    print("-" * 60)

    for name, smiles in test_compounds:
        print(f"\n{name}: {smiles[:50]}...")

        # Validate SMILES
        if not predictor.validate_smiles(smiles):
            print(f"  Invalid SMILES!")
            continue

        std_smiles = predictor.standardize_smiles(smiles)
        print(f"  Canonical: {std_smiles[:50]}...")

        # Test each available endpoint
        for endpoint in predictor.get_available_endpoints():
            result = predictor.predict_single(smiles, endpoint)
            if result["success"]:
                if result["task_type"] == "classification":
                    prob_str = f", prob={result['probability']:.3f}" if result.get("probability") else ""
                    ad_str = "✓" if result["applicability_domain"]["in_domain"] else "✗"
                    print(f"  {endpoint}: {result['label']} (pred={result['prediction']}{prob_str}) [AD: {ad_str}]")
                else:
                    print(f"  {endpoint}: {result['prediction']:.3f} {result.get('unit', '')} [AD: {'✓' if result['applicability_domain']['in_domain'] else '✗'}]")
            else:
                print(f"  {endpoint}: ERROR - {result.get('error', 'Unknown')}")

    print("\n" + "-" * 60)
    print("Multi-endpoint prediction")
    print("-" * 60)

    result = predictor.predict_multi_endpoint("CC(=O)Nc1ccc(O)cc1")  # Acetaminophen
    print(f"\nAcetaminophen multi-endpoint:")
    if result["success"]:
        for endpoint, pred in result["predictions"].items():
            if "error" not in pred:
                if pred["task_type"] == "classification":
                    print(f"  {endpoint}: {pred['label']}")
                else:
                    print(f"  {endpoint}: {pred['prediction']:.3f}")
            else:
                print(f"  {endpoint}: {pred['error']}")

    print("\n" + "-" * 60)
    print("Batch prediction")
    print("-" * 60)

    smiles_list = [s for _, s in test_compounds[:3]]
    for endpoint in predictor.get_available_endpoints()[:1]:  # Just test first endpoint
        results = predictor.predict_batch(smiles_list, endpoint)
        print(f"\nBatch {endpoint} ({len(results)} compounds):")
        for i, r in enumerate(results):
            if r["success"]:
                if r["task_type"] == "classification":
                    print(f"  {i+1}. {r['label']}")
                else:
                    print(f"  {i+1}. {r['prediction']:.3f}")
            else:
                print(f"  {i+1}. ERROR")

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_predictor()
