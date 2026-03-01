#!/usr/bin/env python3
"""
Command-line toxicity prediction script.

Usage:
    python scripts/predict.py --smiles "CCO" --endpoint herg
    python scripts/predict.py --smiles "CCO" --all
    python scripts/predict.py --file compounds.txt --endpoint hepatotox
"""

import argparse
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.predictor import ToxicityPredictor


def main():
    parser = argparse.ArgumentParser(
        description="Multi-endpoint toxicity prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/predict.py --smiles "CC(=O)Nc1ccc(O)cc1" --endpoint hepatotox
  python scripts/predict.py --smiles "CC(=O)Nc1ccc(O)cc1" --all
  python scripts/predict.py --file compounds.txt --endpoint herg --output results.json
        """
    )

    parser.add_argument("--smiles", "-s", type=str, help="SMILES string to predict")
    parser.add_argument("--file", "-f", type=str, help="File with SMILES (one per line)")
    parser.add_argument("--endpoint", "-e", type=str, help="Toxicity endpoint")
    parser.add_argument("--all", "-a", action="store_true", help="Predict all endpoints")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file")
    parser.add_argument("--list", "-l", action="store_true", help="List available endpoints")

    args = parser.parse_args()

    # Initialize predictor
    predictor = ToxicityPredictor(models_dir=str(PROJECT_ROOT / "models"))

    # List endpoints
    if args.list:
        print("\nAvailable endpoints:")
        for endpoint in predictor.get_available_endpoints():
            info = ToxicityPredictor.ENDPOINTS.get(endpoint, {})
            print(f"  {endpoint}: {info.get('description', '')} ({info.get('task', '')})")
        print()
        return

    # Check inputs
    if not args.smiles and not args.file:
        parser.error("Either --smiles or --file is required")

    if not args.endpoint and not args.all:
        parser.error("Either --endpoint or --all is required")

    # Get SMILES list
    smiles_list = []
    if args.smiles:
        smiles_list = [args.smiles]
    elif args.file:
        with open(args.file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]

    # Make predictions
    results = []

    for smiles in smiles_list:
        if args.all:
            result = predictor.predict_multi_endpoint(smiles)
        else:
            result = predictor.predict_single(smiles, args.endpoint)
        results.append(result)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        # Print to console
        for i, result in enumerate(results):
            if len(smiles_list) > 1:
                print(f"\n[{i+1}/{len(smiles_list)}] {smiles_list[i]}")

            if not result.get("success", False):
                print(f"  ERROR: {result.get('error', 'Unknown error')}")
                continue

            if args.all:
                # Multi-endpoint result
                print(f"  SMILES: {result['smiles']}")
                for endpoint, pred in result.get("predictions", {}).items():
                    if "error" in pred:
                        print(f"  {endpoint}: ERROR - {pred['error']}")
                    elif pred.get("task_type") == "classification":
                        prob = f" (prob={pred['probability']:.3f})" if pred.get("probability") else ""
                        print(f"  {endpoint}: {pred['label']}{prob}")
                    else:
                        print(f"  {endpoint}: {pred['prediction']:.4f} {pred.get('unit', '')}")
            else:
                # Single endpoint result
                print(f"  SMILES: {result['smiles']}")
                print(f"  Endpoint: {result['endpoint']}")
                if result.get("task_type") == "classification":
                    print(f"  Prediction: {result['label']}")
                    if result.get("probability"):
                        print(f"  Probability: {result['probability']:.3f}")
                else:
                    print(f"  Prediction: {result['prediction']:.4f} {result.get('unit', '')}")

                ad = result.get("applicability_domain", {})
                print(f"  In AD: {ad.get('in_domain', 'N/A')}")


if __name__ == "__main__":
    main()
