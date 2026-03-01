"""
MultiEndpointTox API Usage Examples
===================================

This script demonstrates how to use the MultiEndpointTox API for
toxicity prediction of chemical compounds.

Prerequisites:
    1. Start the API server:
       python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

    2. Install requests:
       pip install requests

Usage:
    python docs/examples.py
"""

import requests
import json
from typing import List, Dict, Any

BASE_URL = "http://127.0.0.1:8000"


def check_health() -> Dict[str, Any]:
    """Check if the API is healthy and models are loaded."""
    response = requests.get(f"{BASE_URL}/health")
    response.raise_for_status()
    return response.json()


def validate_smiles(smiles: str) -> Dict[str, Any]:
    """Validate a SMILES string."""
    response = requests.post(
        f"{BASE_URL}/validate",
        json={"smiles": smiles}
    )
    response.raise_for_status()
    return response.json()


def predict_single(smiles: str, endpoint: str) -> Dict[str, Any]:
    """
    Predict toxicity for a single compound on a single endpoint.

    Args:
        smiles: SMILES string of the compound
        endpoint: One of 'herg', 'hepatotox', 'nephrotox', 'ames', 'skin_sens', 'cytotox'

    Returns:
        Prediction result with probability and applicability domain
    """
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"smiles": smiles, "endpoint": endpoint}
    )
    response.raise_for_status()
    return response.json()


def predict_multi(smiles: str, endpoints: List[str] = None) -> Dict[str, Any]:
    """
    Predict toxicity across multiple endpoints.

    Args:
        smiles: SMILES string of the compound
        endpoints: List of endpoints (None = all endpoints)

    Returns:
        Predictions for all requested endpoints
    """
    response = requests.post(
        f"{BASE_URL}/predict/multi",
        json={"smiles": smiles, "endpoints": endpoints}
    )
    response.raise_for_status()
    return response.json()


def predict_batch(smiles_list: List[str], endpoint: str) -> Dict[str, Any]:
    """
    Predict toxicity for multiple compounds on a single endpoint.

    Args:
        smiles_list: List of SMILES strings (max 1000)
        endpoint: Toxicity endpoint

    Returns:
        Results for all compounds
    """
    response = requests.post(
        f"{BASE_URL}/predict/batch",
        json={"smiles_list": smiles_list, "endpoint": endpoint}
    )
    response.raise_for_status()
    return response.json()


def predict_integrated(
    smiles: str,
    include_interpretation: bool = True,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Get comprehensive multi-endpoint prediction with SHAP interpretation.

    Args:
        smiles: SMILES string of the compound
        include_interpretation: Include SHAP explanations
        top_k: Number of top features to show

    Returns:
        Full prediction with risk assessment and interpretations
    """
    response = requests.post(
        f"{BASE_URL}/predict/integrated",
        json={
            "smiles": smiles,
            "include_interpretation": include_interpretation,
            "top_k": top_k
        }
    )
    response.raise_for_status()
    return response.json()


def print_prediction_summary(result: Dict[str, Any]) -> None:
    """Print a human-readable summary of an integrated prediction."""
    print("\n" + "=" * 60)
    print(f"COMPOUND: {result['smiles']}")
    print("=" * 60)

    # Print predictions
    print("\nPREDICTIONS:")
    print("-" * 40)
    for endpoint, pred in result['predictions'].items():
        if pred['task_type'] == 'classification':
            print(f"  {endpoint:12} | {pred['label']:20} | P={pred['probability']:.3f}")
        else:
            print(f"  {endpoint:12} | {pred['prediction']:.2f} {pred.get('unit', '')}")

    # Print risk assessment
    if 'integrated_assessment' in result:
        assessment = result['integrated_assessment']
        print("\nRISK ASSESSMENT:")
        print("-" * 40)
        print(f"  Overall Risk Level: {assessment['overall_risk_level'].upper()}")
        print(f"  Overall Risk Score: {assessment['overall_risk_score']:.3f}")
        print(f"  Critical Endpoint:  {assessment['critical_endpoint']}")
        print(f"  Recommendation:     {assessment.get('recommendation', 'N/A')}")

        print("\n  Endpoint Risk Ranking:")
        for i, ep in enumerate(assessment['endpoint_ranking'], 1):
            level = assessment['endpoint_risk_levels'][ep]
            score = assessment['endpoint_risk_scores'][ep]
            print(f"    {i}. {ep:12} - {level:8} ({score:.3f})")


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("MultiEndpointTox API Examples")
    print("=" * 60)

    # Check API health
    print("\n1. Checking API health...")
    try:
        health = check_health()
        print(f"   Status: {health['status']}")
        print(f"   Models loaded: {health['models_loaded']}")
        print(f"   Endpoints: {', '.join(health['available_endpoints'])}")
    except requests.exceptions.ConnectionError:
        print("   ERROR: Could not connect to API. Is the server running?")
        print("   Start with: python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000")
        exit(1)

    # Example compounds
    compounds = {
        "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "Ethanol": "CCO",
        "Ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
    }

    # Validate SMILES
    print("\n2. Validating SMILES strings...")
    for name, smiles in compounds.items():
        result = validate_smiles(smiles)
        status = "✓ Valid" if result['valid'] else "✗ Invalid"
        print(f"   {name}: {status}")

    # Single prediction
    print("\n3. Single endpoint prediction (hepatotox for Acetaminophen)...")
    result = predict_single(compounds["Acetaminophen"], "hepatotox")
    print(f"   Prediction: {result['label']}")
    print(f"   Probability: {result['probability']:.3f}")
    print(f"   In domain: {result['applicability_domain']['in_domain']}")

    # Multi-endpoint prediction
    print("\n4. Multi-endpoint prediction (all endpoints for Caffeine)...")
    result = predict_multi(compounds["Caffeine"])
    for endpoint, pred in result['predictions'].items():
        if pred['task_type'] == 'classification':
            print(f"   {endpoint}: {pred['label']} (P={pred['probability']:.3f})")
        else:
            print(f"   {endpoint}: {pred['prediction']:.2f} {pred.get('unit', '')}")

    # Batch prediction
    print("\n5. Batch prediction (hepatotox for all compounds)...")
    smiles_list = list(compounds.values())
    result = predict_batch(smiles_list, "hepatotox")
    print(f"   Total: {result['total']}, Successful: {result['successful']}")
    for i, (name, r) in enumerate(zip(compounds.keys(), result['results'])):
        if r['success']:
            print(f"   {name}: {r['label']} (P={r['probability']:.3f})")
        else:
            print(f"   {name}: FAILED - {r.get('error', 'Unknown error')}")

    # Integrated prediction with interpretation
    print("\n6. Integrated prediction with SHAP interpretation...")
    result = predict_integrated(compounds["Acetaminophen"], top_k=5)
    print_prediction_summary(result)

    # Show top SHAP features for critical endpoint
    if 'interpretations' in result:
        critical = result['integrated_assessment']['critical_endpoint']
        interp = result['interpretations'].get(critical, {})
        if 'shap_explanation' in interp:
            print(f"\n  Top SHAP Features for {critical}:")
            for feat in interp['shap_explanation'].get('top_features', [])[:5]:
                direction = "↑" if feat['contribution'] == 'increases' else "↓"
                print(f"    {direction} {feat['feature']}: {feat['shap_value']:.4f}")

    # Example with multiple compounds
    print("\n7. Comparing multiple compounds...")
    print("-" * 60)
    for name, smiles in list(compounds.items())[:3]:
        result = predict_integrated(smiles, include_interpretation=False)
        assessment = result['integrated_assessment']
        print(f"   {name:15} | Risk: {assessment['overall_risk_level']:8} | "
              f"Critical: {assessment['critical_endpoint']}")

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
