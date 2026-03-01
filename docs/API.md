# MultiEndpointTox API Documentation

## Base URL

```
http://127.0.0.1:8000
```

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## Endpoints

### Health & Information

#### GET `/`

Returns basic API information.

**Response:**
```json
{
  "name": "MultiEndpointTox API",
  "version": "0.3.0",
  "description": "Multi-endpoint toxicity prediction for drug safety",
  "endpoints": ["herg", "hepatotox", "nephrotox", "ames", "skin_sens", "cytotox"],
  "features": [
    "Multi-endpoint prediction",
    "SHAP interpretability",
    "Integrated risk assessment",
    "Structural alerts",
    "Design recommendations"
  ],
  "shap_available": true
}
```

---

#### GET `/health`

Health check endpoint for monitoring.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 6,
  "available_endpoints": ["herg", "hepatotox", "nephrotox", "ames", "skin_sens", "cytotox"]
}
```

---

#### GET `/endpoints`

List all available toxicity endpoints with descriptions.

**Response:**
```json
[
  {
    "name": "herg",
    "task": "regression",
    "description": "hERG channel inhibition (cardiotoxicity)",
    "available": true
  },
  {
    "name": "hepatotox",
    "task": "classification",
    "description": "Drug-induced liver injury",
    "available": true
  },
  {
    "name": "nephrotox",
    "task": "classification",
    "description": "Drug-induced kidney injury",
    "available": true
  },
  {
    "name": "ames",
    "task": "classification",
    "description": "Ames mutagenicity",
    "available": true
  },
  {
    "name": "skin_sens",
    "task": "classification",
    "description": "Skin sensitization",
    "available": true
  },
  {
    "name": "cytotox",
    "task": "classification",
    "description": "General cytotoxicity",
    "available": true
  }
]
```

---

### Prediction Endpoints

#### POST `/predict`

Predict toxicity for a single compound on a single endpoint.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| smiles | string | Yes | SMILES string of the compound |
| endpoint | string | Yes | Toxicity endpoint name |

**Example Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "endpoint": "hepatotox"
}
```

**Response (Classification):**
```json
{
  "success": true,
  "endpoint": "hepatotox",
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "task_type": "classification",
  "prediction": 1,
  "label": "Hepatotoxic",
  "probability": 0.809,
  "applicability_domain": {
    "in_domain": true,
    "confidence": 1.0
  },
  "model": "randomforest_optimized"
}
```

**Response (Regression - hERG):**
```json
{
  "success": true,
  "endpoint": "herg",
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "task_type": "regression",
  "prediction": 5.49,
  "unit": "pIC50",
  "applicability_domain": {
    "in_domain": true,
    "confidence": 1.0
  },
  "model": "xgboost_optimized"
}
```

**Error Response:**
```json
{
  "detail": "Invalid SMILES string"
}
```

---

#### GET `/predict/{endpoint}`

Predict via GET request (convenient for browser/curl).

**Query Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| smiles | string | Yes | URL-encoded SMILES string |

**Example:**
```
GET /predict/hepatotox?smiles=CCO
```

---

#### POST `/predict/multi`

Predict toxicity across multiple endpoints for a single compound.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| smiles | string | Yes | SMILES string |
| endpoints | array | No | List of endpoints (null = all) |

**Example Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "endpoints": ["hepatotox", "nephrotox", "cytotox"]
}
```

**Response:**
```json
{
  "success": true,
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "predictions": {
    "hepatotox": {
      "task_type": "classification",
      "prediction": 1,
      "label": "Hepatotoxic",
      "probability": 0.809,
      "applicability_domain": {"in_domain": true, "confidence": 1.0},
      "model": "randomforest_optimized"
    },
    "nephrotox": {
      "task_type": "classification",
      "prediction": 0,
      "label": "Non-nephrotoxic",
      "probability": 0.294,
      "applicability_domain": {"in_domain": true, "confidence": 1.0},
      "model": "randomforest_optimized"
    },
    "cytotox": {
      "task_type": "classification",
      "prediction": 0,
      "label": "Non-cytotoxic",
      "probability": 0.026,
      "applicability_domain": {"in_domain": true, "confidence": 1.0},
      "model": "xgboost_optimized"
    }
  }
}
```

---

#### POST `/predict/batch`

Predict toxicity for multiple compounds on a single endpoint.

**Request Body:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| smiles_list | array | Yes | List of SMILES strings (max 1000) |
| endpoint | string | Yes | Toxicity endpoint name |

**Example Request:**
```json
{
  "smiles_list": [
    "CC(=O)Nc1ccc(O)cc1",
    "CC(=O)OC1=CC=CC=C1C(=O)O",
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
  ],
  "endpoint": "hepatotox"
}
```

**Response:**
```json
{
  "endpoint": "hepatotox",
  "total": 3,
  "successful": 3,
  "results": [
    {
      "success": true,
      "smiles": "CC(=O)Nc1ccc(O)cc1",
      "prediction": 1,
      "label": "Hepatotoxic",
      "probability": 0.809
    },
    {
      "success": true,
      "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O",
      "prediction": 0,
      "label": "Non-hepatotoxic",
      "probability": 0.342
    },
    {
      "success": true,
      "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
      "prediction": 0,
      "label": "Non-hepatotoxic",
      "probability": 0.287
    }
  ]
}
```

---

### Interpretability Endpoints

#### POST `/predict/interpret`

Predict with SHAP feature importance explanation.

**Request Body:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| smiles | string | Yes | - | SMILES string |
| endpoint | string | Yes | - | Toxicity endpoint |
| top_k | integer | No | 10 | Number of top features |

**Example Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "endpoint": "hepatotox",
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "endpoint": "hepatotox",
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "task_type": "classification",
  "prediction": 1,
  "label": "Hepatotoxic",
  "probability": 0.809,
  "interpretation": {
    "available": true,
    "shap_explanation": {
      "success": true,
      "endpoint": "hepatotox",
      "base_value": 0.501,
      "prediction_contribution": 0.308,
      "top_features": [
        {
          "feature": "Morgan_156",
          "shap_value": 0.017,
          "contribution": "increases",
          "importance": 0.017
        },
        {
          "feature": "Morgan_155",
          "shap_value": -0.017,
          "contribution": "decreases",
          "importance": 0.017
        }
      ],
      "feature_categories": {
        "physicochemical": [],
        "structural_fingerprints": [
          {"feature": "Morgan_156", "shap_value": 0.017, "contribution": "increases"}
        ],
        "substructure_keys": []
      },
      "n_features_analyzed": 500,
      "interpretation": "Features increasing hepatotoxicity risk: Morgan_156; Features decreasing: Morgan_155"
    },
    "structural_alerts": []
  }
}
```

---

#### POST `/predict/integrated`

Comprehensive multi-endpoint prediction with full interpretability.

**Request Body:**
| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| smiles | string | Yes | - | SMILES string |
| endpoints | array | No | null | Endpoints (null = all) |
| top_k | integer | No | 10 | Top features per endpoint |
| include_interpretation | boolean | No | true | Include SHAP analysis |

**Example Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "include_interpretation": true,
  "top_k": 5
}
```

**Response:**
```json
{
  "success": true,
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "predictions": {
    "herg": {"prediction": 5.49, "unit": "pIC50", ...},
    "hepatotox": {"prediction": 1, "label": "Hepatotoxic", "probability": 0.809, ...},
    "nephrotox": {"prediction": 0, "label": "Non-nephrotoxic", ...},
    "ames": {"prediction": 0, "label": "Non-mutagenic", ...},
    "skin_sens": {"prediction": 1, "label": "Sensitizer", ...},
    "cytotox": {"prediction": 0, "label": "Non-cytotoxic", ...}
  },
  "interpretations": {
    "herg": {"shap_explanation": {...}, "structural_alerts": []},
    "hepatotox": {"shap_explanation": {...}, "structural_alerts": []},
    ...
  },
  "integrated_assessment": {
    "overall_risk_score": 0.796,
    "overall_risk_level": "high",
    "critical_endpoint": "hepatotox",
    "critical_endpoint_score": 0.796,
    "endpoint_risk_scores": {
      "herg": 0.104,
      "hepatotox": 0.796,
      "nephrotox": 0.301,
      "ames": 0.020,
      "skin_sens": 0.641,
      "cytotox": 0.026
    },
    "endpoint_risk_levels": {
      "herg": "low",
      "hepatotox": "high",
      "nephrotox": "moderate",
      "ames": "low",
      "skin_sens": "moderate",
      "cytotox": "low"
    },
    "endpoint_ranking": ["hepatotox", "skin_sens", "nephrotox", "herg", "cytotox", "ames"],
    "recommendation": "Caution: High hepatotox risk - consider structural optimization"
  },
  "structural_alerts": [],
  "cross_endpoint_analysis": {
    "common_features": [],
    "insights": []
  },
  "design_recommendations": []
}
```

---

#### GET `/predict/integrated/{smiles}`

Quick integrated prediction via GET.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| top_k | integer | 10 | Number of top features |

**Example:**
```
GET /predict/integrated/CCO?top_k=5
```

---

#### GET `/interpret/{endpoint}`

Get SHAP interpretation via GET request.

**Query Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| smiles | string | Yes | - | SMILES string |
| top_k | integer | No | 10 | Top features |

**Example:**
```
GET /interpret/hepatotox?smiles=CCO&top_k=5
```

---

#### GET `/shap/status`

Check SHAP explainer status.

**Response:**
```json
{
  "shap_available": true,
  "explainers_initialized": ["hepatotox", "herg"],
  "endpoints_available": ["herg", "hepatotox", "nephrotox", "ames", "skin_sens", "cytotox"]
}
```

---

### Utility Endpoints

#### POST `/validate`

Validate a SMILES string.

**Request Body:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1"
}
```

**Response:**
```json
{
  "input": "CC(=O)Nc1ccc(O)cc1",
  "valid": true,
  "canonical_smiles": "CC(=O)Nc1ccc(O)cc1"
}
```

---

#### GET `/validate`

Validate SMILES via GET.

**Example:**
```
GET /validate?smiles=CCO
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (invalid SMILES, invalid endpoint) |
| 422 | Validation Error (missing required fields) |
| 503 | Service Unavailable (predictor not initialized) |

### Error Response Format

```json
{
  "detail": "Error message describing the issue"
}
```

---

## Risk Levels

The integrated assessment uses the following risk levels:

| Level | Score Range | Description |
|-------|-------------|-------------|
| low | 0.0 - 0.3 | Low toxicity risk |
| moderate | 0.3 - 0.7 | Moderate toxicity risk |
| high | 0.7 - 1.0 | High toxicity risk |

---

## Applicability Domain

Each prediction includes an applicability domain assessment:

| Field | Description |
|-------|-------------|
| in_domain | Whether the compound is within the model's training domain |
| confidence | Confidence score (0.0 - 1.0) |

Compounds outside the applicability domain may have less reliable predictions.

---

## Rate Limits

Currently, no rate limits are enforced. For batch predictions, the maximum is 1000 compounds per request.

---

## Examples

### Python

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# Single prediction
response = requests.post(f"{BASE_URL}/predict", json={
    "smiles": "CC(=O)Nc1ccc(O)cc1",
    "endpoint": "hepatotox"
})
print(response.json())

# Batch prediction
response = requests.post(f"{BASE_URL}/predict/batch", json={
    "smiles_list": ["CCO", "CC", "CCC", "CCCC"],
    "endpoint": "hepatotox"
})
results = response.json()
for r in results["results"]:
    print(f"{r['smiles']}: {r['label']} ({r['probability']:.2f})")

# Integrated prediction
response = requests.post(f"{BASE_URL}/predict/integrated", json={
    "smiles": "CC(=O)Nc1ccc(O)cc1",
    "include_interpretation": True
})
result = response.json()
print(f"Overall Risk: {result['integrated_assessment']['overall_risk_level']}")
print(f"Critical Endpoint: {result['integrated_assessment']['critical_endpoint']}")
```

### cURL

```bash
# Health check
curl http://127.0.0.1:8000/health

# Single prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "endpoint": "hepatotox"}'

# Integrated prediction
curl -X POST "http://127.0.0.1:8000/predict/integrated" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO", "include_interpretation": true}'
```

### PowerShell

```powershell
# Single prediction
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"smiles": "CCO", "endpoint": "hepatotox"}'
$response | ConvertTo-Json

# Integrated prediction
$body = @{
    smiles = "CCO"
    include_interpretation = $true
} | ConvertTo-Json

$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict/integrated" `
  -ContentType "application/json" -Body $body
$response.integrated_assessment
```
