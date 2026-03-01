# MultiEndpointTox

**Multi-Endpoint Toxicity Prediction API for Drug Safety Assessment**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

MultiEndpointTox is a machine learning-powered REST API for predicting multiple drug toxicity endpoints from molecular structures. It provides interpretable predictions with SHAP explanations, structural alerts, and integrated risk assessment.

### Toxicity Endpoints

| Endpoint | Type | Description | Model |
|----------|------|-------------|-------|
| **hERG** | Regression | Cardiotoxicity (hERG channel inhibition) | XGBoost |
| **Hepatotox** | Classification | Drug-induced liver injury (DILI) | RandomForest |
| **Nephrotox** | Classification | Drug-induced kidney injury | RandomForest |
| **Ames** | Classification | Ames mutagenicity test | XGBoost |
| **Skin Sens** | Classification | Skin sensitization potential | RandomForest |
| **Cytotox** | Classification | General cytotoxicity | XGBoost |

### Key Features

- **Multi-endpoint prediction** - Assess all toxicity risks in a single API call
- **SHAP interpretability** - Understand which molecular features drive predictions
- **Structural alerts** - Detection of known toxic substructures
- **Applicability domain** - Confidence assessment for each prediction
- **Integrated risk assessment** - Overall risk score and critical endpoint identification
- **Batch processing** - Predict up to 1000 compounds at once

---

## Installation

### Requirements

- Python 3.10+
- RDKit
- ~500MB disk space (including models)

### Setup

```bash
# Clone the repository
git clone https://github.com/sharhabileltahir/MultiEndpointTox.git
cd MultiEndpointTox

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from rdkit import Chem; print('RDKit OK')"
```

---

## Quick Start

### Start the API Server

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### Make Your First Prediction

```bash
# Single endpoint prediction
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "endpoint": "hepatotox"}'

# Multi-endpoint prediction with interpretation
curl -X POST "http://127.0.0.1:8000/predict/integrated" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "include_interpretation": true}'
```

### Using PowerShell

```powershell
# Single prediction
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"smiles": "CC(=O)Nc1ccc(O)cc1", "endpoint": "hepatotox"}'

# Multi-endpoint with interpretation
$body = @{
    smiles = "CC(=O)Nc1ccc(O)cc1"
    include_interpretation = $true
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict/integrated" `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"smiles": "CC(=O)Nc1ccc(O)cc1", "endpoint": "hepatotox"}
)
print(response.json())

# Multi-endpoint with interpretation
response = requests.post(
    "http://127.0.0.1:8000/predict/integrated",
    json={
        "smiles": "CC(=O)Nc1ccc(O)cc1",
        "include_interpretation": True
    }
)
result = response.json()
print(f"Overall Risk: {result['integrated_assessment']['overall_risk_level']}")
```

---

## API Reference

### Interactive Documentation

Once the server is running:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

### Endpoints

#### Health & Info

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and available endpoints |
| GET | `/health` | Health check |
| GET | `/endpoints` | List all toxicity endpoints |

#### Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict` | Single compound, single endpoint |
| GET | `/predict/{endpoint}?smiles=...` | Single prediction via GET |
| POST | `/predict/multi` | Single compound, multiple endpoints |
| POST | `/predict/batch` | Multiple compounds, single endpoint |
| POST | `/predict/integrated` | Full prediction with interpretation |

#### Interpretability

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/predict/interpret` | Prediction with SHAP explanation |
| GET | `/interpret/{endpoint}?smiles=...` | SHAP interpretation via GET |
| GET | `/shap/status` | SHAP explainer status |

#### Utilities

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/validate` | Validate SMILES string |
| GET | `/validate?smiles=...` | Validate via GET |

### Request/Response Examples

#### Single Prediction

**Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "endpoint": "hepatotox"
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
  "applicability_domain": {
    "in_domain": true,
    "confidence": 1.0
  },
  "model": "randomforest_optimized"
}
```

#### Integrated Prediction

**Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "include_interpretation": true,
  "top_k": 5
}
```

**Response (abbreviated):**
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
  "integrated_assessment": {
    "overall_risk_score": 0.796,
    "overall_risk_level": "high",
    "critical_endpoint": "hepatotox",
    "endpoint_ranking": ["hepatotox", "skin_sens", "nephrotox", "herg", "cytotox", "ames"],
    "recommendation": "Caution: High hepatotox risk - consider structural optimization"
  },
  "interpretations": {
    "hepatotox": {
      "shap_explanation": {
        "top_features": [
          {"feature": "Morgan_156", "shap_value": 0.017, "contribution": "increases"},
          ...
        ]
      },
      "structural_alerts": []
    }
  }
}
```

#### Batch Prediction

**Request:**
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
    {"success": true, "smiles": "CC(=O)Nc1ccc(O)cc1", "prediction": 1, ...},
    {"success": true, "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O", "prediction": 0, ...},
    {"success": true, "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "prediction": 0, ...}
  ]
}
```

---

## Project Structure

```
MultiEndpointTox/
├── config/
│   └── config.yaml           # Main configuration
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned data
│   └── external/             # External validation sets
├── models/                   # Trained ML models
│   ├── herg/
│   ├── hepatotox/
│   ├── nephrotox/
│   ├── ames/
│   ├── skin_sens/
│   └── cytotox/
├── reports/                  # Validation reports
├── src/
│   ├── api/
│   │   ├── app.py           # FastAPI application
│   │   ├── predictor.py     # Prediction logic
│   │   └── explainer.py     # SHAP explanations
│   ├── data_curation/       # Data collection & cleaning
│   ├── feature_engineering/ # Molecular descriptors
│   ├── modeling/            # ML training
│   ├── validation/          # Model validation
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── requirements.txt
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v
```

**Test Coverage:**
- 56 tests covering API endpoints, predictor functionality, and edge cases
- Tests for all 6 toxicity endpoints
- SHAP interpretation tests
- Invalid input handling

---

## Model Performance

| Endpoint | Metric | Value | Dataset Size |
|----------|--------|-------|--------------|
| hERG | R² | 0.72 | 7,889 |
| Hepatotox | AUC-ROC | 0.82 | 1,597 |
| Nephrotox | AUC-ROC | 0.78 | 565 |
| Ames | AUC-ROC | 0.85 | 6,512 |
| Skin Sens | AUC-ROC | 0.79 | 1,100 |
| Cytotox | AUC-ROC | 0.81 | 8,371 |

---

## Configuration

Edit `config/config.yaml` to customize:

```yaml
project:
  name: MultiEndpointTox
  random_seed: 42

modeling:
  test_size: 0.2
  cv_folds: 5
  optimization_trials: 100

api:
  host: "127.0.0.1"
  port: 8000
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use MultiEndpointTox in your research, please cite:

```bibtex
@software{multiendpointtox2026,
  author = {Sharhabil Eltahir},
  title = {MultiEndpointTox: Multi-Endpoint Toxicity Prediction API},
  year = {2026},
  url = {https://github.com/sharhabileltahir/MultiEndpointTox}
}
```

---

## Author

**Sharhabil Eltahir**
MSc Biotechnology, Alexandria University
Research Assistant | Computational Pharmaceutical Chemistry

---

## Acknowledgments

- RDKit for cheminformatics
- SHAP for model interpretability
- FastAPI for the web framework
- scikit-learn, XGBoost, LightGBM for ML models
