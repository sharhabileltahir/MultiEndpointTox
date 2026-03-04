# MultiEndpointTox

**Multi-Endpoint Toxicity Prediction API for Drug Safety Assessment**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/sharhabileltahir/MultiEndpointTox/actions/workflows/ci.yml/badge.svg)](https://github.com/sharhabileltahir/MultiEndpointTox/actions)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://github.com/sharhabileltahir/MultiEndpointTox)

---

## Overview

MultiEndpointTox is a machine learning-powered REST API for predicting multiple drug toxicity endpoints from molecular structures. It provides interpretable predictions with SHAP explanations, structural alerts, molecular docking, 3D descriptors, and integrated risk assessment.

### Toxicity Endpoints

| Endpoint | Type | Description | Model |
|----------|------|-------------|-------|
| **hERG** | Regression | Cardiotoxicity (hERG channel inhibition) | XGBoost |
| **Hepatotox** | Classification | Drug-induced liver injury (DILI) | RandomForest |
| **Nephrotox** | Classification | Drug-induced kidney injury | RandomForest |
| **Ames** | Classification | Ames mutagenicity test | XGBoost |
| **Skin Sens** | Classification | Skin sensitization potential | RandomForest |
| **Cytotox** | Classification | General cytotoxicity | XGBoost |
| **Reproductive Tox** | Classification | Reproductive/developmental toxicity | LightGBM |

### Key Features

- **Multi-endpoint prediction** - Assess all toxicity risks in a single API call
- **SHAP interpretability** - Understand which molecular features drive predictions
- **Structural alerts** - Detection of known toxic substructures
- **Molecular docking** - AutoDock Vina integration for binding affinity prediction
- **3D descriptors** - Shape, surface, and volume calculations
- **2D/3D/GNN benchmarking workflows** - Reproducible scripts for classical ML and Chemprop D-MPNN comparison
- **Pharmacophore analysis** - Feature extraction and similarity comparison
- **Applicability domain** - Confidence assessment for each prediction
- **Integrated risk assessment** - Overall risk score and critical endpoint identification
- **Batch processing** - Predict up to 1000 compounds at once
- **PDF reports** - Comprehensive toxicity assessment reports

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

#### Molecular Docking

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/docking/status` | Check docking availability |
| GET | `/docking/targets` | List available protein targets |
| POST | `/dock` | Dock compound against target |
| GET | `/dock/{target}?smiles=...` | Dock via GET |
| POST | `/dock/batch` | Batch docking |
| POST | `/dock/enhanced` | Enhanced docking with 3D descriptors |
| POST | `/predict/ensemble` | ML + docking ensemble prediction |

#### 3D Descriptors & Pharmacophores

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/descriptors/3d` | Calculate 3D molecular descriptors |
| GET | `/descriptors/3d?smiles=...` | 3D descriptors via GET |
| POST | `/pharmacophore/features` | Extract pharmacophore features |
| POST | `/pharmacophore/compare` | Compare two compounds |

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

#### Molecular Docking

**Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "target": "herg",
  "exhaustiveness": 8
}
```

**Response:**
```json
{
  "success": true,
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "protein_id": "herg",
  "affinity": -6.4,
  "normalized_score": 0.32,
  "num_poses": 9,
  "interpretation": {
    "risk_level": "moderate",
    "description": "Moderate binding affinity"
  }
}
```

#### Enhanced Docking with 3D Descriptors

**Request:**
```json
{
  "smiles": "CC(=O)Nc1ccc(O)cc1",
  "target": "herg"
}
```

**Response:**
```json
{
  "docking": {
    "success": true,
    "affinity": -6.4,
    "normalized_score": 0.32
  },
  "descriptors_3d": {
    "shape_descriptors": {
      "asphericity": 0.6534,
      "eccentricity": 0.9911,
      "radius_of_gyration": 2.65
    },
    "volume_descriptors": {
      "molecular_volume": 140.02
    },
    "pharmacophore_counts": {
      "h_bond_acceptors": 2,
      "h_bond_donors": 2,
      "aromatic_rings": 1,
      "hydrophobic_centers": 6
    }
  },
  "binding_compatibility": {
    "shape_compatibility": "good",
    "size_compatibility": "good",
    "pharmacophore_match": "high",
    "binding_risk": "moderate"
  },
  "enhanced_score": {
    "score": 0.3795,
    "risk_level": "moderate",
    "shape_modifier": 1.1,
    "pharmacophore_modifier": 1.15
  }
}
```

#### Pharmacophore Comparison

**Request:**
```json
{
  "smiles1": "CC(=O)Nc1ccc(O)cc1",
  "smiles2": "CC(=O)OC1=CC=CC=C1C(=O)O"
}
```

**Response:**
```json
{
  "success": true,
  "pharmacophore_similarity": 0.087,
  "feature_comparison": {
    "h_bond_acceptors": [2, 4],
    "h_bond_donors": [2, 1],
    "aromatic_rings": [1, 1],
    "hydrophobic_centers": [6, 7]
  },
  "shape_comparison": {
    "asphericity": [0.653, 0.197],
    "molecular_volume": [140.0, 156.9]
  }
}
```

---

## Molecular Docking

MultiEndpointTox integrates AutoDock Vina for physics-based binding affinity prediction.

### Setup

1. **Install AutoDock Vina:**
   - Download from https://vina.scripps.edu/downloads/
   - Add to system PATH

2. **Enable docking in config:**
   ```yaml
   docking:
     enabled: true
     engine: "vina"
   ```

### Available Protein Targets

| Target | PDB ID | Description | Endpoints |
|--------|--------|-------------|-----------|
| hERG | 5VA1 | hERG potassium channel | herg |
| CYP3A4 | 1TQN | Cytochrome P450 3A4 | hepatotox |
| CYP2D6 | 4WNT | Cytochrome P450 2D6 | hepatotox |
| CYP2C9 | 1OG5 | Cytochrome P450 2C9 | hepatotox |
| AR | 2AM9 | Androgen receptor | reproductive_tox |
| ER-alpha | 1ERE | Estrogen receptor alpha | reproductive_tox |

### Docking Examples

```bash
# Single compound docking
curl -X POST "http://127.0.0.1:8000/dock" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "target": "herg"}'

# Enhanced docking with 3D descriptors
curl -X POST "http://127.0.0.1:8000/dock/enhanced" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "target": "herg"}'

# Ensemble prediction (ML + docking)
curl -X POST "http://127.0.0.1:8000/predict/ensemble" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "endpoint": "herg", "include_docking": true}'
```

### Batch Docking Script

```bash
# Dock compound library against multiple targets
python scripts/batch_docking.py \
  --input compounds.csv \
  --targets herg,hepatotox,cyp2d6 \
  --output results/docking_results.csv
```

---

## 3D Descriptors & Pharmacophores

Calculate 3D molecular properties and pharmacophore features for binding analysis.

### 3D Descriptor Categories

| Category | Descriptors |
|----------|-------------|
| **Shape** | Asphericity, eccentricity, PMI ratios, spherocity |
| **Surface** | TPSA, SASA (solvent accessible surface area) |
| **Volume** | Molecular volume |
| **Pharmacophore** | HBA, HBD, aromatic, hydrophobic, ionizable |

### Python Usage

```python
from src.docking import Descriptors3DCalculator, DockingManager

# Calculate 3D descriptors
calc = Descriptors3DCalculator()
result = calc.calculate_all("CC(=O)Nc1ccc(O)cc1")

print(f"Asphericity: {result.asphericity:.3f}")
print(f"Volume: {result.molecular_volume:.1f} Å³")
print(f"H-Bond Acceptors: {result.n_hba}")
print(f"Aromatic Rings: {result.n_aromatic}")

# Compare pharmacophores
dm = DockingManager(config=config)
comparison = dm.compare_pharmacophores(
    "CC(=O)Nc1ccc(O)cc1",  # Acetaminophen
    "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
)
print(f"Similarity: {comparison['pharmacophore_similarity']:.3f}")
```

---

## hERG Benchmark Workflows

This repository includes reproducible, publication-oriented benchmarking pipelines for hERG:

- **3D descriptor matrix generation** from SMILES with ETKDG + UFF optimization
- **Classical ML comparisons** for 2D vs 3D vs Hybrid features (classification and regression)
- **Graph Neural Network (D-MPNN)** benchmarking via Chemprop for regression

### 1) Build 3D Descriptor Matrix (ETKDG/UFF)

```bash
python scripts/build_3d_descriptor_matrix.py \
  --input-csv data/processed/herg_3d_binary_input.csv \
  --smiles-col SMILES \
  --target-col target \
  --output-dir data/interim/herg_3d_matrix
```

Outputs:
- `data/interim/herg_3d_matrix/X_3D.npy`
- `data/interim/herg_3d_matrix/y.npy`

### 2) Classical 2D/3D/Hybrid Comparison (Classification)

```bash
python scripts/compare_2d_3d_models.py \
  --x2d data/interim/herg_2d_matrix/X_2D.npy \
  --x3d data/interim/herg_3d_matrix/X_3D.npy \
  --y data/interim/herg_3d_matrix/y.npy \
  --output-dir results/model_comparison_2d_3d
```

### 3) Classical 2D/3D/Hybrid Comparison (Regression)

```bash
python scripts/compare_2d_3d_regression.py \
  --x2d data/interim/herg_2d_matrix/X_2D.npy \
  --x3d data/interim/herg_3d_matrix/X_3D.npy \
  --y data/interim/herg_regression/y.npy \
  --output-dir results/model_comparison_2d_3d_regression
```

### 4) Chemprop D-MPNN Workflow (Regression)

```bash
# Create isolated environment (recommended)
conda create -n chemprop python=3.11 -y
conda activate chemprop
python -m pip install --upgrade pip
python -m pip install "chemprop==2.2.2"

# Prepare Chemprop input CSV (smiles, pchembl_value)
python scripts/prepare_chemprop_herg_regression_dataset.py \
  --input-csv data/processed/herg_curated.csv \
  --output-csv data/processed/herg_chemprop_regression.csv \
  --smiles-col std_smiles \
  --target-col pchembl_value

# Run 80/20 split + 5-fold CV on training + external test
python scripts/run_chemprop_herg_regression.py \
  --data-csv data/processed/herg_chemprop_regression.csv \
  --chemprop-bin chemprop \
  --smiles-col smiles \
  --target-col pchembl_value \
  --test-size 0.2 \
  --n-splits 5 \
  --seed 42 \
  --output-dir results/model_comparison_2d_3d_regression/chemprop_dmpnn
```

Chemprop outputs:
- `cv_fold_predictions.csv`
- `test_set_predictions.csv`
- `cv_performance_summary.csv`
- `external_test_performance.csv`
- `fold_r2_values.json`
- Fold/final checkpoints under `cv/` and `final_model/`

---

## PDF Report Generation

Generate comprehensive toxicity assessment reports.

```bash
# Single compound report
python scripts/generate_report.py \
  --smiles "CC(=O)Nc1ccc(O)cc1" \
  --name "Acetaminophen" \
  --include-docking \
  --output reports/acetaminophen_report.pdf

# Multi-compound report from CSV
python scripts/generate_report.py \
  --input compounds.csv \
  --include-docking \
  --output reports/toxicity_report.pdf
```

### Report Contents

- **Compound Summary**: Structure image, properties, SMILES
- **Risk Assessment**: Overall risk score and gauge visualization
- **Endpoint Predictions**: Color-coded toxicity predictions
- **Docking Results**: Binding affinities and enhanced scores
- **3D Descriptors**: Shape and pharmacophore analysis
- **Binding Compatibility**: Target-specific compatibility analysis
- **Structural Alerts**: Detected toxicophores
- **Methodology**: Model and analysis descriptions

---

## Project Structure

```
MultiEndpointTox/
├── config/
│   └── config.yaml           # Main configuration
├── data/
│   ├── raw/                  # Original datasets
│   ├── processed/            # Cleaned data
│   ├── structures/           # Protein PDB/PDBQT files
│   └── external/             # External validation sets
├── models/                   # Trained ML models
│   ├── herg/
│   ├── hepatotox/
│   ├── nephrotox/
│   ├── ames/
│   ├── skin_sens/
│   ├── cytotox/
│   └── reproductive_tox/
├── reports/                  # Generated PDF reports
├── scripts/
│   ├── batch_docking.py     # Batch docking workflow
│   ├── generate_report.py   # PDF report generator
│   └── test_3d_descriptors.py
├── src/
│   ├── api/
│   │   ├── app.py           # FastAPI application
│   │   ├── predictor.py     # Prediction logic
│   │   └── explainer.py     # SHAP explanations
│   ├── data_curation/       # Data collection & cleaning
│   ├── docking/             # Molecular docking module
│   │   ├── docking_engine.py    # Vina/Smina interface
│   │   ├── docking_manager.py   # High-level docking API
│   │   ├── structure_manager.py # Protein preparation
│   │   └── descriptors_3d.py    # 3D descriptors & pharmacophores
│   ├── feature_engineering/ # Molecular descriptors
│   ├── modeling/            # ML training
│   ├── validation/          # Model validation
│   └── utils/               # Utilities
├── tests/                   # Test suite
├── requirements.txt
└── README.md
```

### Newly Added Benchmark Scripts

- `scripts/build_3d_descriptor_matrix.py`
- `scripts/compare_2d_3d_models.py`
- `scripts/compare_2d_3d_regression.py`
- `scripts/prepare_chemprop_herg_regression_dataset.py`
- `scripts/run_chemprop_herg_regression.py`

---

## Docker Deployment

```bash
# Build the image
docker build -t multiendpointtox .

# Run the container
docker run -p 8000:8000 multiendpointtox

# Run with environment variables
docker run -p 8000:8000 -e API_KEY=your-key multiendpointtox

# Docker Compose (if docker-compose.yml exists)
docker-compose up -d
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
- Tests for all 7 toxicity endpoints
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
| Reproductive Tox* | Accuracy | 0.75 | 117 |

> **Note on Reproductive Toxicity:** This endpoint has a limited training dataset (117 compounds). Predictions should be interpreted with caution and validated experimentally. The model uses a curated set of known reproductive toxicants and safe compounds from literature.

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

- RDKit for cheminformatics and 3D conformer generation
- SHAP for model interpretability
- FastAPI for the web framework
- scikit-learn, XGBoost, LightGBM for ML models
- AutoDock Vina for molecular docking
- ReportLab for PDF generation
