# MultiEndpointTox v1.0.0

## Multi-endpoint Toxicity Prediction API for Drug Safety Assessment

### Features

- **6 Toxicity Prediction Endpoints**
  - hERG (Cardiotoxicity) - Regression
  - Hepatotoxicity (DILI) - Classification
  - Nephrotoxicity - Classification
  - Ames Mutagenicity - Classification
  - Skin Sensitization - Classification
  - Cytotoxicity - Classification

- **SHAP Interpretability**
  - Feature importance explanations for each prediction
  - Categorized features (physicochemical, fingerprints, substructures)
  - Human-readable interpretation summaries

- **Integrated Risk Assessment**
  - Overall risk score across all endpoints
  - Critical endpoint identification
  - Risk level classification (low/moderate/high)
  - Design recommendations

- **Structural Alerts**
  - Detection of known toxic substructures
  - PAINS filter alerts
  - Endpoint-specific alerts

- **API Endpoints**
  - Single compound prediction
  - Multi-endpoint prediction
  - Batch prediction (up to 1000 compounds)
  - SMILES validation
  - Applicability domain assessment

### Technical Details

- **Models**: XGBoost and RandomForest classifiers
- **Optimization**: Hyperparameter tuning via Optuna
- **Features**: Morgan fingerprints + RDKit descriptors (500 features)
- **Validation**: Cross-validation with applicability domain

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start

```bash
# Start the API server
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000

# Test prediction
curl -X POST "http://127.0.0.1:8000/predict/integrated" \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Nc1ccc(O)cc1", "include_interpretation": true}'
```

### API Documentation

Once the server is running, visit:
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

### Test Suite

```bash
pytest tests/ -v
```

56 tests covering API endpoints, predictor functionality, and edge cases.

### License

MIT License
