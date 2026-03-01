# Multi-Endpoint Computational Toxicology Platform

## Integrated ML Prediction of Cardiotoxicity, Hepatotoxicity & Nephrotoxicity

---

## Project Overview

This platform develops integrated QSAR models for predicting three critical drug toxicity endpoints:

| Endpoint | Target | Dataset Size | Timeline |
|----------|--------|-------------|----------|
| hERG Cardiotoxicity | R2 > 0.85 | 8,000-10,000 | Months 1-3 |
| Hepatotoxicity (DILI) | R2 > 0.80 | 5,000-8,000 | Months 3-6 |
| Nephrotoxicity | R2 > 0.70 | 2,000-5,000 | Months 6-9 |
| Integrated Platform | Multi-endpoint | Combined | Months 9-12 |

---

## Quick Start

```bash
# 1. Open in VS Code
code MultiEndpointTox/

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import rdkit; print(f'RDKit {rdkit.__version__}')"

# 5. Run the pipeline
python src/main.py --endpoint herg --phase data_curation
```

---

## Project Structure

```
MultiEndpointTox/
|
|-- .vscode/                    # VS Code settings
|   |-- settings.json           # Project-specific settings
|   |-- launch.json             # Debug configurations
|   |-- extensions.json         # Recommended extensions
|   |-- tasks.json              # Build/run tasks
|
|-- config/                     # Configuration files
|   |-- config.yaml             # Main project config
|
|-- data/
|   |-- raw/                    # Original immutable data
|   |   |-- herg/
|   |   |-- hepatotox/
|   |   |-- nephrotox/
|   |-- processed/              # Cleaned, curated data
|   |-- external/               # External validation sets
|   |-- interim/                # Intermediate transforms
|
|-- src/                        # Source code
|   |-- main.py                 # Main pipeline entry point
|   |-- data_curation/          # Phase 1: Data collection
|   |-- feature_engineering/    # Phase 2: Descriptors
|   |-- modeling/               # Phase 3: ML training
|   |-- validation/             # Phase 4: Validation
|   |-- platform_integration/   # Phase 5: Multi-platform
|   |-- visualization/          # Plotting
|   |-- utils/                  # Shared utilities
|
|-- notebooks/                  # Jupyter notebooks
|-- models/                     # Saved trained models
|-- reports/                    # Generated reports
|-- tests/                      # Unit tests
|-- requirements.txt
|-- Makefile
|-- pyproject.toml
|-- README.md
```

---

## Pipeline Phases

### Phase 1: Data Curation (Months 1-2)
```bash
python src/main.py --endpoint herg --phase data_curation
```

### Phase 2: Feature Engineering (Month 2-3)
```bash
python src/main.py --endpoint herg --phase feature_engineering
```

### Phase 3: Model Training (Month 3-4)
```bash
python src/main.py --endpoint herg --phase modeling
```

### Phase 4: Validation (Month 4-5)
```bash
python src/main.py --endpoint herg --phase validation
```

### Phase 5: Platform Integration (Month 9-12)
```bash
python src/main.py --phase platform_integration
```

---

## Expected Outputs

- 4-6 peer-reviewed publications
- 3 validated QSAR models (hERG, hepatotox, nephrotox)
- 1 integrated multi-endpoint platform
- Open-source Python package + GitHub repository

---

## Author

Sharhabil - MSc Biotechnology, Alexandria University
Research Assistant | Computational Pharmaceutical Chemistry
