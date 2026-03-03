# Contributing to MultiEndpointTox

Thank you for your interest in contributing to MultiEndpointTox! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Adding New Endpoints](#adding-new-endpoints)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone regardless of experience level.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/MultiEndpointTox.git
   cd MultiEndpointTox
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/sharhabileltahir/MultiEndpointTox.git
   ```

## Development Setup

### Prerequisites

- Python 3.10+
- Conda (recommended) or virtualenv
- Git

### Environment Setup

```bash
# Create conda environment (recommended)
conda create -n toxpred python=3.11
conda activate toxpred

# Or use virtualenv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Create requirements-dev.txt if needed:

```bash
pip install pytest pytest-cov pytest-asyncio httpx
pip install black isort ruff mypy
pip install pre-commit
```

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Code Style

We follow these conventions:

### Python Style

- **Formatter**: Black (line length 88)
- **Import sorting**: isort
- **Linter**: Ruff
- **Type hints**: Required for public APIs

### Running Formatters

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint
ruff check src/ --fix
```

### Naming Conventions

- **Files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private methods**: `_leading_underscore`

### Docstrings

Use Google-style docstrings:

```python
def predict_toxicity(smiles: str, endpoint: str) -> dict:
    """
    Predict toxicity for a compound.

    Args:
        smiles: SMILES string of the compound
        endpoint: Toxicity endpoint (e.g., 'hepatotox')

    Returns:
        Dictionary containing prediction results

    Raises:
        ValueError: If SMILES is invalid
    """
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py -v

# Run specific test
pytest tests/test_api.py::test_predict_single -v
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use fixtures for common setup
- Test both success and error cases

Example:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_predict_valid_smiles():
    response = client.post(
        "/predict",
        json={"smiles": "CCO", "endpoint": "hepatotox"}
    )
    assert response.status_code == 200
    assert response.json()["success"] == True

def test_predict_invalid_smiles():
    response = client.post(
        "/predict",
        json={"smiles": "invalid", "endpoint": "hepatotox"}
    )
    assert response.status_code == 400
```

### Coverage Requirements

- Aim for >80% coverage on new code
- Critical paths (predictions, API endpoints) should have >90% coverage

## Submitting Changes

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation
- `refactor/description` - Code refactoring

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(docking): add CYP2D6 protein target
fix(api): handle empty SMILES in batch prediction
docs(readme): update installation instructions
```

### Pull Request Process

1. Update your fork:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. Create a feature branch:
   ```bash
   git checkout -b feature/my-feature
   ```

3. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

4. Push to your fork:
   ```bash
   git push origin feature/my-feature
   ```

5. Open a Pull Request on GitHub

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted (black, isort)
- [ ] No linting errors (ruff)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventions
- [ ] PR description explains changes

## Adding New Endpoints

To add a new toxicity endpoint:

### 1. Data Curation

Create a data fetcher in `src/data_curation/`:

```python
# src/data_curation/new_endpoint_fetcher.py
class NewEndpointDataFetcher:
    def fetch_data(self) -> pd.DataFrame:
        # Fetch and clean data
        pass
```

### 2. Configuration

Add endpoint to `config/config.yaml`:

```yaml
endpoints:
  new_endpoint:
    type: classification  # or regression
    task: binary  # or multi-class, regression
    models:
      - randomforest
      - xgboost
```

### 3. Training

Run the training pipeline:

```bash
python src/main.py --endpoint new_endpoint --mode train
```

### 4. API Integration

The endpoint will be automatically available via the API.

### 5. Testing

Add tests in `tests/test_new_endpoint.py`.

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for questions
- Email: [maintainer email]

Thank you for contributing!
