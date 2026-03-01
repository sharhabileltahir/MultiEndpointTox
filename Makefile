.PHONY: setup install test lint clean herg hepatotox nephrotox platform

setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

install:
	pip install -r requirements.txt

herg-data:
	python src/main.py --endpoint herg --phase data_curation

herg-features:
	python src/main.py --endpoint herg --phase feature_engineering

herg-model:
	python src/main.py --endpoint herg --phase modeling

herg-validate:
	python src/main.py --endpoint herg --phase validation

herg:
	python src/main.py --endpoint herg --phase all

hepatotox:
	python src/main.py --endpoint hepatotox --phase all

nephrotox:
	python src/main.py --endpoint nephrotox --phase all

platform:
	python src/main.py --phase platform_integration

all: herg hepatotox nephrotox platform

test:
	pytest tests/ -v --tb=short

lint:
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

verify:
	python -c "import rdkit, sklearn, xgboost, pandas; print('All packages OK')"
