#!/usr/bin/env python3
"""
Multi-Endpoint Computational Toxicology Platform
=================================================
Main pipeline entry point.

Usage:
    python src/main.py --endpoint herg --phase data_curation
    python src/main.py --endpoint herg --phase all
    python src/main.py --phase platform_integration
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import yaml
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger


def load_config(config_path="config/config.yaml"):
    config_file = PROJECT_ROOT / config_path
    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)
    with open(config_file, "r") as f:
        return yaml.safe_load(f)


def run_data_curation(config, endpoint):
    logger.info("=" * 60)
    logger.info(f"PHASE 1: Data Curation - {endpoint.upper()}")
    logger.info("=" * 60)

    from src.data_curation.chembl_fetcher import ChEMBLFetcher
    from src.data_curation.data_cleaner import DataCleaner
    from src.data_curation.data_splitter import DataSplitter

    # Use specialized fetchers for specific endpoints
    if endpoint == "nephrotox":
        logger.info("Step 1/3: Fetching FDA DIRIL nephrotoxicity dataset...")
        from src.data_curation.diril_fetcher import DIRILFetcher
        fetcher = DIRILFetcher(config)
        raw_data = fetcher.fetch()
        logger.success(f"Fetched {len(raw_data)} drugs from DIRIL")
    elif endpoint == "ames":
        logger.info("Step 1/3: Fetching Ames mutagenicity dataset...")
        from src.data_curation.ames_fetcher import AmesFetcher
        fetcher = AmesFetcher(config)
        raw_data = fetcher.fetch()
        logger.success(f"Fetched {len(raw_data)} compounds for Ames")
    elif endpoint == "skin_sens":
        logger.info("Step 1/3: Fetching skin sensitization dataset...")
        from src.data_curation.skin_sensitization_fetcher import SkinSensitizationFetcher
        fetcher = SkinSensitizationFetcher(config)
        raw_data = fetcher.fetch()
        logger.success(f"Fetched {len(raw_data)} compounds for skin sensitization")
    elif endpoint == "cytotox":
        logger.info("Step 1/3: Fetching cytotoxicity dataset...")
        from src.data_curation.cytotox_fetcher import CytotoxFetcher
        fetcher = CytotoxFetcher(config)
        raw_data = fetcher.fetch()
        logger.success(f"Fetched {len(raw_data)} compounds for cytotoxicity")
    elif endpoint == "reproductive_tox":
        logger.info("Step 1/3: Fetching reproductive toxicity dataset...")
        from src.data_curation.reproductive_tox_fetcher import ReproductiveToxFetcher
        fetcher = ReproductiveToxFetcher(config)
        raw_data = fetcher.fetch()
        logger.success(f"Fetched {len(raw_data)} compounds for reproductive toxicity")
    else:
        logger.info("Step 1/3: Fetching bioactivity data from ChEMBL...")
        fetcher = ChEMBLFetcher(config)
        raw_data = fetcher.fetch_endpoint(endpoint)
        logger.success(f"Fetched {len(raw_data)} records for {endpoint}")

    logger.info("Step 2/3: Cleaning and standardizing data...")
    cleaner = DataCleaner(config)
    clean_data = cleaner.clean(raw_data, endpoint)
    logger.success(f"Cleaned dataset: {len(clean_data)} compounds")

    logger.info("Step 3/3: Splitting into train/test/validation sets...")
    splitter = DataSplitter(config)
    splits = splitter.split(clean_data, endpoint)
    logger.success(
        f"Split complete - Train: {len(splits['train'])}, "
        f"Test: {len(splits['test'])}, Val: {len(splits['val'])}"
    )
    return splits


def run_feature_engineering(config, endpoint):
    logger.info("=" * 60)
    logger.info(f"PHASE 2: Feature Engineering - {endpoint.upper()}")
    logger.info("=" * 60)

    from src.feature_engineering.descriptor_calculator import DescriptorCalculator
    from src.feature_engineering.fingerprint_generator import FingerprintGenerator
    from src.feature_engineering.feature_selector import FeatureSelector

    logger.info("Step 1/3: Calculating molecular descriptors...")
    calc = DescriptorCalculator(config)
    descriptors = calc.calculate(endpoint)
    logger.success(f"Calculated {descriptors.shape[1]} descriptors")

    logger.info("Step 2/3: Generating molecular fingerprints...")
    fp_gen = FingerprintGenerator(config)
    fingerprints = fp_gen.generate(endpoint)
    logger.success(f"Generated fingerprints: {fingerprints.shape}")

    logger.info("Step 3/3: Selecting optimal features...")
    selector = FeatureSelector(config)
    selected = selector.select(descriptors, fingerprints, endpoint)
    logger.success(f"Selected {selected.shape[1]} features")
    return selected


def run_modeling(config, endpoint):
    logger.info("=" * 60)
    logger.info(f"PHASE 3: Model Training - {endpoint.upper()}")
    logger.info("=" * 60)

    from src.modeling.model_trainer import ModelTrainer
    from src.modeling.hyperparameter_tuner import HyperparameterTuner

    logger.info("Step 1/2: Training baseline models (RF, XGBoost, SVM, LightGBM)...")
    trainer = ModelTrainer(config)
    baseline_results = trainer.train_all(endpoint)

    # Check task type from first model's metrics
    first_metrics = next(iter(baseline_results.values()))
    is_classification = "accuracy" in first_metrics

    for name, metrics in baseline_results.items():
        if is_classification:
            logger.info(f"  {name}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}")
        else:
            logger.info(f"  {name}: R2={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")

    logger.info("Step 2/2: Optimizing hyperparameters with Optuna...")
    tuner = HyperparameterTuner(config)
    best_model = tuner.optimize(endpoint)

    if is_classification:
        logger.success(
            f"Best model: {best_model['name']} - "
            f"Acc={best_model.get('accuracy', 0):.4f}, F1={best_model.get('f1', 0):.4f}"
        )
    else:
        logger.success(
            f"Best model: {best_model['name']} - "
            f"R2={best_model['r2']:.4f}, RMSE={best_model['rmse']:.4f}"
        )
    return best_model


def run_validation(config, endpoint):
    logger.info("=" * 60)
    logger.info(f"PHASE 4: Validation - {endpoint.upper()}")
    logger.info("=" * 60)

    from src.validation.cross_validator import CrossValidator
    from src.validation.applicability_domain import ApplicabilityDomain
    from src.validation.metrics_calculator import MetricsCalculator

    logger.info("Step 1/3: Running 5-fold cross-validation (3 repeats)...")
    cv = CrossValidator(config)
    cv_results = cv.validate(endpoint)

    logger.info("Step 2/3: Assessing applicability domain...")
    ad = ApplicabilityDomain(config)
    ad_results = ad.assess(endpoint)

    logger.info("Step 3/3: Generating validation report...")
    metrics = MetricsCalculator(config)
    report = metrics.generate_report(endpoint, cv_results, ad_results)
    logger.success(f"Validation complete for {endpoint}")
    return report


def run_platform_integration(config):
    logger.info("=" * 60)
    logger.info("PHASE 5: Platform Integration and Comparison")
    logger.info("=" * 60)

    from src.platform_integration.platform_comparator import PlatformComparator
    from src.platform_integration.consensus_predictor import ConsensusPredictor

    logger.info("Comparing predictions across external platforms...")
    comparator = PlatformComparator(config)
    comparison = comparator.compare_all()

    logger.info("Building consensus prediction model...")
    consensus = ConsensusPredictor(config)
    consensus_results = consensus.build(comparison)
    logger.success("Platform integration complete")
    return consensus_results


def run_full_pipeline(config, endpoint):
    logger.info("=" * 60)
    logger.info(f"FULL PIPELINE: {endpoint.upper()}")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    run_data_curation(config, endpoint)
    run_feature_engineering(config, endpoint)
    run_modeling(config, endpoint)
    run_validation(config, endpoint)
    logger.success(f"Full pipeline complete for {endpoint}!")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Endpoint Computational Toxicology Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --endpoint herg --phase data_curation
  python src/main.py --endpoint herg --phase all
  python src/main.py --endpoint hepatotox --phase modeling
  python src/main.py --phase platform_integration
        """,
    )
    parser.add_argument(
        "--endpoint",
        choices=["herg", "hepatotox", "nephrotox", "ames", "skin_sens", "cytotox", "reproductive_tox"],
        help="Toxicity endpoint to process",
    )
    parser.add_argument(
        "--phase",
        choices=[
            "data_curation", "feature_engineering", "modeling",
            "validation", "platform_integration", "all",
        ],
        required=True,
        help="Pipeline phase to execute",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.phase != "platform_integration" and not args.endpoint:
        parser.error("--endpoint is required for all phases except platform_integration")

    config = load_config(args.config)
    log_level = "DEBUG" if args.verbose else config.get("logging", {}).get("level", "INFO")
    setup_logger(log_level)

    logger.info(f"Pipeline started: endpoint={args.endpoint}, phase={args.phase}")

    phase_map = {
        "data_curation": lambda: run_data_curation(config, args.endpoint),
        "feature_engineering": lambda: run_feature_engineering(config, args.endpoint),
        "modeling": lambda: run_modeling(config, args.endpoint),
        "validation": lambda: run_validation(config, args.endpoint),
        "platform_integration": lambda: run_platform_integration(config),
        "all": lambda: run_full_pipeline(config, args.endpoint),
    }

    try:
        result = phase_map[args.phase]()
        logger.success("Pipeline completed successfully!")
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
