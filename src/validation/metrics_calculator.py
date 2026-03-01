"""Calculate and report comprehensive QSAR validation metrics."""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, balanced_accuracy_score,
    confusion_matrix
)

from src.utils.scaler import FeatureScaler


class MetricsCalculator:
    """
    Generate comprehensive validation reports for QSAR models.

    Aggregates CV results, AD assessment, and test set performance
    into detailed JSON and markdown reports.
    """

    def __init__(self, config):
        self.config = config
        self.metrics_config = config["validation"]["metrics"]
        self.targets = config["targets"]

        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.models_dir = Path(config["paths"]["models"])
        self.reports_dir = Path(config["paths"]["reports"])
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _get_task_type(self, endpoint):
        """Get task type from saved metadata."""
        meta_path = self.models_dir / endpoint / "task_type.txt"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return f.read().strip()
        return "regression"

    def generate_report(self, endpoint, cv_results, ad_results):
        """
        Generate comprehensive validation report.

        Args:
            endpoint: Toxicity endpoint name
            cv_results: Cross-validation results from CrossValidator
            ad_results: Applicability domain results from ApplicabilityDomain

        Returns:
            dict: Complete validation report
        """
        logger.info(f"Generating validation report for {endpoint}...")

        task_type = self._get_task_type(endpoint)
        logger.info(f"Task type: {task_type}")

        report = {
            "endpoint": endpoint,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "cv_folds": self.config["validation"]["cross_validation"]["n_splits"],
                "cv_repeats": self.config["validation"]["cross_validation"]["n_repeats"],
                "ad_method": self.config["validation"]["applicability_domain"]["method"],
            },
            "targets": self.targets.get(endpoint, {}),
            "models": {},
            "applicability_domain": ad_results if ad_results else {},
            "summary": {},
        }

        # Load test data and predictions
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        test_data = pd.read_csv(self.processed_dir / f"{endpoint}_test.csv")

        indices = np.load(self.processed_dir / f"{endpoint}_split_indices.npz")
        test_idx = indices["test"]

        X_test = features.iloc[test_idx].values

        # Determine target column based on task type
        if task_type == "classification":
            target_col = "activity_label"
        else:
            target_col = "pchembl_value"

        y_test = test_data[target_col].values

        scaler = FeatureScaler.load(self.config, endpoint)
        X_test_scaled = scaler.transform(X_test)

        # Find and evaluate all models
        from sklearn.base import is_classifier, is_regressor

        model_dir = self.models_dir / endpoint
        model_files = list(model_dir.glob("*.pkl"))
        model_files = [f for f in model_files if f.name not in [
            "feature_scaler.pkl", "feature_selection.pkl",
            "applicability_domain.pkl", "cv_results.pkl",
            "optuna_results.pkl", "ad_results.pkl"
        ]]

        if task_type == "classification":
            best_model = {"name": None, "f1": -np.inf}
        else:
            best_model = {"name": None, "r2": -np.inf}

        n_features = X_test_scaled.shape[1]

        for model_path in model_files:
            model_name = model_path.stem
            model = joblib.load(model_path)

            # Skip models that don't match the task type
            if task_type == "classification" and not is_classifier(model):
                logger.warning(f"Skipping {model_name} (not a classifier)")
                continue
            elif task_type == "regression" and not is_regressor(model):
                logger.warning(f"Skipping {model_name} (not a regressor)")
                continue

            # Skip models with incompatible feature dimensions
            if hasattr(model, "n_features_in_") and model.n_features_in_ != n_features:
                logger.warning(
                    f"Skipping {model_name} (feature mismatch: "
                    f"model expects {model.n_features_in_}, got {n_features})"
                )
                continue

            y_pred = model.predict(X_test_scaled)

            # Calculate comprehensive metrics based on task type
            if task_type == "classification":
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                metrics = self._calculate_classification_metrics(y_test, y_pred, y_prob)
            else:
                metrics = self._calculate_all_metrics(y_test, y_pred)

            # Add CV results if available
            if cv_results and model_name in cv_results:
                metrics["cv"] = cv_results[model_name]

            # Check against targets
            targets = self.targets.get(endpoint, {})
            metrics["meets_targets"] = self._check_targets(metrics, targets)

            report["models"][model_name] = metrics

            # Track best model
            if task_type == "classification":
                if metrics["f1"] > best_model["f1"]:
                    best_model = {"name": model_name, "f1": metrics["f1"]}
            else:
                if metrics["r2"] > best_model["r2"]:
                    best_model = {"name": model_name, "r2": metrics["r2"]}

        # Summary statistics
        if task_type == "classification":
            report["summary"] = {
                "best_model": best_model["name"],
                "best_f1": float(best_model["f1"]) if best_model["f1"] != -np.inf else None,
                "n_models_evaluated": len(report["models"]),
                "ad_coverage": ad_results.get("coverage", 0) if ad_results else 0,
                "n_test_samples": len(y_test),
            }
        else:
            report["summary"] = {
                "best_model": best_model["name"],
                "best_r2": float(best_model["r2"]) if best_model["r2"] != -np.inf else None,
                "n_models_evaluated": len(report["models"]),
                "ad_coverage": ad_results.get("coverage", 0) if ad_results else 0,
                "n_test_samples": len(y_test),
            }

        # Save JSON report
        report_path = self.reports_dir / f"{endpoint}_validation_report.json"
        with open(report_path, "w") as f:
            json.dump(self._make_serializable(report), f, indent=2)
        logger.info(f"JSON report saved to {report_path}")

        # Generate markdown summary
        if task_type == "classification":
            self._generate_classification_markdown_report(endpoint, report)
        else:
            self._generate_markdown_report(endpoint, report)

        return report

    def _calculate_all_metrics(self, y_true, y_pred):
        """Calculate all regression metrics."""
        residuals = y_true - y_pred

        metrics = {
            # Primary metrics
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),

            # Additional metrics
            "explained_variance": float(explained_variance_score(y_true, y_pred)),
            "mse": float(mean_squared_error(y_true, y_pred)),

            # Q2 (predictive R2)
            "q2": float(self._calculate_q2(y_true, y_pred)),

            # Concordance correlation coefficient
            "ccc": float(self._calculate_ccc(y_true, y_pred)),

            # Residual statistics
            "residual_mean": float(np.mean(residuals)),
            "residual_std": float(np.std(residuals)),
            "max_error": float(np.max(np.abs(residuals))),

            # Sample size
            "n_samples": len(y_true),
        }

        return metrics

    def _calculate_classification_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate all classification metrics."""
        metrics = {
            # Primary metrics
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),

            # Sample size
            "n_samples": len(y_true),
        }

        # AUC if probability estimates available
        if y_prob is not None and len(np.unique(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        else:
            metrics["auc"] = 0.0

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics["true_negatives"] = int(tn)
            metrics["false_positives"] = int(fp)
            metrics["false_negatives"] = int(fn)
            metrics["true_positives"] = int(tp)
            metrics["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            metrics["sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return metrics

    def _calculate_q2(self, y_true, y_pred, y_train_mean=None):
        """Calculate Q2 (predictive squared correlation coefficient)."""
        if y_train_mean is None:
            y_train_mean = np.mean(y_true)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_train_mean) ** 2)

        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def _calculate_ccc(self, y_true, y_pred):
        """Calculate Lin's Concordance Correlation Coefficient."""
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))

        numerator = 2 * cov
        denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

        return numerator / denominator if denominator > 0 else 0

    def _check_targets(self, metrics, targets):
        """Check if metrics meet target thresholds."""
        results = {}

        # Regression targets
        if "r2" in targets and "r2" in metrics:
            results["r2"] = metrics["r2"] >= targets["r2"]

        if "rmse" in targets and "rmse" in metrics:
            results["rmse"] = metrics["rmse"] <= targets["rmse"]

        # Classification targets
        if "accuracy" in targets and "accuracy" in metrics:
            results["accuracy"] = metrics["accuracy"] >= targets["accuracy"]

        if "f1" in targets and "f1" in metrics:
            results["f1"] = metrics["f1"] >= targets["f1"]

        if "auc" in targets and "auc" in metrics:
            results["auc"] = metrics["auc"] >= targets["auc"]

        if "sensitivity" in targets and "sensitivity" in metrics:
            results["sensitivity"] = metrics["sensitivity"] >= targets["sensitivity"]

        if "specificity" in targets and "specificity" in metrics:
            results["specificity"] = metrics["specificity"] >= targets["specificity"]

        if "mcc" in targets and "mcc" in metrics:
            results["mcc"] = metrics["mcc"] >= targets["mcc"]

        results["all"] = all(results.values()) if results else False

        return results

    def _generate_markdown_report(self, endpoint, report):
        """Generate a human-readable markdown report."""
        lines = [
            f"# Validation Report: {endpoint.upper()}",
            "",
            f"**Generated:** {report['timestamp']}",
            "",
            "## Summary",
            "",
            f"- **Best Model:** {report['summary']['best_model']}",
            f"- **Best R²:** {report['summary']['best_r2']:.4f}" if report['summary']['best_r2'] else "- **Best R²:** N/A",
            f"- **Models Evaluated:** {report['summary']['n_models_evaluated']}",
            f"- **Test Samples:** {report['summary']['n_test_samples']}",
            f"- **AD Coverage:** {report['summary']['ad_coverage']:.2%}",
            "",
            "## Target Metrics",
            "",
            "| Metric | Target | Best Achieved | Status |",
            "|--------|--------|---------------|--------|",
        ]

        targets = report["targets"]
        best_model_name = report["summary"]["best_model"]
        best_metrics = report["models"].get(best_model_name, {}) if best_model_name else {}

        for metric, target in targets.items():
            achieved = best_metrics.get(metric, "N/A")
            if isinstance(achieved, float):
                achieved_str = f"{achieved:.4f}"
            else:
                achieved_str = str(achieved)

            meets = best_metrics.get("meets_targets", {}).get(metric, False)
            status = "PASS" if meets else "FAIL"
            lines.append(f"| {metric} | {target} | {achieved_str} | {status} |")

        lines.extend([
            "",
            "## Model Comparison",
            "",
            "| Model | R² | RMSE | MAE | Q² | CCC |",
            "|-------|-----|------|-----|-----|-----|",
        ])

        for model_name, metrics in report["models"].items():
            lines.append(
                f"| {model_name} | {metrics['r2']:.4f} | {metrics['rmse']:.4f} | "
                f"{metrics['mae']:.4f} | {metrics['q2']:.4f} | {metrics['ccc']:.4f} |"
            )

        # CV Results section
        if any("cv" in m for m in report["models"].values()):
            lines.extend([
                "",
                "## Cross-Validation Results",
                "",
                "| Model | CV R² (mean ± std) | CV RMSE (mean ± std) |",
                "|-------|-------------------|---------------------|",
            ])

            for model_name, metrics in report["models"].items():
                if "cv" in metrics:
                    cv = metrics["cv"]
                    lines.append(
                        f"| {model_name} | {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f} | "
                        f"{cv['rmse_mean']:.4f} ± {cv['rmse_std']:.4f} |"
                    )

        # AD section
        if report.get("applicability_domain"):
            ad = report["applicability_domain"]
            lines.extend([
                "",
                "## Applicability Domain",
                "",
                f"- **Method:** {ad.get('method', 'N/A')}",
                f"- **Coverage:** {ad.get('coverage', 0):.2%}",
                f"- **Threshold:** {ad.get('threshold', 'N/A')}",
                f"- **Outliers:** {len(ad.get('outliers', []))} compounds outside AD",
            ])

        lines.extend([
            "",
            "---",
            "*Report generated by MultiEndpointTox Pipeline*",
        ])

        md_path = self.reports_dir / f"{endpoint}_validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved to {md_path}")

    def _generate_classification_markdown_report(self, endpoint, report):
        """Generate a human-readable markdown report for classification."""
        lines = [
            f"# Validation Report: {endpoint.upper()} (Classification)",
            "",
            f"**Generated:** {report['timestamp']}",
            f"**Task Type:** Classification",
            "",
            "## Summary",
            "",
            f"- **Best Model:** {report['summary']['best_model']}",
            f"- **Best F1:** {report['summary']['best_f1']:.4f}" if report['summary'].get('best_f1') else "- **Best F1:** N/A",
            f"- **Models Evaluated:** {report['summary']['n_models_evaluated']}",
            f"- **Test Samples:** {report['summary']['n_test_samples']}",
            f"- **AD Coverage:** {report['summary']['ad_coverage']:.2%}",
            "",
            "## Model Comparison",
            "",
            "| Model | Accuracy | Bal. Acc | Precision | Recall | F1 | MCC | AUC |",
            "|-------|----------|----------|-----------|--------|-----|-----|-----|",
        ]

        for model_name, metrics in report["models"].items():
            lines.append(
                f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['balanced_accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
                f"{metrics['mcc']:.4f} | {metrics['auc']:.4f} |"
            )

        # Confusion matrix for best model
        best_model_name = report["summary"]["best_model"]
        if best_model_name:
            best_metrics = report["models"].get(best_model_name, {})
            if "true_positives" in best_metrics:
                lines.extend([
                    "",
                    f"## Confusion Matrix (Best Model: {best_model_name})",
                    "",
                    "| | Predicted Negative | Predicted Positive |",
                    "|---|---|---|",
                    f"| Actual Negative | {best_metrics['true_negatives']} (TN) | {best_metrics['false_positives']} (FP) |",
                    f"| Actual Positive | {best_metrics['false_negatives']} (FN) | {best_metrics['true_positives']} (TP) |",
                    "",
                    f"- **Sensitivity (Recall):** {best_metrics.get('sensitivity', 0):.4f}",
                    f"- **Specificity:** {best_metrics.get('specificity', 0):.4f}",
                ])

        # CV Results section
        if any("cv" in m for m in report["models"].values()):
            lines.extend([
                "",
                "## Cross-Validation Results",
                "",
                "| Model | CV Accuracy (mean +/- std) | CV F1 (mean +/- std) | CV AUC (mean +/- std) |",
                "|-------|---------------------------|---------------------|----------------------|",
            ])

            for model_name, metrics in report["models"].items():
                if "cv" in metrics:
                    cv = metrics["cv"]
                    lines.append(
                        f"| {model_name} | {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f} | "
                        f"{cv['f1_mean']:.4f} +/- {cv['f1_std']:.4f} | "
                        f"{cv['auc_mean']:.4f} +/- {cv['auc_std']:.4f} |"
                    )

        # AD section
        if report.get("applicability_domain"):
            ad = report["applicability_domain"]
            lines.extend([
                "",
                "## Applicability Domain",
                "",
                f"- **Method:** {ad.get('method', 'N/A')}",
                f"- **Coverage:** {ad.get('coverage', 0):.2%}",
                f"- **Threshold:** {ad.get('threshold', 'N/A')}",
                f"- **Outliers:** {len(ad.get('outliers', []))} compounds outside AD",
            ])

        lines.extend([
            "",
            "---",
            "*Report generated by MultiEndpointTox Pipeline*",
        ])

        md_path = self.reports_dir / f"{endpoint}_validation_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"Markdown report saved to {md_path}")

    def _make_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj
