"""Cross-validation module for QSAR model validation."""

import copy
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score
)
from sklearn.base import clone

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class CrossValidator:
    """
    Perform rigorous cross-validation with repeated k-fold.

    Supports both regression and classification tasks.
    """

    def __init__(self, config):
        self.config = config
        self.cv_config = config["validation"]["cross_validation"]
        self.n_splits = self.cv_config["n_splits"]
        self.n_repeats = self.cv_config["n_repeats"]
        self.random_state = config["modeling"]["random_state"]

        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.models_dir = Path(config["paths"]["models"])

    def _get_task_type(self, endpoint):
        """Get task type from saved metadata or infer from data."""
        meta_path = self.models_dir / endpoint / "task_type.txt"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return f.read().strip()

        # Infer from data
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")
        if "activity_label" in train_data.columns:
            return "classification"
        return "regression"

    def _load_full_training_data(self, endpoint):
        """Load combined train+val data for cross-validation."""
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")
        val_data = pd.read_csv(self.processed_dir / f"{endpoint}_val.csv")

        indices = np.load(self.processed_dir / f"{endpoint}_split_indices.npz")
        train_idx = indices["train"]
        val_idx = indices["val"]

        # Combine train and validation for CV
        combined_idx = np.concatenate([train_idx, val_idx])
        X = features.iloc[combined_idx].values

        # Determine target column
        task_type = self._get_task_type(endpoint)
        if task_type == "classification":
            target_col = "activity_label"
        else:
            target_col = "pchembl_value"

        y = np.concatenate([
            train_data[target_col].values,
            val_data[target_col].values
        ])

        return X, y, task_type

    def validate(self, endpoint):
        """
        Run repeated k-fold cross-validation on all trained models.
        """
        logger.info(f"Running {self.n_splits}-fold CV with {self.n_repeats} repeats...")

        X, y, task_type = self._load_full_training_data(endpoint)
        logger.info(f"CV data: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Task type: {task_type}")

        # Find all trained models
        model_dir = self.models_dir / endpoint
        model_files = list(model_dir.glob("*.pkl"))
        model_files = [f for f in model_files if f.name not in [
            "feature_scaler.pkl", "feature_selection.pkl",
            "applicability_domain.pkl", "cv_results.pkl",
            "optuna_results.pkl", "ad_results.pkl"
        ]]

        # Filter models based on task type to avoid mixing regression/classification
        if task_type == "classification":
            # Exclude regression model names
            regression_names = ["svr", "randomforestregressor", "xgbregressor", "lgbmregressor"]
            model_files = [f for f in model_files if not any(
                rn in f.stem.lower() for rn in regression_names
            )]
        else:
            # Exclude classification model names
            classification_names = ["svc", "randomforestclassifier", "xgbclassifier", "lgbmclassifier"]
            model_files = [f for f in model_files if not any(
                cn in f.stem.lower() for cn in classification_names
            )]

        if not model_files:
            logger.warning(f"No models found in {model_dir}")
            return {}

        logger.info(f"Found {len(model_files)} models to validate")

        # Use stratified CV for classification
        if task_type == "classification":
            cv = RepeatedStratifiedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )
            results = self._validate_classification(X, y, model_files, cv)
        else:
            cv = RepeatedKFold(
                n_splits=self.n_splits,
                n_repeats=self.n_repeats,
                random_state=self.random_state
            )
            results = self._validate_regression(X, y, model_files, cv)

        # Save CV results
        cv_path = self.models_dir / endpoint / "cv_results.pkl"
        joblib.dump(results, cv_path)
        logger.info(f"CV results saved to {cv_path}")

        return results

    def _validate_regression(self, X, y, model_files, cv):
        """Validate regression models."""
        from sklearn.base import is_regressor
        results = {}

        for model_path in model_files:
            model_name = model_path.stem
            model_template = joblib.load(model_path)

            # Skip classification models
            if not is_regressor(model_template):
                logger.warning(f"  Skipping {model_name} (not a regressor)")
                continue

            logger.info(f"  Validating {model_name}...")
            fold_metrics = {"r2": [], "rmse": [], "mae": [], "q2": []}

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)

                try:
                    model_clone = clone(model_template)
                except Exception:
                    model_clone = copy.deepcopy(model_template)
                model_clone.fit(X_train_scaled, y_train_fold)

                y_pred_fold = model_clone.predict(X_val_scaled)

                r2 = r2_score(y_val_fold, y_pred_fold)
                rmse = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                mae = mean_absolute_error(y_val_fold, y_pred_fold)

                ss_res = np.sum((y_val_fold - y_pred_fold) ** 2)
                ss_tot = np.sum((y_val_fold - np.mean(y_train_fold)) ** 2)
                q2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                fold_metrics["r2"].append(r2)
                fold_metrics["rmse"].append(rmse)
                fold_metrics["mae"].append(mae)
                fold_metrics["q2"].append(q2)

            results[model_name] = {
                "r2_mean": float(np.mean(fold_metrics["r2"])),
                "r2_std": float(np.std(fold_metrics["r2"])),
                "rmse_mean": float(np.mean(fold_metrics["rmse"])),
                "rmse_std": float(np.std(fold_metrics["rmse"])),
                "mae_mean": float(np.mean(fold_metrics["mae"])),
                "mae_std": float(np.std(fold_metrics["mae"])),
                "q2_mean": float(np.mean(fold_metrics["q2"])),
                "q2_std": float(np.std(fold_metrics["q2"])),
                "n_folds": self.n_splits * self.n_repeats,
                "task_type": "regression",
            }

            logger.info(
                f"    R2: {results[model_name]['r2_mean']:.4f} ± {results[model_name]['r2_std']:.4f}"
            )

        return results

    def _validate_classification(self, X, y, model_files, cv):
        """Validate classification models."""
        from sklearn.base import is_classifier
        results = {}

        for model_path in model_files:
            model_name = model_path.stem
            model_template = joblib.load(model_path)

            # Skip regression models
            if not is_classifier(model_template):
                logger.warning(f"  Skipping {model_name} (not a classifier)")
                continue

            logger.info(f"  Validating {model_name}...")
            fold_metrics = {
                "accuracy": [], "balanced_accuracy": [], "precision": [],
                "recall": [], "f1": [], "mcc": [], "auc": []
            }

            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                X_train_fold = X[train_idx]
                X_val_fold = X[val_idx]
                y_train_fold = y[train_idx]
                y_val_fold = y[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_fold)
                X_val_scaled = scaler.transform(X_val_fold)

                # Apply SMOTE to training fold if imbalanced
                X_train_resampled, y_train_resampled = X_train_scaled, y_train_fold
                if SMOTE_AVAILABLE:
                    unique, counts = np.unique(y_train_fold, return_counts=True)
                    if len(unique) == 2:
                        minority_count = min(counts)
                        majority_count = max(counts)
                        if majority_count / minority_count > 2 and minority_count > 1:
                            try:
                                k_neighbors = min(5, minority_count - 1)
                                if k_neighbors >= 1:
                                    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                    X_train_resampled, y_train_resampled = smote.fit_resample(
                                        X_train_scaled, y_train_fold
                                    )
                            except Exception:
                                pass  # Use original data if SMOTE fails

                try:
                    model_clone = clone(model_template)
                except Exception:
                    model_clone = copy.deepcopy(model_template)
                model_clone.fit(X_train_resampled, y_train_resampled)

                y_pred_fold = model_clone.predict(X_val_scaled)

                fold_metrics["accuracy"].append(accuracy_score(y_val_fold, y_pred_fold))
                fold_metrics["balanced_accuracy"].append(balanced_accuracy_score(y_val_fold, y_pred_fold))
                fold_metrics["precision"].append(precision_score(y_val_fold, y_pred_fold, zero_division=0))
                fold_metrics["recall"].append(recall_score(y_val_fold, y_pred_fold, zero_division=0))
                fold_metrics["f1"].append(f1_score(y_val_fold, y_pred_fold, zero_division=0))
                fold_metrics["mcc"].append(matthews_corrcoef(y_val_fold, y_pred_fold))

                # AUC
                if hasattr(model_clone, "predict_proba") and len(np.unique(y_val_fold)) > 1:
                    y_prob = model_clone.predict_proba(X_val_scaled)[:, 1]
                    fold_metrics["auc"].append(roc_auc_score(y_val_fold, y_prob))
                else:
                    fold_metrics["auc"].append(0.0)

            results[model_name] = {
                "accuracy_mean": float(np.mean(fold_metrics["accuracy"])),
                "accuracy_std": float(np.std(fold_metrics["accuracy"])),
                "balanced_accuracy_mean": float(np.mean(fold_metrics["balanced_accuracy"])),
                "balanced_accuracy_std": float(np.std(fold_metrics["balanced_accuracy"])),
                "precision_mean": float(np.mean(fold_metrics["precision"])),
                "precision_std": float(np.std(fold_metrics["precision"])),
                "recall_mean": float(np.mean(fold_metrics["recall"])),
                "recall_std": float(np.std(fold_metrics["recall"])),
                "f1_mean": float(np.mean(fold_metrics["f1"])),
                "f1_std": float(np.std(fold_metrics["f1"])),
                "mcc_mean": float(np.mean(fold_metrics["mcc"])),
                "mcc_std": float(np.std(fold_metrics["mcc"])),
                "auc_mean": float(np.mean(fold_metrics["auc"])),
                "auc_std": float(np.std(fold_metrics["auc"])),
                "n_folds": self.n_splits * self.n_repeats,
                "task_type": "classification",
            }

            logger.info(
                f"    Acc: {results[model_name]['accuracy_mean']:.4f} ± {results[model_name]['accuracy_std']:.4f}, "
                f"F1: {results[model_name]['f1_mean']:.4f} ± {results[model_name]['f1_std']:.4f}"
            )

        return results
