"""Hyperparameter Tuner - Bayesian optimization with Optuna."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger

import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score
)
import xgboost as xgb
import lightgbm as lgb

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from src.utils.scaler import FeatureScaler

optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterTuner:
    """
    Bayesian hyperparameter optimization using Optuna.

    Optimizes hyperparameters for RF, XGBoost, LightGBM, and SVR/SVC
    using cross-validation on properly aligned training data.
    Supports both regression and classification tasks.
    """

    def __init__(self, config):
        self.config = config
        self.n_trials = config["modeling"]["hyperparameter_tuning"]["n_trials"]
        self.cv_folds = config["modeling"]["hyperparameter_tuning"]["cv_folds"]
        self.random_state = config["modeling"]["random_state"]

        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.models_dir = Path(config["paths"]["models"])

    def _get_task_type(self, endpoint):
        """Get task type from saved metadata."""
        meta_path = self.models_dir / endpoint / "task_type.txt"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return f.read().strip()
        return "regression"

    def _load_training_data(self, endpoint):
        """Load and align training data with proper scaling."""
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")

        # Load indices for alignment
        indices_path = self.processed_dir / f"{endpoint}_split_indices.npz"
        if indices_path.exists():
            indices = np.load(indices_path)
            train_idx = indices["train"]
            X_train = features.iloc[train_idx].values
        elif "original_index" in train_data.columns:
            X_train = features.iloc[train_data["original_index"].values].values
        else:
            raise ValueError(
                f"Cannot align features for {endpoint}. Re-run data curation."
            )

        # Determine target column based on task type
        task_type = self._get_task_type(endpoint)
        if task_type == "classification":
            y_train = train_data["activity_label"].values
        else:
            y_train = train_data["pchembl_value"].values

        # Load or fit scaler
        scaler_path = self.models_dir / endpoint / "feature_scaler.pkl"
        if scaler_path.exists():
            scaler = FeatureScaler.load(self.config, endpoint)
            X_train_scaled = scaler.transform(X_train)
        else:
            scaler = FeatureScaler(self.config, method="standard")
            X_train_scaled = scaler.fit_transform(X_train, endpoint)

        return X_train_scaled, y_train, task_type

    # ==================== REGRESSION OBJECTIVES ====================

    def _create_rf_objective_regression(self, X, y):
        """Create Optuna objective for Random Forest Regressor."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "random_state": self.random_state,
                "n_jobs": -1,
            }

            model = RandomForestRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring="r2", n_jobs=-1
            )
            return scores.mean()

        return objective

    def _create_xgb_objective_regression(self, X, y):
        """Create Optuna objective for XGBoost Regressor."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "random_state": self.random_state,
                "verbosity": 0,
                "n_jobs": -1,
            }

            model = xgb.XGBRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring="r2", n_jobs=-1
            )
            return scores.mean()

        return objective

    def _create_lgb_objective_regression(self, X, y):
        """Create Optuna objective for LightGBM Regressor."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }

            model = lgb.LGBMRegressor(**params)
            scores = cross_val_score(
                model, X, y, cv=self.cv_folds, scoring="r2", n_jobs=-1
            )
            return scores.mean()

        return objective

    # ==================== CLASSIFICATION OBJECTIVES ====================

    def _create_rf_objective_classification(self, X, y):
        """Create Optuna objective for Random Forest Classifier."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "random_state": self.random_state,
                "n_jobs": -1,
            }

            model = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="f1", n_jobs=-1
            )
            return scores.mean()

        return objective

    def _create_xgb_objective_classification(self, X, y):
        """Create Optuna objective for XGBoost Classifier."""
        # Calculate scale_pos_weight for imbalanced data
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        # Handle both cases: more negatives or more positives
        if n_neg > n_pos:
            # Standard case: more negatives, scale up positives
            scale_weight_low = 1.0
            scale_weight_high = max(2.0, (n_neg / n_pos) * 2)
        else:
            # Inverted case: more positives, scale down (use values < 1)
            scale_weight_low = max(0.1, (n_neg / n_pos) / 2)
            scale_weight_high = 1.0

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "scale_pos_weight": trial.suggest_float("scale_pos_weight", scale_weight_low, scale_weight_high),
                "random_state": self.random_state,
                "verbosity": 0,
                "n_jobs": -1,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }

            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="f1", n_jobs=-1
            )
            return scores.mean()

        return objective

    def _create_lgb_objective_classification(self, X, y):
        """Create Optuna objective for LightGBM Classifier."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
                "random_state": self.random_state,
                "verbose": -1,
                "n_jobs": -1,
            }

            model = lgb.LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = cross_val_score(
                model, X, y, cv=cv, scoring="f1", n_jobs=-1
            )
            return scores.mean()

        return objective

    def optimize(self, endpoint):
        """
        Run hyperparameter optimization for all models.

        Args:
            endpoint: Toxicity endpoint name

        Returns:
            dict: Best model info with metrics and parameters
        """
        logger.info(f"Starting Optuna optimization ({self.n_trials} trials per model)...")

        X_train, y_train, task_type = self._load_training_data(endpoint)
        logger.info(f"Training data shape: {X_train.shape}, Task type: {task_type}")

        if task_type == "classification":
            return self._optimize_classification(endpoint, X_train, y_train)
        else:
            return self._optimize_regression(endpoint, X_train, y_train)

    def _optimize_regression(self, endpoint, X_train, y_train):
        """Run optimization for regression models."""
        logger.info("Optimizing: RandomForest, XGBoost, LightGBM (Regression)")

        model_objectives = {
            "RandomForest": (self._create_rf_objective_regression, RandomForestRegressor),
            "XGBoost": (self._create_xgb_objective_regression, xgb.XGBRegressor),
            "LightGBM": (self._create_lgb_objective_regression, lgb.LGBMRegressor),
        }

        all_results = {}
        best_overall = {"name": None, "cv_score": -np.inf, "params": {}}

        for model_name, (objective_fn, model_class) in model_objectives.items():
            logger.info(f"  Optimizing {model_name}...")

            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                objective_fn(X_train, y_train),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_score = study.best_value

            all_results[model_name] = {
                "best_params": best_params,
                "best_cv_r2": best_score,
                "n_trials": self.n_trials,
            }

            logger.info(f"    Best CV R2: {best_score:.4f}")

            if best_score > best_overall["cv_score"]:
                best_overall = {
                    "name": f"{model_name}_optimized",
                    "cv_score": best_score,
                    "params": best_params,
                    "model_class": model_class,
                }

        # Save all optimization results
        results_path = self.models_dir / endpoint / "optuna_results.pkl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(all_results, results_path)
        logger.info(f"Optimization results saved to {results_path}")

        # Train and evaluate best model on test set
        best_model_result = self._train_best_model_regression(endpoint, best_overall)

        return {
            "name": best_overall["name"],
            "r2": best_model_result["test_r2"],
            "rmse": best_model_result["test_rmse"],
            "cv_r2": best_overall["cv_score"],
            "params": best_overall["params"],
        }

    def _optimize_classification(self, endpoint, X_train, y_train):
        """Run optimization for classification models with SMOTE."""
        logger.info("Optimizing: RandomForest, XGBoost, LightGBM (Classification)")

        # Apply SMOTE before optimization if imbalanced
        X_train_resampled, y_train_resampled = X_train, y_train
        if SMOTE_AVAILABLE:
            unique, counts = np.unique(y_train, return_counts=True)
            if len(unique) == 2:
                minority_count = min(counts)
                majority_count = max(counts)
                if majority_count / minority_count > 2 and minority_count > 1:
                    logger.info(f"Applying SMOTE for optimization (ratio: {majority_count/minority_count:.1f}:1)...")
                    try:
                        k_neighbors = min(5, minority_count - 1)
                        if k_neighbors >= 1:
                            smote = SMOTE(random_state=self.random_state, k_neighbors=k_neighbors)
                            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
                            unique_new, counts_new = np.unique(y_train_resampled, return_counts=True)
                            logger.info(f"After SMOTE: {dict(zip(unique_new, counts_new))}")
                    except Exception as e:
                        logger.warning(f"SMOTE failed: {e}, using original data")

        model_objectives = {
            "RandomForest": (self._create_rf_objective_classification, RandomForestClassifier),
            "XGBoost": (self._create_xgb_objective_classification, xgb.XGBClassifier),
            "LightGBM": (self._create_lgb_objective_classification, lgb.LGBMClassifier),
        }

        all_results = {}
        best_overall = {"name": None, "cv_score": -np.inf, "params": {}}

        for model_name, (objective_fn, model_class) in model_objectives.items():
            logger.info(f"  Optimizing {model_name}...")

            sampler = TPESampler(seed=self.random_state)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(
                objective_fn(X_train_resampled, y_train_resampled),
                n_trials=self.n_trials,
                show_progress_bar=False,
            )

            best_params = study.best_params
            best_score = study.best_value

            all_results[model_name] = {
                "best_params": best_params,
                "best_cv_f1": best_score,
                "n_trials": self.n_trials,
            }

            logger.info(f"    Best CV F1: {best_score:.4f}")

            if best_score > best_overall["cv_score"]:
                best_overall = {
                    "name": f"{model_name}_optimized",
                    "cv_score": best_score,
                    "params": best_params,
                    "model_class": model_class,
                }

        # Save all optimization results
        results_path = self.models_dir / endpoint / "optuna_results.pkl"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(all_results, results_path)
        logger.info(f"Optimization results saved to {results_path}")

        # Train and evaluate best model on test set
        best_model_result = self._train_best_model_classification(endpoint, best_overall)

        return {
            "name": best_overall["name"],
            "accuracy": best_model_result["test_accuracy"],
            "f1": best_model_result["test_f1"],
            "auc": best_model_result["test_auc"],
            "cv_f1": best_overall["cv_score"],
            "params": best_overall["params"],
        }

    def _train_best_model_regression(self, endpoint, best_info):
        """Train the best regression model on full training data and evaluate on test set."""
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")
        test_data = pd.read_csv(self.processed_dir / f"{endpoint}_test.csv")

        indices = np.load(self.processed_dir / f"{endpoint}_split_indices.npz")
        train_idx, test_idx = indices["train"], indices["test"]

        X_train = features.iloc[train_idx].values
        X_test = features.iloc[test_idx].values
        y_train = train_data["pchembl_value"].values
        y_test = test_data["pchembl_value"].values

        # Scale features
        scaler = FeatureScaler.load(self.config, endpoint)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reconstruct model with best params
        model_class = best_info["model_class"]
        params = best_info["params"].copy()

        # Add fixed params based on model type
        if model_class == RandomForestRegressor:
            params["random_state"] = self.random_state
            params["n_jobs"] = -1
        elif model_class == xgb.XGBRegressor:
            params["random_state"] = self.random_state
            params["verbosity"] = 0
            params["n_jobs"] = -1
        elif model_class == lgb.LGBMRegressor:
            params["random_state"] = self.random_state
            params["verbose"] = -1
            params["n_jobs"] = -1

        model = model_class(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        test_r2 = r2_score(y_test, y_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        # Save optimized model
        model_path = self.models_dir / endpoint / f"{best_info['name'].lower()}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Best model saved to {model_path}")

        logger.info(f"  Test R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

        return {"test_r2": test_r2, "test_rmse": test_rmse}

    def _train_best_model_classification(self, endpoint, best_info):
        """Train the best classification model on full training data and evaluate on test set."""
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")
        test_data = pd.read_csv(self.processed_dir / f"{endpoint}_test.csv")

        indices = np.load(self.processed_dir / f"{endpoint}_split_indices.npz")
        train_idx, test_idx = indices["train"], indices["test"]

        X_train = features.iloc[train_idx].values
        X_test = features.iloc[test_idx].values
        y_train = train_data["activity_label"].values
        y_test = test_data["activity_label"].values

        # Scale features
        scaler = FeatureScaler.load(self.config, endpoint)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Reconstruct model with best params
        model_class = best_info["model_class"]
        params = best_info["params"].copy()

        # Add fixed params based on model type
        if model_class == RandomForestClassifier:
            params["random_state"] = self.random_state
            params["n_jobs"] = -1
        elif model_class == xgb.XGBClassifier:
            params["random_state"] = self.random_state
            params["verbosity"] = 0
            params["n_jobs"] = -1
            params["use_label_encoder"] = False
            params["eval_metric"] = "logloss"
        elif model_class == lgb.LGBMClassifier:
            params["random_state"] = self.random_state
            params["verbose"] = -1
            params["n_jobs"] = -1

        model = model_class(**params)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, zero_division=0)

        # AUC if probability estimates available
        test_auc = 0.0
        if hasattr(model, "predict_proba") and len(np.unique(y_test)) > 1:
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            test_auc = roc_auc_score(y_test, y_prob)

        # Save optimized model
        model_path = self.models_dir / endpoint / f"{best_info['name'].lower()}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Best model saved to {model_path}")

        logger.info(f"  Test Accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")

        return {"test_accuracy": test_accuracy, "test_f1": test_f1, "test_auc": test_auc}
