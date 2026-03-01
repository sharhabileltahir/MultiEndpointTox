"""Model Trainer - Train multiple ML models for QSAR prediction."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, balanced_accuracy_score
)
import xgboost as xgb
import lightgbm as lgb

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from src.utils.scaler import FeatureScaler


class ModelTrainer:
    """
    Train and evaluate multiple ML models for QSAR prediction.

    Supports both regression (pchembl_value) and classification (activity_label).
    """

    REGRESSION_MODELS = {
        "RandomForest": lambda: RandomForestRegressor(
            n_estimators=300, max_depth=20, n_jobs=-1, random_state=42
        ),
        "XGBoost": lambda: xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42, verbosity=0
        ),
        "LightGBM": lambda: lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42, verbose=-1
        ),
        "SVR": lambda: SVR(kernel="rbf", C=10, gamma="scale"),
    }

    CLASSIFICATION_MODELS = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=20, n_jobs=-1, random_state=42
        ),
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric="logloss"
        ),
        "LightGBM": lambda: lgb.LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, random_state=42, verbose=-1
        ),
        "SVC": lambda: SVC(kernel="rbf", C=10, gamma="scale", probability=True),
    }

    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.models_dir = Path(config["paths"]["models"])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = None
        self.task_type = None

    def train_all(self, endpoint):
        """
        Train all baseline models on the endpoint data.

        Automatically detects regression vs classification based on data.
        """
        X_train, X_test, y_train, y_test, self.task_type = self._load_data(endpoint)

        # Scale features - fit ONLY on training data
        self.scaler = FeatureScaler(self.config, method="standard")
        X_train_scaled = self.scaler.fit_transform(X_train, endpoint)
        X_test_scaled = self.scaler.transform(X_test)

        logger.info(f"Training data: {X_train_scaled.shape}, Test data: {X_test_scaled.shape}")
        logger.info(f"Task type: {self.task_type}")

        if self.task_type == "classification":
            return self._train_classification(endpoint, X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            return self._train_regression(endpoint, X_train_scaled, X_test_scaled, y_train, y_test)

    def _train_regression(self, endpoint, X_train, X_test, y_train, y_test):
        """Train regression models."""
        results = {}
        for name, model_fn in self.REGRESSION_MODELS.items():
            logger.info(f"Training {name}...")
            model = model_fn()
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            metrics = {
                "r2": r2_score(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mae": mean_absolute_error(y_test, y_pred),
            }
            results[name] = metrics

            # Save trained model
            model_path = self.models_dir / endpoint / f"{name.lower()}_baseline.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)

            logger.info(
                f"  {name}: R2={metrics['r2']:.4f}, "
                f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}"
            )

        # Save task type
        self._save_task_type(endpoint, "regression")
        return results

    def _train_classification(self, endpoint, X_train, X_test, y_train, y_test):
        """Train classification models with SMOTE for class imbalance."""
        results = {}

        # Log original class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist = dict(zip(unique, counts))
        logger.info(f"Original class distribution: {class_dist}")

        # Apply SMOTE if available and class imbalance exists
        X_train_resampled, y_train_resampled = X_train, y_train
        if SMOTE_AVAILABLE and len(unique) == 2:
            minority_count = min(counts)
            majority_count = max(counts)
            imbalance_ratio = majority_count / minority_count

            if imbalance_ratio > 2:  # Only apply SMOTE if significant imbalance
                logger.info(f"Applying SMOTE (imbalance ratio: {imbalance_ratio:.1f}:1)...")
                try:
                    # k_neighbors must be less than minority class count
                    k_neighbors = min(5, minority_count - 1)
                    if k_neighbors >= 1:
                        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

                        unique_new, counts_new = np.unique(y_train_resampled, return_counts=True)
                        logger.info(f"After SMOTE: {dict(zip(unique_new, counts_new))}")
                    else:
                        logger.warning("Not enough minority samples for SMOTE, skipping...")
                except Exception as e:
                    logger.warning(f"SMOTE failed: {e}, using original data")
        elif not SMOTE_AVAILABLE:
            logger.warning("imbalanced-learn not installed, skipping SMOTE")

        for name, model_fn in self.CLASSIFICATION_MODELS.items():
            logger.info(f"Training {name}...")
            model = model_fn()
            model.fit(X_train_resampled, y_train_resampled)

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }

            # AUC only if we have both classes in test set
            if len(np.unique(y_test)) > 1:
                metrics["auc"] = roc_auc_score(y_test, y_prob)
            else:
                metrics["auc"] = 0.0

            results[name] = metrics

            # Save trained model
            model_path = self.models_dir / endpoint / f"{name.lower()}_baseline.pkl"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)

            logger.info(
                f"  {name}: Acc={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, MCC={metrics['mcc']:.4f}"
            )

        # Save task type
        self._save_task_type(endpoint, "classification")
        return results

    def _save_task_type(self, endpoint, task_type):
        """Save task type metadata."""
        meta_path = self.models_dir / endpoint / "task_type.txt"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            f.write(task_type)

    def get_task_type(self, endpoint):
        """Load task type for endpoint."""
        meta_path = self.models_dir / endpoint / "task_type.txt"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                return f.read().strip()
        return "regression"  # default

    def _load_data(self, endpoint):
        """
        Load and properly align features with targets.

        Returns:
            tuple: X_train, X_test, y_train, y_test, task_type
        """
        # Load feature matrix
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")

        # Load target data
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")
        test_data = pd.read_csv(self.processed_dir / f"{endpoint}_test.csv")

        # Determine task type
        if "activity_label" in train_data.columns:
            task_type = "classification"
            target_col = "activity_label"
        else:
            task_type = "regression"
            target_col = "pchembl_value"

        # Load split indices for proper alignment
        indices_path = self.processed_dir / f"{endpoint}_split_indices.npz"

        if indices_path.exists():
            indices = np.load(indices_path)
            train_idx = indices["train"]
            test_idx = indices["test"]

            X_train = features.iloc[train_idx].values
            X_test = features.iloc[test_idx].values
            y_train = train_data[target_col].values
            y_test = test_data[target_col].values

            logger.info(f"Loaded data with index alignment: train={len(train_idx)}, test={len(test_idx)}")

        elif "original_index" in train_data.columns:
            X_train = features.iloc[train_data["original_index"].values].values
            X_test = features.iloc[test_data["original_index"].values].values
            y_train = train_data[target_col].values
            y_test = test_data[target_col].values

            logger.warning("Using original_index column for alignment")

        else:
            raise ValueError(
                f"Cannot align features with targets for {endpoint}. "
                f"Please re-run data curation."
            )

        return X_train, X_test, y_train, y_test, task_type

    def load_model(self, endpoint, model_name):
        """Load a trained model."""
        model_path = self.models_dir / endpoint / f"{model_name}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        return joblib.load(model_path)

    def predict(self, endpoint, model_name, X_new):
        """Make predictions with a trained model."""
        model = self.load_model(endpoint, model_name)
        scaler = FeatureScaler.load(self.config, endpoint)
        X_scaled = scaler.transform(X_new)
        return model.predict(X_scaled)

    # Keep MODELS as alias for backward compatibility
    MODELS = REGRESSION_MODELS
