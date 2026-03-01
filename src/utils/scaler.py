"""Feature scaling utilities for QSAR modeling."""

import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import StandardScaler, RobustScaler


class FeatureScaler:
    """
    Handle feature scaling with proper train-only fitting.

    This ensures no data leakage: scaler statistics are computed
    only from training data and applied to test/validation sets.
    """

    def __init__(self, config, method="standard"):
        """
        Initialize the feature scaler.

        Args:
            config: Configuration dictionary with paths
            method: Scaling method - "standard" (StandardScaler) or "robust" (RobustScaler)
        """
        self.config = config
        self.method = method
        self.models_dir = Path(config["paths"]["models"])

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'robust'.")

    def fit(self, X_train, endpoint):
        """
        Fit scaler on training data ONLY.

        Args:
            X_train: Training feature matrix (numpy array or DataFrame)
            endpoint: Endpoint name for saving scaler

        Returns:
            self for method chaining
        """
        self.scaler.fit(X_train)

        # Save scaler for later use in predictions
        scaler_path = self.models_dir / endpoint / "feature_scaler.pkl"
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "method": self.method,
            "n_features": X_train.shape[1],
            "n_samples_fitted": X_train.shape[0],
        }, scaler_path)

        logger.info(
            f"Scaler ({self.method}) fitted on {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features. Saved to {scaler_path}"
        )
        return self

    def transform(self, X):
        """
        Transform features using fitted scaler.

        Args:
            X: Feature matrix to transform

        Returns:
            Scaled feature matrix (numpy array)
        """
        return self.scaler.transform(X)

    def fit_transform(self, X_train, endpoint):
        """
        Fit on training data and transform in one call.

        Args:
            X_train: Training feature matrix
            endpoint: Endpoint name for saving scaler

        Returns:
            Scaled training features (numpy array)
        """
        self.fit(X_train, endpoint)
        return self.transform(X_train)

    @classmethod
    def load(cls, config, endpoint):
        """
        Load a previously fitted scaler.

        Args:
            config: Configuration dictionary
            endpoint: Endpoint name

        Returns:
            FeatureScaler instance with loaded scaler
        """
        models_dir = Path(config["paths"]["models"])
        scaler_path = models_dir / endpoint / "feature_scaler.pkl"

        if not scaler_path.exists():
            raise FileNotFoundError(
                f"No fitted scaler found at {scaler_path}. "
                f"Run model training first to fit the scaler."
            )

        scaler_data = joblib.load(scaler_path)

        instance = cls(config, method=scaler_data["method"])
        instance.scaler = scaler_data["scaler"]

        return instance

    def inverse_transform(self, X_scaled):
        """
        Reverse the scaling transformation.

        Args:
            X_scaled: Scaled feature matrix

        Returns:
            Original-scale feature matrix
        """
        return self.scaler.inverse_transform(X_scaled)
