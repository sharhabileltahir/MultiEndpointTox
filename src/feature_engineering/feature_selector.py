"""Feature Selector - Variance filtering, correlation removal (train-only fitting)."""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from loguru import logger
from sklearn.feature_selection import VarianceThreshold


class FeatureSelector:
    """
    Select features based on variance and correlation.

    Critical: All statistics (variance, correlation) are computed ONLY on
    training data to prevent data leakage from test/validation sets.
    """

    def __init__(self, config):
        self.config = config
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.models_dir = Path(config["paths"]["models"])
        self.fs_config = config["features"]["feature_selection"]

        # Store selection state
        self.selected_columns = None
        self.variance_selector = None
        self.dropped_corr_cols = None

    def select(self, descriptors, fingerprints, endpoint):
        """
        Select features using variance and correlation filtering.

        Fits selection criteria on training data only, then applies
        to full dataset for consistency.

        Args:
            descriptors: DataFrame with molecular descriptors
            fingerprints: DataFrame with fingerprints
            endpoint: Toxicity endpoint name

        Returns:
            DataFrame with selected features
        """
        # Combine all features
        features = pd.concat([descriptors, fingerprints], axis=1)
        logger.info(f"Combined features: {features.shape}")

        # Load training indices
        indices_path = self.processed_dir / f"{endpoint}_split_indices.npz"
        if indices_path.exists():
            indices = np.load(indices_path)
            train_idx = indices["train"]
            logger.info(f"Using {len(train_idx)} training samples for feature selection")
        else:
            # Fallback: use first 70% as approximation (not ideal but handles legacy data)
            n_train = int(len(features) * 0.7)
            train_idx = np.arange(n_train)
            logger.warning(
                f"Split indices not found, using first {n_train} samples for feature selection. "
                f"Re-run data curation for proper train-only feature selection."
            )

        # Extract training subset for fitting selection criteria
        train_features = features.iloc[train_idx]

        # Step 1: Variance threshold (fit on training only)
        var_thresh = self.fs_config["variance_threshold"]
        self.variance_selector = VarianceThreshold(threshold=var_thresh)
        self.variance_selector.fit(train_features)
        variance_mask = self.variance_selector.get_support()

        features_after_var = features.loc[:, variance_mask]
        train_after_var = train_features.loc[:, variance_mask]
        logger.info(f"After variance filter ({var_thresh}): {features_after_var.shape[1]} features")

        # Step 2: Correlation filtering (computed on training only)
        corr_thresh = self.fs_config["correlation_threshold"]
        corr_matrix = train_after_var.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        self.dropped_corr_cols = [col for col in upper.columns if any(upper[col] > corr_thresh)]

        features_after_corr = features_after_var.drop(columns=self.dropped_corr_cols)
        train_after_corr = train_after_var.drop(columns=self.dropped_corr_cols)
        logger.info(f"After correlation filter ({corr_thresh}): {features_after_corr.shape[1]} features")

        # Step 3: Cap at max features (based on training variance)
        max_features = self.fs_config.get("max_features", 500)
        if features_after_corr.shape[1] > max_features:
            # Rank by variance computed on training data only
            variances = train_after_corr.var().sort_values(ascending=False)
            top_cols = variances.head(max_features).index.tolist()
            features_after_corr = features_after_corr[top_cols]
            logger.info(f"Capped at {max_features} features")

        # Store selected columns for reproducibility
        self.selected_columns = features_after_corr.columns.tolist()

        # Save selection metadata
        selection_path = self.models_dir / endpoint / "feature_selection.pkl"
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "selected_columns": self.selected_columns,
            "variance_threshold": var_thresh,
            "correlation_threshold": corr_thresh,
            "n_original_features": features.shape[1],
            "n_selected_features": len(self.selected_columns),
            "dropped_variance_cols": list(features.columns[~variance_mask]),
            "dropped_corr_cols": self.dropped_corr_cols,
        }, selection_path)
        logger.info(f"Feature selection metadata saved to {selection_path}")

        # Save selected features
        output_path = self.interim_dir / f"{endpoint}_selected_features.csv"
        features_after_corr.to_csv(output_path, index=False)

        return features_after_corr

    @classmethod
    def load(cls, config, endpoint):
        """
        Load a previously fitted feature selector.

        Args:
            config: Configuration dictionary
            endpoint: Endpoint name

        Returns:
            FeatureSelector instance with loaded selection state
        """
        instance = cls(config)
        models_dir = Path(config["paths"]["models"])
        selection_path = models_dir / endpoint / "feature_selection.pkl"

        if not selection_path.exists():
            raise FileNotFoundError(
                f"Feature selection metadata not found at {selection_path}. "
                f"Run feature engineering first."
            )

        metadata = joblib.load(selection_path)
        instance.selected_columns = metadata["selected_columns"]

        return instance

    def transform(self, descriptors, fingerprints):
        """
        Apply saved feature selection to new data.

        Args:
            descriptors: DataFrame with molecular descriptors
            fingerprints: DataFrame with fingerprints

        Returns:
            DataFrame with selected features only
        """
        if self.selected_columns is None:
            raise ValueError("FeatureSelector not fitted. Call select() first or use load().")

        features = pd.concat([descriptors, fingerprints], axis=1)

        # Check for missing columns
        missing = set(self.selected_columns) - set(features.columns)
        if missing:
            raise ValueError(f"Missing {len(missing)} expected columns: {list(missing)[:5]}...")

        return features[self.selected_columns]
