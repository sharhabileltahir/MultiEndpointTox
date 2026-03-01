"""Applicability domain assessment using leverage and distance methods."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from loguru import logger
from scipy.spatial.distance import cdist

from src.utils.scaler import FeatureScaler


class ApplicabilityDomain:
    """
    Assess prediction reliability using applicability domain methods.

    Supports leverage-based and distance-based AD assessment.
    Identifies compounds that fall outside the model's reliable prediction space.
    """

    def __init__(self, config):
        self.config = config
        self.ad_config = config["validation"]["applicability_domain"]
        self.method = self.ad_config["method"]
        self.threshold = self.ad_config["threshold"]

        self.processed_dir = Path(config["paths"]["processed_data"])
        self.interim_dir = Path(config["paths"]["interim_data"])
        self.models_dir = Path(config["paths"]["models"])

        # Store training data statistics
        self.X_train = None
        self.leverage_threshold = None
        self.centroid = None
        self.max_distance = None

    def fit(self, endpoint):
        """
        Fit applicability domain on training data.

        Args:
            endpoint: Toxicity endpoint name

        Returns:
            self for method chaining
        """
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        train_data = pd.read_csv(self.processed_dir / f"{endpoint}_train.csv")

        indices = np.load(self.processed_dir / f"{endpoint}_split_indices.npz")
        train_idx = indices["train"]

        X_train = features.iloc[train_idx].values

        # Scale features
        scaler = FeatureScaler.load(self.config, endpoint)
        self.X_train = scaler.transform(X_train)

        n, p = self.X_train.shape

        if self.method == "leverage":
            # Warning threshold for leverage: h* = threshold * (p+1) / n
            self.leverage_threshold = self.threshold * (p + 1) / n
            logger.info(
                f"Leverage AD fitted: threshold h* = {self.leverage_threshold:.4f} "
                f"(n={n}, p={p})"
            )
        else:
            # Distance-based AD: compute centroid and max distance
            self.method = "distance"
            self.centroid = np.mean(self.X_train, axis=0)
            distances = cdist(
                self.X_train, self.centroid.reshape(1, -1), metric="euclidean"
            ).flatten()
            self.max_distance = np.percentile(distances, 95)
            logger.info(f"Distance AD fitted: max distance = {self.max_distance:.4f}")

        # Save AD model
        ad_path = self.models_dir / endpoint / "applicability_domain.pkl"
        ad_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "method": self.method,
            "X_train": self.X_train,
            "leverage_threshold": self.leverage_threshold,
            "centroid": self.centroid,
            "max_distance": self.max_distance,
            "n_train": n,
            "n_features": p,
        }, ad_path)

        return self

    def assess(self, endpoint):
        """
        Assess applicability domain for test compounds.

        Args:
            endpoint: Toxicity endpoint name

        Returns:
            dict: AD assessment results including coverage and reliability
        """
        logger.info(f"Assessing applicability domain using {self.method} method...")

        # Load test data first to check dimensions
        features = pd.read_csv(self.interim_dir / f"{endpoint}_selected_features.csv")
        test_data = pd.read_csv(self.processed_dir / f"{endpoint}_test.csv")

        indices = np.load(self.processed_dir / f"{endpoint}_split_indices.npz")
        test_idx = indices["test"]

        X_test = features.iloc[test_idx].values
        n_features = X_test.shape[1]

        scaler = FeatureScaler.load(self.config, endpoint)
        X_test_scaled = scaler.transform(X_test)

        # Load AD model if not fitted, but re-fit if feature dimensions don't match
        ad_path = self.models_dir / endpoint / "applicability_domain.pkl"
        needs_refit = True

        if ad_path.exists():
            ad_data = joblib.load(ad_path)
            # Check if the saved AD has matching dimensions
            if ad_data["X_train"].shape[1] == n_features:
                self.method = ad_data["method"]
                self.X_train = ad_data["X_train"]
                self.leverage_threshold = ad_data["leverage_threshold"]
                self.centroid = ad_data["centroid"]
                self.max_distance = ad_data["max_distance"]
                needs_refit = False
            else:
                logger.warning(
                    f"AD feature mismatch: saved={ad_data['X_train'].shape[1]}, "
                    f"current={n_features}. Re-fitting AD..."
                )

        if needs_refit:
            self.fit(endpoint)

        # Calculate AD metrics for test set
        if self.method == "leverage":
            results = self._assess_leverage(X_test_scaled)
        else:
            results = self._assess_distance(X_test_scaled)

        # Add metadata
        results["training_samples"] = len(self.X_train)
        results["test_samples"] = len(X_test_scaled)
        results["n_features"] = X_test_scaled.shape[1]

        # Save results
        results_path = self.models_dir / endpoint / "ad_results.pkl"
        joblib.dump(results, results_path)

        logger.info(f"AD Coverage: {results['coverage']:.2%} of test compounds within AD")
        logger.info(f"Outliers: {len(results['outliers'])} compounds outside AD")

        return results

    def _assess_leverage(self, X_test):
        """Assess test compounds using leverage method."""
        n_train, p = self.X_train.shape

        # Calculate leverage for test compounds
        # h_i = x_i' (X'X)^(-1) x_i
        try:
            XtX_inv = np.linalg.pinv(self.X_train.T @ self.X_train)
            leverages = np.array([
                float(x @ XtX_inv @ x.T) for x in X_test
            ])
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in leverage calculation, using distance method")
            return self._assess_distance(X_test)

        # Determine which compounds are within AD
        within_ad = leverages <= self.leverage_threshold

        return {
            "method": "leverage",
            "leverages": leverages.tolist(),
            "threshold": float(self.leverage_threshold),
            "within_ad": within_ad.tolist(),
            "coverage": float(np.mean(within_ad)),
            "mean_leverage": float(np.mean(leverages)),
            "max_leverage": float(np.max(leverages)),
            "outliers": np.where(~within_ad)[0].tolist(),
        }

    def _assess_distance(self, X_test):
        """Assess test compounds using distance method."""
        if self.centroid is None:
            self.centroid = np.mean(self.X_train, axis=0)
            distances_train = cdist(
                self.X_train, self.centroid.reshape(1, -1), metric="euclidean"
            ).flatten()
            self.max_distance = np.percentile(distances_train, 95)

        distances = cdist(
            X_test, self.centroid.reshape(1, -1), metric="euclidean"
        ).flatten()

        within_ad = distances <= self.max_distance

        return {
            "method": "distance",
            "distances": distances.tolist(),
            "threshold": float(self.max_distance),
            "within_ad": within_ad.tolist(),
            "coverage": float(np.mean(within_ad)),
            "mean_distance": float(np.mean(distances)),
            "max_distance_test": float(np.max(distances)),
            "outliers": np.where(~within_ad)[0].tolist(),
        }

    def predict_reliability(self, X_new, endpoint):
        """
        Predict reliability scores for new compounds.

        Args:
            X_new: New feature matrix (already scaled)
            endpoint: Endpoint name

        Returns:
            array: Reliability scores (0-1, higher is more reliable)
        """
        # Load AD model
        ad_path = self.models_dir / endpoint / "applicability_domain.pkl"
        ad_data = joblib.load(ad_path)

        if ad_data["method"] == "leverage":
            XtX_inv = np.linalg.pinv(ad_data["X_train"].T @ ad_data["X_train"])
            leverages = np.array([float(x @ XtX_inv @ x.T) for x in X_new])

            # Convert leverage to reliability (inverse relationship)
            threshold = ad_data["leverage_threshold"]
            reliability = 1 - np.clip(leverages / (2 * threshold), 0, 1)
        else:
            distances = cdist(
                X_new, ad_data["centroid"].reshape(1, -1), metric="euclidean"
            ).flatten()

            # Convert distance to reliability
            threshold = ad_data["max_distance"]
            reliability = 1 - np.clip(distances / (2 * threshold), 0, 1)

        return reliability
