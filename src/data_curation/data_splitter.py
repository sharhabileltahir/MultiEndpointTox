"""Data Splitter - Split curated datasets into train/test/validation sets."""

import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from sklearn.model_selection import train_test_split


class DataSplitter:

    def __init__(self, config):
        self.config = config
        self.processed_dir = Path(config["paths"]["processed_data"])
        self.test_size = config["modeling"]["test_size"]
        self.val_size = config["modeling"]["val_size"]
        self.random_state = config["modeling"]["random_state"]

    def split(self, df, endpoint):
        """
        Split data into train/test/val sets while preserving indices for feature alignment.

        The key fix: We track original row indices so features can be properly
        aligned with their corresponding targets during model training.
        """
        # Reset index to ensure consistent integer indices
        df = df.reset_index(drop=True)

        # Create index array to track original positions
        indices = np.arange(len(df))

        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            indices, test_size=self.test_size, random_state=self.random_state
        )

        # Second split: separate validation from training
        relative_val_size = self.val_size / (1 - self.test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=relative_val_size, random_state=self.random_state
        )

        # Create DataFrames using tracked indices
        train = df.iloc[train_idx].copy()
        test = df.iloc[test_idx].copy()
        val = df.iloc[val_idx].copy()

        # Add original_index column for backup alignment method
        train["original_index"] = train_idx
        test["original_index"] = test_idx
        val["original_index"] = val_idx

        # Save split DataFrames
        splits = {"train": train, "test": test, "val": val}
        for split_name, split_df in splits.items():
            path = self.processed_dir / f"{endpoint}_{split_name}.csv"
            split_df.to_csv(path, index=False)

        # Save indices as numpy arrays for efficient loading during training
        indices_path = self.processed_dir / f"{endpoint}_split_indices.npz"
        np.savez(
            indices_path,
            train=train_idx,
            test=test_idx,
            val=val_idx
        )
        logger.info(f"Split indices saved to {indices_path}")

        logger.info(f"Splits saved - Train: {len(train)}, Test: {len(test)}, Val: {len(val)}")
        return splits
