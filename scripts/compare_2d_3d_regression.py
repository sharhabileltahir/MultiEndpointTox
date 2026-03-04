#!/usr/bin/env python3
"""
Compare 2D-only, 3D-only, and Hybrid (2D+3D) models for hERG regression.

Design principles for publication-grade benchmarking:
- Identical train/test split indices across all feature representations
- Repeated CV performed on training data only
- Fixed CV splits reused across representations
- Leakage-safe preprocessing (scaling only inside SVR pipeline)
- Fresh estimator clone per fold and final test fit
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError(
        "xgboost is required for this script. Install with `pip install xgboost`."
    ) from exc


SEED = 42


@dataclass
class FoldMetrics:
    r2: float
    rmse: float
    mae: float
    ccc: float


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def concordance_correlation_coefficient(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """
    Lin's Concordance Correlation Coefficient (CCC).

    CCC = 2 * cov(x, y) / (var(x) + var(y) + (mean_x - mean_y)^2)
    """
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)

    if x.size < 2:
        return np.nan

    mean_x = np.mean(x)
    mean_y = np.mean(y)
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)

    if var_x == 0.0 and var_y == 0.0:
        return 1.0 if np.allclose(mean_x, mean_y) else 0.0

    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    denominator = var_x + var_y + (mean_x - mean_y) ** 2

    if denominator == 0.0:
        return np.nan

    return float((2.0 * cov_xy) / denominator)


def load_arrays(
    x2d_path: Path, x3d_path: Path, y_path: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger = logging.getLogger("load_arrays")

    X_2D = np.load(x2d_path)
    X_3D = np.load(x3d_path)
    y = np.load(y_path)

    if y.ndim > 1:
        y = np.ravel(y)

    if X_2D.ndim != 2 or X_3D.ndim != 2:
        raise ValueError("X_2D and X_3D must be 2D arrays.")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array.")
    if not (X_2D.shape[0] == X_3D.shape[0] == y.shape[0]):
        raise ValueError(
            "Mismatched sample counts: "
            f"X_2D={X_2D.shape[0]}, X_3D={X_3D.shape[0]}, y={y.shape[0]}"
        )

    y = y.astype(float)
    if np.isnan(y).any():
        raise ValueError("y contains NaN values.")

    logger.info(
        "Loaded arrays | X_2D=%s, X_3D=%s, y=%s, y_range=(%.4f, %.4f)",
        X_2D.shape,
        X_3D.shape,
        y.shape,
        float(np.min(y)),
        float(np.max(y)),
    )
    return X_2D, X_3D, y


def get_feature_sets(X_2D: np.ndarray, X_3D: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "2D": X_2D,
        "3D": X_3D,
        "Hybrid": np.hstack([X_2D, X_3D]),
    }


def build_models(seed: int = SEED) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "RandomForest": Pipeline(
            steps=[
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=500,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                (
                    "reg",
                    XGBRegressor(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        ),
        "SVR": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("reg", SVR(kernel="rbf", C=10.0, epsilon=0.1)),
            ]
        ),
    }
    return models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> FoldMetrics:
    return FoldMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        ccc=float(concordance_correlation_coefficient(y_true, y_pred)),
    )


def evaluate_representation_with_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    models: Dict[str, Pipeline],
) -> Tuple[Dict[str, List[FoldMetrics]], Dict[str, List[float]]]:
    """
    Evaluate each model on fixed CV splits.

    Returns:
        - fold_metrics: model -> list[FoldMetrics]
        - fold_r2: model -> list[r2] (for paired tests)
    """
    fold_metrics: Dict[str, List[FoldMetrics]] = {name: [] for name in models}
    fold_r2: Dict[str, List[float]] = {name: [] for name in models}

    for model_name, model in models.items():
        for tr_idx, val_idx in cv_splits:
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            y_pred = fold_model.predict(X_val)

            metrics = compute_metrics(y_val, y_pred)
            fold_metrics[model_name].append(metrics)
            fold_r2[model_name].append(metrics.r2)

    return fold_metrics, fold_r2


def evaluate_on_test(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Pipeline],
) -> Dict[str, FoldMetrics]:
    test_results: Dict[str, FoldMetrics] = {}
    for model_name, model in models.items():
        final_model = clone(model)
        final_model.fit(X_train, y_train)
        y_pred = final_model.predict(X_test)
        test_results[model_name] = compute_metrics(y_test, y_pred)
    return test_results


def summarize_cv_metrics(
    cv_results: Dict[str, Dict[str, List[FoldMetrics]]]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    metric_names = ["r2", "rmse", "mae", "ccc"]

    for representation, model_metrics in cv_results.items():
        for model_name, folds in model_metrics.items():
            row: Dict[str, object] = {
                "representation": representation,
                "model": model_name,
                "n_folds": len(folds),
            }
            for metric in metric_names:
                values = np.array([getattr(f, metric) for f in folds], dtype=float)
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd"] = float(np.std(values, ddof=1))
                row[f"{metric}_mean_sd"] = (
                    f"{np.mean(values):.4f} +/- {np.std(values, ddof=1):.4f}"
                )
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["model", "representation"]).reset_index(
        drop=True
    )


def summarize_test_metrics(
    test_results: Dict[str, Dict[str, FoldMetrics]]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for representation, model_results in test_results.items():
        for model_name, metrics in model_results.items():
            rows.append(
                {
                    "representation": representation,
                    "model": model_name,
                    "r2": metrics.r2,
                    "rmse": metrics.rmse,
                    "mae": metrics.mae,
                    "ccc": metrics.ccc,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "representation"]).reset_index(
        drop=True
    )


def paired_r2_tests(
    fold_r2: Dict[str, Dict[str, List[float]]]
) -> pd.DataFrame:
    """
    Perform paired t-tests on fold-wise R²:
    - 2D vs 3D
    - 2D vs Hybrid
    for each model.
    """
    rows: List[Dict[str, object]] = []
    comparisons = [("2D", "3D"), ("2D", "Hybrid")]
    model_names = sorted(fold_r2["2D"].keys())

    for model_name in model_names:
        for left, right in comparisons:
            r2_left = np.array(fold_r2[left][model_name], dtype=float)
            r2_right = np.array(fold_r2[right][model_name], dtype=float)

            if len(r2_left) != len(r2_right):
                raise ValueError(
                    f"Fold count mismatch for {model_name}: {left}={len(r2_left)}, {right}={len(r2_right)}"
                )

            test = ttest_rel(r2_left, r2_right)
            rows.append(
                {
                    "model": model_name,
                    "comparison": f"{left} vs {right}",
                    "n_folds": len(r2_left),
                    "mean_r2_left": float(np.mean(r2_left)),
                    "mean_r2_right": float(np.mean(r2_right)),
                    "mean_difference_left_minus_right": float(np.mean(r2_left - r2_right)),
                    "t_statistic": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "significant_p_lt_0_05": bool(test.pvalue < 0.05),
                }
            )

    return pd.DataFrame(rows).sort_values(["model", "comparison"]).reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    cv_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    stats_summary: pd.DataFrame,
    fold_r2: Dict[str, Dict[str, List[float]]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_summary.to_csv(output_dir / "cv_performance_summary.csv", index=False)
    test_summary.to_csv(output_dir / "external_test_performance.csv", index=False)
    stats_summary.to_csv(output_dir / "paired_ttest_r2.csv", index=False)

    with open(output_dir / "fold_r2_values.json", "w", encoding="utf-8") as f:
        json.dump(fold_r2, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare 2D, 3D, and Hybrid models for hERG regression."
    )
    parser.add_argument("--x2d", type=str, required=True, help="Path to X_2D.npy")
    parser.add_argument("--x3d", type=str, required=True, help="Path to X_3D.npy")
    parser.add_argument("--y", type=str, required=True, help="Path to y.npy (continuous)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison_2d_3d_regression",
        help="Directory for output tables.",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test fraction.")
    parser.add_argument("--n-splits", type=int, default=5, help="CV folds per repeat.")
    parser.add_argument("--n-repeats", type=int, default=3, help="Number of repeats.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_reproducibility(args.seed)
    logger = logging.getLogger("compare_2d_3d_regression")

    X_2D, X_3D, y = load_arrays(Path(args.x2d), Path(args.x3d), Path(args.y))
    feature_sets = get_feature_sets(X_2D, X_3D)

    # Identical split indices reused across all feature representations.
    all_idx = np.arange(y.shape[0])
    train_idx, test_idx = train_test_split(
        all_idx,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    y_train = y[train_idx]
    y_test = y[test_idx]
    logger.info(
        "Train/test split complete | train=%d, test=%d",
        len(train_idx),
        len(test_idx),
    )

    # Fixed repeated CV splits for fair paired comparisons across representations.
    cv = RepeatedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.seed,
    )
    cv_splits = list(cv.split(np.zeros_like(y_train), y_train))
    logger.info(
        "Using repeated CV: %d folds total (%d x %d)",
        len(cv_splits),
        args.n_splits,
        args.n_repeats,
    )

    models = build_models(args.seed)
    cv_results: Dict[str, Dict[str, List[FoldMetrics]]] = {}
    fold_r2: Dict[str, Dict[str, List[float]]] = {}
    test_results: Dict[str, Dict[str, FoldMetrics]] = {}

    for representation, X in feature_sets.items():
        X_train = X[train_idx]
        X_test = X[test_idx]
        logger.info(
            "Evaluating representation '%s' | train_shape=%s test_shape=%s",
            representation,
            X_train.shape,
            X_test.shape,
        )

        rep_cv_results, rep_fold_r2 = evaluate_representation_with_cv(
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            models=models,
        )
        rep_test_results = evaluate_on_test(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
        )

        cv_results[representation] = rep_cv_results
        fold_r2[representation] = rep_fold_r2
        test_results[representation] = rep_test_results

    cv_summary = summarize_cv_metrics(cv_results)
    test_summary = summarize_test_metrics(test_results)
    stats_summary = paired_r2_tests(fold_r2)

    write_outputs(
        output_dir=Path(args.output_dir),
        cv_summary=cv_summary,
        test_summary=test_summary,
        stats_summary=stats_summary,
        fold_r2=fold_r2,
    )

    print("\n=== Mean +/- SD Performance (CV on Training Set) ===")
    print(
        cv_summary[
            [
                "representation",
                "model",
                "r2_mean_sd",
                "rmse_mean_sd",
                "mae_mean_sd",
                "ccc_mean_sd",
            ]
        ].to_string(index=False)
    )

    print("\n=== Statistical Significance (Paired t-test on Fold-wise R2) ===")
    print(
        stats_summary[
            [
                "model",
                "comparison",
                "n_folds",
                "mean_r2_left",
                "mean_r2_right",
                "mean_difference_left_minus_right",
                "t_statistic",
                "p_value",
                "significant_p_lt_0_05",
            ]
        ].to_string(index=False)
    )

    print("\n=== External Test Performance (Holdout 20%) ===")
    print(test_summary.to_string(index=False))

    print(f"\nSaved outputs to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
