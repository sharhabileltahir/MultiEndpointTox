#!/usr/bin/env python3
"""
Train and evaluate Chemprop D-MPNN for hERG regression with:
- 80/20 external split
- 5-fold CV on training set
- fixed seed (42) for reproducibility

Outputs:
- CV fold predictions
- Test set predictions
- CV summary metrics
- External test metrics
- Model checkpoints (per fold + final model)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


SEED = 42


@dataclass
class RegressionMetrics:
    r2: float
    rmse: float
    mae: float


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_reproducibility(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def run_command(cmd: Sequence[str], env: Dict[str, str] | None = None) -> None:
    logger = logging.getLogger("subprocess")
    logger.info("Running command: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    return RegressionMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )


def summarize_metrics(metrics_list: List[RegressionMetrics]) -> Dict[str, float]:
    r2_vals = np.array([m.r2 for m in metrics_list], dtype=float)
    rmse_vals = np.array([m.rmse for m in metrics_list], dtype=float)
    mae_vals = np.array([m.mae for m in metrics_list], dtype=float)
    return {
        "n_folds": int(len(metrics_list)),
        "r2_mean": float(np.mean(r2_vals)),
        "r2_sd": float(np.std(r2_vals, ddof=1)),
        "rmse_mean": float(np.mean(rmse_vals)),
        "rmse_sd": float(np.std(rmse_vals, ddof=1)),
        "mae_mean": float(np.mean(mae_vals)),
        "mae_sd": float(np.std(mae_vals, ddof=1)),
    }


def find_model_checkpoint(model_dir: Path) -> Path:
    candidates = sorted(
        list(model_dir.rglob("*.ckpt")) + list(model_dir.rglob("*.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint found in {model_dir}")
    return candidates[0]


def find_prediction_csv(base_output: Path) -> Path:
    """
    Chemprop may append an index to output stem (e.g., preds_0.csv).
    Resolve the actual produced file robustly.
    """
    parent = base_output.parent
    stem = base_output.stem
    candidates = sorted(parent.glob(f"{stem}*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No prediction CSV produced matching {base_output}")
    return candidates[0]


def infer_prediction_column(df: pd.DataFrame, target_col: str, smiles_col: str) -> str:
    # Prefer columns that include target name but are not the true target column.
    preferred = [
        c
        for c in df.columns
        if (target_col in c.lower()) and (c != target_col) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if preferred:
        return preferred[0]

    numeric_cols = [
        c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {target_col}
    ]
    if numeric_cols:
        return numeric_cols[-1]

    # Chemprop v2 commonly writes predictions directly under the target column name.
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        return target_col

    raise ValueError(
        f"Could not infer prediction column. Columns were: {df.columns.tolist()}"
    )


def build_chemprop_train_cmd(
    chemprop_bin: str,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    output_dir: Path,
    smiles_col: str,
    target_col: str,
    seed: int,
) -> List[str]:
    return [
        chemprop_bin,
        "train",
        "--data-path",
        str(train_path),
        str(val_path),
        str(test_path),
        "--task-type",
        "regression",
        "--smiles-columns",
        smiles_col,
        "--target-columns",
        target_col,
        "--output-dir",
        str(output_dir),
        "--message-hidden-dim",
        "300",
        "--depth",
        "3",
        "--dropout",
        "0.2",
        "--batch-size",
        "64",
        "--epochs",
        "50",
        "--patience",
        "10",
        "--metrics",
        "rmse",
        "mae",
        "r2",
        "--tracking-metric",
        "rmse",
        "--data-seed",
        str(seed),
        "--pytorch-seed",
        str(seed),
    ]


def build_chemprop_predict_cmd(
    chemprop_bin: str,
    test_path: Path,
    model_path: Path,
    output_path: Path,
    smiles_col: str,
) -> List[str]:
    return [
        chemprop_bin,
        "predict",
        "--test-path",
        str(test_path),
        "--model-paths",
        str(model_path),
        "--smiles-columns",
        smiles_col,
        "--output",
        str(output_path),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Chemprop D-MPNN hERG regression with CV + external test."
    )
    parser.add_argument(
        "--data-csv",
        type=str,
        default="data/processed/herg_chemprop_regression.csv",
        help="Input CSV with columns 'smiles' and 'pchembl_value'.",
    )
    parser.add_argument("--smiles-col", type=str, default="smiles")
    parser.add_argument("--target-col", type=str, default="pchembl_value")
    parser.add_argument("--chemprop-bin", type=str, default="chemprop")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison_2d_3d_regression/chemprop_dmpnn",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    set_reproducibility(args.seed)
    logger = logging.getLogger("chemprop_regression")

    output_dir = Path(args.output_dir)
    split_dir = output_dir / "splits"
    cv_dir = output_dir / "cv"
    final_dir = output_dir / "final_model"
    split_dir.mkdir(parents=True, exist_ok=True)
    cv_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_csv)
    for col in [args.smiles_col, args.target_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {args.data_csv}")

    work = df[[args.smiles_col, args.target_col]].copy()
    work = work.dropna(subset=[args.smiles_col, args.target_col]).reset_index(drop=True)
    work[args.target_col] = pd.to_numeric(work[args.target_col], errors="coerce")
    work = work.dropna(subset=[args.target_col]).reset_index(drop=True)
    logger.info("Using %d molecules after cleaning.", len(work))

    train_df, test_df = train_test_split(
        work,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_full_path = split_dir / "train_full.csv"
    test_path = split_dir / "test.csv"
    train_df.to_csv(train_full_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info("Saved train/test splits: train=%d, test=%d", len(train_df), len(test_df))

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(args.seed)

    # 5-fold CV on the training set.
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    cv_fold_metrics: List[RegressionMetrics] = []
    cv_preds_rows: List[pd.DataFrame] = []
    fold_r2_values: List[float] = []

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(train_df), start=1):
        fold_root = cv_dir / f"fold_{fold_idx:02d}"
        fold_split_dir = fold_root / "data"
        fold_model_dir = fold_root / "checkpoints"
        fold_pred_dir = fold_root / "predictions"
        fold_split_dir.mkdir(parents=True, exist_ok=True)
        fold_model_dir.mkdir(parents=True, exist_ok=True)
        fold_pred_dir.mkdir(parents=True, exist_ok=True)

        fold_train = train_df.iloc[tr_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)

        fold_train_path = fold_split_dir / "train.csv"
        fold_val_path = fold_split_dir / "val.csv"
        fold_train.to_csv(fold_train_path, index=False)
        fold_val.to_csv(fold_val_path, index=False)

        train_cmd = build_chemprop_train_cmd(
            chemprop_bin=args.chemprop_bin,
            train_path=fold_train_path,
            val_path=fold_val_path,
            test_path=fold_val_path,  # duplicate for explicit file-based setup
            output_dir=fold_model_dir,
            smiles_col=args.smiles_col,
            target_col=args.target_col,
            seed=args.seed,
        )
        run_command(train_cmd, env=env)

        model_ckpt = find_model_checkpoint(fold_model_dir)
        val_pred_output = fold_pred_dir / "val_predictions.csv"
        predict_cmd = build_chemprop_predict_cmd(
            chemprop_bin=args.chemprop_bin,
            test_path=fold_val_path,
            model_path=model_ckpt,
            output_path=val_pred_output,
            smiles_col=args.smiles_col,
        )
        run_command(predict_cmd, env=env)

        pred_file = find_prediction_csv(val_pred_output)
        pred_df = pd.read_csv(pred_file)
        pred_col = infer_prediction_column(pred_df, args.target_col, args.smiles_col)

        # Always use the source split file for ground truth to avoid schema ambiguity.
        y_true = fold_val[args.target_col].to_numpy(dtype=float)
        y_pred = pred_df[pred_col].to_numpy(dtype=float)

        metrics = compute_metrics(y_true, y_pred)
        cv_fold_metrics.append(metrics)
        fold_r2_values.append(metrics.r2)
        logger.info(
            "Fold %d | R2=%.4f RMSE=%.4f MAE=%.4f",
            fold_idx,
            metrics.r2,
            metrics.rmse,
            metrics.mae,
        )

        cv_rows = pd.DataFrame(
            {
                "fold": fold_idx,
                "split": "cv_val",
                "smiles": pred_df.get(args.smiles_col, fold_val[args.smiles_col]).astype(str),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        cv_preds_rows.append(cv_rows)

    cv_summary = summarize_metrics(cv_fold_metrics)
    cv_summary_df = pd.DataFrame(
        [
            {
                "representation": "GNN",
                "model": "D-MPNN",
                "n_folds": cv_summary["n_folds"],
                "r2_mean": cv_summary["r2_mean"],
                "r2_sd": cv_summary["r2_sd"],
                "r2_mean_sd": f"{cv_summary['r2_mean']:.4f} +/- {cv_summary['r2_sd']:.4f}",
                "rmse_mean": cv_summary["rmse_mean"],
                "rmse_sd": cv_summary["rmse_sd"],
                "rmse_mean_sd": f"{cv_summary['rmse_mean']:.4f} +/- {cv_summary['rmse_sd']:.4f}",
                "mae_mean": cv_summary["mae_mean"],
                "mae_sd": cv_summary["mae_sd"],
                "mae_mean_sd": f"{cv_summary['mae_mean']:.4f} +/- {cv_summary['mae_sd']:.4f}",
            }
        ]
    )

    # Final model for external test evaluation.
    final_train_df, final_val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=args.seed,
        shuffle=True,
    )
    final_train_df = final_train_df.reset_index(drop=True)
    final_val_df = final_val_df.reset_index(drop=True)

    final_train_path = final_dir / "train.csv"
    final_val_path = final_dir / "val.csv"
    final_test_path = final_dir / "test.csv"
    final_train_df.to_csv(final_train_path, index=False)
    final_val_df.to_csv(final_val_path, index=False)
    test_df.to_csv(final_test_path, index=False)

    final_ckpt_dir = final_dir / "checkpoints"
    final_ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_train_cmd = build_chemprop_train_cmd(
        chemprop_bin=args.chemprop_bin,
        train_path=final_train_path,
        val_path=final_val_path,
        test_path=final_test_path,
        output_dir=final_ckpt_dir,
        smiles_col=args.smiles_col,
        target_col=args.target_col,
        seed=args.seed,
    )
    run_command(final_train_cmd, env=env)

    final_model_ckpt = find_model_checkpoint(final_ckpt_dir)
    test_pred_output = final_dir / "test_predictions.csv"
    final_predict_cmd = build_chemprop_predict_cmd(
        chemprop_bin=args.chemprop_bin,
        test_path=final_test_path,
        model_path=final_model_ckpt,
        output_path=test_pred_output,
        smiles_col=args.smiles_col,
    )
    run_command(final_predict_cmd, env=env)

    test_pred_file = find_prediction_csv(test_pred_output)
    test_pred_df = pd.read_csv(test_pred_file)
    test_pred_col = infer_prediction_column(test_pred_df, args.target_col, args.smiles_col)

    # Always use held-out split file for true labels.
    y_test_true = test_df[args.target_col].to_numpy(dtype=float)
    y_test_pred = test_pred_df[test_pred_col].to_numpy(dtype=float)
    test_metrics = compute_metrics(y_test_true, y_test_pred)

    test_summary_df = pd.DataFrame(
        [
            {
                "representation": "GNN",
                "model": "D-MPNN",
                "r2": test_metrics.r2,
                "rmse": test_metrics.rmse,
                "mae": test_metrics.mae,
            }
        ]
    )

    cv_preds_df = pd.concat(cv_preds_rows, axis=0, ignore_index=True)
    out_cv_preds = output_dir / "cv_fold_predictions.csv"
    out_test_preds = output_dir / "test_set_predictions.csv"
    out_cv_summary = output_dir / "cv_performance_summary.csv"
    out_test_summary = output_dir / "external_test_performance.csv"
    out_fold_r2 = output_dir / "fold_r2_values.json"

    cv_preds_df.to_csv(out_cv_preds, index=False)
    test_export = pd.DataFrame(
        {
            "split": "test",
            "smiles": test_pred_df.get(args.smiles_col, test_df[args.smiles_col]).astype(str),
            "y_true": y_test_true,
            "y_pred": y_test_pred,
        }
    )
    test_export.to_csv(out_test_preds, index=False)
    cv_summary_df.to_csv(out_cv_summary, index=False)
    test_summary_df.to_csv(out_test_summary, index=False)
    with open(out_fold_r2, "w", encoding="utf-8") as f:
        json.dump({"D-MPNN": fold_r2_values}, f, indent=2)

    logger.info("Saved CV predictions to %s", out_cv_preds)
    logger.info("Saved test predictions to %s", out_test_preds)
    logger.info("Saved CV metrics summary to %s", out_cv_summary)
    logger.info("Saved test metrics summary to %s", out_test_summary)
    logger.info("Saved fold R2 values to %s", out_fold_r2)
    logger.info(
        "External test metrics | R2=%.4f RMSE=%.4f MAE=%.4f",
        test_metrics.r2,
        test_metrics.rmse,
        test_metrics.mae,
    )


if __name__ == "__main__":
    main()
