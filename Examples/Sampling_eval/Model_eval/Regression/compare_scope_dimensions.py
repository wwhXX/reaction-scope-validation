# -*- coding: utf-8 -*-
"""Quick experiment: compare 2D scope features with higher-dimensional features.

This script keeps the model and data split fixed, and changes only the feature
dimension used for regression. It is intentionally lightweight and depends only
on numpy/pandas so it can run before the full ScopeMap ML environment is ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DIMS = ["2", "4", "8", "16", "32", "64", "128", "full"]
DEFAULT_ALPHAS = np.logspace(-4, 4, 17)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare model performance across 2D and higher-dimensional scope spaces."
    )
    parser.add_argument(
        "--data",
        default="1700_final_norepeat.csv",
        help="Input CSV containing smiles, conv, and numeric descriptor columns.",
    )
    parser.add_argument(
        "--target",
        default="conv",
        help="Target column name.",
    )
    parser.add_argument(
        "--non-feature-cols",
        nargs="+",
        default=["smiles", "conv"],
        help="Columns excluded from descriptor features.",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        default=DEFAULT_DIMS,
        help="Dimensions to compare. Use integers plus optional 'full'.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of deterministic train/test folds.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction held out as test in each fold.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.2,
        help="Fraction of the training split used for ridge alpha selection.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.",
    )
    parser.add_argument(
        "--out-prefix",
        default="scope_dimension_comparison",
        help="Prefix for detail and summary CSV outputs.",
    )
    return parser.parse_args()


def load_data(path: Path, target: str, non_feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    data = pd.read_csv(path)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {path}")

    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    numeric_features = data[feature_cols].select_dtypes(include=[np.number])
    if numeric_features.empty:
        raise ValueError("No numeric feature columns were found.")

    X = numeric_features.to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    return data, X, y, list(numeric_features.columns)


def shuffled_split(n_rows: int, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_rows)
    n_test = max(1, int(round(n_rows * test_size)))
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return train_idx, test_idx


def train_val_split(train_idx: np.ndarray, val_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_idx)
    n_val = max(1, int(round(len(shuffled) * val_size)))
    return shuffled[n_val:], shuffled[:n_val]


def standardize_fit_transform(X_train: np.ndarray, X_other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale[scale == 0] = 1.0
    return (X_train - mean) / scale, (X_other - mean) / scale, mean, scale


def standardize_transform(X: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - mean) / scale


def pca_fit_transform(
    X_train: np.ndarray, X_other: np.ndarray, n_components: int
) -> tuple[np.ndarray, np.ndarray, float]:
    max_components = min(X_train.shape[0], X_train.shape[1])
    n_components = min(n_components, max_components)
    _, singular_values, vt = np.linalg.svd(X_train, full_matrices=False)
    components = vt[:n_components]
    explained = singular_values**2
    explained_ratio = float(explained[:n_components].sum() / explained.sum()) if explained.sum() > 0 else 0.0
    return X_train @ components.T, X_other @ components.T, explained_ratio


def ridge_fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    alpha: float,
) -> np.ndarray:
    x_mean = X_train.mean(axis=0)
    y_mean = y_train.mean()
    X_centered = X_train - x_mean
    y_centered = y_train - y_mean

    gram = X_centered.T @ X_centered
    penalty = alpha * np.eye(gram.shape[0])
    coef = np.linalg.solve(gram + penalty, X_centered.T @ y_centered)
    return (X_pred - x_mean) @ coef + y_mean


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(residual**2) / total) if total > 0 else np.nan
    return {"rmse": rmse, "mae": mae, "r2": r2}


def choose_alpha(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> tuple[float, float]:
    best_alpha = float(DEFAULT_ALPHAS[0])
    best_rmse = np.inf
    for alpha in DEFAULT_ALPHAS:
        pred = ridge_fit_predict(X_train, y_train, X_val, float(alpha))
        rmse = metrics(y_val, pred)["rmse"]
        if rmse < best_rmse:
            best_alpha = float(alpha)
            best_rmse = rmse
    return best_alpha, float(best_rmse)


def evaluate_dimension(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    inner_train_idx: np.ndarray,
    val_idx: np.ndarray,
    dim_label: str,
) -> dict[str, float | str | int]:
    X_inner_raw = X[inner_train_idx]
    X_val_raw = X[val_idx]
    X_train_raw = X[train_idx]
    X_test_raw = X[test_idx]

    X_inner_std, X_val_std, inner_mean, inner_scale = standardize_fit_transform(X_inner_raw, X_val_raw)
    X_train_std = standardize_transform(X_train_raw, inner_mean, inner_scale)
    X_test_std = standardize_transform(X_test_raw, inner_mean, inner_scale)

    if dim_label == "full":
        X_inner_model = X_inner_std
        X_val_model = X_val_std
        explained_ratio = 1.0
        effective_dim = X_inner_model.shape[1]
    else:
        requested_dim = int(dim_label)
        X_inner_model, X_val_model, explained_ratio = pca_fit_transform(X_inner_std, X_val_std, requested_dim)
        X_train_model, X_test_model, _ = pca_fit_transform(X_train_std, X_test_std, requested_dim)
        effective_dim = X_inner_model.shape[1]

    best_alpha, val_rmse = choose_alpha(X_inner_model, y[inner_train_idx], X_val_model, y[val_idx])

    if dim_label == "full":
        final_mean = X_train_raw.mean(axis=0)
        final_scale = X_train_raw.std(axis=0)
        final_scale[final_scale == 0] = 1.0
        X_train_model = standardize_transform(X_train_raw, final_mean, final_scale)
        X_test_model = standardize_transform(X_test_raw, final_mean, final_scale)
    else:
        X_train_std_final, X_test_std_final, _, _ = standardize_fit_transform(X_train_raw, X_test_raw)
        X_train_model, X_test_model, explained_ratio = pca_fit_transform(
            X_train_std_final, X_test_std_final, int(dim_label)
        )

    test_pred = ridge_fit_predict(X_train_model, y[train_idx], X_test_model, best_alpha)
    result = metrics(y[test_idx], test_pred)
    result.update(
        {
            "dimension": dim_label,
            "effective_dim": int(effective_dim),
            "explained_variance_ratio": float(explained_ratio),
            "best_alpha": best_alpha,
            "val_rmse": val_rmse,
        }
    )
    return result


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    grouped = results.groupby("dimension", sort=False)
    summary = grouped.agg(
        effective_dim=("effective_dim", "max"),
        explained_variance_ratio_mean=("explained_variance_ratio", "mean"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        r2_mean=("r2", "mean"),
        best_alpha_median=("best_alpha", "median"),
    ).reset_index()
    full_rmse = summary.loc[summary["dimension"] == "full", "rmse_mean"]
    if not full_rmse.empty:
        summary["rmse_delta_vs_full"] = summary["rmse_mean"] - float(full_rmse.iloc[0])
    return summary


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    _, X, y, feature_cols = load_data(data_path, args.target, args.non_feature_cols)

    dims = [dim.lower() for dim in args.dims]
    valid_dims = []
    for dim in dims:
        if dim == "full":
            valid_dims.append(dim)
            continue
        parsed = int(dim)
        if parsed < 1:
            raise ValueError(f"Dimension must be positive: {dim}")
        valid_dims.append(str(parsed))

    print("=== Scope dimension comparison ===")
    print(f"Data: {data_path}")
    print(f"Rows: {X.shape[0]}, numeric features: {len(feature_cols)}")
    print(f"Dimensions: {', '.join(valid_dims)}")
    print("Model: closed-form Ridge regression with validation-selected alpha")

    records = []
    for fold in range(args.folds):
        train_idx, test_idx = shuffled_split(len(y), args.test_size, args.seed + fold)
        inner_train_idx, val_idx = train_val_split(train_idx, args.val_size, args.seed + 1000 + fold)
        print(f"\nFold {fold + 1}/{args.folds}: train={len(train_idx)}, test={len(test_idx)}")

        for dim in valid_dims:
            row = evaluate_dimension(X, y, train_idx, test_idx, inner_train_idx, val_idx, dim)
            row["fold"] = fold + 1
            records.append(row)
            print(
                f"  dim={dim:>4} RMSE={row['rmse']:.4f} MAE={row['mae']:.4f} "
                f"R2={row['r2']:.4f} explained={row['explained_variance_ratio']:.3f}"
            )

    detail = pd.DataFrame(records)
    summary = summarize(detail)

    detail_path = Path(f"{args.out_prefix}_detail.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Summary ===")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {detail_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
