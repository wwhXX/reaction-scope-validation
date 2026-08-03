# -*- coding: utf-8 -*-
"""Repeated split benchmark with bootstrap confidence intervals.

This script complements ``compare_scope_dimensions_models.py``. It uses many
random stratified train/test splits and summarizes each metric with a bootstrap
confidence interval across repeated split scores.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DIMS = ["2", "64", "128", "full"]
REGRESSION_METRICS = ["rmse", "mae", "r2"]
CLASSIFICATION_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "negative_precision_npv",
    "negative_recall_specificity",
    "f1",
    "mcc",
    "roc_auc",
    "average_precision",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated dimension benchmarks and summarize bootstrap confidence intervals."
    )
    parser.add_argument("--data", default="1700_final_norepeat.csv")
    parser.add_argument("--target", default="conv")
    parser.add_argument("--non-feature-cols", nargs="+", default=["smiles", "conv"])
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=70.0)
    parser.add_argument("--model-set", choices=["minimal", "fast", "all"], default="fast")
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--out-prefix", default="scope_dimension_repeated_ci")
    return parser.parse_args()


def load_xy(path: Path, target: str, non_feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = pd.read_csv(path)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {path}")

    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    features = data[feature_cols].select_dtypes(include=[np.number])
    if features.empty:
        raise ValueError("No numeric feature columns were found.")

    return features.to_numpy(dtype=float), data[target].to_numpy(dtype=float), list(features.columns)


def transform_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    dim: str,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    if dim == "full":
        return X_train_std, X_test_std, 1.0, X_train_std.shape[1]

    n_components = min(int(dim), X_train_std.shape[0], X_train_std.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    return X_train_pca, X_test_pca, float(np.sum(pca.explained_variance_ratio_)), n_components


def regression_models(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "DummyMeanRegressor": DummyRegressor(strategy="mean"),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if args.model_set in {"fast", "all"}:
        models["HistGradientBoostingRegressor"] = HistGradientBoostingRegressor(
            max_iter=max(100, args.n_estimators // 2),
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=seed,
        )
    
    if args.model_set == "all":
        models["ExtraTreesRegressor"] = ExtraTreesRegressor(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )

    if args.include_xgboost and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        models["XGBRegressor"] = XGBRegressor(
            n_estimators=args.n_estimators,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
    return models


def classification_models(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "DummyMostFrequentClassifier": DummyClassifier(strategy="most_frequent"),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
    }
    if args.model_set in {"fast", "all"}:
        models["HistGradientBoostingClassifier"] = HistGradientBoostingClassifier(
            max_iter=max(100, args.n_estimators // 2),
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=seed,
        )

    if args.model_set == "all":
        models["ExtraTreesClassifier"] = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )

    if args.include_xgboost and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        models["XGBClassifier"] = XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    return models


def evaluate_regression(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    pred = fitted.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
        "r2": float(r2_score(y_test, pred)),
    }


def evaluate_classification(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    fitted = clone(model)
    fitted.fit(X_train, y_train)
    pred = fitted.predict(X_test)
    result = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "negative_precision_npv": float(precision_score(y_test, pred, pos_label=0, zero_division=0)),
        "negative_recall_specificity": float(recall_score(y_test, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_test, pred)),
    }
    if hasattr(fitted, "predict_proba") and len(np.unique(y_test)) == 2:
        prob = fitted.predict_proba(X_test)[:, 1]
        result["roc_auc"] = float(roc_auc_score(y_test, prob))
        result["average_precision"] = float(average_precision_score(y_test, prob))
    return result


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int, ci: float) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    draws = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def summarize_ci(detail: pd.DataFrame, task: str, args: argparse.Namespace) -> pd.DataFrame:
    metrics = REGRESSION_METRICS if task == "regression" else CLASSIFICATION_METRICS
    rng = np.random.default_rng(args.seed + 100000)
    rows: list[dict[str, Any]] = []
    for (task_name, model, dimension), group in detail.groupby(["task", "model", "dimension"], sort=False):
        row: dict[str, Any] = {
            "task": task_name,
            "model": model,
            "dimension": dimension,
            "n_repeats": int(group["repeat"].nunique()),
            "effective_dim": int(group["effective_dim"].max()),
            "explained_variance_ratio_mean": float(group["explained_variance_ratio"].mean()),
        }
        for metric in metrics:
            if metric not in group:
                continue
            values = group[metric].to_numpy(dtype=float)
            clean = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(np.mean(clean)) if len(clean) else np.nan
            row[f"{metric}_std"] = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
            row[f"{metric}_sem"] = float(row[f"{metric}_std"] / np.sqrt(len(clean))) if len(clean) else np.nan
            low, high = bootstrap_mean_ci(values, rng, args.bootstrap_samples, args.ci)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dims = [dim.lower() for dim in args.dims]
    for dim in dims:
        if dim != "full" and int(dim) < 1:
            raise ValueError(f"Dimension must be positive: {dim}")

    X, y_reg, feature_cols = load_xy(Path(args.data), args.target, args.non_feature_cols)
    y_cls = (y_reg >= args.success_threshold).astype(int)

    print("=== Repeated dimension benchmark with bootstrap CI ===")
    print(f"Data: {args.data}")
    print(f"Rows: {X.shape[0]}, numeric features: {len(feature_cols)}")
    print(f"Classification threshold: {args.target} >= {args.success_threshold}")
    print(f"Positive class ratio: {y_cls.mean():.3f}")
    print(f"Dimensions: {', '.join(dims)}")
    print(f"Repeats: {args.repeats}, test size: {args.test_size}, model set: {args.model_set}")

    reg_records: list[dict[str, Any]] = []
    cls_records: list[dict[str, Any]] = []
    indices = np.arange(len(y_reg))

    for repeat_idx in range(1, args.repeats + 1):
        split_seed = args.seed + repeat_idx - 1
        train_idx, test_idx = train_test_split(
            indices,
            test_size=args.test_size,
            random_state=split_seed,
            stratify=y_cls,
        )
        reg_models = regression_models(args, split_seed)
        cls_models = classification_models(args, split_seed)
        print(f"\nRepeat {repeat_idx}/{args.repeats}: train={len(train_idx)}, test={len(test_idx)}")

        for dim in dims:
            X_train, X_test, explained, effective_dim = transform_features(X[train_idx], X[test_idx], dim)
            print(f"  dim={dim:>4}, effective={effective_dim}, explained={explained:.3f}")

            for model_name, model in reg_models.items():
                result = evaluate_regression(model, X_train, y_reg[train_idx], X_test, y_reg[test_idx])
                result.update(
                    {
                        "task": "regression",
                        "model": model_name,
                        "dimension": dim,
                        "repeat": repeat_idx,
                        "effective_dim": effective_dim,
                        "explained_variance_ratio": explained,
                    }
                )
                reg_records.append(result)

            for model_name, model in cls_models.items():
                result = evaluate_classification(model, X_train, y_cls[train_idx], X_test, y_cls[test_idx])
                result.update(
                    {
                        "task": "classification",
                        "model": model_name,
                        "dimension": dim,
                        "repeat": repeat_idx,
                        "effective_dim": effective_dim,
                        "explained_variance_ratio": explained,
                    }
                )
                cls_records.append(result)

    reg_detail = pd.DataFrame(reg_records)
    cls_detail = pd.DataFrame(cls_records)
    reg_summary = summarize_ci(reg_detail, "regression", args)
    cls_summary = summarize_ci(cls_detail, "classification", args)

    reg_detail_path = Path(f"{args.out_prefix}_regression_detail.csv")
    reg_summary_path = Path(f"{args.out_prefix}_regression_summary_ci.csv")
    cls_detail_path = Path(f"{args.out_prefix}_classification_detail.csv")
    cls_summary_path = Path(f"{args.out_prefix}_classification_summary_ci.csv")

    reg_detail.to_csv(reg_detail_path, index=False)
    reg_summary.to_csv(reg_summary_path, index=False)
    cls_detail.to_csv(cls_detail_path, index=False)
    cls_summary.to_csv(cls_summary_path, index=False)

    print("\n=== Regression CI summary ===")
    cols = ["task", "model", "dimension", "r2_mean", "r2_ci_low", "r2_ci_high", "rmse_mean", "rmse_ci_low", "rmse_ci_high"]
    print(reg_summary[[col for col in cols if col in reg_summary]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== Classification CI summary ===")
    cols = [
        "task",
        "model",
        "dimension",
        "balanced_accuracy_mean",
        "balanced_accuracy_ci_low",
        "balanced_accuracy_ci_high",
        "mcc_mean",
        "mcc_ci_low",
        "mcc_ci_high",
        "roc_auc_mean",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
    ]
    print(cls_summary[[col for col in cols if col in cls_summary]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:")
    print(f"  {reg_detail_path}")
    print(f"  {reg_summary_path}")
    print(f"  {cls_detail_path}")
    print(f"  {cls_summary_path}")


if __name__ == "__main__":
    main()
