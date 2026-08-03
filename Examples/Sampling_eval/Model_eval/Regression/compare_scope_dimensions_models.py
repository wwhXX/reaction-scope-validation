# -*- coding: utf-8 -*-
"""Compare 2D vs higher-dimensional scope spaces with stronger ML models.

The goal is to isolate representation effects. Every model sees the same
train/validation/test split; only the feature space dimension changes.
Regression predicts conversion directly, while classification predicts whether
conversion passes a user-defined threshold.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare low-dimensional and high-dimensional scope spaces using tree/boosting models."
    )
    parser.add_argument("--data", default="1700_final_norepeat.csv")
    parser.add_argument("--target", default="conv")
    parser.add_argument("--non-feature-cols", nargs="+", default=["smiles", "conv"])
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--val-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=70.0)
    parser.add_argument("--include-xgboost", action="store_true")
    parser.add_argument("--out-prefix", default="scope_dimension_model_comparison")
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


def make_splits(
    n_rows: int,
    y_binary: np.ndarray,
    folds: int,
    test_size: float,
    val_size: float,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(n_rows)
    splits = []
    for fold in range(folds):
        fold_seed = seed + fold
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=fold_seed,
            stratify=y_binary,
        )
        relative_val_size = val_size / (1.0 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=relative_val_size,
            random_state=fold_seed + 1000,
            stratify=y_binary[train_val_idx],
        )
        splits.append((train_idx, val_idx, test_idx))
    return splits


def transform_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    dim: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int]:
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_val_std = scaler.transform(X_val)
    X_test_std = scaler.transform(X_test)

    if dim == "full":
        return X_train_std, X_val_std, X_test_std, 1.0, X_train_std.shape[1]

    n_components = min(int(dim), X_train_std.shape[0], X_train_std.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_train_pca = pca.fit_transform(X_train_std)
    X_val_pca = pca.transform(X_val_std)
    X_test_pca = pca.transform(X_test_std)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return X_train_pca, X_val_pca, X_test_pca, explained, n_components


def regression_models(seed: int, include_xgboost: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "DummyMeanRegressor": DummyRegressor(strategy="mean"),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTreesRegressor": ExtraTreesRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=seed,
        ),
    }

    if include_xgboost and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBRegressor

        models["XGBRegressor"] = XGBRegressor(
            n_estimators=500,
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


def classification_models(seed: int, include_xgboost: bool) -> dict[str, Any]:
    models: dict[str, Any] = {
        "DummyMostFrequentClassifier": DummyClassifier(strategy="most_frequent"),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "ExtraTreesClassifier": ExtraTreesClassifier(
            n_estimators=500,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            l2_regularization=0.05,
            random_state=seed,
        ),
    }

    if include_xgboost and importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        models["XGBClassifier"] = XGBClassifier(
            n_estimators=500,
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


def evaluate_classification(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float]:
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

    if hasattr(fitted, "predict_proba"):
        prob = fitted.predict_proba(X_test)[:, 1]
        if len(np.unique(y_test)) == 2:
            result["roc_auc"] = float(roc_auc_score(y_test, prob))
            result["average_precision"] = float(average_precision_score(y_test, prob))
    return result


def summarize(detail: pd.DataFrame, task: str) -> pd.DataFrame:
    metric_cols = {
        "regression": ["rmse", "mae", "r2"],
        "classification": [
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
        ],
    }[task]
    existing_metrics = [col for col in metric_cols if col in detail.columns]

    aggregations: dict[str, tuple[str, str]] = {
        "effective_dim": ("effective_dim", "max"),
        "explained_variance_ratio_mean": ("explained_variance_ratio", "mean"),
    }
    for metric in existing_metrics:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")

    return detail.groupby(["task", "model", "dimension"], sort=False).agg(**aggregations).reset_index()


def main() -> None:
    args = parse_args()
    dims = [dim.lower() for dim in args.dims]
    for dim in dims:
        if dim != "full" and int(dim) < 1:
            raise ValueError(f"Dimension must be positive: {dim}")

    X, y_reg, feature_cols = load_xy(Path(args.data), args.target, args.non_feature_cols)
    y_cls = (y_reg >= args.success_threshold).astype(int)
    splits = make_splits(len(y_reg), y_cls, args.folds, args.test_size, args.val_size, args.seed)

    print("=== Scope dimension model comparison ===")
    print(f"Data: {args.data}")
    print(f"Rows: {X.shape[0]}, numeric features: {len(feature_cols)}")
    print(f"Classification threshold: {args.target} >= {args.success_threshold}")
    print(f"Positive class ratio: {y_cls.mean():.3f}")
    print(f"Dimensions: {', '.join(dims)}")

    reg_records: list[dict[str, Any]] = []
    cls_records: list[dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx, test_idx) in enumerate(splits, start=1):
        print(f"\nFold {fold_idx}/{len(splits)}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        fold_seed = args.seed + fold_idx
        reg_models = regression_models(fold_seed, args.include_xgboost)
        cls_models = classification_models(fold_seed, args.include_xgboost)

        for dim in dims:
            X_train, _, X_test, explained, effective_dim = transform_features(
                X[train_idx],
                X[val_idx],
                X[test_idx],
                dim,
            )
            print(f"  dim={dim:>4}, effective={effective_dim}, explained={explained:.3f}")

            for model_name, model in reg_models.items():
                result = evaluate_regression(model, X_train, y_reg[train_idx], X_test, y_reg[test_idx])
                result.update(
                    {
                        "task": "regression",
                        "model": model_name,
                        "dimension": dim,
                        "fold": fold_idx,
                        "effective_dim": effective_dim,
                        "explained_variance_ratio": explained,
                    }
                )
                reg_records.append(result)
                print(f"    REG {model_name}: RMSE={result['rmse']:.3f}, R2={result['r2']:.3f}")

            for model_name, model in cls_models.items():
                result = evaluate_classification(model, X_train, y_cls[train_idx], X_test, y_cls[test_idx])
                result.update(
                    {
                        "task": "classification",
                        "model": model_name,
                        "dimension": dim,
                        "fold": fold_idx,
                        "effective_dim": effective_dim,
                        "explained_variance_ratio": explained,
                    }
                )
                cls_records.append(result)
                auc_text = f", AUC={result['roc_auc']:.3f}" if "roc_auc" in result else ""
                print(
                    f"    CLS {model_name}: F1={result['f1']:.3f}, "
                    f"BalAcc={result['balanced_accuracy']:.3f}, "
                    f"Spec={result['negative_recall_specificity']:.3f}, "
                    f"MCC={result['mcc']:.3f}{auc_text}"
                )

    reg_detail = pd.DataFrame(reg_records)
    cls_detail = pd.DataFrame(cls_records)
    reg_summary = summarize(reg_detail, "regression")
    cls_summary = summarize(cls_detail, "classification")

    reg_detail_path = Path(f"{args.out_prefix}_regression_detail.csv")
    reg_summary_path = Path(f"{args.out_prefix}_regression_summary.csv")
    cls_detail_path = Path(f"{args.out_prefix}_classification_detail.csv")
    cls_summary_path = Path(f"{args.out_prefix}_classification_summary.csv")

    reg_detail.to_csv(reg_detail_path, index=False)
    reg_summary.to_csv(reg_summary_path, index=False)
    cls_detail.to_csv(cls_detail_path, index=False)
    cls_summary.to_csv(cls_summary_path, index=False)

    print("\n=== Regression summary ===")
    print(reg_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\n=== Classification summary ===")
    print(cls_summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:")
    print(f"  {reg_detail_path}")
    print(f"  {reg_summary_path}")
    print(f"  {cls_detail_path}")
    print(f"  {cls_summary_path}")


if __name__ == "__main__":
    main()
