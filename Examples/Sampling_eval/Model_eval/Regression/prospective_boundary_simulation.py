# -*- coding: utf-8 -*-
"""Retrospective prospective simulation for reaction-boundary discovery.

The full labelled table is treated as an oracle. Within each representation,
each method starts from the same initial labelled set, selects substrates batch
by batch, and only receives labels for the selected substrates. Evaluation is
retrospective: failure clusters and near-boundary points are defined from the
full labels in the same representation, then used to measure how quickly each
simulated campaign discovers them.

For strict cross-representation claims, use ``experiments/decision_space_contrast.py``.
That paired contrast uses a shared initial set across decision spaces and scores
all campaigns in one fixed full-space evaluation geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

from boundary_sampling_benchmark import (
    boundary_mask_from_labels,
    distances_between,
    evaluate_sample,
    failure_cluster_labels,
    load_data,
    local_nearest_unique_points,
    repulsion_gradient,
    sample_cvt,
    sample_fps,
    sample_kennard_stone,
    sample_qmc,
    sample_random,
    sample_ward,
    scopemap_repulsion_scale,
    stable_method_seed,
    transform_features,
    euclidean_cvt_gradient,
    tanimoto_cvt_gradient,
    clip_gradient_rows,
)


DEFAULT_DIMS = ["2", "64", "128", "full", "tanimoto"]
DEFAULT_METHODS = [
    "random",
    "cvt",
    "weighted_itr_cvt",
    "fps",
    "kennard_stone",
    "ward",
    "repulsive_fps",
    "uncertainty",
    "uncertainty_diversity",
]
STATIC_METHODS = {"random", "cvt", "fps", "kennard_stone", "ward", "lhs", "sobol"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate batch-by-batch boundary discovery on a fully labelled retrospective dataset."
    )
    parser.add_argument("--data", default="1700_final_norepeat.csv")
    parser.add_argument("--target", default="conv")
    parser.add_argument("--non-feature-cols", nargs="+", default=["smiles", "conv"])
    parser.add_argument("--success-threshold", type=float, default=70.0)
    parser.add_argument("--failure-threshold", type=float, default=None)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budget", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--initial-size", type=int, default=None)
    parser.add_argument("--initial-method", choices=["random", "cvt", "fps"], default="cvt")
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-quantile", type=float, default=0.25)
    parser.add_argument("--failure-clusters", type=int, default=8)
    parser.add_argument("--repulsion-strength", type=float, default=1.0)
    parser.add_argument("--weighted-iters", type=int, default=20)
    parser.add_argument("--cvt-init-iters", type=int, default=10)
    parser.add_argument("--weighted-learning-rate", type=float, default=0.01)
    parser.add_argument("--uncertainty-trees", type=int, default=200)
    parser.add_argument("--candidate-multiplier", type=int, default=10)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out-prefix", default="prospective_boundary_simulation")
    return parser.parse_args()


def bootstrap_ci(values: pd.Series, samples: int, seed: int) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if samples <= 0 or len(clean) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(samples, len(clean)))
    means = clean[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def normalize(values: np.ndarray) -> np.ndarray:
    values = values.astype(float, copy=False)
    span = float(np.nanmax(values) - np.nanmin(values))
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - float(np.nanmin(values))) / span


def make_initial_selection(
    X: np.ndarray,
    initial_size: int,
    method: str,
    rng: np.random.Generator,
    metric: str,
) -> np.ndarray:
    if initial_size <= 0:
        return np.array([], dtype=int)
    initial_size = min(initial_size, len(X))
    if method == "random":
        return sample_random(X, initial_size, rng)
    if method == "fps":
        return sample_fps(X, initial_size, rng, metric)
    return sample_cvt(X, initial_size, rng, metric)


def select_static_batch(
    method: str,
    X: np.ndarray,
    selected: list[int],
    batch_size: int,
    rng: np.random.Generator,
    metric: str,
) -> np.ndarray:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    next_batch = min(batch_size, len(remaining))
    if next_batch <= 0:
        return np.array([], dtype=int)
    if method == "random":
        return rng.choice(remaining, size=next_batch, replace=False)
    if method == "fps":
        if not selected:
            local = sample_fps(X[remaining], next_batch, rng, metric)
            return remaining[local]
        min_dist = distances_between(X[remaining], X[np.array(selected, dtype=int)], metric).min(axis=1)
        chosen: list[int] = []
        available = remaining.copy()
        for _ in range(next_batch):
            local_idx = int(np.argmax(min_dist))
            global_idx = int(available[local_idx])
            chosen.append(global_idx)
            new_dist = distances_between(X[available], X[[global_idx]], metric).ravel()
            min_dist = np.minimum(min_dist, new_dist)
            min_dist[local_idx] = -np.inf
        return np.array(chosen, dtype=int)
    if method == "kennard_stone":
        if not selected:
            local = sample_kennard_stone(X[remaining], next_batch, rng, metric)
            return remaining[local]
        distances = distances_between(X[remaining], X[np.array(selected, dtype=int)], metric)
        ranked = remaining[np.argsort(distances.min(axis=1))[::-1]]
        return ranked[:next_batch].astype(int)
    if method == "cvt":
        local = sample_cvt(X[remaining], next_batch, rng, metric)
        return remaining[local]
    if method == "ward":
        local = sample_ward(X[remaining], next_batch, rng, metric)
        return remaining[local]
    if method in {"lhs", "sobol"}:
        if metric == "tanimoto":
            local = sample_fps(X[remaining], next_batch, rng, metric)
        else:
            local = sample_qmc(X[remaining], next_batch, rng, method)
        return remaining[local]
    raise ValueError(f"Unknown static method: {method}")


def select_weighted_itr_batch(
    X: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    batch_size: int,
    rng: np.random.Generator,
    repulsion_strength: float,
    metric: str,
    weighted_iters: int,
    cvt_init_iters: int,
    learning_rate: float,
) -> np.ndarray:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    next_batch = min(batch_size, len(remaining))
    if next_batch <= 0:
        return np.array([], dtype=int)
    if not selected:
        local = sample_cvt(X[remaining], next_batch, rng, metric)
        return remaining[local]

    failed_selected = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)
    fixed_centers = X[np.array(selected, dtype=int)]
    repulsion_points = X[failed_selected] if len(failed_selected) else np.empty((0, X.shape[1]))

    if metric == "tanimoto":
        local = sample_ward(X[remaining], next_batch, rng, metric)
        centers = X[remaining[local]].astype(float)
    else:
        model = KMeans(
            n_clusters=next_batch,
            n_init=5,
            max_iter=max(1, cvt_init_iters),
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        model.fit(X[remaining])
        centers = model.cluster_centers_.astype(float)

    repulsion_scale = scopemap_repulsion_scale(X, repulsion_strength, metric)
    for _ in range(max(1, weighted_iters)):
        if metric == "tanimoto":
            cvt_gradients = tanimoto_cvt_gradient(X[remaining], centers, fixed_centers)
        else:
            cvt_gradients = euclidean_cvt_gradient(X[remaining], centers, fixed_centers)
        total_gradients = cvt_gradients + repulsion_gradient(centers, repulsion_points, repulsion_scale, metric)
        total_gradients = clip_gradient_rows(total_gradients)
        updated = centers - learning_rate * total_gradients
        if metric == "tanimoto":
            updated = np.clip(updated, 0.0, 1.0)
        if np.linalg.norm(updated - centers) < 1e-6:
            centers = updated
            break
        centers = updated

    return local_nearest_unique_points(X, centers, remaining, next_batch, metric)


def select_repulsive_fps_batch(
    X: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    batch_size: int,
    rng: np.random.Generator,
    repulsion_strength: float,
    metric: str,
) -> np.ndarray:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    next_batch = min(batch_size, len(remaining))
    if next_batch <= 0:
        return np.array([], dtype=int)
    if not selected:
        local = sample_fps(X[remaining], next_batch, rng, metric)
        return remaining[local]

    dist_to_selected = distances_between(X[remaining], X[np.array(selected, dtype=int)], metric).min(axis=1)
    failed_selected = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)
    if len(failed_selected):
        dist_to_failures = distances_between(X[remaining], X[failed_selected], metric).min(axis=1)
        score = dist_to_selected + repulsion_strength * dist_to_failures
    else:
        score = dist_to_selected
    return remaining[np.argsort(score)[::-1][:next_batch]].astype(int)


def select_uncertainty_batch(
    method: str,
    X: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    batch_size: int,
    rng: np.random.Generator,
    metric: str,
    uncertainty_trees: int,
    candidate_multiplier: int,
    diversity_weight: float,
) -> np.ndarray:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    next_batch = min(batch_size, len(remaining))
    if next_batch <= 0:
        return np.array([], dtype=int)
    if len(selected) < 2 or len(np.unique(y_success[np.array(selected, dtype=int)])) < 2:
        return select_static_batch("fps", X, selected, next_batch, rng, metric)

    model = RandomForestClassifier(
        n_estimators=uncertainty_trees,
        class_weight="balanced",
        random_state=int(rng.integers(0, 2**31 - 1)),
        n_jobs=-1,
    )
    selected_array = np.array(selected, dtype=int)
    model.fit(X[selected_array], y_success[selected_array])
    proba = model.predict_proba(X[remaining])
    class_index = list(model.classes_).index(1) if 1 in model.classes_ else 0
    p_success = proba[:, class_index]
    uncertainty = 1.0 - np.abs(p_success - 0.5) * 2.0
    if method == "uncertainty":
        return remaining[np.argsort(uncertainty)[::-1][:next_batch]].astype(int)

    pool_size = min(len(remaining), max(next_batch, next_batch * candidate_multiplier))
    pool = remaining[np.argsort(uncertainty)[::-1][:pool_size]]
    uncertainty_by_index = {int(idx): float(value) for idx, value in zip(remaining.tolist(), uncertainty.tolist())}
    selected_batch: list[int] = []
    current_selected = selected_array.copy()
    while len(selected_batch) < next_batch:
        candidate_pool = np.array([idx for idx in pool if int(idx) not in selected_batch], dtype=int)
        if len(candidate_pool) == 0:
            break
        local_uncertainty = np.array([uncertainty_by_index[int(idx)] for idx in candidate_pool], dtype=float)
        if len(current_selected):
            diversity = distances_between(X[candidate_pool], X[current_selected], metric).min(axis=1)
        else:
            diversity = np.zeros(len(candidate_pool), dtype=float)
        score = normalize(local_uncertainty) + diversity_weight * normalize(diversity)
        next_idx = int(candidate_pool[int(np.argmax(score))])
        selected_batch.append(next_idx)
        current_selected = np.append(current_selected, next_idx)
    return np.array(selected_batch, dtype=int)


def select_next_batch(
    method: str,
    X: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    args: argparse.Namespace,
    rng: np.random.Generator,
    metric: str,
) -> np.ndarray:
    if method in STATIC_METHODS:
        return select_static_batch(method, X, selected, args.batch_size, rng, metric)
    if method == "weighted_itr_cvt":
        return select_weighted_itr_batch(
            X,
            y_success,
            selected,
            args.batch_size,
            rng,
            args.repulsion_strength,
            metric,
            args.weighted_iters,
            args.cvt_init_iters,
            args.weighted_learning_rate,
        )
    if method == "repulsive_fps":
        return select_repulsive_fps_batch(
            X, y_success, selected, args.batch_size, rng, args.repulsion_strength, metric
        )
    if method in {"uncertainty", "uncertainty_diversity"}:
        return select_uncertainty_batch(
            method,
            X,
            y_success,
            selected,
            args.batch_size,
            rng,
            metric,
            args.uncertainty_trees,
            args.candidate_multiplier,
            args.diversity_weight,
        )
    raise ValueError(f"Unknown method: {method}")


def evaluate_checkpoint(
    X: np.ndarray,
    y_target: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    boundary_mask: np.ndarray,
    nearest_opposite: np.ndarray,
    failure_idx: np.ndarray,
    failure_labels: np.ndarray,
    metric: str,
) -> dict[str, float]:
    selected_array = np.array(selected, dtype=int)
    metrics = evaluate_sample(
        X,
        y_target,
        y_success,
        selected_array,
        boundary_mask,
        nearest_opposite,
        failure_idx,
        failure_labels,
        metric,
    )
    metrics["budget_spent"] = float(len(selected_array))
    metrics["known_success_rate"] = float(y_success[selected_array].mean())
    return metrics


def summarize(detail: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    metric_cols = [
        "sampled_success_rate",
        "sampled_failure_rate",
        "failure_recall",
        "failure_cluster_coverage",
        "boundary_coverage",
        "boundary_enrichment",
        "mean_nearest_opposite_distance",
        "coverage_radius",
        "target_mean",
        "known_success_rate",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "n_repeats": ("repeat", "nunique"),
        "effective_dim": ("effective_dim", "max"),
        "explained_variance_ratio_mean": ("explained_variance_ratio", "mean"),
    }
    for metric in metric_cols:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = (
        detail.groupby(["method", "dimension", "budget_spent"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )

    ci_records: list[dict[str, Any]] = []
    grouped = detail.groupby(["method", "dimension", "budget_spent"], sort=False)
    for (method, dimension, budget_spent), group in grouped:
        record: dict[str, Any] = {
            "method": method,
            "dimension": dimension,
            "budget_spent": budget_spent,
        }
        for metric in metric_cols:
            metric_seed = seed + stable_method_seed(f"{method}_{dimension}_{budget_spent}_{metric}")
            low, high = bootstrap_ci(group[metric], bootstrap_samples, metric_seed)
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        ci_records.append(record)
    return summary.merge(pd.DataFrame(ci_records), on=["method", "dimension", "budget_spent"], how="left")


def discovery_times(detail: pd.DataFrame) -> pd.DataFrame:
    thresholds = {
        "failure_cluster_coverage": [0.50, 0.75, 1.00],
        "boundary_coverage": [0.10, 0.25, 0.50],
        "failure_recall": [0.10, 0.25, 0.50],
    }
    records: list[dict[str, Any]] = []
    for (method, dimension, repeat), group in detail.groupby(["method", "dimension", "repeat"], sort=False):
        group = group.sort_values("budget_spent")
        for metric, cutoffs in thresholds.items():
            for cutoff in cutoffs:
                reached = group[group[metric] >= cutoff]
                budget = float(reached.iloc[0]["budget_spent"]) if not reached.empty else np.nan
                records.append(
                    {
                        "method": method,
                        "dimension": dimension,
                        "repeat": repeat,
                        "metric": metric,
                        "threshold": cutoff,
                        "first_budget": budget,
                    }
                )
    return pd.DataFrame(records)


def summarize_discovery_times(times: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    if times.empty:
        return times
    summary = (
        times.groupby(["method", "dimension", "metric", "threshold"], sort=False)
        .agg(
            n_repeats=("repeat", "nunique"),
            reached_rate=("first_budget", lambda values: float(values.notna().mean())),
            first_budget_mean=("first_budget", "mean"),
            first_budget_median=("first_budget", "median"),
        )
        .reset_index()
    )
    ci_records: list[dict[str, Any]] = []
    for (method, dimension, metric, threshold), group in times.groupby(
        ["method", "dimension", "metric", "threshold"], sort=False
    ):
        metric_seed = seed + stable_method_seed(f"{method}_{dimension}_{metric}_{threshold}")
        low, high = bootstrap_ci(group["first_budget"], bootstrap_samples, metric_seed)
        ci_records.append(
            {
                "method": method,
                "dimension": dimension,
                "metric": metric,
                "threshold": threshold,
                "first_budget_ci_low": low,
                "first_budget_ci_high": high,
            }
        )
    return summary.merge(pd.DataFrame(ci_records), on=["method", "dimension", "metric", "threshold"], how="left")


def stable_dimension_seed(dim: str) -> int:
    return zlib.crc32(dim.encode("utf-8")) % 100000


def main() -> None:
    args = parse_args()
    dims = [dim.lower() for dim in args.dims]
    methods = [method.lower() for method in args.methods]
    initial_size = args.batch_size if args.initial_size is None else args.initial_size
    failure_threshold = args.success_threshold if args.failure_threshold is None else args.failure_threshold

    if args.budget < 1 or args.batch_size < 1 or initial_size < 0:
        raise ValueError("Budget, batch size, and initial size must be valid positive values.")
    for dim in dims:
        if dim not in {"full", "tanimoto"} and int(dim) < 1:
            raise ValueError(f"Dimension must be positive: {dim}")

    _, X_raw, y_target, feature_cols = load_data(Path(args.data), args.target, args.non_feature_cols)
    y_success = (y_target >= args.success_threshold).astype(int)
    y_failure = (y_target < failure_threshold).astype(int)
    y_success_for_boundary = 1 - y_failure if args.failure_threshold is not None else y_success

    print("=== Retrospective prospective boundary simulation ===")
    print(f"Data: {args.data}")
    print(f"Rows: {X_raw.shape[0]}, numeric features: {len(feature_cols)}")
    print(f"Success threshold: {args.target} >= {args.success_threshold}")
    print(f"Failure threshold: {args.target} < {failure_threshold}")
    print(f"Initial set: {initial_size} by {args.initial_method}")
    print(f"Batch size: {args.batch_size}, final budget: {args.budget}, repeats: {args.repeats}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Dimensions: {', '.join(dims)}")

    records: list[dict[str, Any]] = []
    for dim in dims:
        X, explained, effective_dim, metric = transform_features(X_raw, dim)
        boundary_mask, nearest_opposite = boundary_mask_from_labels(
            X, y_success_for_boundary, args.boundary_quantile, metric
        )
        failure_idx, failure_labels = failure_cluster_labels(X, y_success_for_boundary, args.failure_clusters, metric)
        print(
            f"\nDimension={dim}, effective={effective_dim}, explained={explained:.3f}, "
            f"metric={metric}, boundary points={int(boundary_mask.sum())}"
        )

        for repeat in range(1, args.repeats + 1):
            init_seed = args.seed + repeat * 1009 + stable_dimension_seed(dim)
            initial_rng = np.random.default_rng(init_seed)
            initial = make_initial_selection(X, initial_size, args.initial_method, initial_rng, metric).astype(int)

            for method in methods:
                method_seed = init_seed + stable_method_seed(method)
                rng = np.random.default_rng(method_seed)
                selected = initial.astype(int).tolist()
                round_idx = 0
                if selected:
                    metrics = evaluate_checkpoint(
                        X,
                        y_target,
                        y_success_for_boundary,
                        selected,
                        boundary_mask,
                        nearest_opposite,
                        failure_idx,
                        failure_labels,
                        metric,
                    )
                    metrics.update(
                        {
                            "method": method,
                            "dimension": dim,
                            "repeat": repeat,
                            "round": round_idx,
                            "effective_dim": effective_dim,
                            "explained_variance_ratio": explained,
                        }
                    )
                    records.append(metrics)

                while len(selected) < args.budget:
                    batch = select_next_batch(method, X, y_success_for_boundary, selected, args, rng, metric)
                    if len(batch) == 0:
                        break
                    for idx in batch.astype(int).tolist():
                        if idx not in selected:
                            selected.append(idx)
                    selected = selected[: args.budget]
                    round_idx += 1
                    metrics = evaluate_checkpoint(
                        X,
                        y_target,
                        y_success_for_boundary,
                        selected,
                        boundary_mask,
                        nearest_opposite,
                        failure_idx,
                        failure_labels,
                        metric,
                    )
                    metrics.update(
                        {
                            "method": method,
                            "dimension": dim,
                            "repeat": repeat,
                            "round": round_idx,
                            "effective_dim": effective_dim,
                            "explained_variance_ratio": explained,
                        }
                    )
                    records.append(metrics)
            print(f"  repeat {repeat}: done")

    detail = pd.DataFrame(records)
    summary = summarize(detail, args.bootstrap_samples, args.seed)
    times = discovery_times(detail)
    times_summary = summarize_discovery_times(times, args.bootstrap_samples, args.seed)

    detail_path = Path(f"{args.out_prefix}_detail.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    times_path = Path(f"{args.out_prefix}_discovery_times.csv")
    times_summary_path = Path(f"{args.out_prefix}_discovery_times_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    times.to_csv(times_path, index=False)
    times_summary.to_csv(times_summary_path, index=False)

    print("\n=== Final-budget preview ===")
    final_budget = summary["budget_spent"].max()
    preview_cols = [
        "method",
        "dimension",
        "budget_spent",
        "failure_cluster_coverage_mean",
        "boundary_coverage_mean",
        "failure_recall_mean",
        "boundary_enrichment_mean",
    ]
    print(
        summary[summary["budget_spent"] == final_budget][preview_cols].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("\nSaved:")
    for path in [detail_path, summary_path, times_path, times_summary_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
