# -*- coding: utf-8 -*-
"""Benchmark sampling methods with boundary-sensitive metrics.

This script evaluates whether a sampling strategy covers failed substrates and
near-boundary regions, instead of only asking whether a downstream regressor can
predict yield. It is intentionally self-contained so it can be reused on Aldol,
cobalt, or any table with numeric descriptors plus one target column.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
import zlib

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, Sobol
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


DEFAULT_METHODS = [
    "random",
    "cvt",
    "weighted_itr_cvt",
    "fps",
    "kennard_stone",
    "ward",
    "lhs",
    "sobol",
    "repulsive_fps",
]
DEFAULT_DIMS = ["2", "64", "128", "full"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sampling methods using failure and boundary coverage metrics."
    )
    parser.add_argument("--data", default="1700_final_norepeat.csv")
    parser.add_argument("--target", default="conv")
    parser.add_argument("--non-feature-cols", nargs="+", default=["smiles", "conv"])
    parser.add_argument("--success-threshold", type=float, default=70.0)
    parser.add_argument("--failure-threshold", type=float, default=None)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--boundary-quantile", type=float, default=0.25)
    parser.add_argument("--failure-clusters", type=int, default=8)
    parser.add_argument("--repulsion-strength", type=float, default=1.0)
    parser.add_argument("--weighted-iters", type=int, default=60)
    parser.add_argument("--cvt-init-iters", type=int, default=20)
    parser.add_argument("--weighted-learning-rate", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--out-prefix", default="boundary_sampling_benchmark")
    return parser.parse_args()


def load_data(path: Path, target: str, non_feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    data = pd.read_csv(path)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in {path}")

    feature_cols = [col for col in data.columns if col not in non_feature_cols]
    features = data[feature_cols].select_dtypes(include=[np.number])
    if features.empty:
        raise ValueError("No numeric feature columns were found.")

    return data, features.to_numpy(dtype=float), data[target].to_numpy(dtype=float), list(features.columns)


def transform_features(X: np.ndarray, dim: str) -> tuple[np.ndarray, float, int, str]:
    if dim == "tanimoto":
        return X.astype(float), 1.0, X.shape[1], "tanimoto"

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    if dim == "full":
        return X_std, 1.0, X_std.shape[1], "euclidean"

    n_components = min(int(dim), X_std.shape[0], X_std.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_std)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return X_pca, explained, n_components, "euclidean"


def tanimoto_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = X.astype(float, copy=False)
    Y = Y.astype(float, copy=False)
    xy = X @ Y.T
    x2 = np.sum(X * X, axis=1, keepdims=True)
    y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    denom = x2 + y2 - xy
    similarity = np.divide(xy, denom, out=np.ones_like(xy), where=denom > 1e-12)
    return 1.0 - similarity


def distances_between(X: np.ndarray, Y: np.ndarray, metric: str) -> np.ndarray:
    if metric == "tanimoto":
        return tanimoto_distances(X, Y)
    return pairwise_distances(X, Y, metric="euclidean")


def nearest_unique_points(X: np.ndarray, centers: np.ndarray, k: int, metric: str = "euclidean") -> np.ndarray:
    distances = distances_between(X, centers, metric)
    selected: list[int] = []
    used: set[int] = set()
    for center_idx in range(centers.shape[0]):
        for candidate in np.argsort(distances[:, center_idx]):
            candidate_int = int(candidate)
            if candidate_int not in used:
                selected.append(candidate_int)
                used.add(candidate_int)
                break

    if len(selected) < k:
        remaining = [idx for idx in range(len(X)) if idx not in used]
        selected.extend(remaining[: k - len(selected)])
    return np.array(selected[:k], dtype=int)


def sample_random(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(len(X), size=k, replace=False)


def sample_cvt(X: np.ndarray, k: int, rng: np.random.Generator, metric: str = "euclidean") -> np.ndarray:
    if metric == "tanimoto":
        return sample_precomputed_medoid(X, k, rng, metric)
    model = KMeans(n_clusters=k, n_init=10, random_state=int(rng.integers(0, 2**31 - 1)))
    model.fit(X)
    return nearest_unique_points(X, model.cluster_centers_, k, metric)


def sample_fps(X: np.ndarray, k: int, rng: np.random.Generator, metric: str = "euclidean") -> np.ndarray:
    selected = [int(rng.integers(0, len(X)))]
    min_dist = distances_between(X, X[selected], metric).ravel()
    min_dist[selected] = -np.inf
    while len(selected) < k:
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        new_dist = distances_between(X, X[[next_idx]], metric).ravel()
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[selected] = -np.inf
    return np.array(selected, dtype=int)


def sample_kennard_stone(X: np.ndarray, k: int, rng: np.random.Generator, metric: str = "euclidean") -> np.ndarray:
    distances = distances_between(X, X, metric)
    first, second = np.unravel_index(np.argmax(distances), distances.shape)
    selected = [int(first), int(second)]
    min_dist = np.min(distances[:, selected], axis=1)
    min_dist[selected] = -np.inf
    while len(selected) < k:
        tied = np.flatnonzero(min_dist == np.max(min_dist))
        next_idx = int(rng.choice(tied))
        selected.append(next_idx)
        min_dist = np.minimum(min_dist, distances[:, next_idx])
        min_dist[selected] = -np.inf
    return np.array(selected[:k], dtype=int)


def sample_ward(X: np.ndarray, k: int, rng: np.random.Generator, metric: str = "euclidean") -> np.ndarray:
    if metric == "tanimoto":
        return sample_precomputed_medoid(X, k, rng, metric)
    clustering = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
    labels = clustering.fit_predict(X)
    selected: list[int] = []
    for cluster_id in range(k):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if len(cluster_indices) == 0:
            continue
        cluster_points = X[cluster_indices]
        center = np.mean(cluster_points, axis=0, keepdims=True)
        nearest = int(cluster_indices[np.argmin(pairwise_distances(cluster_points, center).ravel())])
        selected.append(nearest)
    if len(selected) < k:
        remaining = [idx for idx in range(len(X)) if idx not in set(selected)]
        selected.extend(rng.choice(remaining, size=k - len(selected), replace=False).tolist())
    return np.array(selected[:k], dtype=int)


def sample_precomputed_medoid(X: np.ndarray, k: int, rng: np.random.Generator, metric: str) -> np.ndarray:
    distances = distances_between(X, X, metric)
    clustering = AgglomerativeClustering(n_clusters=k, linkage="average", metric="precomputed")
    labels = clustering.fit_predict(distances)
    selected: list[int] = []
    for cluster_id in range(k):
        cluster_indices = np.flatnonzero(labels == cluster_id)
        if len(cluster_indices) == 0:
            continue
        sub = distances[np.ix_(cluster_indices, cluster_indices)]
        medoid = int(cluster_indices[np.argmin(sub.mean(axis=1))])
        selected.append(medoid)
    if len(selected) < k:
        remaining = [idx for idx in range(len(X)) if idx not in set(selected)]
        selected.extend(rng.choice(remaining, size=k - len(selected), replace=False).tolist())
    return np.array(selected[:k], dtype=int)


def sample_qmc(X: np.ndarray, k: int, rng: np.random.Generator, method: str) -> np.ndarray:
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    span = np.where(maxs > mins, maxs - mins, 1.0)
    seed = int(rng.integers(0, 2**31 - 1))
    if method == "lhs":
        sampler = LatinHypercube(d=X.shape[1], seed=seed)
        design = sampler.random(n=k)
    else:
        sampler = Sobol(d=X.shape[1], scramble=True, seed=seed)
        power = int(np.ceil(np.log2(k)))
        design = sampler.random_base2(m=power)[:k]
    centers = design * span + mins
    return nearest_unique_points(X, centers, k)


def local_nearest_unique_points(
    X: np.ndarray,
    centers: np.ndarray,
    candidate_indices: np.ndarray,
    k: int,
    metric: str,
) -> np.ndarray:
    distances = distances_between(X[candidate_indices], centers, metric)
    selected: list[int] = []
    used: set[int] = set()
    for center_idx in range(centers.shape[0]):
        for local_candidate in np.argsort(distances[:, center_idx]):
            candidate = int(candidate_indices[int(local_candidate)])
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                break
    if len(selected) < k:
        remaining = [idx for idx in candidate_indices.tolist() if int(idx) not in used]
        selected.extend(int(idx) for idx in remaining[: k - len(selected)])
    return np.array(selected[:k], dtype=int)


def scopemap_repulsion_scale(X: np.ndarray, repulsion_strength: float, metric: str) -> float:
    distances = distances_between(X, X, metric)
    max_distance = float(np.max(distances))
    d = X.shape[1]
    expected_sq_distance = (max_distance**2) * d / 12
    return float((expected_sq_distance**2) * len(X) * (10**repulsion_strength))


def euclidean_cvt_gradient(points: np.ndarray, centers: np.ndarray, fixed_centers: np.ndarray) -> np.ndarray:
    all_centers = np.vstack([centers, fixed_centers]) if len(fixed_centers) else centers
    labels = np.argmin(distances_between(points, all_centers, "euclidean"), axis=1)
    gradients = np.zeros_like(centers)
    for center_idx in range(len(centers)):
        assigned = points[labels == center_idx]
        if len(assigned):
            gradients[center_idx] = -2 * np.sum(assigned - centers[center_idx], axis=0)
    return gradients


def tanimoto_cvt_gradient(points: np.ndarray, centers: np.ndarray, fixed_centers: np.ndarray) -> np.ndarray:
    all_centers = np.vstack([centers, fixed_centers]) if len(fixed_centers) else centers
    labels = np.argmin(distances_between(points, all_centers, "tanimoto"), axis=1)
    gradients = np.zeros_like(centers)
    for center_idx in range(len(centers)):
        assigned = points[labels == center_idx]
        if len(assigned) == 0:
            continue
        center = centers[center_idx]
        p_dot_c = assigned @ center
        p_squared = np.sum(assigned * assigned, axis=1)
        c_squared = float(np.sum(center * center))
        denominator = np.maximum(p_squared + c_squared - p_dot_c, 1e-10)
        gradient = np.zeros_like(center)
        for row_idx, point in enumerate(assigned):
            d_similarity = (
                point * denominator[row_idx] - p_dot_c[row_idx] * (2 * center - point)
            ) / (denominator[row_idx] ** 2)
            gradient += -d_similarity
        gradients[center_idx] = gradient
    return gradients


def repulsion_gradient(
    centers: np.ndarray,
    repulsion_points: np.ndarray,
    strength: float,
    metric: str,
) -> np.ndarray:
    gradients = np.zeros_like(centers)
    if len(repulsion_points) == 0 or strength <= 0:
        return gradients
    for center_idx, center in enumerate(centers):
        for repulsion_point in repulsion_points:
            if metric == "tanimoto":
                similarity = 1.0 - float(tanimoto_distances(center.reshape(1, -1), repulsion_point.reshape(1, -1))[0, 0])
                tanimoto_distance = max(1.0 - similarity, 1e-10)
                p = repulsion_point
                c = center
                p_dot_c = float(np.sum(p * c))
                p_squared = float(np.sum(p * p))
                c_squared = float(np.sum(c * c))
                denominator = max(p_squared + c_squared - p_dot_c, 1e-10)
                d_similarity = (p * denominator - p_dot_c * (2 * c - p)) / (denominator**2)
                gradients[center_idx] += strength * d_similarity / (tanimoto_distance**2)
            else:
                diff = center - repulsion_point
                distance_sq = float(np.sum(diff * diff) + 1e-10)
                gradients[center_idx] += -2 * strength * diff / (distance_sq**2)
    return gradients


def clip_gradient_rows(gradients: np.ndarray, max_norm: float = 10.0) -> np.ndarray:
    norms = np.linalg.norm(gradients, axis=1, keepdims=True)
    scale = np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    return gradients * scale


def sample_weighted_itr_cvt(
    X: np.ndarray,
    y_success: np.ndarray,
    k: int,
    batch_size: int,
    rng: np.random.Generator,
    repulsion_strength: float,
    metric: str,
    weighted_iters: int,
    cvt_init_iters: int,
    learning_rate: float,
) -> np.ndarray:
    """ScopeMap-style iterative CVT with fixed sampled centers and failed-sample repulsion."""
    selected: list[int] = []
    base_repulsion = scopemap_repulsion_scale(X, repulsion_strength, metric)
    while len(selected) < k:
        selected_set = set(selected)
        remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
        next_batch = min(batch_size, k - len(selected), len(remaining))
        if next_batch <= 0:
            break

        if not selected:
            batch = sample_cvt(X, next_batch, rng, metric)
            selected.extend(int(idx) for idx in batch)
            continue

        failed_selected = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)
        if metric == "tanimoto":
            local_selected = sample_precomputed_medoid(X[remaining], next_batch, rng, metric)
            centers = X[remaining[local_selected]]
        else:
            model = KMeans(
                n_clusters=next_batch,
                n_init=5,
                random_state=int(rng.integers(0, 2**31 - 1)),
                max_iter=max(1, cvt_init_iters),
            )
            model.fit(X[remaining])
            centers = model.cluster_centers_.astype(float)

        fixed_centers = X[selected] if selected else np.empty((0, X.shape[1]))
        repulsion_points = X[failed_selected] if len(failed_selected) else np.empty((0, X.shape[1]))
        for _ in range(max(1, weighted_iters)):
            if metric == "tanimoto":
                cvt_gradients = tanimoto_cvt_gradient(X[remaining], centers, fixed_centers)
            else:
                cvt_gradients = euclidean_cvt_gradient(X[remaining], centers, fixed_centers)
            total_gradients = cvt_gradients + repulsion_gradient(centers, repulsion_points, base_repulsion, metric)
            total_gradients = clip_gradient_rows(total_gradients)
            new_centers = centers - learning_rate * total_gradients
            if metric == "tanimoto":
                new_centers = np.clip(new_centers, 0.0, 1.0)

            if np.linalg.norm(new_centers - centers) < 1e-6:
                centers = new_centers
                break
            centers = new_centers

        batch = local_nearest_unique_points(X, centers, remaining, next_batch, metric)
        selected.extend(int(idx) for idx in batch)

    return np.array(selected[:k], dtype=int)


def sample_repulsive_fps(
    X: np.ndarray,
    y_success: np.ndarray,
    k: int,
    batch_size: int,
    rng: np.random.Generator,
    repulsion_strength: float,
    metric: str,
) -> np.ndarray:
    """Simulate human-in-the-loop repulsion from known failed sampled points."""
    first_batch = min(batch_size, k)
    selected = sample_fps(X, first_batch, rng, metric).tolist()

    while len(selected) < k:
        selected_set = set(selected)
        remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
        next_batch = min(batch_size, k - len(selected))

        dist_to_selected = distances_between(X[remaining], X[selected], metric).min(axis=1)
        failed_selected = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)
        if len(failed_selected):
            dist_to_failures = distances_between(X[remaining], X[failed_selected], metric).min(axis=1)
            score = dist_to_selected + repulsion_strength * dist_to_failures
        else:
            score = dist_to_selected

        ranked = remaining[np.argsort(score)[::-1]]
        selected.extend(ranked[:next_batch].astype(int).tolist())
    return np.array(selected[:k], dtype=int)


def run_sampler(
    method: str,
    X: np.ndarray,
    y_success: np.ndarray,
    k: int,
    batch_size: int,
    rng: np.random.Generator,
    repulsion_strength: float,
    metric: str,
    weighted_iters: int,
    cvt_init_iters: int,
    learning_rate: float,
) -> np.ndarray:
    if method == "random":
        return sample_random(X, k, rng)
    if method == "cvt":
        return sample_cvt(X, k, rng, metric)
    if method == "weighted_itr_cvt":
        return sample_weighted_itr_cvt(
            X, y_success, k, batch_size, rng, repulsion_strength, metric, weighted_iters, cvt_init_iters, learning_rate
        )
    if method == "fps":
        return sample_fps(X, k, rng, metric)
    if method == "kennard_stone":
        return sample_kennard_stone(X, k, rng, metric)
    if method == "ward":
        return sample_ward(X, k, rng, metric)
    if method in {"lhs", "sobol"}:
        if metric == "tanimoto":
            raise ValueError(f"{method} is not defined for tanimoto dimension.")
        return sample_qmc(X, k, rng, method)
    if method == "repulsive_fps":
        return sample_repulsive_fps(X, y_success, k, batch_size, rng, repulsion_strength, metric)
    raise ValueError(f"Unknown sampling method: {method}")


def boundary_mask_from_labels(
    X: np.ndarray, y_success: np.ndarray, quantile: float, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    success_idx = np.flatnonzero(y_success == 1)
    failure_idx = np.flatnonzero(y_success == 0)
    if len(success_idx) == 0 or len(failure_idx) == 0:
        raise ValueError("Boundary metrics require both success and failure examples.")

    nearest_opposite = np.empty(len(X), dtype=float)
    nearest_opposite[success_idx] = distances_between(X[success_idx], X[failure_idx], metric).min(axis=1)
    nearest_opposite[failure_idx] = distances_between(X[failure_idx], X[success_idx], metric).min(axis=1)
    cutoff = float(np.quantile(nearest_opposite, quantile))
    return nearest_opposite <= cutoff, nearest_opposite


def failure_cluster_labels(
    X: np.ndarray, y_success: np.ndarray, n_clusters: int, metric: str
) -> tuple[np.ndarray, np.ndarray]:
    failure_idx = np.flatnonzero(y_success == 0)
    if len(failure_idx) == 0:
        return failure_idx, np.array([], dtype=int)
    effective_clusters = min(n_clusters, len(failure_idx))
    if effective_clusters == 1:
        return failure_idx, np.zeros(len(failure_idx), dtype=int)
    if metric == "tanimoto":
        distances = distances_between(X[failure_idx], X[failure_idx], metric)
        model = AgglomerativeClustering(n_clusters=effective_clusters, linkage="average", metric="precomputed")
        labels = model.fit_predict(distances)
        return failure_idx, labels
    model = KMeans(n_clusters=effective_clusters, n_init=10, random_state=0)
    labels = model.fit_predict(X[failure_idx])
    return failure_idx, labels


def evaluate_sample(
    X: np.ndarray,
    y_target: np.ndarray,
    y_success: np.ndarray,
    selected: np.ndarray,
    boundary_mask: np.ndarray,
    nearest_opposite: np.ndarray,
    failure_idx: np.ndarray,
    failure_labels: np.ndarray,
    metric: str,
) -> dict[str, float]:
    selected_set = set(selected.astype(int).tolist())
    selected_mask = np.zeros(len(X), dtype=bool)
    selected_mask[selected] = True
    failure_mask = y_success == 0

    sampled_failures = selected_mask & failure_mask
    sampled_boundary = selected_mask & boundary_mask
    population_boundary_rate = float(boundary_mask.mean())
    sample_boundary_rate = float(sampled_boundary.mean() / selected_mask.mean())

    if len(selected) > 1:
        sample_distances = distances_between(X[selected], X[selected], metric)
        np.fill_diagonal(sample_distances, np.inf)
        mean_sample_nn_distance = float(np.min(sample_distances, axis=1).mean())
    else:
        mean_sample_nn_distance = np.nan

    coverage_radius = float(distances_between(X, X[selected], metric).min(axis=1).mean())

    if len(failure_idx):
        sampled_failure_clusters = {
            int(label)
            for idx, label in zip(failure_idx.tolist(), failure_labels.tolist())
            if idx in selected_set
        }
        failure_cluster_coverage = len(sampled_failure_clusters) / max(1, len(np.unique(failure_labels)))
    else:
        failure_cluster_coverage = np.nan

    return {
        "sampled_success_rate": float(y_success[selected].mean()),
        "sampled_failure_rate": float(1.0 - y_success[selected].mean()),
        "failure_recall": float(sampled_failures.sum() / max(1, failure_mask.sum())),
        "failure_cluster_coverage": float(failure_cluster_coverage),
        "boundary_coverage": float(sampled_boundary.sum() / max(1, boundary_mask.sum())),
        "boundary_enrichment": float(sample_boundary_rate / population_boundary_rate) if population_boundary_rate else np.nan,
        "mean_nearest_opposite_distance": float(nearest_opposite[selected].mean()),
        "mean_sample_nn_distance": mean_sample_nn_distance,
        "coverage_radius": coverage_radius,
        "target_mean": float(np.mean(y_target[selected])),
        "target_std": float(np.std(y_target[selected], ddof=1)) if len(selected) > 1 else 0.0,
    }


def bootstrap_ci(values: pd.Series, samples: int, seed: int) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if samples <= 0 or len(clean) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(samples, len(clean)))
    means = clean[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(detail: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    metric_cols = [
        "sampled_success_rate",
        "sampled_failure_rate",
        "failure_recall",
        "failure_cluster_coverage",
        "boundary_coverage",
        "boundary_enrichment",
        "mean_nearest_opposite_distance",
        "mean_sample_nn_distance",
        "coverage_radius",
        "target_mean",
        "target_std",
    ]
    aggregations: dict[str, tuple[str, str]] = {
        "effective_dim": ("effective_dim", "max"),
        "explained_variance_ratio_mean": ("explained_variance_ratio", "mean"),
    }
    for metric in metric_cols:
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_std"] = (metric, "std")
    summary = detail.groupby(["method", "dimension"], sort=False).agg(**aggregations).reset_index()
    ci_records: list[dict[str, Any]] = []
    for (method, dimension), group in detail.groupby(["method", "dimension"], sort=False):
        record: dict[str, Any] = {"method": method, "dimension": dimension}
        for metric in metric_cols:
            metric_seed = seed + stable_method_seed(f"{method}_{dimension}_{metric}")
            low, high = bootstrap_ci(group[metric], bootstrap_samples, metric_seed)
            record[f"{metric}_ci_low"] = low
            record[f"{metric}_ci_high"] = high
        ci_records.append(record)
    ci = pd.DataFrame(ci_records)
    return summary.merge(ci, on=["method", "dimension"], how="left")


def stable_method_seed(method: str) -> int:
    return zlib.crc32(method.encode("utf-8")) % 100000


def main() -> None:
    args = parse_args()
    dims = [dim.lower() for dim in args.dims]
    methods = [method.lower() for method in args.methods]
    failure_threshold = args.success_threshold if args.failure_threshold is None else args.failure_threshold

    for dim in dims:
        if dim not in {"full", "tanimoto"} and int(dim) < 1:
            raise ValueError(f"Dimension must be positive: {dim}")
    if args.budget < 1:
        raise ValueError("Budget must be positive.")

    _, X_raw, y_target, feature_cols = load_data(Path(args.data), args.target, args.non_feature_cols)
    y_success = (y_target >= args.success_threshold).astype(int)
    y_failure = (y_target < failure_threshold).astype(int)
    if args.failure_threshold is not None:
        y_success_for_boundary = 1 - y_failure
    else:
        y_success_for_boundary = y_success

    print("=== Boundary-sensitive sampling benchmark ===")
    print(f"Data: {args.data}")
    print(f"Rows: {X_raw.shape[0]}, numeric features: {len(feature_cols)}")
    print(f"Success threshold: {args.target} >= {args.success_threshold}")
    print(f"Failure threshold: {args.target} < {failure_threshold}")
    print(f"Success ratio: {y_success.mean():.3f}, failure ratio for boundary: {(1 - y_success_for_boundary).mean():.3f}")
    print(f"Methods: {', '.join(methods)}")
    print(f"Dimensions: {', '.join(dims)}")
    print(f"Budget: {args.budget}, repeats: {args.repeats}")

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

        for method in methods:
            for repeat in range(1, args.repeats + 1):
                rng = np.random.default_rng(args.seed + repeat * 1009 + effective_dim * 9176 + stable_method_seed(method))
                selected = run_sampler(
                    method,
                    X,
                    y_success_for_boundary,
                    args.budget,
                    args.batch_size,
                    rng,
                    args.repulsion_strength,
                    metric,
                    args.weighted_iters,
                    args.cvt_init_iters,
                    args.weighted_learning_rate,
                )
                metrics = evaluate_sample(
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
                        "budget": args.budget,
                        "effective_dim": effective_dim,
                        "explained_variance_ratio": explained,
                    }
                )
                records.append(metrics)
            print(f"  {method}: done")

    detail = pd.DataFrame(records)
    summary = summarize(detail, args.bootstrap_samples, args.seed)

    detail_path = Path(f"{args.out_prefix}_detail.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== Summary preview ===")
    preview_cols = [
        "method",
        "dimension",
        "failure_recall_mean",
        "failure_cluster_coverage_mean",
        "boundary_coverage_mean",
        "boundary_enrichment_mean",
        "coverage_radius_mean",
    ]
    print(summary[preview_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nSaved:")
    print(f"  {detail_path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
