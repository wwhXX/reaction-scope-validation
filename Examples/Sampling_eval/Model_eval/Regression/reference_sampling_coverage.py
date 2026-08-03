# -*- coding: utf-8 -*-
"""Dimension-aware sampling coverage against a reference substrate set.

This benchmark is for article examples that provide a large candidate space and
an experimental/reference subset, but not reliable negative labels or yields for
the whole space. It therefore avoids inventing classification targets and asks a
more modest question: how well does a sampling method cover the reference set
under different molecular representations?
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

from boundary_sampling_benchmark import (
    distances_between,
    sample_cvt,
    sample_fps,
    sample_kennard_stone,
    sample_qmc,
    sample_random,
    sample_ward,
    sample_weighted_itr_cvt,
)


DEFAULT_DIMS = ["2", "64", "128", "full", "tanimoto"]
DEFAULT_METHODS = ["random", "cvt", "scopemap_hd", "fps", "kennard_stone", "ward", "lhs", "sobol"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sampling methods by coverage of a known reference substrate subset."
    )
    parser.add_argument("--space", required=True, help="Candidate-space CSV containing a SMILES column.")
    parser.add_argument("--space-features", required=True, help="Numeric descriptor CSV aligned with --space rows.")
    parser.add_argument("--reference", required=True, help="Reference/experimental subset CSV containing SMILES.")
    parser.add_argument(
        "--reference-features",
        required=True,
        help="Numeric descriptor CSV aligned with --reference rows.",
    )
    parser.add_argument("--space-smiles-col", default="smiles")
    parser.add_argument("--reference-smiles-col", default=None)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coverage-quantiles", nargs="+", type=float, default=[0.10, 0.25])
    parser.add_argument("--reference-clusters", type=int, default=6)
    parser.add_argument("--repulsion-strength", type=float, default=1.0)
    parser.add_argument("--weighted-iters", type=int, default=20)
    parser.add_argument("--cvt-init-iters", type=int, default=10)
    parser.add_argument("--weighted-learning-rate", type=float, default=0.01)
    parser.add_argument("--out-prefix", default="reference_sampling_coverage")
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def resolve_smiles_col(frame: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in frame.columns:
        return requested
    for candidate in ["smiles", "SMILES", "Smiles"]:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"No SMILES column found. Columns: {list(frame.columns)}")


def normalize_smiles(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip()


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, str, str]:
    space = read_table(Path(args.space))
    space_features = read_table(Path(args.space_features)).select_dtypes(include=[np.number])
    reference = read_table(Path(args.reference))
    reference_features = read_table(Path(args.reference_features)).select_dtypes(include=[np.number])

    if len(space) != len(space_features):
        raise ValueError(f"Space row mismatch: metadata={len(space)}, features={len(space_features)}")
    if len(reference) != len(reference_features):
        raise ValueError(
            f"Reference row mismatch: metadata={len(reference)}, features={len(reference_features)}"
        )
    if space_features.empty or reference_features.empty:
        raise ValueError("Feature tables must contain numeric descriptor columns.")

    space_smiles_col = resolve_smiles_col(space, args.space_smiles_col)
    reference_smiles_col = resolve_smiles_col(reference, args.reference_smiles_col)
    return (
        space,
        space_features.to_numpy(dtype=float),
        reference,
        reference_features.to_numpy(dtype=float),
        space_smiles_col,
        reference_smiles_col,
    )


def transform_pair(X_space: np.ndarray, X_reference: np.ndarray, dim: str) -> tuple[np.ndarray, np.ndarray, float, int, str]:
    dim = dim.lower()
    if dim == "tanimoto":
        return X_space.astype(float), X_reference.astype(float), 1.0, X_space.shape[1], "tanimoto"

    scaler = StandardScaler()
    X_space_std = scaler.fit_transform(X_space)
    X_reference_std = scaler.transform(X_reference)
    if dim == "full":
        return X_space_std, X_reference_std, 1.0, X_space_std.shape[1], "euclidean"

    n_components = min(int(dim), X_space_std.shape[0], X_space_std.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    X_space_pca = pca.fit_transform(X_space_std)
    X_reference_pca = pca.transform(X_reference_std)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return X_space_pca, X_reference_pca, explained, n_components, "euclidean"


def run_sampler(
    method: str,
    X: np.ndarray,
    budget: int,
    batch_size: int,
    rng: np.random.Generator,
    metric: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if method == "random":
        return sample_random(X, budget, rng)
    if method == "cvt":
        return sample_cvt(X, budget, rng, metric)
    if method == "scopemap_hd":
        sampled_are_repulsion_anchors = np.zeros(len(X), dtype=int)
        return sample_weighted_itr_cvt(
            X,
            sampled_are_repulsion_anchors,
            budget,
            batch_size,
            rng,
            args.repulsion_strength,
            metric,
            args.weighted_iters,
            args.cvt_init_iters,
            args.weighted_learning_rate,
        )
    if method == "fps":
        return sample_fps(X, budget, rng, metric)
    if method == "kennard_stone":
        return sample_kennard_stone(X, budget, rng, metric)
    if method == "ward":
        return sample_ward(X, budget, rng, metric)
    if method in {"lhs", "sobol"}:
        if metric == "tanimoto":
            return sample_fps(X, budget, rng, metric)
        return sample_qmc(X, budget, rng, method)
    raise ValueError(f"Unknown sampling method: {method}")


def mean_sample_nn_distance(X_sample: np.ndarray, metric: str) -> float:
    if len(X_sample) < 2:
        return 0.0
    distances = distances_between(X_sample, X_sample, metric)
    np.fill_diagonal(distances, np.inf)
    return float(np.min(distances, axis=1).mean())


def reference_cluster_coverage(
    X_reference: np.ndarray,
    reference_covered: np.ndarray,
    n_clusters: int,
    metric: str,
) -> float:
    if len(X_reference) == 0:
        return np.nan
    n_clusters = min(max(1, n_clusters), len(X_reference))
    if n_clusters == 1:
        return float(np.any(reference_covered))
    if metric == "tanimoto":
        distances = distances_between(X_reference, X_reference, metric)
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="precomputed").fit_predict(
            distances
        )
    else:
        labels = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward", metric="euclidean").fit_predict(
            X_reference
        )
    covered = 0
    for cluster_id in range(n_clusters):
        covered += int(np.any(reference_covered[labels == cluster_id]))
    return float(covered / n_clusters)


def evaluate_sample(
    X_space: np.ndarray,
    X_reference: np.ndarray,
    sampled_idx: np.ndarray,
    metric: str,
    space_smiles: pd.Series,
    reference_smiles: pd.Series,
    coverage_quantiles: list[float],
    reference_clusters: int,
) -> dict[str, float]:
    X_sample = X_space[sampled_idx]
    space_to_sample = distances_between(X_space, X_sample, metric).min(axis=1)
    reference_to_sample = distances_between(X_reference, X_sample, metric).min(axis=1)

    sampled_smiles = set(space_smiles.iloc[sampled_idx].tolist())
    reference_set = set(reference_smiles.tolist())
    exact_hits = len(sampled_smiles.intersection(reference_set))

    result: dict[str, float] = {
        "reference_exact_recall": float(exact_hits / max(1, len(reference_set))),
        "sample_reference_hit_rate": float(exact_hits / max(1, len(sampled_idx))),
        "reference_mean_nearest_sample_distance": float(reference_to_sample.mean()),
        "reference_median_nearest_sample_distance": float(np.median(reference_to_sample)),
        "coverage_radius": float(space_to_sample.mean()),
        "max_coverage_radius": float(space_to_sample.max()),
        "mean_sample_nn_distance": mean_sample_nn_distance(X_sample, metric),
    }

    for quantile in coverage_quantiles:
        threshold = float(np.quantile(space_to_sample, quantile))
        covered = reference_to_sample <= threshold
        label = f"q{int(round(quantile * 100)):02d}"
        coverage = float(np.mean(covered))
        result[f"reference_coverage_{label}"] = coverage
        result[f"reference_enrichment_{label}"] = float(coverage / quantile) if quantile > 0 else np.nan
        result[f"reference_cluster_coverage_{label}"] = reference_cluster_coverage(
            X_reference, covered, reference_clusters, metric
        )
    return result


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, samples: int = 2000) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return np.nan, np.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0])
    draws = rng.choice(clean, size=(samples, len(clean)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summarize(detail: pd.DataFrame, seed: int) -> pd.DataFrame:
    id_cols = {"dataset", "method", "dimension", "repeat", "metric", "effective_dim", "explained_variance_ratio"}
    metric_cols = [col for col in detail.columns if col not in id_cols and pd.api.types.is_numeric_dtype(detail[col])]
    rng = np.random.default_rng(seed + 100000)
    rows: list[dict[str, Any]] = []
    group_cols = ["method", "dimension"]
    if "metric" in detail.columns:
        group_cols.append("metric")
    for keys, group in detail.groupby(group_cols, sort=False):
        if len(group_cols) == 3:
            method, dimension, metric_name = keys
        else:
            method, dimension = keys
            metric_name = "tanimoto" if str(dimension) == "tanimoto" else "euclidean"
        row: dict[str, Any] = {
            "method": method,
            "dimension": dimension,
            "metric": metric_name,
            "n_repeats": int(group["repeat"].nunique()),
            "effective_dim": int(group["effective_dim"].max()),
            "explained_variance_ratio_mean": float(group["explained_variance_ratio"].mean()),
        }
        for metric in metric_cols:
            values = group[metric].to_numpy(dtype=float)
            clean = values[np.isfinite(values)]
            row[f"{metric}_mean"] = float(clean.mean()) if len(clean) else np.nan
            row[f"{metric}_std"] = float(clean.std(ddof=1)) if len(clean) > 1 else 0.0
            low, high = bootstrap_mean_ci(values, rng)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    dims = [dim.lower() for dim in args.dims]
    methods = [method.lower() for method in args.methods]
    coverage_quantiles = sorted(args.coverage_quantiles)
    space, X_space_raw, reference, X_reference_raw, space_smiles_col, reference_smiles_col = load_inputs(args)
    budget = args.budget or len(reference)
    budget = min(budget, len(space))
    batch_size = args.batch_size or min(10, budget)
    space_smiles = normalize_smiles(space[space_smiles_col])
    reference_smiles = normalize_smiles(reference[reference_smiles_col])

    print("=== Reference sampling coverage benchmark ===")
    print(f"Space: {args.space} rows={len(space)}, features={X_space_raw.shape[1]}")
    print(f"Reference: {args.reference} rows={len(reference)}, features={X_reference_raw.shape[1]}")
    print(f"Budget: {budget}, batch size: {batch_size}")
    print(f"Dimensions: {', '.join(dims)}")
    print(f"Methods: {', '.join(methods)}")

    records: list[dict[str, Any]] = []
    for dim in dims:
        X_space, X_reference, explained, effective_dim, metric = transform_pair(X_space_raw, X_reference_raw, dim)
        print(f"\ndim={dim}, effective={effective_dim}, explained={explained:.3f}, metric={metric}")
        for repeat in range(1, args.repeats + 1):
            for method in methods:
                rng = np.random.default_rng(args.seed + repeat * 1009 + zlib.crc32(f"{dim}:{method}".encode()))
                sampled_idx = run_sampler(method, X_space, budget, batch_size, rng, metric, args)
                result = evaluate_sample(
                    X_space,
                    X_reference,
                    sampled_idx,
                    metric,
                    space_smiles,
                    reference_smiles,
                    coverage_quantiles,
                    args.reference_clusters,
                )
                result.update(
                    {
                        "method": method,
                        "dimension": dim,
                        "repeat": repeat,
                        "metric": metric,
                        "effective_dim": effective_dim,
                        "explained_variance_ratio": explained,
                    }
                )
                records.append(result)
            print(f"  repeat {repeat}/{args.repeats} complete")

    detail = pd.DataFrame(records)
    summary = summarize(detail, args.seed)
    detail_path = Path(f"{args.out_prefix}_detail.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved {detail_path}")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
