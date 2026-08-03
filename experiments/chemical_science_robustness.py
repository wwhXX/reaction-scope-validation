# -*- coding: utf-8 -*-
"""Boundary and failure-cluster robustness checks for the Chemical Science story.

This focused script avoids the local sklearn/SciPy linear-algebra crashes by
using only NumPy/Pandas. It tests whether the main Aldol/Cobalt decision-space
contrast is stable to boundary-quantile and failure-cluster-count choices.
"""

from __future__ import annotations

import argparse
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    data: str
    target: str
    non_feature_cols: tuple[str, ...]
    success_threshold: float
    dims: tuple[str, ...]
    budget: int
    batch_size: int
    initial_size: int
    cluster_counts: tuple[int, ...]


DATASETS = [
    DatasetSpec(
        name="aldol",
        data="Examples/Sampling_eval/Model_eval/Regression/1700_final_norepeat.csv",
        target="conv",
        non_feature_cols=("smiles", "conv"),
        success_threshold=70.0,
        dims=("2", "full", "tanimoto"),
        budget=80,
        batch_size=10,
        initial_size=10,
        cluster_counts=(6, 8, 10),
    ),
    DatasetSpec(
        name="cobalt",
        data="Examples/Co_catalytic/strength=0.0/cobalt_condition1_benchmark.csv",
        target="condition1_yield",
        non_feature_cols=("SMILES", "condition1_yield", "condition1_ee", "condition2_yield", "condition2_ee"),
        success_threshold=1.0,
        dims=("2", "full", "tanimoto"),
        budget=24,
        batch_size=6,
        initial_size=6,
        cluster_counts=(3, 4, 5),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chemical Science robustness sensitivity checks.")
    parser.add_argument("--out-dir", default="experiments/results/chemical_science_robustness")
    parser.add_argument("--out-prefix", default="boundary_cluster_robustness_r20")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--boundary-quantiles", nargs="+", type=float, default=[0.15, 0.25, 0.35])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def load_dataset(spec: DatasetSpec) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = pd.read_csv(resolve(spec.data))
    feature_cols = [col for col in data.columns if col not in set(spec.non_feature_cols)]
    features = data[feature_cols].select_dtypes(include=[np.number])
    if features.empty:
        raise ValueError(f"No numeric features found for {spec.name}")
    return data, features.to_numpy(dtype=float), data[spec.target].to_numpy(dtype=float)


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)


def pca_2d(X_std: np.ndarray) -> tuple[np.ndarray, float]:
    _, singular_values, vt = np.linalg.svd(X_std, full_matrices=False)
    variance = singular_values**2
    explained = float(variance[:2].sum() / variance.sum()) if variance.sum() else 0.0
    return X_std @ vt[:2].T, explained


def euclidean_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    d2 = np.maximum(a2 + b2 - 2.0 * (A @ B.T), 0.0)
    return np.sqrt(d2)


def tanimoto_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(float, copy=False)
    B = B.astype(float, copy=False)
    xy = A @ B.T
    a2 = np.sum(A * A, axis=1, keepdims=True)
    b2 = np.sum(B * B, axis=1, keepdims=True).T
    denom = a2 + b2 - xy
    sim = np.divide(xy, denom, out=np.ones_like(xy), where=denom > 1e-12)
    return 1.0 - sim


def distances(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    if metric == "tanimoto":
        return tanimoto_distances(A, B)
    return euclidean_distances(A, B)


def simple_kmeans(X: np.ndarray, k: int, seed: int, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if k >= len(X):
        labels = np.arange(len(X), dtype=int)
        return labels, X.copy()
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(max(1, iters)):
        labels = np.argmin(euclidean_distances(X, centers), axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        if np.linalg.norm(new_centers - centers) < 1e-8:
            break
        centers = new_centers
    return labels, centers


def nearest_unique(X: np.ndarray, centers: np.ndarray, candidate_idx: np.ndarray, k: int, metric: str) -> list[int]:
    dist = distances(X[candidate_idx], centers, metric)
    chosen: list[int] = []
    used: set[int] = set()
    for center_idx in range(centers.shape[0]):
        for local in np.argsort(dist[:, center_idx]):
            global_idx = int(candidate_idx[int(local)])
            if global_idx not in used:
                chosen.append(global_idx)
                used.add(global_idx)
                break
    if len(chosen) < k:
        remaining = [int(idx) for idx in candidate_idx.tolist() if int(idx) not in used]
        chosen.extend(remaining[: k - len(chosen)])
    return chosen[:k]


def cvt_select(X: np.ndarray, candidate_idx: np.ndarray, k: int, seed: int, metric: str) -> list[int]:
    if metric == "tanimoto":
        return fps_select(X, [], k, metric, np.random.default_rng(seed), candidate_idx)
    _, centers = simple_kmeans(X[candidate_idx], min(k, len(candidate_idx)), seed, iters=25)
    return nearest_unique(X, centers, candidate_idx, k, metric)


def fps_select(
    X: np.ndarray,
    selected: list[int],
    k: int,
    metric: str,
    rng: np.random.Generator,
    candidate_idx: np.ndarray | None = None,
) -> list[int]:
    if candidate_idx is None:
        selected_set = set(selected)
        candidate_idx = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    if len(candidate_idx) == 0:
        return []
    chosen: list[int] = []
    if selected:
        min_dist = distances(X[candidate_idx], X[np.array(selected, dtype=int)], metric).min(axis=1)
    else:
        first = int(rng.choice(candidate_idx))
        chosen.append(first)
        candidate_idx = np.array([idx for idx in candidate_idx if int(idx) != first], dtype=int)
        if len(candidate_idx) == 0:
            return chosen[:k]
        min_dist = distances(X[candidate_idx], X[[first]], metric).ravel()
    for _ in range(k - len(chosen)):
        local = int(np.argmax(min_dist))
        global_idx = int(candidate_idx[local])
        chosen.append(global_idx)
        new_dist = distances(X[candidate_idx], X[[global_idx]], metric).ravel()
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[local] = -np.inf
    return chosen[:k]


def weighted_itr_cvt_like_batch(
    X: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    batch_size: int,
    metric: str,
    rng: np.random.Generator,
) -> list[int]:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    k = min(batch_size, len(remaining))
    if k <= 0:
        return []
    if not selected:
        return cvt_select(X, remaining, k, int(rng.integers(0, 2**31 - 1)), metric)
    if metric == "tanimoto":
        # For binary-like spaces, use a stable repulsive FPS approximation.
        min_to_selected = distances(X[remaining], X[np.array(selected, dtype=int)], metric).min(axis=1)
        failed = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)
        if len(failed):
            min_to_failed = distances(X[remaining], X[failed], metric).min(axis=1)
            score = min_to_selected + 0.5 * min_to_failed
        else:
            score = min_to_selected
        return remaining[np.argsort(score)[::-1][:k]].astype(int).tolist()

    _, centers = simple_kmeans(X[remaining], k, int(rng.integers(0, 2**31 - 1)), iters=15)
    failed = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)
    for _ in range(20):
        labels = np.argmin(euclidean_distances(X[remaining], centers), axis=1)
        target_centers = centers.copy()
        for cluster in range(k):
            members = X[remaining][labels == cluster]
            if len(members):
                target_centers[cluster] = members.mean(axis=0)
        if len(failed):
            repulsion = np.zeros_like(centers)
            diff = centers[:, None, :] - X[failed][None, :, :]
            dist = np.linalg.norm(diff, axis=2, keepdims=True) + 1e-6
            repulsion = (diff / (dist**3)).sum(axis=1)
            repulsion /= np.linalg.norm(repulsion, axis=1, keepdims=True) + 1e-12
            target_centers = target_centers + 0.05 * repulsion
        if np.linalg.norm(target_centers - centers) < 1e-6:
            centers = target_centers
            break
        centers = target_centers
    return nearest_unique(X, centers, remaining, k, metric)


def nearest_opposite_distances(X: np.ndarray, y_success: np.ndarray) -> np.ndarray:
    success_idx = np.flatnonzero(y_success == 1)
    failure_idx = np.flatnonzero(y_success == 0)
    nearest = np.empty(len(X), dtype=float)
    nearest[success_idx] = euclidean_distances(X[success_idx], X[failure_idx]).min(axis=1)
    nearest[failure_idx] = euclidean_distances(X[failure_idx], X[success_idx]).min(axis=1)
    return nearest


def evaluate(
    selected: list[int],
    y_success: np.ndarray,
    boundary_mask: np.ndarray,
    nearest: np.ndarray,
    failure_idx: np.ndarray,
    failure_labels: np.ndarray,
) -> dict[str, float]:
    selected_arr = np.array(selected, dtype=int)
    selected_mask = np.zeros(len(y_success), dtype=bool)
    selected_mask[selected_arr] = True
    failure_mask = y_success == 0
    sampled_boundary = selected_mask & boundary_mask
    sampled_failures = selected_mask & failure_mask
    selected_set = set(selected)
    discovered = {
        int(label)
        for idx, label in zip(failure_idx.tolist(), failure_labels.tolist())
        if int(idx) in selected_set
    }
    population_boundary_rate = float(boundary_mask.mean())
    sample_boundary_rate = float(sampled_boundary.mean() / selected_mask.mean())
    return {
        "failure_recall": float(sampled_failures.sum() / max(1, failure_mask.sum())),
        "failure_cluster_coverage": float(len(discovered) / max(1, len(np.unique(failure_labels)))),
        "boundary_coverage": float(sampled_boundary.sum() / max(1, boundary_mask.sum())),
        "boundary_enrichment": float(sample_boundary_rate / population_boundary_rate) if population_boundary_rate else np.nan,
        "mean_nearest_opposite_distance": float(nearest[selected_arr].mean()),
    }


def bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or samples <= 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def run_dataset(spec: DatasetSpec, boundary_quantiles: list[float], repeats: int, seed: int) -> pd.DataFrame:
    _, X_raw, y_target = load_dataset(spec)
    X_std = standardize(X_raw)
    X_2d, explained_2d = pca_2d(X_std)
    spaces = {
        "2": (X_2d, "euclidean", explained_2d),
        "full": (X_std, "euclidean", 1.0),
        "tanimoto": (X_raw, "tanimoto", 1.0),
    }
    y_success = (y_target >= spec.success_threshold).astype(int)
    nearest = nearest_opposite_distances(X_std, y_success)
    failure_idx = np.flatnonzero(y_success == 0)
    cluster_labels_by_k = {
        k: simple_kmeans(X_std[failure_idx], min(k, len(failure_idx)), seed + k * 1009, iters=30)[0]
        for k in spec.cluster_counts
    }

    selections: dict[tuple[int, str], list[int]] = {}
    for repeat in range(1, repeats + 1):
        init_rng = np.random.default_rng(seed + repeat * 1009)
        initial = cvt_select(X_std, np.arange(len(X_std), dtype=int), spec.initial_size, seed + repeat * 1009, "euclidean")
        for dim_name in spec.dims:
            X_decision, metric, _ = spaces[dim_name]
            selected = list(initial)
            rng = np.random.default_rng(seed + repeat * 1009 + stable_seed(f"{spec.name}_{dim_name}_weighted"))
            while len(selected) < spec.budget:
                batch = weighted_itr_cvt_like_batch(X_decision, y_success, selected, spec.batch_size, metric, rng)
                if not batch:
                    break
                for idx in batch:
                    if idx not in selected:
                        selected.append(idx)
                selected = selected[: spec.budget]
            selections[(repeat, dim_name)] = selected
        print(f"{spec.name} repeat {repeat}: done")

    rows: list[dict[str, float | int | str]] = []
    for repeat in range(1, repeats + 1):
        for dim_name in spec.dims:
            for quantile in boundary_quantiles:
                cutoff = float(np.quantile(nearest, quantile))
                boundary_mask = nearest <= cutoff
                for cluster_count, labels in cluster_labels_by_k.items():
                    metrics = evaluate(
                        selections[(repeat, dim_name)],
                        y_success,
                        boundary_mask,
                        nearest,
                        failure_idx,
                        labels,
                    )
                    rows.append(
                        {
                            "dataset": spec.name,
                            "method": "weighted_itr_cvt_like",
                            "decision_dimension": dim_name,
                            "repeat": repeat,
                            "budget_spent": spec.budget,
                            "boundary_quantile": quantile,
                            "failure_clusters": cluster_count,
                            "explained_variance_ratio": spaces[dim_name][2],
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def summarize(detail: pd.DataFrame, bootstrap_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = ["boundary_coverage", "failure_recall", "failure_cluster_coverage", "boundary_enrichment"]
    group_cols = ["dataset", "method", "decision_dimension", "boundary_quantile", "failure_clusters"]
    grouped = detail.groupby(group_cols, sort=False)
    summary = grouped.agg(
        n_repeats=("repeat", "nunique"),
        budget_spent=("budget_spent", "max"),
        explained_variance_ratio=("explained_variance_ratio", "mean"),
        **{f"{metric}_mean": (metric, "mean") for metric in metric_cols},
    ).reset_index()
    ci_rows = []
    for keys, group in grouped:
        row = dict(zip(group_cols, keys))
        for metric in metric_cols:
            low, high = bootstrap_ci(group[metric].to_numpy(float), bootstrap_samples, seed + stable_seed(str(keys) + metric))
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        ci_rows.append(row)
    summary = summary.merge(pd.DataFrame(ci_rows), on=group_cols)

    pair_rows = []
    final_metrics = ["boundary_coverage", "failure_recall", "failure_cluster_coverage", "boundary_enrichment"]
    for keys, group in detail.groupby(["dataset", "method", "boundary_quantile", "failure_clusters"], sort=False):
        dataset, method, quantile, cluster_count = keys
        base = group[group["decision_dimension"] == "2"][["repeat", *final_metrics]]
        if base.empty:
            continue
        base = base.rename(columns={metric: f"{metric}_2d" for metric in final_metrics})
        for dim_name, high in group[group["decision_dimension"] != "2"].groupby("decision_dimension", sort=False):
            merged = high[["repeat", *final_metrics]].merge(base, on="repeat", how="inner")
            row = {
                "dataset": dataset,
                "method": method,
                "high_dimension": dim_name,
                "boundary_quantile": quantile,
                "failure_clusters": cluster_count,
                "paired_repeats": int(len(merged)),
            }
            for metric in final_metrics:
                delta = merged[metric].to_numpy(float) - merged[f"{metric}_2d"].to_numpy(float)
                row[f"{metric}_delta_mean"] = float(delta.mean())
                low, high_ci = bootstrap_ci(delta, bootstrap_samples, seed + stable_seed(str(keys) + dim_name + metric))
                row[f"{metric}_delta_ci_low"] = low
                row[f"{metric}_delta_ci_high"] = high_ci
            pair_rows.append(row)
    return summary, pd.DataFrame(pair_rows)


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"] * len(shown.columns)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, pairwise: pd.DataFrame, out_dir: Path, out_prefix: str) -> None:
    lead = pairwise[
        ((pairwise["dataset"] == "aldol") & (pairwise["high_dimension"] == "full"))
        | ((pairwise["dataset"] == "cobalt") & (pairwise["high_dimension"].isin(["full", "tanimoto"])))
    ].copy()
    lead = lead.sort_values(["dataset", "high_dimension", "boundary_quantile", "failure_clusters"])
    agg = (
        lead.groupby(["dataset", "high_dimension"], sort=False)
        .agg(
            min_boundary_delta=("boundary_coverage_delta_mean", "min"),
            max_boundary_delta=("boundary_coverage_delta_mean", "max"),
            min_boundary_ci_low=("boundary_coverage_delta_ci_low", "min"),
            max_boundary_ci_high=("boundary_coverage_delta_ci_high", "max"),
            min_cluster_delta=("failure_cluster_coverage_delta_mean", "min"),
            max_cluster_delta=("failure_cluster_coverage_delta_mean", "max"),
        )
        .reset_index()
    )
    lines = [
        "# Chemical Science Robustness: Boundary and Failure-Cluster Sensitivity",
        "",
        "This focused robustness check varies boundary quantile and failure-cluster count while keeping shared initial sets and fixed full-space evaluation.",
        "The acquisition is a NumPy-only weighted-iterative-CVT-like approximation used to avoid local sklearn/SciPy crashes, so it should be read as a robustness stress test rather than a replacement for the r20 main contrast.",
        "",
        "## Lead Contrasts Across All Sensitivity Settings",
        "",
        markdown_table(agg),
        "",
        "## Pairwise Deltas",
        "",
        markdown_table(lead),
        "",
        "## Summary Rows",
        "",
        markdown_table(summary.sort_values(["dataset", "decision_dimension", "boundary_quantile", "failure_clusters"])),
        "",
    ]
    (out_dir / f"{out_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail = pd.concat(
        [run_dataset(spec, args.boundary_quantiles, args.repeats, args.seed) for spec in DATASETS],
        ignore_index=True,
    )
    summary, pairwise = summarize(detail, args.bootstrap_samples, args.seed)
    detail.to_csv(out_dir / f"{args.out_prefix}_detail.csv", index=False)
    summary.to_csv(out_dir / f"{args.out_prefix}_summary.csv", index=False)
    pairwise.to_csv(out_dir / f"{args.out_prefix}_pairwise_deltas.csv", index=False)
    write_report(summary, pairwise, out_dir, args.out_prefix)
    print(f"Saved robustness outputs to {out_dir}")


if __name__ == "__main__":
    main()
