# -*- coding: utf-8 -*-
"""Lightweight paired contrast for drawable external Buchwald-Hartwig data.

This script avoids local SciPy/sklearn linear-algebra crashes by using only
NumPy/Pandas. It is intentionally focused: shared initial set, FPS-style
diversity acquisition, and fixed full-space boundary evaluation.
"""

from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
META_COLS = {
    "reaction_id",
    "yield",
    "smiles",
    "reaction_smiles",
    "ligand_smiles",
    "additive_smiles",
    "base_smiles",
    "aryl_halide_smiles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run drawable external paired FPS contrast.")
    parser.add_argument("--data", default="Data/External/rxn_yields_prepared/buchwald_hartwig_drawable_fp_benchmark.csv")
    parser.add_argument("--out-dir", default="experiments/results/chemical_science_drawable_external")
    parser.add_argument("--out-prefix", default="drawable_external_contrast_r20")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--budget", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--initial-size", type=int, default=20)
    parser.add_argument("--success-threshold", type=float, default=50.0)
    parser.add_argument("--boundary-quantile", type=float, default=0.25)
    parser.add_argument("--failure-clusters", type=int, default=8)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def standardize(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)


def pca_spaces(X_std: np.ndarray) -> dict[str, tuple[np.ndarray, str, float]]:
    _, singular_values, vt = np.linalg.svd(X_std, full_matrices=False)
    variance = singular_values**2
    total = float(variance.sum())
    spaces: dict[str, tuple[np.ndarray, str, float]] = {"full": (X_std, "euclidean", 1.0)}
    for dim in [2, 32, 64]:
        coords = X_std @ vt[:dim].T
        explained = float(variance[:dim].sum() / total) if total else 0.0
        spaces[str(dim)] = (coords, "euclidean", explained)
    return spaces


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


def nearest_opposite_distances(X: np.ndarray, y_success: np.ndarray) -> np.ndarray:
    success_idx = np.flatnonzero(y_success == 1)
    failure_idx = np.flatnonzero(y_success == 0)
    nearest = np.empty(len(X), dtype=float)
    nearest[success_idx] = euclidean_distances(X[success_idx], X[failure_idx]).min(axis=1)
    nearest[failure_idx] = euclidean_distances(X[failure_idx], X[success_idx]).min(axis=1)
    return nearest


def simple_kmeans(X: np.ndarray, k: int, seed: int, iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), size=k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        labels = np.argmin(euclidean_distances(X, centers), axis=1)
        new_centers = centers.copy()
        for cluster in range(k):
            members = X[labels == cluster]
            if len(members):
                new_centers[cluster] = members.mean(axis=0)
        if np.linalg.norm(new_centers - centers) < 1e-8:
            break
        centers = new_centers
    return labels


def fps_batch(X: np.ndarray, selected: list[int], batch_size: int, metric: str, rng: np.random.Generator) -> list[int]:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(X)) if idx not in selected_set], dtype=int)
    if len(remaining) == 0:
        return []
    if not selected:
        first = int(rng.choice(remaining))
        selected = [first]
        selected_set.add(first)
        remaining = np.array([idx for idx in remaining if idx != first], dtype=int)
    min_dist = distances(X[remaining], X[np.array(selected, dtype=int)], metric).min(axis=1)
    chosen: list[int] = []
    for _ in range(min(batch_size, len(remaining))):
        local = int(np.argmax(min_dist))
        global_idx = int(remaining[local])
        chosen.append(global_idx)
        new_dist = distances(X[remaining], X[[global_idx]], metric).ravel()
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[local] = -np.inf
    return chosen


def random_batch(n: int, selected: list[int], batch_size: int, rng: np.random.Generator) -> list[int]:
    remaining = np.array([idx for idx in range(n) if idx not in set(selected)], dtype=int)
    if len(remaining) == 0:
        return []
    return rng.choice(remaining, size=min(batch_size, len(remaining)), replace=False).astype(int).tolist()


def evaluate(
    selected: list[int],
    y_target: np.ndarray,
    y_success: np.ndarray,
    boundary_mask: np.ndarray,
    nearest_opposite: np.ndarray,
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
        "known_success_rate": float(y_success[selected_arr].mean()),
        "mean_nearest_opposite_distance": float(nearest_opposite[selected_arr].mean()),
        "target_mean": float(y_target[selected_arr].mean()),
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


def markdown_table(frame: pd.DataFrame) -> str:
    shown = frame.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"] * len(shown.columns)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data = pd.read_csv(resolve(args.data))
    feature_cols = [col for col in data.columns if col not in META_COLS]
    X_raw = data[feature_cols].to_numpy(dtype=float)
    X_std = standardize(X_raw)
    y_target = data["yield"].to_numpy(dtype=float)
    y_success = (y_target >= args.success_threshold).astype(int)
    spaces = pca_spaces(X_std)
    spaces["tanimoto"] = (X_raw, "tanimoto", 1.0)

    X_eval = X_std
    nearest = nearest_opposite_distances(X_eval, y_success)
    boundary_cutoff = float(np.quantile(nearest, args.boundary_quantile))
    boundary_mask = nearest <= boundary_cutoff
    failure_idx = np.flatnonzero(y_success == 0)
    failure_labels = simple_kmeans(X_eval[failure_idx], min(args.failure_clusters, len(failure_idx)), args.seed)

    rows: list[dict[str, float | int | str]] = []
    for repeat in range(1, args.repeats + 1):
        initial_rng = np.random.default_rng(args.seed + repeat * 1009)
        initial = fps_batch(X_eval, [], args.initial_size, "euclidean", initial_rng)
        for dim_name in ["2", "32", "64", "full", "tanimoto"]:
            X_decision, metric, explained = spaces[dim_name]
            for method in ["random", "diversity_fps"]:
                selected = list(initial)
                seed_key = method if method == "random" else f"{dim_name}_{method}"
                rng = np.random.default_rng(args.seed + repeat * 1009 + stable_seed(seed_key))
                while len(selected) < args.budget:
                    if method == "random":
                        batch = random_batch(len(data), selected, args.batch_size, rng)
                    else:
                        batch = fps_batch(X_decision, selected, args.batch_size, metric, rng)
                    if not batch:
                        break
                    selected.extend(batch)
                    selected = selected[: args.budget]
                metrics = evaluate(selected, y_target, y_success, boundary_mask, nearest, failure_idx, failure_labels)
                rows.append(
                    {
                        "dataset": "buchwald_hartwig_drawable",
                        "method": method,
                        "decision_dimension": dim_name,
                        "repeat": repeat,
                        "budget_spent": len(selected),
                        "explained_variance_ratio": explained,
                        **metrics,
                    }
                )
        print(f"repeat {repeat}: done")

    detail = pd.DataFrame(rows)
    grouped = detail.groupby(["dataset", "method", "decision_dimension"], sort=False)
    summary = grouped.agg(
        n_repeats=("repeat", "nunique"),
        explained_variance_ratio=("explained_variance_ratio", "mean"),
        boundary_coverage_mean=("boundary_coverage", "mean"),
        failure_recall_mean=("failure_recall", "mean"),
        failure_cluster_coverage_mean=("failure_cluster_coverage", "mean"),
        boundary_enrichment_mean=("boundary_enrichment", "mean"),
        known_success_rate_mean=("known_success_rate", "mean"),
        target_mean=("target_mean", "mean"),
    ).reset_index()

    ci_rows = []
    for keys, group in grouped:
        row = dict(zip(["dataset", "method", "decision_dimension"], keys))
        for metric in ["boundary_coverage", "failure_recall", "failure_cluster_coverage", "boundary_enrichment"]:
            low, high = bootstrap_ci(group[metric].to_numpy(float), args.bootstrap_samples, args.seed + len(ci_rows) * 17)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        ci_rows.append(row)
    summary = summary.merge(pd.DataFrame(ci_rows), on=["dataset", "method", "decision_dimension"])

    pair_rows = []
    for method, group in detail.groupby("method", sort=False):
        base = group[group["decision_dimension"] == "2"][["repeat", "boundary_coverage", "failure_recall", "failure_cluster_coverage", "boundary_enrichment"]]
        base = base.rename(columns={col: f"{col}_2d" for col in base.columns if col != "repeat"})
        for dim_name in ["32", "64", "full", "tanimoto"]:
            high = group[group["decision_dimension"] == dim_name][["repeat", "boundary_coverage", "failure_recall", "failure_cluster_coverage", "boundary_enrichment"]]
            merged = high.merge(base, on="repeat")
            row = {"dataset": "buchwald_hartwig_drawable", "method": method, "high_dimension": dim_name, "paired_repeats": len(merged)}
            for metric in ["boundary_coverage", "failure_recall", "failure_cluster_coverage", "boundary_enrichment"]:
                delta = merged[metric].to_numpy(float) - merged[f"{metric}_2d"].to_numpy(float)
                row[f"{metric}_delta_mean"] = float(delta.mean())
                low, high_ci = bootstrap_ci(delta, args.bootstrap_samples, args.seed + len(pair_rows) * 31)
                row[f"{metric}_delta_ci_low"] = low
                row[f"{metric}_delta_ci_high"] = high_ci
            pair_rows.append(row)
    pairwise = pd.DataFrame(pair_rows)

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out_dir / f"{args.out_prefix}_detail.csv", index=False)
    summary.to_csv(out_dir / f"{args.out_prefix}_summary.csv", index=False)
    pairwise.to_csv(out_dir / f"{args.out_prefix}_pairwise_deltas.csv", index=False)

    report = [
        "# Drawable External Buchwald-Hartwig Contrast",
        "",
        "This focused run uses preserved Buchwald-Hartwig component SMILES and concatenated Morgan fingerprints.",
        "Selection uses a shared initial FPS set and an FPS-style diversity acquisition policy; scoring uses fixed full-fingerprint Euclidean boundary geometry.",
        "",
        "## Pairwise High-Dimensional Deltas Over 2D",
        "",
        markdown_table(pairwise.sort_values("boundary_coverage_delta_mean", ascending=False)),
        "",
        "## Summary",
        "",
        markdown_table(summary.sort_values(["method", "decision_dimension"])),
        "",
    ]
    (out_dir / f"{args.out_prefix}_report.md").write_text("\n".join(report), encoding="utf-8")

    reps = data.iloc[failure_idx].copy()
    reps["failure_cluster"] = failure_labels
    reps.sort_values(["failure_cluster", "yield"]).groupby("failure_cluster").head(4).to_csv(
        out_dir / f"{args.out_prefix}_failure_representatives.csv", index=False
    )
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
