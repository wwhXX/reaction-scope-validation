# -*- coding: utf-8 -*-
"""Paired 2D-map versus high-dimensional decision-space contrast.

This script replays retrospective prospective campaigns with a shared initial
set across decision spaces. Selection is made in the requested decision space,
but scoring is always done in one fixed evaluation space. This makes the
comparison stricter than the routine prospective benchmark, where each
dimension has its own initial set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
SCRIPT_DIR = ROOT / "Examples" / "Sampling_eval" / "Model_eval" / "Regression"
sys.path.insert(0, str(SCRIPT_DIR))

from boundary_sampling_benchmark import (  # noqa: E402
    boundary_mask_from_labels,
    failure_cluster_labels,
    transform_features,
)
from prospective_boundary_simulation import (  # noqa: E402
    bootstrap_ci,
    evaluate_checkpoint,
    make_initial_selection,
    select_next_batch,
    stable_dimension_seed,
    stable_method_seed,
)
from boundary_sampling_benchmark import load_data  # noqa: E402


DEFAULT_METHODS = ["diversity", "uncertainty", "uncertainty_diversity", "weighted_itr_cvt", "cvt"]
METHOD_ALIASES = {"diversity": "fps"}
DEFAULT_DATASETS = ["aldol_prospective_boundary", "cobalt_prospective_boundary", "buchwald_hartwig_prospective_boundary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired 2D versus high-dimensional decision-space contrasts.")
    parser.add_argument("--config", default="experiments/configs/prospective_boundary_simulation.json")
    parser.add_argument("--only", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--repeats", type=int, default=None, help="Override repeats for all tasks.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--eval-dim", default="full")
    parser.add_argument("--out-dir", default="experiments/results/chemical_science_sprint")
    parser.add_argument("--out-prefix", default="decision_space_contrast")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def task_dataset_name(task_name: str) -> str:
    return task_name.replace("_prospective_boundary", "")


def contrast_dims(task_params: dict[str, Any]) -> list[str]:
    dims = [str(dim).lower() for dim in task_params["dims"]]
    chosen = ["2"]
    if "32" in dims:
        chosen.append("32")
    if "64" in dims:
        chosen.append("64")
    if "full" in dims:
        chosen.append("full")
    if "tanimoto" in dims:
        chosen.append("tanimoto")
    return list(dict.fromkeys(chosen))


def make_selection_args(params: dict[str, Any], methods: list[str], repeats: int, seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        methods=methods,
        batch_size=int(params["batch_size"]),
        budget=int(params["budget"]),
        initial_size=int(params.get("initial_size") or params["batch_size"]),
        initial_method=params.get("initial_method", "cvt"),
        repeats=repeats,
        seed=seed,
        repulsion_strength=float(params.get("repulsion_strength", 1.0)),
        weighted_iters=int(params.get("weighted_iters", 20)),
        cvt_init_iters=int(params.get("cvt_init_iters", 10)),
        weighted_learning_rate=float(params.get("weighted_learning_rate", 0.01)),
        uncertainty_trees=int(params.get("uncertainty_trees", 200)),
        candidate_multiplier=int(params.get("candidate_multiplier", 10)),
        diversity_weight=float(params.get("diversity_weight", 0.25)),
    )


def cluster_discovery_record(
    dataset: str,
    method: str,
    decision_dim: str,
    repeat: int,
    budget_spent: int,
    selected: list[int],
    failure_idx: np.ndarray,
    failure_labels: np.ndarray,
) -> list[dict[str, Any]]:
    selected_set = set(int(idx) for idx in selected)
    records: list[dict[str, Any]] = []
    for cluster_id in sorted(set(int(label) for label in failure_labels.tolist())):
        members = failure_idx[failure_labels == cluster_id]
        discovered = any(int(idx) in selected_set for idx in members.tolist())
        records.append(
            {
                "dataset": dataset,
                "method": method,
                "decision_dimension": decision_dim,
                "repeat": repeat,
                "budget_spent": budget_spent,
                "failure_cluster": int(cluster_id),
                "cluster_size": int(len(members)),
                "discovered": bool(discovered),
            }
        )
    return records


def summarize_with_ci(detail: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    metric_cols = [
        "failure_recall",
        "failure_cluster_coverage",
        "boundary_coverage",
        "boundary_enrichment",
        "coverage_radius",
        "known_success_rate",
    ]
    grouped = detail.groupby(["dataset", "method", "decision_dimension", "budget_spent"], sort=False)
    summary = grouped.agg(
        n_repeats=("repeat", "nunique"),
        effective_dim=("effective_dim", "max"),
        explained_variance_ratio=("explained_variance_ratio", "mean"),
        **{f"{metric}_mean": (metric, "mean") for metric in metric_cols},
        **{f"{metric}_std": (metric, "std") for metric in metric_cols},
    ).reset_index()

    ci_rows: list[dict[str, Any]] = []
    for keys, group in grouped:
        dataset, method, decision_dim, budget_spent = keys
        row: dict[str, Any] = {
            "dataset": dataset,
            "method": method,
            "decision_dimension": decision_dim,
            "budget_spent": budget_spent,
        }
        for metric in metric_cols:
            metric_seed = seed + stable_method_seed(f"{dataset}_{method}_{decision_dim}_{budget_spent}_{metric}")
            low, high = bootstrap_ci(group[metric], bootstrap_samples, metric_seed)
            row[f"{metric}_ci_low"] = low
            row[f"{metric}_ci_high"] = high
        ci_rows.append(row)
    return summary.merge(pd.DataFrame(ci_rows), on=["dataset", "method", "decision_dimension", "budget_spent"])


def pairwise_deltas(detail: pd.DataFrame, bootstrap_samples: int, seed: int) -> pd.DataFrame:
    final_rows = detail.loc[detail.groupby(["dataset"])["budget_spent"].transform("max") == detail["budget_spent"]]
    metrics = ["boundary_coverage", "failure_cluster_coverage", "failure_recall", "boundary_enrichment"]
    records: list[dict[str, Any]] = []
    for (dataset, method), group in final_rows.groupby(["dataset", "method"], sort=False):
        base = group[group["decision_dimension"] == "2"]
        if base.empty:
            continue
        base = base[["repeat", *metrics]].rename(columns={metric: f"{metric}_2d" for metric in metrics})
        for decision_dim, high in group[group["decision_dimension"] != "2"].groupby("decision_dimension", sort=False):
            merged = high[["repeat", *metrics]].merge(base, on="repeat", how="inner")
            if merged.empty:
                continue
            row: dict[str, Any] = {
                "dataset": dataset,
                "method": method,
                "high_dimension": decision_dim,
                "paired_repeats": int(len(merged)),
            }
            for metric in metrics:
                delta = merged[metric] - merged[f"{metric}_2d"]
                row[f"{metric}_delta_mean"] = float(delta.mean())
                low, high_ci = bootstrap_ci(delta, bootstrap_samples, seed + stable_method_seed(f"{dataset}_{method}_{decision_dim}_{metric}_delta"))
                row[f"{metric}_delta_ci_low"] = low
                row[f"{metric}_delta_ci_high"] = high_ci
            records.append(row)
    return pd.DataFrame(records)


def summarize_cluster_discovery(cluster_detail: pd.DataFrame) -> pd.DataFrame:
    if cluster_detail.empty:
        return cluster_detail
    return (
        cluster_detail.groupby(
            ["dataset", "method", "decision_dimension", "budget_spent", "failure_cluster", "cluster_size"],
            sort=False,
        )
        .agg(discovery_rate=("discovered", "mean"), repeats=("repeat", "nunique"))
        .reset_index()
    )


def plot_pairwise(pairwise: pd.DataFrame, out_dir: Path, out_prefix: str) -> None:
    if pairwise.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = pairwise[pairwise["high_dimension"].isin(["32", "64", "full", "tanimoto"])].copy()
    shown["label"] = shown["dataset"] + "\n" + shown["method"] + "\n" + shown["high_dimension"] + " vs 2D"
    shown = shown.sort_values(["dataset", "method", "high_dimension"])
    fig, ax = plt.subplots(figsize=(max(9.5, 0.42 * len(shown)), 5.5))
    colors = ["#2E8B57" if value >= 0 else "#B85450" for value in shown["boundary_coverage_delta_mean"]]
    ax.bar(range(len(shown)), shown["boundary_coverage_delta_mean"], color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("Final boundary coverage delta")
    ax.set_title("Same initial set, same sampler: high-dimensional decision space minus 2D")
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels(shown["label"], rotation=75, ha="right", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    min_value = float(shown["boundary_coverage_delta_mean"].min())
    max_value = float(shown["boundary_coverage_delta_mean"].max())
    span = max(max_value - min_value, 0.05)
    ax.set_ylim(min_value - 0.12 * span, max_value + 0.12 * span)
    fig.tight_layout()
    fig.savefig(out_dir / f"{out_prefix}_boundary_delta.png", dpi=300)
    fig.savefig(out_dir / f"{out_prefix}_boundary_delta.svg")
    plt.close(fig)

    best = (
        shown.sort_values("boundary_coverage_delta_mean", ascending=False)
        .groupby(["dataset", "method"], sort=False)
        .head(1)
        .sort_values(["dataset", "method"])
    )
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    datasets = list(best["dataset"].drop_duplicates())
    methods = list(best["method"].drop_duplicates())
    width = 0.15
    x = np.arange(len(datasets), dtype=float)
    palette = {
        "diversity": "#4C78A8",
        "uncertainty": "#F58518",
        "uncertainty_diversity": "#B279A2",
        "weighted_itr_cvt": "#54A24B",
        "cvt": "#72B7B2",
    }
    for offset, method in enumerate(methods):
        method_rows = best[best["method"] == method].set_index("dataset")
        values = [method_rows.loc[dataset, "boundary_coverage_delta_mean"] if dataset in method_rows.index else np.nan for dataset in datasets]
        positions = x + (offset - (len(methods) - 1) / 2) * width
        ax.bar(positions, values, width=width, label=method, color=palette.get(method))
        for pos, dataset, value in zip(positions, datasets, values):
            if pd.isna(value) or dataset not in method_rows.index:
                continue
            dim = str(method_rows.loc[dataset, "high_dimension"])
            offset_y = 0.012 if value >= 0 else -0.012
            ax.text(pos, value + offset_y, dim, ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Best high-D minus 2D boundary coverage")
    ax.set_title("Direct decision-space contrast under shared initial sets")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    finite_values = best["boundary_coverage_delta_mean"].dropna()
    if not finite_values.empty:
        min_value = float(finite_values.min())
        max_value = float(finite_values.max())
        span = max(max_value - min_value, 0.05)
        ax.set_ylim(min_value - 0.18 * span, max_value + 0.10 * span)
    fig.tight_layout()
    fig.savefig(out_dir / f"{out_prefix}_best_boundary_delta_by_method.png", dpi=300)
    fig.savefig(out_dir / f"{out_prefix}_best_boundary_delta_by_method.svg")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    shown = frame.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    headers = [str(col) for col in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(pairwise: pd.DataFrame, cluster_summary: pd.DataFrame, out_dir: Path, out_prefix: str) -> None:
    lines = ["# Chemical Science Sprint: Decision-Space Contrast", ""]
    lines.append("This report uses shared initial sets across decision spaces. Selection is done in each decision space; scoring is done in a fixed full-space evaluation geometry.")
    lines.append("")
    if not pairwise.empty:
        top = pairwise.sort_values("boundary_coverage_delta_mean", ascending=False).head(12)
        lines.append("## Largest High-Dimensional Gains Over 2D")
        lines.append("")
        lines.append(markdown_table(top))
        lines.append("")
    if not cluster_summary.empty:
        final = cluster_summary.loc[
            cluster_summary.groupby(["dataset"])["budget_spent"].transform("max") == cluster_summary["budget_spent"]
        ]
        focused = final[final["decision_dimension"].isin(["2", "32", "64", "full", "tanimoto"])]
        lines.append("## Failure-Cluster Discovery Rates At Final Budget")
        lines.append("")
        lines.append(
            markdown_table(
                focused.groupby(["dataset", "method", "decision_dimension"], sort=False)["discovery_rate"]
                .mean()
                .reset_index(name="mean_cluster_discovery_rate")
                .sort_values(["dataset", "method", "decision_dimension"])
            )
        )
        lines.append("")
    (out_dir / f"{out_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_task(task: dict[str, Any], methods: list[str], repeats_override: int | None, eval_dim: str, seed_override: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = task_dataset_name(task["name"])
    params = dict(task["params"])
    seed = int(seed_override if seed_override is not None else params.get("seed", 42))
    repeats = int(repeats_override if repeats_override is not None else params.get("repeats", 20))
    data, X_raw, y_target, _ = load_data(resolve(params["data"]), params["target"], params["non_feature_cols"])
    y_success = (y_target >= float(params["success_threshold"])).astype(int)
    failure_threshold = float(params.get("failure_threshold") or params["success_threshold"])
    y_failure = (y_target < failure_threshold).astype(int)
    y_boundary_success = 1 - y_failure if params.get("failure_threshold") is not None else y_success

    X_eval, _, _, eval_metric = transform_features(X_raw, eval_dim)
    boundary_mask, nearest_opposite = boundary_mask_from_labels(
        X_eval, y_boundary_success, float(params.get("boundary_quantile", 0.25)), eval_metric
    )
    failure_idx, failure_labels = failure_cluster_labels(
        X_eval, y_boundary_success, int(params.get("failure_clusters", 8)), eval_metric
    )

    selection_args = make_selection_args(params, methods, repeats, seed)
    dims = contrast_dims(params)
    records: list[dict[str, Any]] = []
    cluster_records: list[dict[str, Any]] = []
    print(f"\n=== {dataset}: {len(data)} rows, repeats={repeats}, dims={dims}, methods={methods} ===")

    for decision_dim in dims:
        X_decision, explained, effective_dim, decision_metric = transform_features(X_raw, decision_dim)
        for repeat in range(1, repeats + 1):
            init_seed = seed + repeat * 1009
            initial_rng = np.random.default_rng(init_seed)
            initial = make_initial_selection(
                X_eval,
                selection_args.initial_size,
                selection_args.initial_method,
                initial_rng,
                eval_metric,
            ).astype(int)
            for method in methods:
                method_for_selector = METHOD_ALIASES.get(method, method)
                method_seed = init_seed + stable_dimension_seed(decision_dim) + stable_method_seed(method)
                rng = np.random.default_rng(method_seed)
                selected = initial.astype(int).tolist()
                round_idx = 0
                while True:
                    metrics = evaluate_checkpoint(
                        X_eval,
                        y_target,
                        y_boundary_success,
                        selected,
                        boundary_mask,
                        nearest_opposite,
                        failure_idx,
                        failure_labels,
                        eval_metric,
                    )
                    budget_spent = int(len(selected))
                    metrics.update(
                        {
                            "dataset": dataset,
                            "method": method,
                            "selector_method": method_for_selector,
                            "decision_dimension": decision_dim,
                            "evaluation_dimension": eval_dim,
                            "decision_metric": decision_metric,
                            "evaluation_metric": eval_metric,
                            "repeat": repeat,
                            "round": round_idx,
                            "effective_dim": effective_dim,
                            "explained_variance_ratio": explained,
                        }
                    )
                    records.append(metrics)
                    cluster_records.extend(
                        cluster_discovery_record(
                            dataset, method, decision_dim, repeat, budget_spent, selected, failure_idx, failure_labels
                        )
                    )
                    if len(selected) >= selection_args.budget:
                        break
                    batch = select_next_batch(
                        method_for_selector,
                        X_decision,
                        y_boundary_success,
                        selected,
                        selection_args,
                        rng,
                        decision_metric,
                    )
                    if len(batch) == 0:
                        break
                    for idx in batch.astype(int).tolist():
                        if idx not in selected:
                            selected.append(idx)
                    selected = selected[: selection_args.budget]
                    round_idx += 1
            print(f"  {decision_dim} repeat {repeat}: done")
    return pd.DataFrame(records), pd.DataFrame(cluster_records)


def main() -> None:
    args = parse_args()
    config = load_config(resolve(args.config))
    selected = set(args.only or [])
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_frames: list[pd.DataFrame] = []
    cluster_frames: list[pd.DataFrame] = []
    for task in config["tasks"]:
        if selected and task["name"] not in selected:
            continue
        detail, clusters = run_task(task, args.methods, args.repeats, args.eval_dim, args.seed)
        detail_frames.append(detail)
        cluster_frames.append(clusters)

    detail = pd.concat(detail_frames, ignore_index=True)
    cluster_detail = pd.concat(cluster_frames, ignore_index=True)
    summary = summarize_with_ci(detail, args.bootstrap_samples, int(args.seed or 42))
    pairwise = pairwise_deltas(detail, args.bootstrap_samples, int(args.seed or 42))
    cluster_summary = summarize_cluster_discovery(cluster_detail)

    detail.to_csv(out_dir / f"{args.out_prefix}_detail.csv", index=False)
    summary.to_csv(out_dir / f"{args.out_prefix}_summary.csv", index=False)
    pairwise.to_csv(out_dir / f"{args.out_prefix}_pairwise_deltas.csv", index=False)
    cluster_detail.to_csv(out_dir / f"{args.out_prefix}_cluster_detail.csv", index=False)
    cluster_summary.to_csv(out_dir / f"{args.out_prefix}_cluster_summary.csv", index=False)
    plot_pairwise(pairwise, out_dir, args.out_prefix)
    write_report(pairwise, cluster_summary, out_dir, args.out_prefix)

    print("\nSaved:")
    for suffix in ["detail.csv", "summary.csv", "pairwise_deltas.csv", "cluster_summary.csv", "report.md"]:
        print(f"  {out_dir / f'{args.out_prefix}_{suffix}'}")


if __name__ == "__main__":
    main()
