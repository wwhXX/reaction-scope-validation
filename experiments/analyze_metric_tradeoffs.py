# -*- coding: utf-8 -*-
"""Analyze cross-metric tradeoffs for labelled validation outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "failure_recall_mean",
    "failure_cluster_coverage_mean",
    "boundary_coverage_mean",
    "boundary_enrichment_mean",
    "coverage_radius_mean",
]

METRIC_LABELS = {
    "failure_recall_mean": "Failure recall",
    "failure_cluster_coverage_mean": "Failure-cluster coverage",
    "boundary_coverage_mean": "Boundary coverage",
    "boundary_enrichment_mean": "Boundary enrichment",
    "coverage_radius_mean": "Coverage radius",
    "compactness_score": "Coverage compactness",
}

MAXIMIZE_METRICS = {
    "failure_recall_mean": True,
    "failure_cluster_coverage_mean": True,
    "boundary_coverage_mean": True,
    "boundary_enrichment_mean": True,
    "coverage_radius_mean": False,
    "compactness_score": True,
}

PARETO_METRICS = [
    "failure_recall_mean",
    "failure_cluster_coverage_mean",
    "boundary_coverage_mean",
    "boundary_enrichment_mean",
    "compactness_score",
]

METHOD_COLORS = {
    "random": "#8C8C8C",
    "cvt": "#3B73B9",
    "weighted_itr_cvt": "#C23B22",
    "fps": "#2E8B57",
    "kennard_stone": "#7A5195",
    "ward": "#E17C05",
    "repulsive_fps": "#2A9D8F",
    "uncertainty": "#6A4C93",
    "uncertainty_diversity": "#D1495B",
}

METHOD_MARKERS = {
    "random": "o",
    "cvt": "s",
    "weighted_itr_cvt": "*",
    "fps": "^",
    "kennard_stone": "D",
    "ward": "P",
    "repulsive_fps": "X",
    "uncertainty": "v",
    "uncertainty_diversity": "h",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze labelled benchmark metric tradeoffs.")
    parser.add_argument("--summary-dir", default="experiments/results/summary")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--config", default="experiments/configs/prospective_boundary_simulation.json")
    parser.add_argument("--out-dir", default="experiments/results/summary")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def add_compactness(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "coverage_radius_mean" in out.columns:
        out["compactness_score"] = -out["coverage_radius_mean"].astype(float)
    return out


def method_space(row: pd.Series) -> str:
    return f"{row['method']} / {row['dimension']}"


def load_static_boundary(summary_dir: Path) -> pd.DataFrame:
    frame = read_csv_if_exists(summary_dir / "expanded_boundary_sampling_summary.csv")
    if frame.empty:
        return frame
    out = frame.copy()
    out["analysis_type"] = "static_boundary"
    out["budget_spent"] = out["budget"]
    return add_compactness(out)


def prospective_prefixes(config_path: Path) -> list[tuple[str, str, int | None]]:
    if not config_path.exists():
        return []
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    specs: list[tuple[str, str, int | None]] = []
    for task in config.get("tasks", []):
        params = task.get("params", {})
        prefix = str(params.get("out_prefix", task.get("name", "")))
        dataset = str(task.get("name", prefix)).replace("_prospective_boundary", "")
        budget = params.get("budget")
        specs.append((dataset, prefix, int(budget) if budget is not None else None))
    return specs


def load_prospective_final(results_dir: Path, config_path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = results_dir / "prospective_boundary_simulation"
    for dataset, prefix, config_budget in prospective_prefixes(config_path):
        path = base / f"{prefix}_summary.csv"
        frame = read_csv_if_exists(path)
        if frame.empty:
            continue
        final_budget = frame["budget_spent"].max()
        final = frame[frame["budget_spent"] == final_budget].copy()
        final["dataset"] = dataset
        final["budget"] = config_budget if config_budget is not None else final_budget
        final["analysis_type"] = "prospective_final"
        rows.append(final)
    if not rows:
        return pd.DataFrame()
    return add_compactness(pd.concat(rows, ignore_index=True))


def correlation_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [metric for metric in METRIC_COLUMNS if metric in frame.columns]
    for (analysis_type, dataset), group in frame.groupby(["analysis_type", "dataset"], dropna=False):
        if len(group) < 3:
            continue
        corr = group[metrics].corr(method="spearman")
        for metric_x in metrics:
            for metric_y in metrics:
                rows.append(
                    {
                        "analysis_type": analysis_type,
                        "dataset": dataset,
                        "metric_x": metric_x,
                        "metric_y": metric_y,
                        "spearman": corr.loc[metric_x, metric_y],
                    }
                )
    return pd.DataFrame(rows)


def top_by_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = [metric for metric in METRIC_COLUMNS if metric in frame.columns]
    for (analysis_type, dataset), group in frame.groupby(["analysis_type", "dataset"], dropna=False):
        for metric in metrics:
            maximize = MAXIMIZE_METRICS[metric]
            best = group.sort_values(metric, ascending=not maximize).iloc[0]
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "dataset": dataset,
                    "metric": metric,
                    "best_method": best["method"],
                    "best_dimension": best["dimension"],
                    "best_value": best[metric],
                    "best_method_space": method_space(best),
                }
            )
    return pd.DataFrame(rows)


def rank_conflict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        ("boundary_coverage_mean", "failure_cluster_coverage_mean"),
        ("boundary_coverage_mean", "failure_recall_mean"),
        ("boundary_coverage_mean", "coverage_radius_mean"),
        ("failure_cluster_coverage_mean", "coverage_radius_mean"),
    ]
    for (analysis_type, dataset), group in frame.groupby(["analysis_type", "dataset"], dropna=False):
        group = group.copy()
        for primary, secondary in comparisons:
            if primary not in group.columns or secondary not in group.columns:
                continue
            primary_best = group.sort_values(primary, ascending=not MAXIMIZE_METRICS[primary]).iloc[0]
            secondary_best = group.sort_values(secondary, ascending=not MAXIMIZE_METRICS[secondary]).iloc[0]
            secondary_rank = rank_of_row(group, primary_best, secondary)
            primary_rank = rank_of_row(group, secondary_best, primary)
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "dataset": dataset,
                    "primary_metric": primary,
                    "primary_best": method_space(primary_best),
                    "primary_best_value": primary_best[primary],
                    "primary_best_rank_on_secondary": secondary_rank,
                    "secondary_metric": secondary,
                    "secondary_best": method_space(secondary_best),
                    "secondary_best_value": secondary_best[secondary],
                    "secondary_best_rank_on_primary": primary_rank,
                    "same_winner": method_space(primary_best) == method_space(secondary_best),
                }
            )
    return pd.DataFrame(rows)


def rank_of_row(group: pd.DataFrame, row: pd.Series, metric: str) -> int:
    sorted_group = group.sort_values(metric, ascending=not MAXIMIZE_METRICS[metric]).reset_index(drop=True)
    matches = (sorted_group["method"].astype(str) == str(row["method"])) & (
        sorted_group["dimension"].astype(str) == str(row["dimension"])
    )
    return int(np.flatnonzero(matches.to_numpy())[0] + 1)


def pareto_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metrics = [metric for metric in PARETO_METRICS if metric in frame.columns]
    for (analysis_type, dataset), group in frame.groupby(["analysis_type", "dataset"], dropna=False):
        values = group[metrics].to_numpy(dtype=float)
        keep = []
        for idx, row in enumerate(values):
            dominated = False
            for other_idx, other in enumerate(values):
                if idx == other_idx:
                    continue
                if np.all(other >= row) and np.any(other > row):
                    dominated = True
                    break
            keep.append(not dominated)
        frontier = group.loc[keep].copy()
        frontier["pareto_metric_count"] = len(metrics)
        frontier["analysis_type"] = analysis_type
        frontier["dataset"] = dataset
        rows.append(frontier)
    if not rows:
        return pd.DataFrame()
    columns = [
        "analysis_type",
        "dataset",
        "method",
        "dimension",
        "budget",
        "budget_spent",
        *[column for column in METRIC_COLUMNS if column in frame.columns],
        "compactness_score",
        "pareto_metric_count",
    ]
    out = pd.concat(rows, ignore_index=True)
    return out[[column for column in columns if column in out.columns]]


def load_budget_trajectory(results_dir: Path, config_path: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    base = results_dir / "prospective_boundary_simulation"
    for dataset, prefix, config_budget in prospective_prefixes(config_path):
        path = base / f"{prefix}_summary.csv"
        frame = read_csv_if_exists(path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["dataset"] = dataset
        frame["configured_budget"] = config_budget if config_budget is not None else frame["budget_spent"].max()
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    return add_compactness(pd.concat(rows, ignore_index=True))


def budget_summary_rows(trajectory: pd.DataFrame) -> pd.DataFrame:
    if trajectory.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for dataset, group in trajectory.groupby("dataset"):
        final_budget = group["budget_spent"].max()
        final = group[group["budget_spent"] == final_budget].copy()
        best_final = final.sort_values("boundary_coverage_mean", ascending=False).iloc[0]
        mid_budget = sorted(group["budget_spent"].unique())[max(0, len(group["budget_spent"].unique()) // 2 - 1)]
        mid = group[group["budget_spent"] == mid_budget].sort_values("boundary_coverage_mean", ascending=False).iloc[0]
        threshold_hits = group[group["boundary_coverage_mean"] >= 0.10]
        if threshold_hits.empty:
            first_010 = np.nan
            first_010_winner = "not reached"
        else:
            hit = threshold_hits.sort_values(["budget_spent", "boundary_coverage_mean"], ascending=[True, False]).iloc[0]
            first_010 = float(hit["budget_spent"])
            first_010_winner = method_space(hit)
        rows.append(
            {
                "dataset": dataset,
                "configured_budget": best_final.get("configured_budget", final_budget),
                "final_budget": final_budget,
                "mid_budget": mid_budget,
                "best_mid_boundary": method_space(mid),
                "best_mid_boundary_coverage": mid["boundary_coverage_mean"],
                "best_final_boundary": method_space(best_final),
                "best_final_boundary_coverage": best_final["boundary_coverage_mean"],
                "first_budget_boundary_coverage_ge_0_10": first_010,
                "first_boundary_coverage_ge_0_10_method": first_010_winner,
            }
        )
    return pd.DataFrame(rows)


def plot_correlation_heatmap(correlations: pd.DataFrame, out_path: Path) -> None:
    subset = correlations[correlations["analysis_type"] == "static_boundary"].copy()
    datasets = ["aldol", "cobalt_condition1", "cobalt_threshold50"]
    subset = subset[subset["dataset"].isin(datasets)]
    if subset.empty:
        return

    metrics = METRIC_COLUMNS
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.45 * len(datasets), 4.05), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        data = subset[subset["dataset"] == dataset]
        matrix = data.pivot(index="metric_y", columns="metric_x", values="spearman").reindex(index=metrics, columns=metrics)
        im = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(dataset)
        ax.set_xticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(metrics)), [METRIC_LABELS[m] for m in metrics], fontsize=8)
        for row_idx in range(len(metrics)):
            for col_idx in range(len(metrics)):
                value = matrix.iloc[row_idx, col_idx]
                if pd.notna(value):
                    ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.subplots_adjust(left=0.08, right=0.90, bottom=0.27, top=0.82, wspace=0.34)
    colorbar_axis = fig.add_axes([0.925, 0.28, 0.015, 0.52])
    fig.colorbar(im, cax=colorbar_axis, label="Spearman correlation")
    fig.suptitle("Boundary metrics are not interchangeable", y=1.02)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto_scatter(frame: pd.DataFrame, pareto: pd.DataFrame, out_path: Path) -> None:
    subset = frame[
        (frame["analysis_type"] == "static_boundary")
        & (frame["dataset"].isin(["aldol", "cobalt_condition1", "cobalt_threshold50"]))
    ].copy()
    if subset.empty:
        return
    pareto_keys = set(
        zip(
            pareto["analysis_type"].astype(str),
            pareto["dataset"].astype(str),
            pareto["method"].astype(str),
            pareto["dimension"].astype(str),
        )
    )
    datasets = ["aldol", "cobalt_condition1", "cobalt_threshold50"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.15 * len(datasets), 3.95), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        data = subset[subset["dataset"] == dataset]
        for _, row in data.iterrows():
            method = str(row["method"])
            key = (str(row["analysis_type"]), str(row["dataset"]), method, str(row["dimension"]))
            is_pareto = key in pareto_keys
            ax.scatter(
                row["failure_cluster_coverage_mean"],
                row["boundary_coverage_mean"],
                s=125 if is_pareto else 44,
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "#555555"),
                edgecolor="black" if is_pareto else "white",
                linewidth=1.0 if is_pareto else 0.5,
                alpha=0.88 if is_pareto else 0.42,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Failure-cluster coverage")
        ax.set_ylabel("Boundary coverage")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(bottom=-0.01)
        ax.grid(alpha=0.22)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="white",
            label=method,
            markerfacecolor=METHOD_COLORS.get(method, "#555555"),
            markeredgecolor="white",
            linewidth=0,
            markersize=8,
        )
        for method, marker in METHOD_MARKERS.items()
        if method in set(subset["method"].astype(str))
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Pareto frontiers expose objective-dependent sampler choice", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_budget_trajectory(trajectory: pd.DataFrame, out_path: Path) -> None:
    if trajectory.empty:
        return
    datasets = list(trajectory["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 3.8), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        data = trajectory[trajectory["dataset"] == dataset]
        best = (
            data.sort_values("boundary_coverage_mean", ascending=False)
            .groupby("budget_spent", as_index=False, sort=True)
            .head(1)
            .sort_values("budget_spent")
        )
        ax.plot(best["budget_spent"], best["boundary_coverage_mean"], marker="o", linewidth=2.0, color="#2E8B57")
        for _, row in best.iterrows():
            ax.text(
                row["budget_spent"],
                row["boundary_coverage_mean"] + 0.005,
                str(row["dimension"]),
                ha="center",
                fontsize=7,
            )
        ax.set_title(dataset)
        ax.set_xlabel("Simulated assay budget")
        ax.set_ylabel("Best boundary coverage")
        ax.grid(alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Prospective simulations provide budget trajectories, not only final scores", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    out_dir: Path,
    correlations: pd.DataFrame,
    top_by_metric: pd.DataFrame,
    rank_conflicts: pd.DataFrame,
    pareto: pd.DataFrame,
    budget_summary: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
        if frame.empty:
            return "_No data._"
        shown = frame.copy()
        if max_rows is not None:
            shown = shown.head(max_rows)
        for column in shown.select_dtypes(include="number").columns:
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else f"{value:.4f}")
        shown = shown.astype(str)
        headers = list(shown.columns)
        rows = shown.values.tolist()
        widths = [
            max(len(header), *(len(row[col_idx]) for row in rows)) if rows else len(header)
            for col_idx, header in enumerate(headers)
        ]

        def format_row(values: list[str]) -> str:
            cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
            return "| " + " | ".join(cells) + " |"

        separator = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([format_row(headers), separator, *(format_row(row) for row in rows)])

    lines = [
        "# Cross-Metric Trade-Off Analysis",
        "",
        "This analysis uses labelled benchmark outputs. Article-test reference-only datasets are intentionally excluded.",
        "",
        "## Strongest Static Boundary Conflicts",
        "",
    ]
    conflict_subset = rank_conflicts[
        (rank_conflicts["analysis_type"] == "static_boundary")
        & (rank_conflicts["primary_metric"] == "boundary_coverage_mean")
        & (rank_conflicts["secondary_metric"] == "failure_cluster_coverage_mean")
    ].copy()
    if not conflict_subset.empty:
        lines.append(markdown_table(conflict_subset))
    lines.extend(["", "## Top Methods By Metric", ""])
    lines.append(markdown_table(top_by_metric, max_rows=40))
    lines.extend(["", "## Pareto Frontier Counts", ""])
    if not pareto.empty:
        counts = pareto.groupby(["analysis_type", "dataset"], as_index=False).size().rename(columns={"size": "pareto_count"})
        lines.append(markdown_table(counts))
    else:
        lines.append("_No data._")
    lines.extend(["", "## Budget Trajectory Summary", ""])
    lines.append(markdown_table(budget_summary))
    lines.extend(["", "## Correlation Rows", ""])
    lines.append(markdown_table(correlations, max_rows=80))
    (out_dir / "metric_tradeoff_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    summary_dir = resolve(root, args.summary_dir)
    results_dir = resolve(root, args.results_dir)
    config_path = resolve(root, args.config)
    out_dir = resolve(root, args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    static = load_static_boundary(summary_dir)
    prospective_final = load_prospective_final(results_dir, config_path)
    combined = pd.concat([frame for frame in [static, prospective_final] if not frame.empty], ignore_index=True)
    if combined.empty:
        raise FileNotFoundError("No labelled boundary/prospective summaries found.")

    correlations = correlation_rows(combined)
    top_by_metric = top_by_metric_rows(combined)
    rank_conflicts = rank_conflict_rows(combined)
    pareto = pareto_rows(combined)
    trajectory = load_budget_trajectory(results_dir, config_path)
    budget_summary = budget_summary_rows(trajectory)

    correlations.to_csv(out_dir / "metric_tradeoff_correlations.csv", index=False)
    top_by_metric.to_csv(out_dir / "metric_tradeoff_top_by_metric.csv", index=False)
    rank_conflicts.to_csv(out_dir / "metric_tradeoff_rank_conflicts.csv", index=False)
    pareto.to_csv(out_dir / "metric_tradeoff_pareto_frontier.csv", index=False)
    budget_summary.to_csv(out_dir / "budget_trajectory_summary.csv", index=False)

    plot_correlation_heatmap(correlations, out_dir / "metric_tradeoff_correlation_heatmap.png")
    plot_pareto_scatter(combined, pareto, out_dir / "metric_tradeoff_pareto_scatter.png")
    plot_budget_trajectory(trajectory, out_dir / "budget_trajectory_boundary_coverage.png")
    write_markdown(out_dir, correlations, top_by_metric, rank_conflicts, pareto, budget_summary)

    print(f"Saved trade-off tables and figures to {out_dir}")


if __name__ == "__main__":
    main()
