# -*- coding: utf-8 -*-
"""Plot batch-by-batch retrospective prospective simulation curves."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DIMS = ["2", "64", "128", "full", "tanimoto"]
COLORS = {
    "random": "#7f7f7f",
    "cvt": "#1f77b4",
    "weighted_itr_cvt": "#bcbd22",
    "fps": "#2ca02c",
    "kennard_stone": "#d62728",
    "ward": "#9467bd",
    "lhs": "#8c564b",
    "sobol": "#e377c2",
    "repulsive_fps": "#17becf",
    "uncertainty": "#ff7f0e",
    "uncertainty_diversity": "#e377c2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot prospective boundary simulation summaries.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--plot-prefix", required=True)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    return parser.parse_args()


def prepare(summary: pd.DataFrame, dim_order: list[str]) -> pd.DataFrame:
    summary = summary.copy()
    summary["dimension"] = summary["dimension"].astype(str)
    summary["dimension"] = pd.Categorical(summary["dimension"], dim_order, ordered=True)
    return summary.sort_values(["dimension", "method", "budget_spent"])


def plot_metric_for_dimension(
    summary: pd.DataFrame,
    dimension: str,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: str,
) -> None:
    mean_col = f"{metric}_mean"
    if mean_col not in summary.columns:
        return
    dim_summary = summary[summary["dimension"].astype(str) == dimension]
    if dim_summary.empty:
        return

    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for method, group in dim_summary.groupby("method", sort=False):
        group = group.sort_values("budget_spent")
        color = COLORS.get(method)
        ax.plot(
            group["budget_spent"],
            group[mean_col],
            marker="o",
            linewidth=2,
            markersize=4,
            label=method,
            color=color,
        )
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        if low_col in group.columns and high_col in group.columns:
            ax.fill_between(group["budget_spent"], group[low_col], group[high_col], color=color, alpha=0.10)

    ax.set_title(title)
    ax.set_xlabel("Simulated experimental budget")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.svg")
    plt.close(fig)


def plot_discovery_times(prefix: str, plot_prefix: str) -> None:
    path = Path(f"{prefix}_discovery_times_summary.csv")
    if not path.exists():
        return
    data = pd.read_csv(path)
    data = data[data["metric"].isin(["failure_cluster_coverage", "boundary_coverage"])]
    if data.empty:
        return

    for metric, metric_data in data.groupby("metric", sort=False):
        for threshold, group in metric_data.groupby("threshold", sort=False):
            shown = group.dropna(subset=["first_budget_mean"])
            if shown.empty:
                continue
            labels = [f"{row.method}\n{row.dimension}" for row in shown.itertuples()]
            fig, ax = plt.subplots(figsize=(max(8.8, len(labels) * 0.42), 5.2))
            ax.bar(labels, shown["first_budget_mean"], color=[COLORS.get(m, "#7f7f7f") for m in shown["method"]])
            ax.set_ylabel("Mean first budget reached")
            ax.set_title(f"Budget to reach {metric} >= {threshold:g}")
            ax.tick_params(axis="x", labelrotation=75, labelsize=8)
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            stem = f"{plot_prefix}_{metric}_threshold{str(threshold).replace('.', 'p')}_first_budget"
            fig.savefig(f"{stem}.png", dpi=300)
            fig.savefig(f"{stem}.svg")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    dim_order = [dim.lower() for dim in args.dims]
    summary = prepare(pd.read_csv(f"{args.prefix}_summary.csv"), dim_order)
    plots = [
        (
            "failure_cluster_coverage",
            "Failure-cluster coverage",
            "How quickly simulated campaigns discover distinct failure modes",
            "failure_cluster_coverage",
        ),
        (
            "boundary_coverage",
            "Boundary coverage",
            "How quickly simulated campaigns sample near-boundary substrates",
            "boundary_coverage",
        ),
        (
            "failure_recall",
            "Failure recall",
            "How much of the failed-substrate set has been sampled",
            "failure_recall",
        ),
        (
            "boundary_enrichment",
            "Boundary enrichment",
            "Near-boundary enrichment relative to the population rate",
            "boundary_enrichment",
        ),
        (
            "coverage_radius",
            "Mean distance to nearest sampled point",
            "Global coverage radius, lower is better",
            "coverage_radius",
        ),
    ]
    for dim in dim_order:
        for metric, ylabel, title, suffix in plots:
            plot_metric_for_dimension(
                summary,
                dim,
                metric=metric,
                ylabel=ylabel,
                title=f"{title} ({dim})",
                output_stem=f"{args.plot_prefix}_{dim}_{suffix}",
            )

    plot_discovery_times(args.prefix, args.plot_prefix)

    print("Saved plots:")
    for path in sorted(Path(".").glob(f"{args.plot_prefix}_*.*")):
        if path.suffix.lower() in {".png", ".svg"}:
            print(f"  {path}")


if __name__ == "__main__":
    main()
