# -*- coding: utf-8 -*-
"""Plot boundary-sensitive sampling benchmark summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DIMS = ["2", "64", "128", "full"]
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot boundary-sensitive sampling benchmark summary CSV files.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--plot-prefix", required=True)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    return parser.parse_args()


def prepare(summary: pd.DataFrame, dim_order: list[str]) -> pd.DataFrame:
    summary = summary.copy()
    summary["dimension"] = pd.Categorical(summary["dimension"].astype(str), dim_order, ordered=True)
    return summary.sort_values(["method", "dimension"])


def plot_lines(summary: pd.DataFrame, metric: str, ylabel: str, title: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for method, group in summary.groupby("method", sort=False):
        group = group.sort_values("dimension")
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        color = COLORS.get(method)
        ax.plot(
            group["dimension"].astype(str),
            group[mean_col],
            marker="o",
            linewidth=2,
            label=method,
            color=color,
        )
        low_col = f"{metric}_ci_low"
        high_col = f"{metric}_ci_high"
        if low_col in group and high_col in group:
            ax.fill_between(
                group["dimension"].astype(str),
                group[low_col],
                group[high_col],
                color=color,
                alpha=0.10,
            )
        elif std_col in group:
            y = group[mean_col]
            yerr = group[std_col].fillna(0)
            ax.fill_between(group["dimension"].astype(str), y - yerr, y + yerr, color=color, alpha=0.10)

    ax.set_title(title)
    ax.set_xlabel("Representation dimension")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dim_order = [dim.lower() for dim in args.dims]
    summary = prepare(pd.read_csv(f"{args.prefix}_summary.csv"), dim_order)

    plots = [
        (
            "failure_recall",
            "Failure recall",
            "How much of the failed-substrate set was sampled",
            "failure_recall",
        ),
        (
            "failure_cluster_coverage",
            "Failure-cluster coverage",
            "Coverage of distinct failed-substrate clusters",
            "failure_cluster_coverage",
        ),
        (
            "boundary_coverage",
            "Boundary coverage",
            "Coverage of near-boundary substrates",
            "boundary_coverage",
        ),
        (
            "boundary_enrichment",
            "Boundary enrichment",
            "Boundary enrichment relative to random population rate",
            "boundary_enrichment",
        ),
        (
            "coverage_radius",
            "Mean distance to nearest sampled point",
            "Global coverage radius, lower is better",
            "coverage_radius",
        ),
    ]

    for metric, ylabel, title, suffix in plots:
        plot_lines(
            summary,
            metric=metric,
            ylabel=ylabel,
            title=title,
            output_stem=f"{args.plot_prefix}_{suffix}",
        )

    print("Saved plots:")
    for path in sorted(Path(".").glob(f"{args.plot_prefix}_*.*")):
        if path.suffix.lower() in {".png", ".svg"}:
            print(f"  {path}")


if __name__ == "__main__":
    main()
