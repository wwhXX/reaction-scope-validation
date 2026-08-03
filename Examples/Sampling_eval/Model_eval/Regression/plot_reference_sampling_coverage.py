# -*- coding: utf-8 -*-
"""Plot reference sampling coverage benchmark summaries."""

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
    "scopemap_hd": "#bcbd22",
    "fps": "#2ca02c",
    "kennard_stone": "#d62728",
    "ward": "#9467bd",
    "lhs": "#8c564b",
    "sobol": "#e377c2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reference sampling coverage summary CSV files.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--plot-prefix", required=True)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    return parser.parse_args()


def prepare(summary: pd.DataFrame, dim_order: list[str]) -> pd.DataFrame:
    summary = summary.copy()
    summary["dimension"] = pd.Categorical(summary["dimension"].astype(str), dim_order, ordered=True)
    return summary.sort_values(["method", "dimension"])


def plot_lines(summary: pd.DataFrame, metric: str, ylabel: str, title: str, output_stem: str) -> None:
    mean_col = f"{metric}_mean"
    if mean_col not in summary.columns:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for method, group in summary.groupby("method", sort=False):
        group = group.sort_values("dimension")
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
            "reference_coverage_q10",
            "Reference coverage at space q10",
            "Reference set covered by the closest 10% of the sampled-space distance distribution",
            "reference_coverage_q10",
        ),
        (
            "reference_enrichment_q10",
            "Reference enrichment at q10",
            "Reference coverage relative to the 10% space baseline",
            "reference_enrichment_q10",
        ),
        (
            "reference_mean_nearest_sample_distance",
            "Mean reference-to-sample distance",
            "Distance from reference substrates to nearest sampled point, lower is better",
            "reference_mean_nearest_sample_distance",
        ),
        (
            "reference_exact_recall",
            "Exact reference recall",
            "Fraction of reference SMILES directly selected",
            "reference_exact_recall",
        ),
        (
            "coverage_radius",
            "Mean space-to-sample distance",
            "Global coverage radius, lower is better",
            "coverage_radius",
        ),
    ]
    for metric, ylabel, title, suffix in plots:
        plot_lines(summary, metric, ylabel, title, f"{args.plot_prefix}_{suffix}")

    print("Saved plots:")
    for path in sorted(Path(".").glob(f"{args.plot_prefix}_*.*")):
        if path.suffix.lower() in {".png", ".svg"}:
            print(f"  {path}")


if __name__ == "__main__":
    main()
