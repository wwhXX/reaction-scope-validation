# -*- coding: utf-8 -*-
"""Plot boundary coverage versus failure-cluster coverage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DIMENSION_COLORS = {
    "2": "#7f7f7f",
    "8": "#1f77b4",
    "16": "#ff7f0e",
    "64": "#2ca02c",
    "128": "#d62728",
    "full": "#9467bd",
    "tanimoto": "#17becf",
}
METHOD_MARKERS = {
    "random": "o",
    "cvt": "s",
    "weighted_itr_cvt": "*",
    "fps": "^",
    "kennard_stone": "D",
    "ward": "P",
    "repulsive_fps": "X",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot sampling-objective tradeoffs.")
    parser.add_argument("--summary", required=True, help="Expanded boundary summary CSV.")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--datasets", nargs="+", help="Optional dataset labels to include.")
    return parser.parse_args()


def annotate_extremes(ax: plt.Axes, frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    best_boundary = frame.loc[frame["boundary_coverage_mean"].idxmax()]
    best_cluster = frame.loc[frame["failure_cluster_coverage_mean"].idxmax()]
    for row, label in [(best_boundary, "best boundary"), (best_cluster, "best clusters")]:
        ax.annotate(
            f"{label}\n{row['method']} / {row['dimension']}",
            (row["failure_cluster_coverage_mean"], row["boundary_coverage_mean"]),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "alpha": 0.6},
        )


def plot_tradeoff(summary: pd.DataFrame, out_prefix: str) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 5.4), squeeze=False)
    for ax, dataset in zip(axes.ravel(), datasets):
        frame = summary[summary["dataset"] == dataset].copy()
        for _, row in frame.iterrows():
            method = str(row["method"])
            dimension = str(row["dimension"])
            ax.scatter(
                row["failure_cluster_coverage_mean"],
                row["boundary_coverage_mean"],
                s=130 if method == "weighted_itr_cvt" else 70,
                marker=METHOD_MARKERS.get(method, "o"),
                color=DIMENSION_COLORS.get(dimension, "#333333"),
                edgecolor="black" if method == "weighted_itr_cvt" else "white",
                linewidth=0.8,
                alpha=0.88,
            )
        annotate_extremes(ax, frame)
        ax.set_title(dataset)
        ax.set_xlabel("Failure-cluster coverage")
        ax.set_ylabel("Near-boundary coverage")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(bottom=-0.01)
        ax.grid(alpha=0.25)

    method_handles = [
        plt.Line2D(
            [0],
            [0],
            marker=marker,
            color="white",
            label=method,
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
        )
        for method, marker in METHOD_MARKERS.items()
        if method in set(summary["method"].astype(str))
    ]
    dim_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label=dimension,
            markerfacecolor=color,
            markeredgecolor="white",
            markersize=8,
            linewidth=0,
        )
        for dimension, color in DIMENSION_COLORS.items()
        if dimension in set(summary["dimension"].astype(str))
    ]
    fig.legend(handles=method_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.legend(handles=dim_handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Sampling objectives separate: failure discovery versus boundary coverage", y=1.10)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{out_prefix}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    summary = pd.read_csv(args.summary)
    if args.datasets:
        summary = summary[summary["dataset"].isin(args.datasets)].copy()
    plot_tradeoff(summary, args.out_prefix)
    print("Saved:")
    print(f"  {args.out_prefix}.png")
    print(f"  {args.out_prefix}.svg")


if __name__ == "__main__":
    main()
