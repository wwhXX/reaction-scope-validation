# -*- coding: utf-8 -*-
"""Plot dimension-aware ScopeMap benchmark summaries."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_PREFIX = "scope_dimension_model_comparison"
DEFAULT_DIMS = ["2", "64", "128", "full"]
COLORS = {
    "DummyMeanRegressor": "#7f7f7f",
    "DummyMostFrequentClassifier": "#7f7f7f",
    "RandomForestRegressor": "#1f77b4",
    "RandomForestClassifier": "#1f77b4",
    "ExtraTreesRegressor": "#2ca02c",
    "ExtraTreesClassifier": "#2ca02c",
    "HistGradientBoostingRegressor": "#d62728",
    "HistGradientBoostingClassifier": "#d62728",
    "XGBRegressor": "#9467bd",
    "XGBClassifier": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot dimension benchmark summary CSV files.")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--plot-prefix", default="scope_dimension")
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    return parser.parse_args()


def prepare(summary: pd.DataFrame, dim_order: list[str]) -> pd.DataFrame:
    summary = summary.copy()
    summary["dimension"] = pd.Categorical(summary["dimension"].astype(str), dim_order, ordered=True)
    return summary.sort_values(["model", "dimension"])


def plot_lines(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_stem: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for model, group in summary.groupby("model", sort=False):
        group = group.sort_values("dimension")
        color = COLORS.get(model)
        ax.plot(
            group["dimension"].astype(str),
            group[f"{metric}_mean"],
            marker="o",
            linewidth=2,
            label=model,
            color=color,
        )
        std_col = f"{metric}_std"
        if std_col in group:
            y = group[f"{metric}_mean"]
            yerr = group[std_col].fillna(0)
            ax.fill_between(group["dimension"].astype(str), y - yerr, y + yerr, color=color, alpha=0.12)

    ax.set_title(title)
    ax.set_xlabel("Representation dimension")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{output_stem}.png", dpi=300)
    fig.savefig(f"{output_stem}.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dim_order = [dim.lower() for dim in args.dims]
    reg = prepare(pd.read_csv(f"{args.prefix}_regression_summary.csv"), dim_order)
    cls = prepare(pd.read_csv(f"{args.prefix}_classification_summary.csv"), dim_order)

    plot_lines(
        reg,
        metric="r2",
        ylabel="Test R2",
        title="Regression performance across representation dimensions",
        output_stem=f"{args.plot_prefix}_regression_r2",
    )
    plot_lines(
        reg,
        metric="rmse",
        ylabel="Test RMSE",
        title="Regression error across representation dimensions",
        output_stem=f"{args.plot_prefix}_regression_rmse",
    )
    plot_lines(
        cls,
        metric="roc_auc",
        ylabel="Test ROC-AUC",
        title="Classification ROC-AUC across representation dimensions",
        output_stem=f"{args.plot_prefix}_classification_auc",
    )
    plot_lines(
        cls,
        metric="balanced_accuracy",
        ylabel="Test balanced accuracy",
        title="Balanced accuracy exposes class-imbalance effects",
        output_stem=f"{args.plot_prefix}_classification_balanced_accuracy",
    )
    plot_lines(
        cls,
        metric="negative_recall_specificity",
        ylabel="Negative recall / specificity",
        title="Failure-class recall across representation dimensions",
        output_stem=f"{args.plot_prefix}_classification_specificity",
    )
    plot_lines(
        cls,
        metric="mcc",
        ylabel="Matthews correlation coefficient",
        title="MCC across representation dimensions",
        output_stem=f"{args.plot_prefix}_classification_mcc",
    )

    outputs = sorted(Path(".").glob(f"{args.plot_prefix}_*.*"))
    print("Saved plots:")
    for path in outputs:
        if path.suffix.lower() in {".png", ".svg"}:
            print(f"  {path}")


if __name__ == "__main__":
    main()
