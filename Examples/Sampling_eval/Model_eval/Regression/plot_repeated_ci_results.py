# -*- coding: utf-8 -*-
"""Plot repeated benchmark summaries with bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".matplotlib-cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIMS = ["2", "64", "128", "full"]
COLORS = {
    "DummyMeanRegressor": "#7f7f7f",
    "DummyMostFrequentClassifier": "#7f7f7f",
    "RandomForestRegressor": "#1f77b4",
    "RandomForestClassifier": "#1f77b4",
    "HistGradientBoostingRegressor": "#d62728",
    "HistGradientBoostingClassifier": "#d62728",
    "ExtraTreesRegressor": "#2ca02c",
    "ExtraTreesClassifier": "#2ca02c",
    "XGBRegressor": "#9467bd",
    "XGBClassifier": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot repeated benchmark CI summary CSV files.")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--plot-prefix", required=True)
    parser.add_argument("--dims", nargs="+", default=DEFAULT_DIMS)
    return parser.parse_args()


def prepare(summary: pd.DataFrame, dim_order: list[str]) -> pd.DataFrame:
    summary = summary.copy()
    summary["dimension"] = pd.Categorical(summary["dimension"].astype(str), dim_order, ordered=True)
    return summary.sort_values(["model", "dimension"])


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, title: str, output_stem: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    for model, group in summary.groupby("model", sort=False):
        group = group.sort_values("dimension")
        x = np.arange(len(group))
        y = group[f"{metric}_mean"].to_numpy(dtype=float)
        low = group[f"{metric}_ci_low"].to_numpy(dtype=float)
        high = group[f"{metric}_ci_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - low, high - y])
        color = COLORS.get(model)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            capsize=4,
            linewidth=2,
            label=model,
            color=color,
        )

    ax.set_xticks(np.arange(len(summary["dimension"].cat.categories)))
    ax.set_xticklabels(summary["dimension"].cat.categories)
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
    reg = prepare(pd.read_csv(f"{args.prefix}_regression_summary_ci.csv"), dim_order)
    cls = prepare(pd.read_csv(f"{args.prefix}_classification_summary_ci.csv"), dim_order)

    plot_metric(
        reg,
        metric="r2",
        ylabel="Test R2, bootstrap 95% CI",
        title="Repeated regression R2",
        output_stem=f"{args.plot_prefix}_regression_r2_ci",
    )
    plot_metric(
        reg,
        metric="rmse",
        ylabel="Test RMSE, bootstrap 95% CI",
        title="Repeated regression RMSE",
        output_stem=f"{args.plot_prefix}_regression_rmse_ci",
    )
    plot_metric(
        cls,
        metric="balanced_accuracy",
        ylabel="Balanced accuracy, bootstrap 95% CI",
        title="Repeated balanced accuracy",
        output_stem=f"{args.plot_prefix}_classification_balanced_accuracy_ci",
    )
    plot_metric(
        cls,
        metric="mcc",
        ylabel="MCC, bootstrap 95% CI",
        title="Repeated Matthews correlation coefficient",
        output_stem=f"{args.plot_prefix}_classification_mcc_ci",
    )
    plot_metric(
        cls,
        metric="roc_auc",
        ylabel="ROC-AUC, bootstrap 95% CI",
        title="Repeated ROC-AUC",
        output_stem=f"{args.plot_prefix}_classification_auc_ci",
    )

    print("Saved plots:")
    for path in sorted(Path(".").glob(f"{args.plot_prefix}_*_ci.*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
