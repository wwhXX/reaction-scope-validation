# -*- coding: utf-8 -*-
"""Assemble manuscript-facing Chemical Science sprint outputs.

This script does not rerun experiments. It converts the completed r20 paired
contrast and failure-cluster explanation files into compact tables, a combined
figure, and a short status report for manuscript triage.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
OUT_DIR = ROOT / "experiments" / "results" / "chemical_science_sprint"
PAIRWISE = OUT_DIR / "decision_space_contrast_r20_pairwise_deltas.csv"
CLUSTER_SUMMARY = OUT_DIR / "failure_cluster_explanations_summary.csv"
REPRESENTATIVES = OUT_DIR / "failure_cluster_explanations_representatives.csv"
ALDOL_STRUCTURES = OUT_DIR / "failure_cluster_explanations_aldol_structures.png"
COBALT_STRUCTURES = OUT_DIR / "failure_cluster_explanations_cobalt_structures.png"

KEY_DELTAS_OUT = OUT_DIR / "chemical_science_key_boundary_deltas.csv"
FAMILY_TABLE_OUT = OUT_DIR / "chemical_science_failure_family_table.csv"
MAIN_FIGURE_PNG = OUT_DIR / "chemical_science_main_experiment_package.png"
MAIN_FIGURE_SVG = OUT_DIR / "chemical_science_main_experiment_package.svg"
REPORT_OUT = OUT_DIR / "chemical_science_experiment_package_2026-06-30.md"


KEY_DELTA_ROWS = [
    ("cobalt", "weighted_itr_cvt", "tanimoto"),
    ("cobalt", "cvt", "tanimoto"),
    ("cobalt", "weighted_itr_cvt", "full"),
    ("aldol", "weighted_itr_cvt", "full"),
    ("aldol", "cvt", "full"),
    ("buchwald_hartwig", "uncertainty", "32"),
]

FAMILY_LABELS = {
    ("aldol", 1): "Bulky or alkoxy aryl aldehydes",
    ("aldol", 2): "Nitro or EWG aryl aldehydes",
    ("aldol", 3): "Sulfonyl or pyridyl aldehydes",
    ("aldol", 5): "Dimethylamino/anilino aldehydes",
    ("aldol", 7): "Mixed heteroaryl, halogenated, CF3 aldehydes",
    ("cobalt", 0): "Bulky carbamate cyclohexanol",
    ("cobalt", 1): "Diverse alcohol failures",
    ("cobalt", 2): "Nitro heteroaryl chlorohydrin",
    ("cobalt", 3): "Sulfonyl alcohol",
    ("buchwald_hartwig", 0): "Component-encoded brominated aryl halide family",
    ("buchwald_hartwig", 1): "Component-encoded iodinated aryl halide family",
    ("buchwald_hartwig", 2): "Component-encoded chlorinated aryl halide family",
}


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def fmt_ci(mean: float, low: float, high: float) -> str:
    return f"{mean:+.4f} [{low:+.4f}, {high:+.4f}]"


def build_key_deltas(pairwise: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, method, high_dim in KEY_DELTA_ROWS:
        match = pairwise[
            (pairwise["dataset"] == dataset)
            & (pairwise["method"] == method)
            & (pairwise["high_dimension"].astype(str) == high_dim)
        ]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "high_dimension_vs_2d": high_dim,
                "paired_repeats": int(row["paired_repeats"]),
                "boundary_coverage_delta_mean": float(row["boundary_coverage_delta_mean"]),
                "boundary_coverage_delta_ci_low": float(row["boundary_coverage_delta_ci_low"]),
                "boundary_coverage_delta_ci_high": float(row["boundary_coverage_delta_ci_high"]),
                "failure_cluster_coverage_delta_mean": float(row["failure_cluster_coverage_delta_mean"]),
                "failure_recall_delta_mean": float(row["failure_recall_delta_mean"]),
                "manuscript_read": fmt_ci(
                    float(row["boundary_coverage_delta_mean"]),
                    float(row["boundary_coverage_delta_ci_low"]),
                    float(row["boundary_coverage_delta_ci_high"]),
                ),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(KEY_DELTAS_OUT, index=False)
    return out


def representative_examples(reps: pd.DataFrame, dataset: str, cluster_id: int, max_items: int = 3) -> str:
    subset = reps[(reps["dataset"] == dataset) & (reps["failure_cluster"] == cluster_id)].head(max_items)
    examples = []
    for _, row in subset.iterrows():
        value = row.get("target_value", np.nan)
        smiles = row.get("smiles")
        if not isinstance(smiles, str) or not smiles:
            smiles = row.get("SMILES")
        if not isinstance(smiles, str) or not smiles:
            smiles = str(row.get("reaction_id", "component encoded"))
        if isinstance(value, (int, float, np.floating)) and not pd.isna(value):
            examples.append(f"{smiles} (y={value:.1f})")
        else:
            examples.append(str(smiles))
    return " | ".join(examples)


def build_family_table(summary: pd.DataFrame, reps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, cluster_id), label in FAMILY_LABELS.items():
        match = summary[(summary["dataset"] == dataset) & (summary["failure_cluster"] == cluster_id)]
        if match.empty:
            continue
        row = match.iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "failure_cluster": int(cluster_id),
                "manuscript_family_label": label,
                "cluster_size": int(row["cluster_size"]),
                "target_range": f"{float(row['target_min']):.1f}-{float(row['target_max']):.1f}",
                "target_median": float(row["target_median"]),
                "auto_tag_summary": row.get("structure_tag_summary", ""),
                "component_enrichment_summary": row.get("component_enrichment_summary", ""),
                "representative_examples": representative_examples(reps, dataset, int(cluster_id)),
                "review_status": "heuristic label; chemistry review required",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FAMILY_TABLE_OUT, index=False)
    return out


def make_label(row: pd.Series) -> str:
    dataset = {
        "aldol": "Aldol",
        "cobalt": "Cobalt",
        "buchwald_hartwig": "Buchwald-Hartwig",
    }.get(str(row["dataset"]), str(row["dataset"]))
    method = str(row["method"]).replace("_", " ")
    return f"{dataset}\n{method}\n{row['high_dimension_vs_2d']} vs 2D"


def build_figure(key_deltas: pd.DataFrame, family_table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.9, 1.1], width_ratios=[1.15, 1.0])

    ax_bar = fig.add_subplot(grid[0, 0])
    shown = key_deltas.sort_values("boundary_coverage_delta_mean", ascending=True)
    colors = ["#4575b4" if ds == "cobalt" else "#74add1" if ds == "aldol" else "#fdae61" for ds in shown["dataset"]]
    y = np.arange(len(shown))
    ax_bar.barh(y, shown["boundary_coverage_delta_mean"], color=colors, height=0.68)
    ax_bar.errorbar(
        shown["boundary_coverage_delta_mean"],
        y,
        xerr=[
            shown["boundary_coverage_delta_mean"] - shown["boundary_coverage_delta_ci_low"],
            shown["boundary_coverage_delta_ci_high"] - shown["boundary_coverage_delta_mean"],
        ],
        fmt="none",
        ecolor="#333333",
        elinewidth=1.0,
        capsize=3,
    )
    ax_bar.axvline(0, color="#333333", linewidth=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([make_label(row) for _, row in shown.iterrows()], fontsize=8)
    ax_bar.set_xlabel("Final-budget boundary coverage delta vs 2D")
    ax_bar.set_title("A. Paired r20 decision-space contrast", loc="left", fontweight="bold")
    ax_bar.grid(axis="x", color="#dddddd", linewidth=0.6)

    ax_table = fig.add_subplot(grid[0, 1])
    ax_table.axis("off")
    focus = family_table[family_table["dataset"].isin(["aldol", "cobalt"])].copy()
    focus = focus.sort_values(["dataset", "failure_cluster"]).head(9)
    cell_text = []
    for _, row in focus.iterrows():
        cell_text.append(
            [
                row["dataset"].replace("_", " ").title(),
                f"C{int(row['failure_cluster'])}",
                row["manuscript_family_label"],
                str(row["cluster_size"]),
            ]
        )
    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Dataset", "Cluster", "Heuristic family", "n"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.18, 0.13, 0.56, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if r == 0:
            cell.set_facecolor("#eeeeee")
            cell.set_text_props(weight="bold")
    ax_table.set_title("B. Failure-family annotation targets", loc="left", fontweight="bold")

    ax_aldol = fig.add_subplot(grid[1, 0])
    if ALDOL_STRUCTURES.exists():
        ax_aldol.imshow(mpimg.imread(ALDOL_STRUCTURES))
    ax_aldol.axis("off")
    ax_aldol.set_title("C. Aldol representative failed aldehydes", loc="left", fontweight="bold")

    ax_cobalt = fig.add_subplot(grid[1, 1])
    if COBALT_STRUCTURES.exists():
        ax_cobalt.imshow(mpimg.imread(COBALT_STRUCTURES))
    ax_cobalt.axis("off")
    ax_cobalt.set_title("D. Cobalt representative zero-yield alcohols", loc="left", fontweight="bold")

    fig.suptitle(
        "Chemical Science sprint package: decision-space gains and interpretable failure families",
        fontsize=14,
        fontweight="bold",
    )
    fig.savefig(MAIN_FIGURE_PNG, dpi=220)
    fig.savefig(MAIN_FIGURE_SVG)
    plt.close(fig)


def write_report(key_deltas: pd.DataFrame, family_table: pd.DataFrame) -> None:
    strongest = key_deltas.sort_values("boundary_coverage_delta_mean", ascending=False)
    aldol_families = family_table[family_table["dataset"] == "aldol"]
    cobalt_families = family_table[family_table["dataset"] == "cobalt"]

    lines = [
        "# Chemical Science Experiment Package, 2026-06-30",
        "",
        "This package assembles the completed r20 paired decision-space contrast and",
        "failure-cluster explanation outputs into manuscript-facing artifacts. It",
        "does not rerun the expensive experiments.",
        "",
        "## Generated Artifacts",
        "",
        "```text",
        str(KEY_DELTAS_OUT.relative_to(ROOT)),
        str(FAMILY_TABLE_OUT.relative_to(ROOT)),
        str(MAIN_FIGURE_PNG.relative_to(ROOT)),
        str(MAIN_FIGURE_SVG.relative_to(ROOT)),
        "```",
        "",
        "## Strongest Paired Decision-Space Signals",
        "",
    ]

    for _, row in strongest.iterrows():
        lines.append(
            "- "
            + f"{row['dataset']} / {row['method']} / {row['high_dimension_vs_2d']} vs 2D: "
            + row["manuscript_read"]
        )

    lines.extend(
        [
            "",
            "## Chemistry-Facing Failure Families",
            "",
            "Aldol families prepared for manual checking:",
            "",
        ]
    )
    for _, row in aldol_families.iterrows():
        lines.append(
            "- "
            + f"cluster {int(row['failure_cluster'])}: {row['manuscript_family_label']} "
            + f"(n={int(row['cluster_size'])}, target {row['target_range']})"
        )

    lines.extend(["", "Cobalt families prepared for manual checking:", ""])
    for _, row in cobalt_families.iterrows():
        lines.append(
            "- "
            + f"cluster {int(row['failure_cluster'])}: {row['manuscript_family_label']} "
            + f"(n={int(row['cluster_size'])}, target {row['target_range']})"
        )

    lines.extend(
        [
            "",
            "## Manuscript Use",
            "",
            "The current experiment package supports the claim that Aldol and Cobalt",
            "show chemically meaningful decision-space failures when 2D maps are used",
            "as acquisition spaces. Buchwald-Hartwig remains useful as an external",
            "full-label support dataset, but it should not be the lead structure figure",
            "until drawable component structures are recovered.",
            "",
            "## Remaining Risk",
            "",
            "The family labels are heuristic annotations derived from SMILES tags and",
            "component enrichment. They are ready for chemistry review, not final",
            "mechanistic claims.",
            "",
        ]
    )
    REPORT_OUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    pairwise = read_required_csv(PAIRWISE)
    summary = read_required_csv(CLUSTER_SUMMARY)
    reps = read_required_csv(REPRESENTATIVES)
    key_deltas = build_key_deltas(pairwise)
    family_table = build_family_table(summary, reps)
    build_figure(key_deltas, family_table)
    write_report(key_deltas, family_table)
    print(f"Wrote {KEY_DELTAS_OUT}")
    print(f"Wrote {FAMILY_TABLE_OUT}")
    print(f"Wrote {MAIN_FIGURE_PNG}")
    print(f"Wrote {MAIN_FIGURE_SVG}")
    print(f"Wrote {REPORT_OUT}")


if __name__ == "__main__":
    main()
