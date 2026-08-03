# -*- coding: utf-8 -*-
"""Summarize reaction-scope validation outputs into compact report tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generated benchmark CSV files.")
    parser.add_argument("--results-dir", default="experiments/results")
    parser.add_argument("--out-dir", default="experiments/results/summary")
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def add_threshold(frame: pd.DataFrame, threshold: int | str) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "threshold", threshold)
    return out


def summarize_repeated_ci(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        (
            "condition1",
            results_dir / "cobalt_core" / "cobalt_condition1_repeated100_minimal_ci_classification_summary_ci.csv",
        ),
        (
            1,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold1_repeated100_minimal_ci_classification_summary_ci.csv",
        ),
        (
            30,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold30_repeated100_minimal_ci_classification_summary_ci.csv",
        ),
        (
            50,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold50_repeated100_minimal_ci_classification_summary_ci.csv",
        ),
    ]
    for threshold, path in specs:
        if path.exists():
            frame = read_csv(path)
            frame = frame[frame["model"] == "RandomForestClassifier"]
            keep = [
                "dimension",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "mcc_mean",
                "mcc_ci_low",
                "mcc_ci_high",
                "roc_auc_mean",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
            ]
            rows.append(add_threshold(frame[keep], threshold))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_regression_ci(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    specs = [
        (
            "condition1",
            results_dir / "cobalt_core" / "cobalt_condition1_repeated100_minimal_ci_regression_summary_ci.csv",
        ),
        (
            1,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold1_repeated100_minimal_ci_regression_summary_ci.csv",
        ),
        (
            30,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold30_repeated100_minimal_ci_regression_summary_ci.csv",
        ),
        (
            50,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold50_repeated100_minimal_ci_regression_summary_ci.csv",
        ),
    ]
    for threshold, path in specs:
        if path.exists():
            frame = read_csv(path)
            frame = frame[frame["model"] == "RandomForestRegressor"]
            keep = ["dimension", "r2_mean", "r2_ci_low", "r2_ci_high", "rmse_mean", "rmse_ci_low", "rmse_ci_high"]
            rows.append(add_threshold(frame[keep], threshold))
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def best_by_metric(frame: pd.DataFrame, group_cols: list[str], metric: str, maximize: bool = True) -> pd.DataFrame:
    if frame.empty:
        return frame
    sorted_frame = frame.sort_values(metric, ascending=not maximize)
    return sorted_frame.groupby(group_cols, as_index=False, sort=False).head(1).reset_index(drop=True)


def summarize_boundary(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    specs = [
        (
            "condition1",
            results_dir / "cobalt_core" / "cobalt_condition1_boundary_sampling_budget12_repeat20_summary.csv",
        ),
        (
            1,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold1_boundary_sampling_budget12_repeat20_summary.csv",
        ),
        (
            30,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold30_boundary_sampling_budget12_repeat20_summary.csv",
        ),
        (
            50,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold50_boundary_sampling_budget12_repeat20_summary.csv",
        ),
    ]
    keep = [
        "method",
        "dimension",
        "failure_recall_mean",
        "failure_cluster_coverage_mean",
        "boundary_coverage_mean",
        "boundary_enrichment_mean",
        "coverage_radius_mean",
    ]
    for threshold, path in specs:
        if path.exists():
            rows.append(add_threshold(read_csv(path)[keep], threshold))
    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty
    boundary = pd.concat(rows, ignore_index=True)
    best_boundary = best_by_metric(boundary, ["threshold"], "boundary_coverage_mean")
    best_clusters = best_by_metric(boundary, ["threshold"], "failure_cluster_coverage_mean")
    return boundary, best_boundary, best_clusters


def summarize_expanded_boundary(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    specs = [
        (
            "aldol",
            40,
            10,
            results_dir / "aldol_core" / "aldol_boundary_expanded_budget40_repeat10_summary.csv",
        ),
        (
            "cobalt_condition1",
            12,
            20,
            results_dir / "cobalt_core" / "cobalt_condition1_boundary_expanded_budget12_repeat20_summary.csv",
        ),
        (
            "cobalt_threshold1",
            12,
            20,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold1_boundary_expanded_budget12_repeat20_summary.csv",
        ),
        (
            "cobalt_threshold30",
            12,
            20,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold30_boundary_expanded_budget12_repeat20_summary.csv",
        ),
        (
            "cobalt_threshold50",
            12,
            20,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold50_boundary_expanded_budget12_repeat20_summary.csv",
        ),
    ]
    keep = [
        "method",
        "dimension",
        "failure_recall_mean",
        "failure_recall_ci_low",
        "failure_recall_ci_high",
        "failure_cluster_coverage_mean",
        "failure_cluster_coverage_ci_low",
        "failure_cluster_coverage_ci_high",
        "boundary_coverage_mean",
        "boundary_coverage_ci_low",
        "boundary_coverage_ci_high",
        "boundary_enrichment_mean",
        "boundary_enrichment_ci_low",
        "boundary_enrichment_ci_high",
        "coverage_radius_mean",
        "coverage_radius_ci_low",
        "coverage_radius_ci_high",
    ]
    for dataset, budget, repeats, path in specs:
        if path.exists():
            source = read_csv(path)
            available_keep = [column for column in keep if column in source.columns]
            frame = source[available_keep].copy()
            frame.insert(0, "repeats", repeats)
            frame.insert(0, "budget", budget)
            frame.insert(0, "dataset", dataset)
            rows.append(frame)
    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty
    expanded = pd.concat(rows, ignore_index=True)
    best_boundary = best_by_metric(expanded, ["dataset"], "boundary_coverage_mean")
    best_clusters = best_by_metric(expanded, ["dataset"], "failure_cluster_coverage_mean")
    return expanded, best_boundary, best_clusters


def expanded_boundary_specs(results_dir: Path) -> list[tuple[str, int, int, Path, Path]]:
    return [
        (
            "aldol",
            40,
            10,
            results_dir / "aldol_core" / "aldol_boundary_expanded_budget40_repeat10_summary.csv",
            results_dir / "aldol_core" / "aldol_boundary_expanded_budget40_repeat10_detail.csv",
        ),
        (
            "cobalt_condition1",
            12,
            20,
            results_dir / "cobalt_core" / "cobalt_condition1_boundary_expanded_budget12_repeat20_summary.csv",
            results_dir / "cobalt_core" / "cobalt_condition1_boundary_expanded_budget12_repeat20_detail.csv",
        ),
        (
            "cobalt_threshold1",
            12,
            20,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold1_boundary_expanded_budget12_repeat20_summary.csv",
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold1_boundary_expanded_budget12_repeat20_detail.csv",
        ),
        (
            "cobalt_threshold30",
            12,
            20,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold30_boundary_expanded_budget12_repeat20_summary.csv",
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold30_boundary_expanded_budget12_repeat20_detail.csv",
        ),
        (
            "cobalt_threshold50",
            12,
            20,
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold50_boundary_expanded_budget12_repeat20_summary.csv",
            results_dir
            / "cobalt_threshold_sensitivity"
            / "cobalt_threshold50_boundary_expanded_budget12_repeat20_detail.csv",
        ),
    ]


def bootstrap_mean_ci(values: np.ndarray, samples: int = 5000, seed: int = 42) -> tuple[float, float]:
    clean = values[~np.isnan(values)]
    if len(clean) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(clean), size=(samples, len(clean)))
    means = clean[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize_expanded_boundary_pairwise(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, budget, repeats, summary_path, detail_path in expanded_boundary_specs(results_dir):
        if not summary_path.exists() or not detail_path.exists():
            continue
        summary = read_csv(summary_path)
        detail = read_csv(detail_path)
        best = summary.sort_values("boundary_coverage_mean", ascending=False).iloc[0]
        best_method = str(best["method"])
        best_dimension = str(best["dimension"])
        best_detail = detail[
            (detail["method"].astype(str) == best_method) & (detail["dimension"].astype(str) == best_dimension)
        ][["repeat", "boundary_coverage"]].rename(columns={"boundary_coverage": "best_boundary_coverage"})
        for _, candidate in summary.iterrows():
            method = str(candidate["method"])
            dimension = str(candidate["dimension"])
            if method == best_method and dimension == best_dimension:
                continue
            candidate_detail = detail[
                (detail["method"].astype(str) == method) & (detail["dimension"].astype(str) == dimension)
            ][["repeat", "boundary_coverage"]].rename(columns={"boundary_coverage": "candidate_boundary_coverage"})
            paired = best_detail.merge(candidate_detail, on="repeat", how="inner")
            if paired.empty:
                continue
            delta = (
                paired["best_boundary_coverage"].to_numpy(dtype=float)
                - paired["candidate_boundary_coverage"].to_numpy(dtype=float)
            )
            ci_low, ci_high = bootstrap_mean_ci(delta, seed=42 + len(rows))
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "repeats": repeats,
                    "best_method": best_method,
                    "best_dimension": best_dimension,
                    "candidate_method": method,
                    "candidate_dimension": dimension,
                    "boundary_coverage_delta_mean": float(np.mean(delta)),
                    "boundary_coverage_delta_ci_low": ci_low,
                    "boundary_coverage_delta_ci_high": ci_high,
                    "paired_repeats": len(paired),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "boundary_coverage_delta_mean"], ascending=[True, False]
    )


def article_reference_specs(results_dir: Path) -> list[tuple[str, int, int, Path]]:
    return [
        (
            "alcohol",
            60,
            20,
            results_dir / "article_reference_sampling" / "alcohol_reference_sampling_budget60_repeat20_summary.csv",
        ),
        (
            "thiol",
            21,
            20,
            results_dir / "article_reference_sampling" / "thiol_reference_sampling_budget21_repeat20_summary.csv",
        ),
        (
            "oxo_carboxide",
            21,
            20,
            results_dir / "article_reference_sampling" / "oxo_carboxide_reference_sampling_budget21_repeat20_summary.csv",
        ),
        (
            "styrene",
            21,
            20,
            results_dir / "article_reference_sampling" / "styrene_reference_sampling_budget21_repeat20_summary.csv",
        ),
    ]


def article_reference_detail_specs(results_dir: Path) -> list[tuple[str, int, int, Path, Path]]:
    return [
        (
            "alcohol",
            60,
            20,
            results_dir / "article_reference_sampling" / "alcohol_reference_sampling_budget60_repeat20_summary.csv",
            results_dir / "article_reference_sampling" / "alcohol_reference_sampling_budget60_repeat20_detail.csv",
        ),
        (
            "thiol",
            21,
            20,
            results_dir / "article_reference_sampling" / "thiol_reference_sampling_budget21_repeat20_summary.csv",
            results_dir / "article_reference_sampling" / "thiol_reference_sampling_budget21_repeat20_detail.csv",
        ),
        (
            "oxo_carboxide",
            21,
            20,
            results_dir / "article_reference_sampling" / "oxo_carboxide_reference_sampling_budget21_repeat20_summary.csv",
            results_dir / "article_reference_sampling" / "oxo_carboxide_reference_sampling_budget21_repeat20_detail.csv",
        ),
        (
            "styrene",
            21,
            20,
            results_dir / "article_reference_sampling" / "styrene_reference_sampling_budget21_repeat20_summary.csv",
            results_dir / "article_reference_sampling" / "styrene_reference_sampling_budget21_repeat20_detail.csv",
        ),
    ]


def summarize_article_reference_sampling(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[pd.DataFrame] = []
    keep = [
        "method",
        "dimension",
        "metric",
        "reference_coverage_q10_mean",
        "reference_coverage_q10_ci_low",
        "reference_coverage_q10_ci_high",
        "reference_enrichment_q10_mean",
        "reference_exact_recall_mean",
        "reference_mean_nearest_sample_distance_mean",
        "reference_mean_nearest_sample_distance_ci_low",
        "reference_mean_nearest_sample_distance_ci_high",
        "coverage_radius_mean",
    ]
    for dataset, budget, repeats, path in article_reference_specs(results_dir):
        if path.exists():
            source = read_csv(path)
            if "metric" not in source.columns:
                source["metric"] = np.where(source["dimension"].astype(str) == "tanimoto", "tanimoto", "euclidean")
            available_keep = [column for column in keep if column in source.columns]
            frame = source[available_keep].copy()
            frame.insert(0, "repeats", repeats)
            frame.insert(0, "budget", budget)
            frame.insert(0, "dataset", dataset)
            rows.append(frame)
    if not rows:
        empty = pd.DataFrame()
        return empty, empty, empty

    article = pd.concat(rows, ignore_index=True)
    best_coverage = best_by_metric(article, ["dataset"], "reference_coverage_q10_mean")
    best_distance = best_by_metric(
        article,
        ["dataset", "metric"],
        "reference_mean_nearest_sample_distance_mean",
        maximize=False,
    )
    return article, best_coverage, best_distance


def summarize_article_reference_pairwise(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metric = "reference_coverage_q10"
    for dataset, budget, repeats, summary_path, detail_path in article_reference_detail_specs(results_dir):
        if not summary_path.exists() or not detail_path.exists():
            continue
        summary = read_csv(summary_path)
        detail = read_csv(detail_path)
        if "metric" not in summary.columns:
            summary["metric"] = np.where(summary["dimension"].astype(str) == "tanimoto", "tanimoto", "euclidean")
        if "metric" not in detail.columns:
            detail["metric"] = np.where(detail["dimension"].astype(str) == "tanimoto", "tanimoto", "euclidean")

        best = summary.sort_values(f"{metric}_mean", ascending=False).iloc[0]
        best_method = str(best["method"])
        best_dimension = str(best["dimension"])
        best_metric = str(best["metric"])
        best_detail = detail[
            (detail["method"].astype(str) == best_method)
            & (detail["dimension"].astype(str) == best_dimension)
            & (detail["metric"].astype(str) == best_metric)
        ][["repeat", metric]].rename(columns={metric: "best_reference_coverage_q10"})

        for _, candidate in summary.iterrows():
            method = str(candidate["method"])
            dimension = str(candidate["dimension"])
            metric_name = str(candidate["metric"])
            if method == best_method and dimension == best_dimension and metric_name == best_metric:
                continue
            candidate_detail = detail[
                (detail["method"].astype(str) == method)
                & (detail["dimension"].astype(str) == dimension)
                & (detail["metric"].astype(str) == metric_name)
            ][["repeat", metric]].rename(columns={metric: "candidate_reference_coverage_q10"})
            paired = best_detail.merge(candidate_detail, on="repeat", how="inner")
            if paired.empty:
                continue
            delta = (
                paired["best_reference_coverage_q10"].to_numpy(dtype=float)
                - paired["candidate_reference_coverage_q10"].to_numpy(dtype=float)
            )
            ci_low, ci_high = bootstrap_mean_ci(delta, seed=4200 + len(rows))
            rows.append(
                {
                    "dataset": dataset,
                    "budget": budget,
                    "repeats": repeats,
                    "best_method": best_method,
                    "best_dimension": best_dimension,
                    "best_metric": best_metric,
                    "candidate_method": method,
                    "candidate_dimension": dimension,
                    "candidate_metric": metric_name,
                    "reference_coverage_q10_delta_mean": float(np.mean(delta)),
                    "reference_coverage_q10_delta_ci_low": ci_low,
                    "reference_coverage_q10_delta_ci_high": ci_high,
                    "paired_repeats": len(paired),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["dataset", "reference_coverage_q10_delta_mean"], ascending=[True, False]
    )


def summarize_external_yield_repeated_ci(results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    specs = [
        (
            "buchwald_hartwig",
            results_dir
            / "external_yield_benchmarks"
            / "buchwald_hartwig_repeated30_minimal_ci_regression_summary_ci.csv",
            results_dir
            / "external_yield_benchmarks"
            / "buchwald_hartwig_repeated30_minimal_ci_classification_summary_ci.csv",
        ),
        (
            "suzuki_miyaura",
            results_dir
            / "external_yield_benchmarks"
            / "suzuki_miyaura_repeated30_minimal_ci_regression_summary_ci.csv",
            results_dir
            / "external_yield_benchmarks"
            / "suzuki_miyaura_repeated30_minimal_ci_classification_summary_ci.csv",
        ),
    ]
    regression_rows: list[pd.DataFrame] = []
    classification_rows: list[pd.DataFrame] = []
    for dataset, regression_path, classification_path in specs:
        if regression_path.exists():
            regression = read_csv(regression_path)
            regression = regression[regression["model"] == "RandomForestRegressor"].copy()
            keep = ["dimension", "r2_mean", "r2_ci_low", "r2_ci_high", "rmse_mean", "rmse_ci_low", "rmse_ci_high"]
            regression.insert(0, "dataset", dataset)
            regression_rows.append(regression[["dataset", *keep]])
        if classification_path.exists():
            classification = read_csv(classification_path)
            classification = classification[classification["model"] == "RandomForestClassifier"].copy()
            keep = [
                "dimension",
                "balanced_accuracy_mean",
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "mcc_mean",
                "mcc_ci_low",
                "mcc_ci_high",
                "roc_auc_mean",
                "roc_auc_ci_low",
                "roc_auc_ci_high",
            ]
            classification.insert(0, "dataset", dataset)
            classification_rows.append(classification[["dataset", *keep]])

    regression_out = pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame()
    classification_out = pd.concat(classification_rows, ignore_index=True) if classification_rows else pd.DataFrame()
    return regression_out, classification_out


def markdown_table(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No matching data found._"
    shown = frame.copy()
    if max_rows is not None:
        shown = shown.head(max_rows)
    for column in shown.select_dtypes(include="number").columns:
        shown[column] = shown[column].map(lambda value: f"{value:.4f}")
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


def write_report(
    out_dir: Path,
    classification: pd.DataFrame,
    regression: pd.DataFrame,
    boundary: pd.DataFrame,
    best_boundary: pd.DataFrame,
    best_clusters: pd.DataFrame,
    expanded_boundary: pd.DataFrame,
    expanded_best_boundary: pd.DataFrame,
    expanded_best_clusters: pd.DataFrame,
    expanded_pairwise: pd.DataFrame,
    article_reference: pd.DataFrame,
    article_reference_best_coverage: pd.DataFrame,
    article_reference_best_distance: pd.DataFrame,
    article_reference_pairwise: pd.DataFrame,
    external_regression: pd.DataFrame,
    external_classification: pd.DataFrame,
) -> None:
    lines = [
        "# Reaction-Scope Validation Benchmark Summary",
        "",
        "## Cobalt Repeated-CI Classification",
        "",
        markdown_table(classification),
        "",
        "## Cobalt Repeated-CI Regression",
        "",
        markdown_table(regression),
        "",
        "## Best Boundary Coverage By Threshold",
        "",
        markdown_table(best_boundary),
        "",
        "## Best Failure-Cluster Coverage By Threshold",
        "",
        markdown_table(best_clusters),
        "",
        "## Expanded Boundary Sampling: Best Boundary Coverage",
        "",
        markdown_table(expanded_best_boundary),
        "",
        "## Expanded Boundary Sampling: Best Failure-Cluster Coverage",
        "",
        markdown_table(expanded_best_clusters),
        "",
        "## Expanded Boundary Sampling: Pairwise Boundary-Coverage Deltas",
        "",
        markdown_table(expanded_pairwise, max_rows=20),
        "",
        "## Article Reference Sampling: Best Reference Coverage",
        "",
        markdown_table(article_reference_best_coverage),
        "",
        "## Article Reference Sampling: Best Reference Distance",
        "",
        markdown_table(article_reference_best_distance),
        "",
        "## Article Reference Sampling: Pairwise Reference-Coverage Deltas",
        "",
        markdown_table(article_reference_pairwise, max_rows=20),
        "",
        "## External HTE Yield Benchmarks: Regression",
        "",
        markdown_table(external_regression),
        "",
        "## External HTE Yield Benchmarks: Classification",
        "",
        markdown_table(external_classification),
        "",
        "## Article Reference Sampling Detail",
        "",
        markdown_table(article_reference),
        "",
        "## Expanded Boundary Sampling Detail",
        "",
        markdown_table(expanded_boundary),
        "",
        "## Boundary Sampling Detail",
        "",
        markdown_table(boundary),
        "",
        "## Reading Notes",
        "",
        "- RandomForest repeated-CI rows are the cleanest dimension-sensitivity signal.",
        "- Boundary sampling metrics should be read as separate objectives: failure recall, failure-cluster coverage, and near-boundary coverage do not select the same methods.",
        "- Expanded boundary rows add `weighted_itr_cvt` and the `tanimoto` representation to the method/dimension screen.",
        "- Article reference sampling uses experimental/reference subsets only as coverage targets; it does not treat untested molecules as negative labels.",
        "- External HTE yield benchmarks use one-hot reaction-component features; they test dimension-sensitive evaluation beyond the two primary case-study datasets.",
        "- `condition1` duplicates threshold 1 from the core run and is retained as a reproducibility check.",
    ]
    (out_dir / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = project_root()
    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = root / results_dir
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    classification = summarize_repeated_ci(results_dir)
    regression = summarize_regression_ci(results_dir)
    boundary, best_boundary, best_clusters = summarize_boundary(results_dir)
    expanded_boundary, expanded_best_boundary, expanded_best_clusters = summarize_expanded_boundary(results_dir)
    expanded_pairwise = summarize_expanded_boundary_pairwise(results_dir)
    article_reference, article_reference_best_coverage, article_reference_best_distance = (
        summarize_article_reference_sampling(results_dir)
    )
    article_reference_pairwise = summarize_article_reference_pairwise(results_dir)
    external_regression, external_classification = summarize_external_yield_repeated_ci(results_dir)

    classification.to_csv(out_dir / "cobalt_repeated_ci_classification_summary.csv", index=False)
    regression.to_csv(out_dir / "cobalt_repeated_ci_regression_summary.csv", index=False)
    boundary.to_csv(out_dir / "cobalt_boundary_sampling_summary.csv", index=False)
    best_boundary.to_csv(out_dir / "cobalt_best_boundary_coverage.csv", index=False)
    best_clusters.to_csv(out_dir / "cobalt_best_failure_cluster_coverage.csv", index=False)
    expanded_boundary.to_csv(out_dir / "expanded_boundary_sampling_summary.csv", index=False)
    expanded_best_boundary.to_csv(out_dir / "expanded_best_boundary_coverage.csv", index=False)
    expanded_best_clusters.to_csv(out_dir / "expanded_best_failure_cluster_coverage.csv", index=False)
    expanded_pairwise.to_csv(out_dir / "expanded_boundary_pairwise_deltas.csv", index=False)
    article_reference.to_csv(out_dir / "article_reference_sampling_summary.csv", index=False)
    article_reference_best_coverage.to_csv(out_dir / "article_reference_best_coverage.csv", index=False)
    article_reference_best_distance.to_csv(out_dir / "article_reference_best_distance.csv", index=False)
    article_reference_pairwise.to_csv(out_dir / "article_reference_pairwise_deltas.csv", index=False)
    external_regression.to_csv(out_dir / "external_yield_repeated_ci_regression_summary.csv", index=False)
    external_classification.to_csv(out_dir / "external_yield_repeated_ci_classification_summary.csv", index=False)
    write_report(
        out_dir,
        classification,
        regression,
        boundary,
        best_boundary,
        best_clusters,
        expanded_boundary,
        expanded_best_boundary,
        expanded_best_clusters,
        expanded_pairwise,
        article_reference,
        article_reference_best_coverage,
        article_reference_best_distance,
        article_reference_pairwise,
        external_regression,
        external_classification,
    )

    print(f"Saved summary report to {out_dir / 'benchmark_summary.md'}")
    print(f"Saved summary tables to {out_dir}")


if __name__ == "__main__":
    main()
