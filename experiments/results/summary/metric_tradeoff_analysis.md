# Cross-Metric Trade-Off Analysis

This analysis uses labelled benchmark outputs. Article-test reference-only datasets are intentionally excluded.

## Strongest Static Boundary Conflicts

| analysis_type   | dataset            | primary_metric         | primary_best                | primary_best_value | primary_best_rank_on_secondary | secondary_metric              | secondary_best       | secondary_best_value | secondary_best_rank_on_primary | same_winner |
| --------------- | ------------------ | ---------------------- | --------------------------- | ------------------ | ------------------------------ | ----------------------------- | -------------------- | -------------------- | ------------------------------ | ----------- |
| static_boundary | aldol              | boundary_coverage_mean | weighted_itr_cvt / full     | 0.0956             | 26.0000                        | failure_cluster_coverage_mean | kennard_stone / 128  | 1.0000               | 32.0000                        | False       |
| static_boundary | cobalt_condition1  | boundary_coverage_mean | weighted_itr_cvt / tanimoto | 0.2500             | 29.0000                        | failure_cluster_coverage_mean | weighted_itr_cvt / 2 | 1.0000               | 32.0000                        | False       |
| static_boundary | cobalt_threshold1  | boundary_coverage_mean | weighted_itr_cvt / tanimoto | 0.2500             | 29.0000                        | failure_cluster_coverage_mean | weighted_itr_cvt / 2 | 1.0000               | 32.0000                        | False       |
| static_boundary | cobalt_threshold30 | boundary_coverage_mean | weighted_itr_cvt / tanimoto | 0.2500             | 29.0000                        | failure_cluster_coverage_mean | weighted_itr_cvt / 2 | 1.0000               | 31.0000                        | False       |
| static_boundary | cobalt_threshold50 | boundary_coverage_mean | cvt / tanimoto              | 0.3333             | 26.0000                        | failure_cluster_coverage_mean | cvt / 2              | 1.0000               | 12.0000                        | False       |

## Top Methods By Metric

| analysis_type     | dataset            | metric                        | best_method           | best_dimension | best_value | best_method_space                |
| ----------------- | ------------------ | ----------------------------- | --------------------- | -------------- | ---------- | -------------------------------- |
| prospective_final | aldol              | failure_recall_mean           | uncertainty_diversity | tanimoto       | 0.1803     | uncertainty_diversity / tanimoto |
| prospective_final | aldol              | failure_cluster_coverage_mean | fps                   | 64             | 1.0000     | fps / 64                         |
| prospective_final | aldol              | boundary_coverage_mean        | cvt                   | full           | 0.1928     | cvt / full                       |
| prospective_final | aldol              | boundary_enrichment_mean      | cvt                   | full           | 2.6293     | cvt / full                       |
| prospective_final | aldol              | coverage_radius_mean          | random                | 2              | 0.4486     | random / 2                       |
| prospective_final | buchwald_hartwig   | failure_recall_mean           | ward                  | 32             | 0.0343     | ward / 32                        |
| prospective_final | buchwald_hartwig   | failure_cluster_coverage_mean | random                | 2              | 1.0000     | random / 2                       |
| prospective_final | buchwald_hartwig   | boundary_coverage_mean        | weighted_itr_cvt      | 32             | 0.0613     | weighted_itr_cvt / 32            |
| prospective_final | buchwald_hartwig   | boundary_enrichment_mean      | weighted_itr_cvt      | 32             | 2.0193     | weighted_itr_cvt / 32            |
| prospective_final | buchwald_hartwig   | coverage_radius_mean          | fps                   | 2              | 0.0002     | fps / 2                          |
| prospective_final | cobalt             | failure_recall_mean           | uncertainty_diversity | tanimoto       | 0.4686     | uncertainty_diversity / tanimoto |
| prospective_final | cobalt             | failure_cluster_coverage_mean | cvt                   | 2              | 1.0000     | cvt / 2                          |
| prospective_final | cobalt             | boundary_coverage_mean        | random                | tanimoto       | 0.4550     | random / tanimoto                |
| prospective_final | cobalt             | boundary_enrichment_mean      | random                | tanimoto       | 1.1375     | random / tanimoto                |
| prospective_final | cobalt             | coverage_radius_mean          | fps                   | 2              | 0.3423     | fps / 2                          |
| prospective_final | suzuki_miyaura     | failure_recall_mean           | kennard_stone         | 2              | 0.0286     | kennard_stone / 2                |
| prospective_final | suzuki_miyaura     | failure_cluster_coverage_mean | random                | 16             | 1.0000     | random / 16                      |
| prospective_final | suzuki_miyaura     | boundary_coverage_mean        | uncertainty           | 32             | 0.0253     | uncertainty / 32                 |
| prospective_final | suzuki_miyaura     | boundary_enrichment_mean      | uncertainty           | 32             | 1.2130     | uncertainty / 32                 |
| prospective_final | suzuki_miyaura     | coverage_radius_mean          | cvt                   | 2              | 0.0000     | cvt / 2                          |
| static_boundary   | aldol              | failure_recall_mean           | fps                   | 64             | 0.0952     | fps / 64                         |
| static_boundary   | aldol              | failure_cluster_coverage_mean | kennard_stone         | 128            | 1.0000     | kennard_stone / 128              |
| static_boundary   | aldol              | boundary_coverage_mean        | weighted_itr_cvt      | full           | 0.0956     | weighted_itr_cvt / full          |
| static_boundary   | aldol              | boundary_enrichment_mean      | weighted_itr_cvt      | full           | 2.6076     | weighted_itr_cvt / full          |
| static_boundary   | aldol              | coverage_radius_mean          | ward                  | 2              | 0.4395     | ward / 2                         |
| static_boundary   | cobalt_condition1  | failure_recall_mean           | fps                   | tanimoto       | 0.2143     | fps / tanimoto                   |
| static_boundary   | cobalt_condition1  | failure_cluster_coverage_mean | weighted_itr_cvt      | 2              | 1.0000     | weighted_itr_cvt / 2             |
| static_boundary   | cobalt_condition1  | boundary_coverage_mean        | weighted_itr_cvt      | tanimoto       | 0.2500     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_condition1  | boundary_enrichment_mean      | weighted_itr_cvt      | tanimoto       | 1.2500     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_condition1  | coverage_radius_mean          | weighted_itr_cvt      | tanimoto       | 0.4914     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_threshold1  | failure_recall_mean           | fps                   | tanimoto       | 0.2143     | fps / tanimoto                   |
| static_boundary   | cobalt_threshold1  | failure_cluster_coverage_mean | weighted_itr_cvt      | 2              | 1.0000     | weighted_itr_cvt / 2             |
| static_boundary   | cobalt_threshold1  | boundary_coverage_mean        | weighted_itr_cvt      | tanimoto       | 0.2500     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_threshold1  | boundary_enrichment_mean      | weighted_itr_cvt      | tanimoto       | 1.2500     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_threshold1  | coverage_radius_mean          | weighted_itr_cvt      | tanimoto       | 0.4914     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_threshold30 | failure_recall_mean           | kennard_stone         | tanimoto       | 0.2222     | kennard_stone / tanimoto         |
| static_boundary   | cobalt_threshold30 | failure_cluster_coverage_mean | weighted_itr_cvt      | 2              | 1.0000     | weighted_itr_cvt / 2             |
| static_boundary   | cobalt_threshold30 | boundary_coverage_mean        | weighted_itr_cvt      | tanimoto       | 0.2500     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_threshold30 | boundary_enrichment_mean      | weighted_itr_cvt      | tanimoto       | 1.2500     | weighted_itr_cvt / tanimoto      |
| static_boundary   | cobalt_threshold30 | coverage_radius_mean          | weighted_itr_cvt      | tanimoto       | 0.4914     | weighted_itr_cvt / tanimoto      |

## Pareto Frontier Counts

| analysis_type     | dataset            | pareto_count |
| ----------------- | ------------------ | ------------ |
| prospective_final | aldol              | 29.0000      |
| prospective_final | buchwald_hartwig   | 10.0000      |
| prospective_final | cobalt             | 19.0000      |
| prospective_final | suzuki_miyaura     | 14.0000      |
| static_boundary   | aldol              | 16.0000      |
| static_boundary   | cobalt_condition1  | 15.0000      |
| static_boundary   | cobalt_threshold1  | 15.0000      |
| static_boundary   | cobalt_threshold30 | 13.0000      |
| static_boundary   | cobalt_threshold50 | 13.0000      |

## Budget Trajectory Summary

| dataset          | configured_budget | final_budget | mid_budget | best_mid_boundary | best_mid_boundary_coverage | best_final_boundary   | best_final_boundary_coverage | first_budget_boundary_coverage_ge_0_10 | first_boundary_coverage_ge_0_10_method |
| ---------------- | ----------------- | ------------ | ---------- | ----------------- | -------------------------- | --------------------- | ---------------------------- | -------------------------------------- | -------------------------------------- |
| aldol            | 80.0000           | 80.0000      | 40.0000    | cvt / full        | 0.1010                     | cvt / full            | 0.1928                       | 40.0000                                | cvt / full                             |
| buchwald_hartwig | 120.0000          | 120.0000     | 60.0000    | cvt / 32          | 0.0327                     | weighted_itr_cvt / 32 | 0.0613                       |                                        | not reached                            |
| cobalt           | 24.0000           | 24.0000      | 12.0000    | random / tanimoto | 0.2775                     | random / tanimoto     | 0.4550                       | 6.0000                                 | random / tanimoto                      |
| suzuki_miyaura   | 120.0000          | 120.0000     | 60.0000    | ward / 32         | 0.0137                     | uncertainty / 32      | 0.0253                       |                                        | not reached                            |

## Correlation Rows

| analysis_type     | dataset          | metric_x                      | metric_y                      | spearman |
| ----------------- | ---------------- | ----------------------------- | ----------------------------- | -------- |
| prospective_final | aldol            | failure_recall_mean           | failure_recall_mean           | 1.0000   |
| prospective_final | aldol            | failure_recall_mean           | failure_cluster_coverage_mean | 0.4996   |
| prospective_final | aldol            | failure_recall_mean           | boundary_coverage_mean        | -0.6294  |
| prospective_final | aldol            | failure_recall_mean           | boundary_enrichment_mean      | -0.6294  |
| prospective_final | aldol            | failure_recall_mean           | coverage_radius_mean          | -0.0514  |
| prospective_final | aldol            | failure_cluster_coverage_mean | failure_recall_mean           | 0.4996   |
| prospective_final | aldol            | failure_cluster_coverage_mean | failure_cluster_coverage_mean | 1.0000   |
| prospective_final | aldol            | failure_cluster_coverage_mean | boundary_coverage_mean        | -0.6041  |
| prospective_final | aldol            | failure_cluster_coverage_mean | boundary_enrichment_mean      | -0.6041  |
| prospective_final | aldol            | failure_cluster_coverage_mean | coverage_radius_mean          | 0.0059   |
| prospective_final | aldol            | boundary_coverage_mean        | failure_recall_mean           | -0.6294  |
| prospective_final | aldol            | boundary_coverage_mean        | failure_cluster_coverage_mean | -0.6041  |
| prospective_final | aldol            | boundary_coverage_mean        | boundary_coverage_mean        | 1.0000   |
| prospective_final | aldol            | boundary_coverage_mean        | boundary_enrichment_mean      | 1.0000   |
| prospective_final | aldol            | boundary_coverage_mean        | coverage_radius_mean          | 0.1932   |
| prospective_final | aldol            | boundary_enrichment_mean      | failure_recall_mean           | -0.6294  |
| prospective_final | aldol            | boundary_enrichment_mean      | failure_cluster_coverage_mean | -0.6041  |
| prospective_final | aldol            | boundary_enrichment_mean      | boundary_coverage_mean        | 1.0000   |
| prospective_final | aldol            | boundary_enrichment_mean      | boundary_enrichment_mean      | 1.0000   |
| prospective_final | aldol            | boundary_enrichment_mean      | coverage_radius_mean          | 0.1932   |
| prospective_final | aldol            | coverage_radius_mean          | failure_recall_mean           | -0.0514  |
| prospective_final | aldol            | coverage_radius_mean          | failure_cluster_coverage_mean | 0.0059   |
| prospective_final | aldol            | coverage_radius_mean          | boundary_coverage_mean        | 0.1932   |
| prospective_final | aldol            | coverage_radius_mean          | boundary_enrichment_mean      | 0.1932   |
| prospective_final | aldol            | coverage_radius_mean          | coverage_radius_mean          | 1.0000   |
| prospective_final | buchwald_hartwig | failure_recall_mean           | failure_recall_mean           | 1.0000   |
| prospective_final | buchwald_hartwig | failure_recall_mean           | failure_cluster_coverage_mean | -0.2491  |
| prospective_final | buchwald_hartwig | failure_recall_mean           | boundary_coverage_mean        | -0.0749  |
| prospective_final | buchwald_hartwig | failure_recall_mean           | boundary_enrichment_mean      | -0.0749  |
| prospective_final | buchwald_hartwig | failure_recall_mean           | coverage_radius_mean          | -0.1395  |
| prospective_final | buchwald_hartwig | failure_cluster_coverage_mean | failure_recall_mean           | -0.2491  |
| prospective_final | buchwald_hartwig | failure_cluster_coverage_mean | failure_cluster_coverage_mean | 1.0000   |
| prospective_final | buchwald_hartwig | failure_cluster_coverage_mean | boundary_coverage_mean        | 0.0721   |
| prospective_final | buchwald_hartwig | failure_cluster_coverage_mean | boundary_enrichment_mean      | 0.0721   |
| prospective_final | buchwald_hartwig | failure_cluster_coverage_mean | coverage_radius_mean          | -0.2192  |
| prospective_final | buchwald_hartwig | boundary_coverage_mean        | failure_recall_mean           | -0.0749  |
| prospective_final | buchwald_hartwig | boundary_coverage_mean        | failure_cluster_coverage_mean | 0.0721   |
| prospective_final | buchwald_hartwig | boundary_coverage_mean        | boundary_coverage_mean        | 1.0000   |
| prospective_final | buchwald_hartwig | boundary_coverage_mean        | boundary_enrichment_mean      | 1.0000   |
| prospective_final | buchwald_hartwig | boundary_coverage_mean        | coverage_radius_mean          | 0.5495   |
| prospective_final | buchwald_hartwig | boundary_enrichment_mean      | failure_recall_mean           | -0.0749  |
| prospective_final | buchwald_hartwig | boundary_enrichment_mean      | failure_cluster_coverage_mean | 0.0721   |
| prospective_final | buchwald_hartwig | boundary_enrichment_mean      | boundary_coverage_mean        | 1.0000   |
| prospective_final | buchwald_hartwig | boundary_enrichment_mean      | boundary_enrichment_mean      | 1.0000   |
| prospective_final | buchwald_hartwig | boundary_enrichment_mean      | coverage_radius_mean          | 0.5495   |
| prospective_final | buchwald_hartwig | coverage_radius_mean          | failure_recall_mean           | -0.1395  |
| prospective_final | buchwald_hartwig | coverage_radius_mean          | failure_cluster_coverage_mean | -0.2192  |
| prospective_final | buchwald_hartwig | coverage_radius_mean          | boundary_coverage_mean        | 0.5495   |
| prospective_final | buchwald_hartwig | coverage_radius_mean          | boundary_enrichment_mean      | 0.5495   |
| prospective_final | buchwald_hartwig | coverage_radius_mean          | coverage_radius_mean          | 1.0000   |
| prospective_final | cobalt           | failure_recall_mean           | failure_recall_mean           | 1.0000   |
| prospective_final | cobalt           | failure_recall_mean           | failure_cluster_coverage_mean | -0.3075  |
| prospective_final | cobalt           | failure_recall_mean           | boundary_coverage_mean        | 0.0179   |
| prospective_final | cobalt           | failure_recall_mean           | boundary_enrichment_mean      | 0.0179   |
| prospective_final | cobalt           | failure_recall_mean           | coverage_radius_mean          | 0.0032   |
| prospective_final | cobalt           | failure_cluster_coverage_mean | failure_recall_mean           | -0.3075  |
| prospective_final | cobalt           | failure_cluster_coverage_mean | failure_cluster_coverage_mean | 1.0000   |
| prospective_final | cobalt           | failure_cluster_coverage_mean | boundary_coverage_mean        | -0.5096  |
| prospective_final | cobalt           | failure_cluster_coverage_mean | boundary_enrichment_mean      | -0.5096  |
| prospective_final | cobalt           | failure_cluster_coverage_mean | coverage_radius_mean          | -0.2602  |
| prospective_final | cobalt           | boundary_coverage_mean        | failure_recall_mean           | 0.0179   |
| prospective_final | cobalt           | boundary_coverage_mean        | failure_cluster_coverage_mean | -0.5096  |
| prospective_final | cobalt           | boundary_coverage_mean        | boundary_coverage_mean        | 1.0000   |
| prospective_final | cobalt           | boundary_coverage_mean        | boundary_enrichment_mean      | 1.0000   |
| prospective_final | cobalt           | boundary_coverage_mean        | coverage_radius_mean          | 0.0498   |
| prospective_final | cobalt           | boundary_enrichment_mean      | failure_recall_mean           | 0.0179   |
| prospective_final | cobalt           | boundary_enrichment_mean      | failure_cluster_coverage_mean | -0.5096  |
| prospective_final | cobalt           | boundary_enrichment_mean      | boundary_coverage_mean        | 1.0000   |
| prospective_final | cobalt           | boundary_enrichment_mean      | boundary_enrichment_mean      | 1.0000   |
| prospective_final | cobalt           | boundary_enrichment_mean      | coverage_radius_mean          | 0.0498   |
| prospective_final | cobalt           | coverage_radius_mean          | failure_recall_mean           | 0.0032   |
| prospective_final | cobalt           | coverage_radius_mean          | failure_cluster_coverage_mean | -0.2602  |
| prospective_final | cobalt           | coverage_radius_mean          | boundary_coverage_mean        | 0.0498   |
| prospective_final | cobalt           | coverage_radius_mean          | boundary_enrichment_mean      | 0.0498   |
| prospective_final | cobalt           | coverage_radius_mean          | coverage_radius_mean          | 1.0000   |
| prospective_final | suzuki_miyaura   | failure_recall_mean           | failure_recall_mean           | 1.0000   |
| prospective_final | suzuki_miyaura   | failure_recall_mean           | failure_cluster_coverage_mean | -0.3277  |
| prospective_final | suzuki_miyaura   | failure_recall_mean           | boundary_coverage_mean        | -0.1175  |
| prospective_final | suzuki_miyaura   | failure_recall_mean           | boundary_enrichment_mean      | -0.1175  |
| prospective_final | suzuki_miyaura   | failure_recall_mean           | coverage_radius_mean          | -0.3274  |