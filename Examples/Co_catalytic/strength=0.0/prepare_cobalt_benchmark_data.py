# -*- coding: utf-8 -*-
"""Build a labeled descriptor table for cobalt benchmark experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
LABELS_PATH = BASE / "experimental_data_results_of_alcohol.csv"
FEATURES_PATH = BASE / "fp_spoc_morgan41024_Maccs_experimental_data_results_of_alcohol_alcohol.csv"
OUTPUT_PATH = BASE / "cobalt_condition1_benchmark.csv"


def main() -> None:
    labels = pd.read_csv(LABELS_PATH)
    features = pd.read_csv(FEATURES_PATH)
    if len(labels) != len(features):
        raise ValueError(f"Row mismatch: labels={len(labels)}, features={len(features)}")

    output = pd.concat(
        [
            labels[["SMILES", "condition1_yield", "condition1_ee", "condition2_yield", "condition2_ee"]],
            features,
        ],
        axis=1,
    )
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {OUTPUT_PATH.name}: rows={len(output)}, columns={len(output.columns)}")
    print(output["condition1_yield"].describe().to_string())
    print(f"Reactive ratio, condition1_yield >= 1: {(output['condition1_yield'] >= 1).mean():.3f}")


if __name__ == "__main__":
    main()
