# -*- coding: utf-8 -*-
"""Draw failure-cluster representative structures from an existing CSV.

This intentionally avoids importing sklearn-heavy benchmark modules so it can
be run with the conda environment that has RDKit installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw representative structures for failure clusters.")
    parser.add_argument(
        "--representatives",
        default="experiments/results/chemical_science_sprint/failure_cluster_explanations_representatives.csv",
    )
    parser.add_argument("--out-dir", default="experiments/results/chemical_science_sprint")
    parser.add_argument("--out-prefix", default="failure_cluster_explanations")
    parser.add_argument("--mols-per-row", type=int, default=4)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def smiles_col(frame: pd.DataFrame) -> str | None:
    for col in ["smiles", "SMILES"]:
        if col in frame.columns and frame[col].notna().any():
            return col
    return None


def draw_dataset(dataset: str, frame: pd.DataFrame, out_dir: Path, out_prefix: str, mols_per_row: int) -> Path | None:
    col = smiles_col(frame)
    if col is None:
        return None
    mols = []
    legends = []
    for _, row in frame.iterrows():
        smiles = row[col]
        if pd.isna(smiles):
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        Chem.rdDepictor.Compute2DCoords(mol)
        mols.append(mol)
        cluster_id = row["failure_cluster"]
        target = row["target_value"] if "target_value" in row.index else row.get("yield", float("nan"))
        tags = row.get("structure_tags", "")
        short_tags = "" if pd.isna(tags) else ", ".join(str(tags).split("; ")[:2])
        legends.append(f"C{cluster_id} y={target:.1f}\n{short_tags}")
    if not mols:
        return None
    image = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=(300, 220),
        legends=legends,
        useSVG=False,
    )
    out_path = out_dir / f"{out_prefix}_{dataset}_structures.png"
    image.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    reps = pd.read_csv(resolve(args.representatives))
    if "dataset" not in reps.columns:
        reps["dataset"] = "representatives"
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, group in reps.groupby("dataset", sort=False):
        path = draw_dataset(dataset, group, out_dir, args.out_prefix, args.mols_per_row)
        if path is not None:
            print(path)
        else:
            print(f"{dataset}: no drawable SMILES column")


if __name__ == "__main__":
    main()
