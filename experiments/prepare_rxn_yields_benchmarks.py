# -*- coding: utf-8 -*-
"""Prepare external rxn_yields HTE datasets for validation benchmarks.

The source workbooks come from the public rxn4chemistry/rxn_yields repository.
Buchwald-Hartwig contains reagent/component SMILES. Suzuki-Miyaura uses names
and short-hand condition labels, so this preparation intentionally uses
one-hot reaction-condition features rather than inventing missing structures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Data" / "External" / "rxn_yields_raw"
OUT_DIR = ROOT / "Data" / "External" / "rxn_yields_prepared"


def clean_column_name(value: object) -> str:
    text = str(value).strip().lower()
    for old, new in [
        (" ", "_"),
        ("-", "_"),
        ("/", "_"),
        ("(", ""),
        (")", ""),
        ("%", "pct"),
        (".", "_"),
    ]:
        text = text.replace(old, new)
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def one_hot_frame(data: pd.DataFrame, categorical_cols: list[str], numeric_cols: list[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if numeric_cols:
        numeric = data[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        numeric.columns = [clean_column_name(col) for col in numeric.columns]
        pieces.append(numeric)
    encoded = pd.get_dummies(
        data[categorical_cols].fillna("missing").astype(str),
        prefix=[clean_column_name(col) for col in categorical_cols],
        dtype=float,
    )
    encoded.columns = [clean_column_name(col) for col in encoded.columns]
    pieces.append(encoded)
    return pd.concat(pieces, axis=1)


def read_excel_or_cached_csv(source: Path, sheet_name: str) -> pd.DataFrame:
    cached = source.with_name(f"{source.stem}_{sheet_name}.csv")
    try:
        return pd.read_excel(source, sheet_name=sheet_name)
    except ImportError:
        if cached.exists():
            return pd.read_csv(cached)
        raise


def prepare_buchwald_hartwig() -> Path:
    source = RAW_DIR / "Dreher_and_Doyle_input_data.xlsx"
    data = read_excel_or_cached_csv(source, sheet_name="FullCV_01")
    data = data.rename(columns={"Output": "yield"})
    component_cols = ["Ligand", "Additive", "Base", "Aryl halide"]
    features = one_hot_frame(data, component_cols, [])
    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "reaction_id": [f"buchwald_hartwig_{idx + 1}" for idx in range(len(data))],
                    "yield": pd.to_numeric(data["yield"], errors="coerce").fillna(0.0),
                }
            ),
            features,
        ],
        axis=1,
    )
    path = OUT_DIR / "buchwald_hartwig_onehot_benchmark.csv"
    output.to_csv(path, index=False)
    return path


def component_fingerprint_frame(data: pd.DataFrame, component_cols: list[str], n_bits: int = 256) -> pd.DataFrame:
    """Concatenate Morgan fingerprints for drawable reaction components."""
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator
    except ImportError as exc:
        raise RuntimeError(
            "RDKit is required for the drawable Buchwald-Hartwig fingerprint benchmark. "
            "Use the reaction-scope-validation Conda environment."
        ) from exc

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    pieces: list[pd.DataFrame] = []
    for col in component_cols:
        rows: list[np.ndarray] = []
        for smiles in data[col].fillna("").astype(str):
            mol = Chem.MolFromSmiles(smiles)
            arr = np.zeros((n_bits,), dtype=np.int8)
            if mol is not None:
                fp = generator.GetFingerprint(mol)
                DataStructs.ConvertToNumpyArray(fp, arr)
            rows.append(arr.astype(float))
        prefix = clean_column_name(col)
        pieces.append(pd.DataFrame(rows, columns=[f"{prefix}_morgan_{idx}" for idx in range(n_bits)]))
    return pd.concat(pieces, axis=1)


def prepare_buchwald_hartwig_drawable() -> Path:
    source = RAW_DIR / "Dreher_and_Doyle_input_data.xlsx"
    data = read_excel_or_cached_csv(source, sheet_name="FullCV_01")
    data = data.rename(columns={"Output": "yield"})
    component_cols = ["Ligand", "Additive", "Base", "Aryl halide"]
    features = component_fingerprint_frame(data, component_cols)
    reaction_smiles = data[component_cols].fillna("").astype(str).agg(".".join, axis=1)
    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "reaction_id": [f"buchwald_hartwig_drawable_{idx + 1}" for idx in range(len(data))],
                    "yield": pd.to_numeric(data["yield"], errors="coerce").fillna(0.0),
                    "smiles": data["Aryl halide"].fillna("").astype(str),
                    "reaction_smiles": reaction_smiles,
                    "ligand_smiles": data["Ligand"].fillna("").astype(str),
                    "additive_smiles": data["Additive"].fillna("").astype(str),
                    "base_smiles": data["Base"].fillna("").astype(str),
                    "aryl_halide_smiles": data["Aryl halide"].fillna("").astype(str),
                }
            ),
            features,
        ],
        axis=1,
    )
    path = OUT_DIR / "buchwald_hartwig_drawable_fp_benchmark.csv"
    output.to_csv(path, index=False)
    return path


def prepare_suzuki_miyaura() -> Path:
    source = RAW_DIR / "aap9112_Data_File_S1.xlsx"
    data = read_excel_or_cached_csv(source, sheet_name="Sheet1")
    data = data.rename(columns={"Product_Yield_PCT_Area_UV": "yield"})
    categorical_cols = [
        "Reactant_1_Name",
        "Reactant_1_Short_Hand",
        "Reactant_2_Name",
        "Catalyst_1_Short_Hand",
        "Ligand_Short_Hand",
        "Reagent_1_Short_Hand",
        "Solvent_1_Short_Hand",
    ]
    numeric_cols = [
        "Reactant_1_eq",
        "Reactant_1_mmol",
        "Reactant_2_eq",
        "Catalyst_1_eq",
        "Ligand_eq",
        "Reagent_1_eq",
    ]
    features = one_hot_frame(data, categorical_cols, numeric_cols)
    output = pd.concat(
        [
            pd.DataFrame(
                {
                    "reaction_id": [f"suzuki_miyaura_{idx + 1}" for idx in range(len(data))],
                    "yield": pd.to_numeric(data["yield"], errors="coerce").fillna(0.0),
                }
            ),
            features,
        ],
        axis=1,
    )
    path = OUT_DIR / "suzuki_miyaura_onehot_benchmark.csv"
    output.to_csv(path, index=False)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [prepare_buchwald_hartwig(), prepare_buchwald_hartwig_drawable(), prepare_suzuki_miyaura()]
    for path in outputs:
        data = pd.read_csv(path)
        metadata_cols = [
            "reaction_id",
            "yield",
            "smiles",
            "reaction_smiles",
            "ligand_smiles",
            "additive_smiles",
            "base_smiles",
            "aryl_halide_smiles",
        ]
        feature_count = len([col for col in data.columns if col not in metadata_cols])
        print(f"Saved {path.relative_to(ROOT)}: rows={len(data)}, features={feature_count}")
        print(data["yield"].describe().to_string())
        print(f"success ratio, yield >= 50: {(data['yield'] >= 50).mean():.3f}")


if __name__ == "__main__":
    main()
