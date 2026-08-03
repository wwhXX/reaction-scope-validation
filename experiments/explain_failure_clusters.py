# -*- coding: utf-8 -*-
"""Explain chemically interpretable failure clusters for sprint figures.

The script defines failure clusters in a fixed high-dimensional evaluation
space and then emits representative substrates/reaction components, simple
structure tags, and optional RDKit structure panels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "Examples" / "Sampling_eval" / "Model_eval" / "Regression"
sys.path.insert(0, str(SCRIPT_DIR))

from boundary_sampling_benchmark import (  # noqa: E402
    distances_between,
    failure_cluster_labels,
    load_data,
    transform_features,
)

try:
    from rdkit import Chem
    from rdkit.Chem import Draw

    HAS_RDKIT = True
except Exception:
    Chem = None
    Draw = None
    HAS_RDKIT = False


DEFAULT_DATASETS = ["aldol_prospective_boundary", "cobalt_prospective_boundary", "buchwald_hartwig_prospective_boundary"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and draw representative failure clusters.")
    parser.add_argument("--config", default="experiments/configs/prospective_boundary_simulation.json")
    parser.add_argument("--only", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--eval-dim", default="full")
    parser.add_argument("--representatives", type=int, default=4)
    parser.add_argument("--out-dir", default="experiments/results/chemical_science_sprint")
    parser.add_argument("--out-prefix", default="failure_cluster_explanations")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def dataset_name(task_name: str) -> str:
    return task_name.replace("_prospective_boundary", "")


def smiles_column(data: pd.DataFrame) -> str | None:
    for col in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
        if col in data.columns:
            return col
    return None


def mol_from_smiles(smiles: str):
    if not HAS_RDKIT or not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Chem.rdDepictor.Compute2DCoords(mol)
    return mol


def has_substructure(mol, smarts: str) -> bool:
    if mol is None:
        return False
    patt = Chem.MolFromSmarts(smarts)
    return bool(patt is not None and mol.HasSubstructMatch(patt))


def structure_tags(smiles: str) -> list[str]:
    fallback_tags = smiles_text_tags(smiles)
    mol = mol_from_smiles(smiles)
    if mol is None:
        return fallback_tags
    tags: list[str] = []
    if has_substructure(mol, "[F,Cl,Br,I]"):
        tags.append("halogenated")
    if has_substructure(mol, "[N+](=O)[O-]"):
        tags.append("nitro/EWG")
    if has_substructure(mol, "C#N"):
        tags.append("nitrile/EWG")
    if has_substructure(mol, "C(F)(F)F"):
        tags.append("CF3/EWG")
    if has_substructure(mol, "[a;!#6]"):
        tags.append("heteroaryl")
    if has_substructure(mol, "n"):
        tags.append("basic heteroaryl N")
    if has_substructure(mol, "[OX2][CH3]") or has_substructure(mol, "[OX2][CX4]"):
        tags.append("alkoxy/EDG")
    if has_substructure(mol, "[NX3;!$(N=O)]"):
        tags.append("amine/EDG")
    if has_substructure(mol, "[CX3H1](=O)") or has_substructure(mol, "[CX3](=O)[#6]"):
        tags.append("carbonyl")
    if has_substructure(mol, "[OX2H]"):
        tags.append("alcohol/phenol")
    if mol.GetNumHeavyAtoms() >= 14:
        tags.append("bulky")
    if mol.GetRingInfo().NumRings() >= 2:
        tags.append("polycyclic")
    if ortho_disubstituted_arene(mol):
        tags.append("ortho-substituted arene")
    return list(dict.fromkeys(tags + fallback_tags))


def smiles_text_tags(smiles: str) -> list[str]:
    if not isinstance(smiles, str) or not smiles or smiles.lower() == "nan":
        return []
    tags: list[str] = []
    if any(token in smiles for token in ["Cl", "Br", "F", "I"]):
        tags.append("halogenated")
    if "[N+](=O)[O-]" in smiles or "N(=O)=O" in smiles:
        tags.append("nitro/EWG")
    if "C#N" in smiles:
        tags.append("nitrile/EWG")
    if "C(F)(F)F" in smiles or "(F)(F)F" in smiles:
        tags.append("CF3/EWG")
    if re.search(r"[nso]", smiles):
        tags.append("heteroaryl")
    if "n" in smiles:
        tags.append("basic heteroaryl N")
    if "OC" in smiles or "CO" in smiles:
        tags.append("alkoxy/alcohol")
    if "N(" in smiles or "[NH" in smiles:
        tags.append("amine/carbamate")
    if "C=O" in smiles or "=O" in smiles:
        tags.append("carbonyl/sulfonyl")
    if len(smiles) >= 28:
        tags.append("bulky")
    return list(dict.fromkeys(tags))


def ortho_disubstituted_arene(mol) -> bool:
    if mol is None:
        return False
    for ring in mol.GetRingInfo().AtomRings():
        aromatic_atoms = [idx for idx in ring if mol.GetAtomWithIdx(idx).GetIsAromatic()]
        if len(aromatic_atoms) < 6:
            continue
        substituted = []
        ring_set = set(ring)
        for idx in aromatic_atoms:
            atom = mol.GetAtomWithIdx(idx)
            outside = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetIdx() not in ring_set]
            if outside:
                substituted.append(idx)
        for idx in substituted:
            atom = mol.GetAtomWithIdx(idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() in substituted and nbr.GetIdx() in ring_set:
                    return True
    return False


def tag_summary(tags: list[list[str]]) -> str:
    counts: dict[str, int] = {}
    for row_tags in tags:
        for tag in row_tags:
            counts[tag] = counts.get(tag, 0) + 1
    if not counts:
        return ""
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return "; ".join(f"{tag} ({count})" for tag, count in ordered[:6])


def onehot_component_summary(data: pd.DataFrame, cluster_indices: np.ndarray, failure_indices: np.ndarray) -> str:
    numeric = data.select_dtypes(include=[np.number])
    meaningful_prefixes = (
        "ligand_",
        "additive_",
        "base_",
        "aryl_halide_",
        "reactant_",
        "catalyst_",
        "reagent_",
        "solvent_",
    )
    component_cols = [
        col
        for col in numeric.columns
        if col.startswith(meaningful_prefixes)
        and numeric[col].dropna().isin([0, 1, 0.0, 1.0]).all()
    ]
    if not component_cols:
        return ""
    cluster_mean = numeric.iloc[cluster_indices][component_cols].mean()
    failure_mean = numeric.iloc[failure_indices][component_cols].mean()
    enriched = (cluster_mean - failure_mean).sort_values(ascending=False)
    enriched = enriched[enriched > 0.10].head(8)
    if enriched.empty:
        enriched = cluster_mean[cluster_mean > 0.5].sort_values(ascending=False).head(8)
    return "; ".join(f"{clean_component_name(col)} ({value:.2f})" for col, value in enriched.items())


def clean_component_name(name: str) -> str:
    name = re.sub(r"^(ligand|additive|base|aryl_halide|reactant_1_name|reactant_2_name|catalyst_1_short_hand|ligand_short_hand|reagent_1_name|solvent_1_short_hand)_", "", name)
    return name.replace("_", " ")


def representative_indices(
    X: np.ndarray,
    y_target: np.ndarray,
    cluster_indices: np.ndarray,
    count: int,
    metric: str,
) -> list[int]:
    if len(cluster_indices) <= count:
        return [int(idx) for idx in cluster_indices]
    X_cluster = X[cluster_indices]
    distances = distances_between(X_cluster, X_cluster, metric)
    medoid = int(cluster_indices[int(np.argmin(distances.mean(axis=1)))])
    lowest = [int(idx) for idx in cluster_indices[np.argsort(y_target[cluster_indices])[: count + 1]]]
    reps = [medoid]
    for idx in lowest:
        if idx not in reps:
            reps.append(idx)
        if len(reps) >= count:
            break
    return reps


def draw_representatives(dataset: str, rows: pd.DataFrame, smiles_col: str, out_dir: Path, out_prefix: str) -> str | None:
    if not HAS_RDKIT or rows.empty:
        return None
    mols = []
    legends = []
    for row in rows.itertuples(index=False):
        smiles = getattr(row, smiles_col)
        mol = mol_from_smiles(smiles)
        if mol is None:
            continue
        mols.append(mol)
        legends.append(f"C{getattr(row, 'failure_cluster')} | y={getattr(row, 'target_value'):.1f}")
    if not mols:
        return None
    image = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 180), legends=legends, useSVG=False)
    out_path = out_dir / f"{out_prefix}_{dataset}_structures.png"
    image.save(out_path)
    return str(out_path)


def analyze_task(task: dict[str, Any], eval_dim: str, representatives: int, out_dir: Path, out_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    name = dataset_name(task["name"])
    params = task["params"]
    data, X_raw, y_target, _ = load_data(resolve(params["data"]), params["target"], params["non_feature_cols"])
    failure_threshold = float(params.get("failure_threshold") or params["success_threshold"])
    y_failure = (y_target < failure_threshold).astype(int)
    y_boundary_success = 1 - y_failure if params.get("failure_threshold") is not None else (y_target >= float(params["success_threshold"])).astype(int)
    X_eval, explained, effective_dim, metric = transform_features(X_raw, eval_dim)
    failure_idx, labels = failure_cluster_labels(X_eval, y_boundary_success, int(params.get("failure_clusters", 8)), metric)
    smiles_col = smiles_column(data)

    cluster_rows: list[dict[str, Any]] = []
    rep_rows: list[dict[str, Any]] = []
    for cluster_id in sorted(set(int(label) for label in labels.tolist())):
        cluster_indices = failure_idx[labels == cluster_id]
        reps = representative_indices(X_eval, y_target, cluster_indices, representatives, metric)
        tag_lists = [structure_tags(str(data.iloc[idx][smiles_col])) for idx in cluster_indices] if smiles_col else []
        component_summary = onehot_component_summary(data, cluster_indices, failure_idx)
        cluster_rows.append(
            {
                "dataset": name,
                "failure_cluster": cluster_id,
                "cluster_size": int(len(cluster_indices)),
                "target_min": float(np.min(y_target[cluster_indices])),
                "target_median": float(np.median(y_target[cluster_indices])),
                "target_max": float(np.max(y_target[cluster_indices])),
                "evaluation_dimension": eval_dim,
                "effective_dim": effective_dim,
                "explained_variance_ratio": explained,
                "structure_tag_summary": tag_summary(tag_lists),
                "component_enrichment_summary": component_summary,
            }
        )
        for rank, idx in enumerate(reps, start=1):
            row = {
                "dataset": name,
                "failure_cluster": cluster_id,
                "representative_rank": rank,
                "row_index": int(idx),
                "target_value": float(y_target[idx]),
                "structure_tags": "; ".join(structure_tags(str(data.iloc[idx][smiles_col]))) if smiles_col else "",
            }
            if smiles_col:
                row[smiles_col] = data.iloc[idx][smiles_col]
            for meta_col in params["non_feature_cols"]:
                if meta_col in data.columns and meta_col not in row:
                    row[meta_col] = data.iloc[idx][meta_col]
            rep_rows.append(row)

    reps_frame = pd.DataFrame(rep_rows)
    if smiles_col and not reps_frame.empty:
        draw_representatives(name, reps_frame.rename(columns={smiles_col: "smiles_for_drawing"}), "smiles_for_drawing", out_dir, out_prefix)
    return pd.DataFrame(cluster_rows), reps_frame


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    shown = frame.copy()
    for col in shown.columns:
        if pd.api.types.is_float_dtype(shown[col]):
            shown[col] = shown[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
    headers = [str(col) for col in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in shown.itertuples(index=False):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(clusters: pd.DataFrame, reps: pd.DataFrame, out_dir: Path, out_prefix: str) -> None:
    lines = ["# Chemical Science Sprint: Failure-Cluster Explanations", ""]
    lines.append("Failure clusters are defined in the full high-dimensional evaluation space. Structure tags are simple RDKit substructure heuristics for manuscript triage, not final mechanistic assignments.")
    lines.append("")
    for dataset, group in clusters.groupby("dataset", sort=False):
        lines.append(f"## {dataset}")
        lines.append("")
        cols = [
            "failure_cluster",
            "cluster_size",
            "target_min",
            "target_median",
            "structure_tag_summary",
            "component_enrichment_summary",
        ]
        lines.append(markdown_table(group[cols]))
        lines.append("")
        reps_group = reps[reps["dataset"] == dataset].head(12)
        keep = [col for col in ["failure_cluster", "representative_rank", "target_value", "smiles", "SMILES", "structure_tags", "component_enrichment_summary"] if col in reps_group.columns]
        if keep:
            lines.append("Representative failures:")
            lines.append("")
            lines.append(markdown_table(reps_group[keep]))
            lines.append("")
    (out_dir / f"{out_prefix}_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = load_config(resolve(args.config))
    selected = set(args.only or [])
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_frames: list[pd.DataFrame] = []
    rep_frames: list[pd.DataFrame] = []
    for task in config["tasks"]:
        if selected and task["name"] not in selected:
            continue
        clusters, reps = analyze_task(task, args.eval_dim, args.representatives, out_dir, args.out_prefix)
        cluster_frames.append(clusters)
        rep_frames.append(reps)
        print(f"{task['name']}: {len(clusters)} clusters, {len(reps)} representatives")

    cluster_summary = pd.concat(cluster_frames, ignore_index=True)
    representatives = pd.concat(rep_frames, ignore_index=True)
    cluster_summary.to_csv(out_dir / f"{args.out_prefix}_summary.csv", index=False)
    representatives.to_csv(out_dir / f"{args.out_prefix}_representatives.csv", index=False)
    write_report(cluster_summary, representatives, out_dir, args.out_prefix)

    print("\nRDKit drawing:", "enabled" if HAS_RDKIT else "not available")
    print("Saved:")
    print(f"  {out_dir / f'{args.out_prefix}_summary.csv'}")
    print(f"  {out_dir / f'{args.out_prefix}_representatives.csv'}")
    print(f"  {out_dir / f'{args.out_prefix}_report.md'}")


if __name__ == "__main__":
    main()
