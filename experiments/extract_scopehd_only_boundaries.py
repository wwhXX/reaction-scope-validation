# -*- coding: utf-8 -*-
"""Extract row-level full-space boundary examples missed by paired 2D.

The original r20 summary files do not store final selected row indices. This
script therefore replays the Aldol paired comparison with a NumPy-only
full-space weighted iterative CVT selector so individual examples can be
labelled honestly:

    selected by full-space high-D replay, not selected by the paired 2D replay.

The replay keeps the key comparison contract: shared initial set, shared budget,
2D versus full decision spaces, and fixed full-space boundary evaluation.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Examples" / "Sampling_eval" / "Model_eval" / "Regression" / "1700_final_norepeat.csv"
OUT_DIR = ROOT / "experiments" / "results" / "manuscript"
OUT_CSV = OUT_DIR / "scopehd_only_boundary_examples.csv"
OUT_COORDS = OUT_DIR / "scopehd_only_aldol_pca2_coords.csv"
OUT_REPEAT_SUMMARY = OUT_DIR / "scopehd_vs_2d_boundary_replay_summary.csv"


def stable_seed(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0xFFFFFFFF


def standardize(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = 1.0
    return np.nan_to_num((x - mu) / sd)


def pca2(x_std: np.ndarray) -> np.ndarray:
    centered = x_std - x_std.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    d2 = np.maximum(x2 + y2 - 2 * (x @ y.T), 0.0)
    return np.sqrt(d2)


def nearest_unique(x: np.ndarray, centers: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
    d = distances(x[candidates], centers)
    selected: list[int] = []
    used: set[int] = set()
    for center_idx in range(centers.shape[0]):
        for local_idx in np.argsort(d[:, center_idx]):
            idx = int(candidates[int(local_idx)])
            if idx not in used:
                selected.append(idx)
                used.add(idx)
                break
    if len(selected) < k:
        selected.extend(int(idx) for idx in candidates if int(idx) not in used)
    return np.array(selected[:k], dtype=int)


def simple_kmeans(x: np.ndarray, k: int, seed: int, iters: int = 20) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = min(k, len(x))
    centers = x[rng.choice(len(x), size=k, replace=False)].astype(float)
    for _ in range(max(1, iters)):
        labels = np.argmin(distances(x, centers), axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = x[labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = members.mean(axis=0)
        if np.linalg.norm(new_centers - centers) < 1e-8:
            break
        centers = new_centers
    return centers


def cvt_select(x: np.ndarray, candidates: np.ndarray, k: int, seed: int, iters: int = 20) -> np.ndarray:
    centers = simple_kmeans(x[candidates], k, seed, iters)
    return nearest_unique(x, centers, candidates, k)


def boundary_mask_and_nearest(x_full: np.ndarray, y_success: np.ndarray, quantile: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    success_idx = np.flatnonzero(y_success == 1)
    failure_idx = np.flatnonzero(y_success == 0)
    nearest = np.empty(len(y_success), dtype=float)
    nearest[success_idx] = distances(x_full[success_idx], x_full[failure_idx]).min(axis=1)
    nearest[failure_idx] = distances(x_full[failure_idx], x_full[success_idx]).min(axis=1)
    cutoff = float(np.quantile(nearest, quantile))
    return nearest <= cutoff, nearest


def weighted_itr_cvt_like_batch(
    x: np.ndarray,
    y_success: np.ndarray,
    selected: list[int],
    batch_size: int,
    seed: int,
    weighted_iters: int = 20,
) -> np.ndarray:
    selected_set = set(selected)
    remaining = np.array([idx for idx in range(len(x)) if idx not in selected_set], dtype=int)
    k = min(batch_size, len(remaining))
    if k <= 0:
        return np.array([], dtype=int)
    centers = simple_kmeans(x[remaining], k, seed, iters=15)
    failed = np.array([idx for idx in selected if y_success[idx] == 0], dtype=int)

    for _ in range(weighted_iters):
        labels = np.argmin(distances(x[remaining], centers), axis=1)
        target_centers = centers.copy()
        for center_idx in range(len(centers)):
            assigned = x[remaining][labels == center_idx]
            if len(assigned):
                target_centers[center_idx] = assigned.mean(axis=0)
        if len(failed):
            diff = centers[:, None, :] - x[failed][None, :, :]
            dist = np.linalg.norm(diff, axis=2, keepdims=True) + 1e-6
            repulsion = (diff / (dist**3)).sum(axis=1)
            repulsion /= np.linalg.norm(repulsion, axis=1, keepdims=True) + 1e-12
            target_centers = target_centers + 0.05 * repulsion
        if np.linalg.norm(target_centers - centers) < 1e-6:
            centers = target_centers
            break
        centers = target_centers
    return nearest_unique(x, centers, remaining, k)


def run_replay(
    x_initial: np.ndarray,
    x_decision: np.ndarray,
    y_success: np.ndarray,
    decision_name: str,
    repeat: int,
    budget: int = 80,
    initial_size: int = 10,
    batch_size: int = 10,
    seed: int = 42,
) -> list[int]:
    init_seed = seed + repeat * 1009
    initial = cvt_select(x_initial, np.arange(len(x_initial), dtype=int), initial_size, init_seed, iters=10)
    selected = initial.astype(int).tolist()
    round_idx = 0
    while len(selected) < budget:
        batch_seed = init_seed + stable_seed(decision_name) + round_idx * 7919
        batch = weighted_itr_cvt_like_batch(x_decision, y_success, selected, batch_size, batch_seed)
        if len(batch) == 0:
            break
        for idx in batch:
            if int(idx) not in selected:
                selected.append(int(idx))
        selected = selected[:budget]
        round_idx += 1
    return selected


def nearest_success_info(idx: int, x_full: np.ndarray, y_success: np.ndarray, data: pd.DataFrame) -> dict[str, Any]:
    success_idx = np.flatnonzero(y_success == 1)
    d = distances(x_full[[idx]], x_full[success_idx]).ravel()
    nearest_idx = int(success_idx[int(np.argmin(d))])
    return {
        "nearest_success_idx": nearest_idx,
        "nearest_success_smiles": data.loc[nearest_idx, "smiles"],
        "nearest_success_conv": float(data.loc[nearest_idx, "conv"]),
        "nearest_success_distance": float(np.min(d)),
    }


def build() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    feature_cols = [col for col in data.columns if col not in {"smiles", "conv"}]
    x_raw = data[feature_cols].select_dtypes(include=[np.number]).to_numpy(float)
    x_full = standardize(x_raw)
    x_2d = pca2(x_full)
    y_target = data["conv"].to_numpy(float)
    y_success = (y_target >= 70).astype(int)
    boundary_mask, nearest_opposite = boundary_mask_and_nearest(x_full, y_success, 0.25)

    rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    for repeat in range(1, 11):
        selected_2d = set(run_replay(x_full, x_2d, y_success, "2d", repeat))
        selected_full = set(run_replay(x_full, x_full, y_success, "full", repeat))
        full_only_boundary = sorted(idx for idx in (selected_full - selected_2d) if bool(boundary_mask[idx]))
        two_d_only_boundary = sorted(idx for idx in (selected_2d - selected_full) if bool(boundary_mask[idx]))
        shared_boundary = sorted(idx for idx in (selected_full & selected_2d) if bool(boundary_mask[idx]))
        repeat_rows.append(
            {
                "repeat": repeat,
                "full_boundary_count": int(sum(bool(boundary_mask[idx]) for idx in selected_full)),
                "two_d_boundary_count": int(sum(bool(boundary_mask[idx]) for idx in selected_2d)),
                "full_only_boundary_count": len(full_only_boundary),
                "two_d_only_boundary_count": len(two_d_only_boundary),
                "shared_boundary_count": len(shared_boundary),
            }
        )
        print(f"repeat {repeat}: high-D-only boundary rows = {len(full_only_boundary)}", flush=True)
        for idx in full_only_boundary:
            row = {
                "row_index": int(idx),
                "repeat": repeat,
                "smiles": data.loc[idx, "smiles"],
                "conv": float(data.loc[idx, "conv"]),
                "is_failure": bool(y_success[idx] == 0),
                "nearest_opposite_distance": float(nearest_opposite[idx]),
                "selected_by_full_scopehd_replay": True,
                "selected_by_paired_2d_replay": False,
                "selection_context": "same repeat, shared initial set, final budget 80",
            }
            row.update(nearest_success_info(idx, x_full, y_success, data))
            rows.append(row)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No high-D-only boundary rows found.")
    summary = (
        detail.groupby(
            [
                "row_index",
                "smiles",
                "conv",
                "is_failure",
                "nearest_success_idx",
                "nearest_success_smiles",
                "nearest_success_conv",
            ],
            sort=False,
        )
        .agg(
            highd_only_repeats=("repeat", lambda values: ";".join(str(int(v)) for v in sorted(set(values)))),
            highd_only_count=("repeat", "nunique"),
            nearest_opposite_distance=("nearest_opposite_distance", "mean"),
            nearest_success_distance=("nearest_success_distance", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["is_failure", "highd_only_count", "nearest_opposite_distance"], ascending=[False, False, True])
    summary.to_csv(OUT_CSV, index=False)
    pd.DataFrame({"pca1": x_2d[:, 0], "pca2": x_2d[:, 1], "conv": y_target, "is_boundary": boundary_mask}).to_csv(
        OUT_COORDS, index=False
    )
    pd.DataFrame(repeat_rows).to_csv(OUT_REPEAT_SUMMARY, index=False)
    print(OUT_CSV)
    print(OUT_COORDS)
    print(OUT_REPEAT_SUMMARY)
    return OUT_CSV


if __name__ == "__main__":
    build()
