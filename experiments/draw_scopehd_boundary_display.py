# -*- coding: utf-8 -*-
"""Draw a chemistry-facing full-space boundary display figure.

The figure translates high-dimensional boundary neighborhoods into
success/failure neighbor pairs and a 2D projection used only as an index view.
It does not claim that the shown representatives are exact selected indices
from a particular retrospective campaign.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Examples" / "Sampling_eval" / "Model_eval" / "Regression" / "1700_final_norepeat.csv"
REPS_PATH = ROOT / "experiments" / "results" / "chemical_science_sprint" / "failure_cluster_explanations_representatives.csv"
OUT_DIR = ROOT / "experiments" / "results" / "manuscript"
OUT_PATH = OUT_DIR / "scopehd_high_dim_boundary_display.png"
PCA_COORDS_PATH = OUT_DIR / "scopehd_aldol_pca2_coords.csv"


FONT = Path("C:/Windows/Fonts/arial.ttf")
BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def font(size: int, bold: bool = False):
    path = BOLD if bold and BOLD.exists() else FONT
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def standardize(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x, axis=0)
    sigma = np.nanstd(x, axis=0)
    sigma[sigma == 0] = 1.0
    return np.nan_to_num((x - mu) / sigma)


def two_axis_index_view(raw_x: np.ndarray) -> np.ndarray:
    """Return a cheap 2D index view without calling local BLAS/LAPACK.

    The local RDKit conda environment can crash in NumPy linear algebra calls.
    For this display figure, the 2D panel is only an index/projection view, not
    the boundary definition, so two high-variance descriptor axes are sufficient.
    """

    variances = np.nanvar(raw_x, axis=0)
    order = np.argsort(variances)[::-1]
    coords = raw_x[:, order[:2]].astype(float)
    return standardize(coords)


def load_projection(raw_x: np.ndarray) -> np.ndarray:
    if PCA_COORDS_PATH.exists():
        coords = pd.read_csv(PCA_COORDS_PATH)[["pca1", "pca2"]].to_numpy(float)
        if len(coords) == len(raw_x):
            return coords
    return two_axis_index_view(raw_x)


def mol_image(smiles: str, size: tuple[int, int] = (230, 170)) -> Image.Image:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return Image.new("RGB", size, "white")
    rdDepictor.Compute2DCoords(mol)
    img = Draw.MolToImage(mol, size=size)
    return img.convert("RGB")


def nearest_success_pairs(data: pd.DataFrame, reps: pd.DataFrame) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    feature_cols = [col for col in data.columns if col not in {"smiles", "conv"}]
    raw_x = data[feature_cols].select_dtypes(include=[np.number]).to_numpy(float)
    x = standardize(raw_x)
    y = data["conv"].to_numpy(float)
    success_idx = np.flatnonzero(y >= 70)

    wanted = [
        (2, "Nitro/EWG aryl aldehyde boundary"),
        (1, "Alkoxy/benzyloxy aryl aldehyde boundary"),
        (3, "Sulfonyl/pyridyl aldehyde boundary"),
        (5, "Donor-substituted anilino aldehyde boundary"),
    ]
    pairs = []
    aldol_reps = reps[reps["dataset"] == "aldol"].copy()
    for cluster_id, label in wanted:
        cluster = aldol_reps[aldol_reps["failure_cluster"].astype(int) == cluster_id].sort_values("representative_rank")
        if cluster.empty:
            continue
        failure_row = cluster.iloc[0]
        f_idx = int(failure_row["row_index"])
        distances = np.sqrt(((x[success_idx] - x[f_idx]) ** 2).sum(axis=1))
        s_idx = int(success_idx[int(np.argmin(distances))])
        pairs.append(
            {
                "cluster": cluster_id,
                "label": label,
                "failure_idx": f_idx,
                "success_idx": s_idx,
                "failure_smiles": data.loc[f_idx, "smiles"],
                "success_smiles": data.loc[s_idx, "smiles"],
                "failure_conv": float(data.loc[f_idx, "conv"]),
                "success_conv": float(data.loc[s_idx, "conv"]),
                "distance": float(np.min(distances)),
            }
        )
    return pairs, x, y, load_projection(raw_x)


def project_to_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    c = coords.copy()
    mins = np.percentile(c, 1, axis=0)
    maxs = np.percentile(c, 99, axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    c = (c - mins) / span
    c = np.clip(c, 0, 1)
    out = np.empty_like(c)
    out[:, 0] = left + c[:, 0] * (right - left)
    out[:, 1] = bottom - c[:, 1] * (bottom - top)
    return out


def dashed_line(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], fill: str, width: int = 3, dash: int = 10) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    for start in range(0, length, dash * 2):
        end = min(start + dash, length)
        a = start / length
        b = end / length
        draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=fill, width=width)


def draw_pair_panel(canvas: Image.Image, pairs: list[dict]) -> None:
    draw = ImageDraw.Draw(canvas)
    title = font(34, True)
    subtitle = font(23)
    small = font(20)
    tiny = font(17)
    draw.text((70, 80), "A. High-dimensional boundary neighborhoods", fill="#0B2545", font=title)
    draw.text((70, 126), "Nearest success/failure pairs in full descriptor space.", fill="#555555", font=subtitle)
    draw.text((70, 156), "Structures are the chemistry-facing boundary view.", fill="#555555", font=subtitle)

    y0 = 215
    row_h = 238
    for idx, pair in enumerate(pairs):
        y = y0 + idx * row_h
        draw.rounded_rectangle((70, y - 20, 1120, y + 205), radius=18, outline="#D7DEE8", width=2, fill="#FBFCFE")
        draw.text((95, y - 2), pair["label"], fill="#1F4D78", font=small)

        s_img = mol_image(pair["success_smiles"])
        f_img = mol_image(pair["failure_smiles"])
        canvas.paste(s_img, (105, y + 35))
        canvas.paste(f_img, (575, y + 35))
        draw.text((145, y + 190), f"success: conv {pair['success_conv']:.0f}", fill="#2A6FBB", font=tiny)
        draw.text((620, y + 190), f"failure/boundary: conv {pair['failure_conv']:.0f}", fill="#B54848", font=tiny)

        draw.line((365, y + 110, 545, y + 110), fill="#333333", width=4)
        draw.polygon([(545, y + 110), (522, y + 98), (522, y + 122)], fill="#333333")
        draw.text((385, y + 72), "nearest", fill="#333333", font=tiny)
        draw.text((372, y + 134), f"high-D d={pair['distance']:.1f}", fill="#333333", font=tiny)


def draw_projection_panel(canvas: Image.Image, pairs: list[dict], y: np.ndarray, coords2: np.ndarray) -> None:
    draw = ImageDraw.Draw(canvas)
    title = font(34, True)
    subtitle = font(23)
    small = font(20)
    tiny = font(17)
    draw.text((1235, 80), "B. 2D projection as index", fill="#0B2545", font=title)
    draw.text((1235, 126), "Dashed links are high-D nearest-opposite relations.", fill="#555555", font=subtitle)
    draw.text((1235, 156), "The boundary is not defined by this 2D view.", fill="#555555", font=subtitle)

    box = (1235, 225, 2260, 1040)
    draw.rounded_rectangle((box[0] - 20, box[1] - 20, box[2] + 20, box[3] + 20), radius=18, outline="#D7DEE8", width=2, fill="#FBFCFE")
    xy = project_to_box(coords2, box)

    success = y >= 70
    for i in range(len(y)):
        x_i, y_i = int(xy[i, 0]), int(xy[i, 1])
        color = "#8FBCE6" if success[i] else "#E4A0A0"
        r = 3 if success[i] else 4
        draw.ellipse((x_i - r, y_i - r, x_i + r, y_i + r), fill=color, outline=None)

    for pair in pairs:
        s = pair["success_idx"]
        f = pair["failure_idx"]
        ps = (int(xy[s, 0]), int(xy[s, 1]))
        pf = (int(xy[f, 0]), int(xy[f, 1]))
        dashed_line(draw, ps, pf, "#222222", width=3, dash=12)
        draw.ellipse((ps[0] - 11, ps[1] - 11, ps[0] + 11, ps[1] + 11), outline="#1F5C99", width=4)
        draw.ellipse((pf[0] - 13, pf[1] - 13, pf[0] + 13, pf[1] + 13), outline="#B54848", width=5)
        draw.text((pf[0] + 12, pf[1] - 16), f"C{pair['cluster']}", fill="#8B1E1E", font=tiny)

    legend_x, legend_y = 1235, 1075
    items = [
        ("success", "#8FBCE6"),
        ("failure / low conversion", "#E4A0A0"),
        ("high-D boundary failure", "#B54848"),
        ("nearest high-D success", "#1F5C99"),
    ]
    for idx, (label, color) in enumerate(items):
        x = legend_x + (idx % 2) * 430
        y0 = legend_y + (idx // 2) * 42
        if "boundary" in label or "nearest" in label:
            draw.ellipse((x, y0, x + 22, y0 + 22), outline=color, width=4)
        else:
            draw.ellipse((x, y0, x + 22, y0 + 22), fill=color)
        draw.text((x + 32, y0 - 2), label, fill="#333333", font=small)

    draw.text((1235, 1195), "Display rule:", fill="#0B2545", font=small)
    draw.text(
        (1370, 1195),
        "Full descriptor space defines boundary neighborhoods; the 2D panel is only a readable index view.",
        fill="#333333",
        font=small,
    )


def build() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    reps = pd.read_csv(REPS_PATH)
    pairs, _, y, coords2 = nearest_success_pairs(data, reps)
    canvas = Image.new("RGB", (2400, 1350), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 25), "Full-space reaction boundary display", fill="#0B2545", font=font(42, True))
    draw.text(
        (70, 1370 - 70),
        "Aldol example. Boundary candidates are low-conversion representatives with nearest high-D success neighbors; no mechanistic causality is implied.",
        fill="#555555",
        font=font(19),
    )
    draw_pair_panel(canvas, pairs)
    draw_projection_panel(canvas, pairs, y, coords2)
    canvas.save(OUT_PATH)
    return OUT_PATH


if __name__ == "__main__":
    print(build())
