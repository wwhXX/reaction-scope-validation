# -*- coding: utf-8 -*-
"""Draw a row-level full-space versus paired 2D boundary comparison figure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "manuscript"
EXAMPLES = OUT_DIR / "scopehd_only_boundary_examples.csv"
COORDS = OUT_DIR / "scopehd_only_aldol_pca2_coords.csv"
SUMMARY = OUT_DIR / "scopehd_vs_2d_boundary_replay_summary.csv"
OUT = OUT_DIR / "scopehd_vs_2d_boundary_comparison.png"

FONT = Path("C:/Windows/Fonts/arial.ttf")
BOLD = Path("C:/Windows/Fonts/arialbd.ttf")


def font(size: int, bold: bool = False):
    path = BOLD if bold and BOLD.exists() else FONT
    if path.exists():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def mol_image(smiles: str, size: tuple[int, int] = (225, 145)) -> Image.Image:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return Image.new("RGB", size, "white")
    rdDepictor.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size).convert("RGB")


def project_to_box(coords: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    mins = np.percentile(coords, 1, axis=0)
    maxs = np.percentile(coords, 99, axis=0)
    span = np.maximum(maxs - mins, 1e-9)
    scaled = np.clip((coords - mins) / span, 0, 1)
    out = np.empty_like(scaled)
    out[:, 0] = left + scaled[:, 0] * (right - left)
    out[:, 1] = bottom - scaled[:, 1] * (bottom - top)
    return out


def dashed_line(draw: ImageDraw.ImageDraw, p1: tuple[int, int], p2: tuple[int, int], fill: str, width: int = 3) -> None:
    x1, y1 = p1
    x2, y2 = p2
    length = max(1, int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5))
    dash = 11
    for start in range(0, length, dash * 2):
        end = min(start + dash, length)
        a = start / length
        b = end / length
        draw.line((x1 + (x2 - x1) * a, y1 + (y2 - y1) * a, x1 + (x2 - x1) * b, y1 + (y2 - y1) * b), fill=fill, width=width)


def centered_text(draw: ImageDraw.ImageDraw, center_x: int, y: int, text: str, fill: str, text_font: ImageFont.ImageFont) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((center_x - (box[2] - box[0]) / 2, y), text, fill=fill, font=text_font)


def choose_examples(df: pd.DataFrame, n: int = 4) -> pd.DataFrame:
    failure = df[df["is_failure"] == True].copy()
    # Prefer repeated high-D-only evidence and chemically varied rows.
    return failure.sort_values(["highd_only_count", "nearest_opposite_distance"], ascending=[False, True]).head(n)


def draw_examples(canvas: Image.Image, examples: pd.DataFrame) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 68), "A. Exact high-D-only boundary examples", fill="#0B2545", font=font(32, True))
    draw.text((70, 110), "Full-space acquisition selected these boundary rows; the paired 2D replay did not.", fill="#555555", font=font(22))
    draw.text((70, 140), "Same repeat, shared initial set, final budget 80.", fill="#555555", font=font(22))
    row_h = 226
    y0 = 195
    for pos, row in enumerate(examples.itertuples(index=False)):
        y = y0 + pos * row_h
        draw.rounded_rectangle((70, y - 18, 1125, y + 197), radius=18, outline="#D7DEE8", width=2, fill="#FBFCFE")
        title = f"row {row.row_index} | conv {row.conv:.0f} | HD-only in repeats {row.highd_only_repeats}"
        draw.text((92, y - 1), title, fill="#1F4D78", font=font(19, True))
        s_img = mol_image(row.nearest_success_smiles)
        f_img = mol_image(row.smiles)
        canvas.paste(s_img, (105, y + 32))
        canvas.paste(f_img, (580, y + 32))
        centered_text(draw, 217, y + 172, "success neighbor", "#2A6FBB", font(15))
        centered_text(draw, 692, y + 172, "HD found / 2D miss", "#B54848", font(15, True))
        draw.line((363, y + 104, 548, y + 104), fill="#333333", width=4)
        draw.polygon([(548, y + 104), (524, y + 92), (524, y + 116)], fill="#333333")
        draw.text((388, y + 70), "high-D nearest", fill="#333333", font=font(16))
        draw.text((395, y + 128), f"d={row.nearest_success_distance:.1f}", fill="#333333", font=font(16))


def draw_projection(canvas: Image.Image, examples: pd.DataFrame, coords: pd.DataFrame) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((1250, 68), "B. 2D projection as location index", fill="#0B2545", font=font(30, True))
    draw.text((1250, 110), "Red rings are exact HD-selected / 2D-missed rows.", fill="#555555", font=font(20))
    draw.text((1250, 138), "Dashed links are high-D nearest-success relations.", fill="#555555", font=font(20))
    box = (1250, 190, 2290, 780)
    draw.rounded_rectangle((box[0] - 18, box[1] - 18, box[2] + 18, box[3] + 18), radius=18, outline="#D7DEE8", width=2, fill="#FBFCFE")
    xy = project_to_box(coords[["pca1", "pca2"]].to_numpy(float), box)
    conv = coords["conv"].to_numpy(float)
    is_boundary = coords["is_boundary"].astype(bool).to_numpy()
    for i in range(len(coords)):
        x, y = int(xy[i, 0]), int(xy[i, 1])
        color = "#8FBCE6" if conv[i] >= 70 else "#E4A0A0"
        r = 3 if not is_boundary[i] else 4
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    for row in examples.itertuples(index=False):
        f_idx = int(row.row_index)
        s_idx = int(row.nearest_success_idx)
        pf = (int(xy[f_idx, 0]), int(xy[f_idx, 1]))
        ps = (int(xy[s_idx, 0]), int(xy[s_idx, 1]))
        dashed_line(draw, ps, pf, "#222222", width=3)
        draw.ellipse((ps[0] - 11, ps[1] - 11, ps[0] + 11, ps[1] + 11), outline="#1F5C99", width=4)
        draw.ellipse((pf[0] - 14, pf[1] - 14, pf[0] + 14, pf[1] + 14), outline="#B54848", width=5)
        draw.text((pf[0] + 12, pf[1] - 15), f"{f_idx}", fill="#8B1E1E", font=font(14, True))

    legend_y = 820
    items = [("success", "#8FBCE6", False), ("low conversion", "#E4A0A0", False), ("HD selected / 2D missed", "#B54848", True), ("nearest high-D success", "#1F5C99", True)]
    for i, (label, color, ring) in enumerate(items):
        x = 1220 + (i % 2) * 420
        y = legend_y + (i // 2) * 42
        if ring:
            draw.ellipse((x, y, x + 22, y + 22), outline=color, width=4)
        else:
            draw.ellipse((x, y, x + 22, y + 22), fill=color)
        draw.text((x + 34, y - 3), label, fill="#333333", font=font(18))


def draw_stats(canvas: Image.Image, summary: pd.DataFrame) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.text((1250, 955), "C. Row-level replay check", fill="#0B2545", font=font(30, True))
    mean_full = float(summary["full_boundary_count"].mean())
    mean_2d = float(summary["two_d_boundary_count"].mean())
    mean_only = float(summary["full_only_boundary_count"].mean())
    max_bar = max(mean_full, mean_2d, 1.0)
    x0, y0 = 1250, 1020
    bar_w = 620
    rows = [("2D decision space", mean_2d, "#8FBCE6"), ("Full-space acquisition", mean_full, "#2E8B57")]
    for i, (label, value, color) in enumerate(rows):
        y = y0 + i * 75
        draw.text((x0, y), label, fill="#333333", font=font(20))
        w = int(bar_w * value / max_bar)
        draw.rounded_rectangle((x0 + 360, y - 2, x0 + 360 + w, y + 28), radius=6, fill=color)
        draw.text((x0 + 370 + w, y - 2), f"{value:.1f} boundary rows", fill="#333333", font=font(18))
    draw.text((x0, y0 + 165), f"Mean full-only boundary rows per paired repeat: {mean_only:.1f}", fill="#B54848", font=font(20, True))
    draw.text((x0, y0 + 207), "This panel supports the example labels; the r20 benchmark remains the main statistical result.", fill="#555555", font=font(17))


def build() -> Path:
    examples = choose_examples(pd.read_csv(EXAMPLES), 4)
    coords = pd.read_csv(COORDS)
    summary = pd.read_csv(SUMMARY)
    canvas = Image.new("RGB", (2400, 1350), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 24), "Full-space vs 2D: exact paired-miss boundary examples", fill="#0B2545", font=font(39, True))
    draw_examples(canvas, examples)
    draw_projection(canvas, examples, coords)
    draw_stats(canvas, summary)
    draw.text((70, 1302), "Aldol row-level paired replay. Examples are selected by full-space high-D replay and not by the paired 2D replay in the listed repeat(s).", fill="#555555", font=font(18))
    canvas.save(OUT)
    return OUT


if __name__ == "__main__":
    print(build())
