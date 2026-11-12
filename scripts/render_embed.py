"""임베딩 확인용 이미지. Poster World 의 표시 문법을 따른다(레퍼런스: EPFLECALLab_PosterWorld_JPG/).

  out/embed/global_map.jpg     UMAP 2D 를 격자에 맞춰 커버 타일을 깐 지도 (Global analysis)
  out/embed/triptych_*.jpg     기준 커버 + 임베딩상 가장 가까운 두 장 (세 폭 짝)
  out/embed/technique_*.jpg    기법 제로샷 라벨별 상위 커버

사용 (.venv-ml)
  .venv-ml/bin/python scripts/render_embed.py
  .venv-ml/bin/python scripts/render_embed.py --anchors 0002 0005 0010
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
EMB = ROOT / "data" / "analysis" / "embed"
OUT = ROOT / "out" / "embed"
BG = (0, 0, 0)
FG = (235, 235, 235)


def font(size: int) -> ImageFont.FreeTypeFont:
    for f in ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Supplemental/Arial.ttf"]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def thumb(name: str, w: int, h: int) -> Image.Image:
    with Image.open(POSTERS / name) as im:
        return im.convert("RGB").resize((w, h), Image.LANCZOS)


def title(name: str) -> str:
    return name[5:-4].replace("-", " ")


def global_map(files: list[str], vecs: np.ndarray, seed: int = 0) -> None:
    import umap

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed).fit_transform(vecs)
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    n = len(files)
    # 세 패널을 합친 가로 격자. 빈 칸이 남아야 Poster World 처럼 흩어진 느낌이 난다
    cols = int(np.ceil(np.sqrt(n * 2.2 * 1.6)))
    rows = int(np.ceil(n * 2.2 / cols))
    gx, gy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    cost = ((xy[:, None, :] - grid[None]) ** 2).sum(-1)
    _, cell = linear_sum_assignment(cost)

    tw, th, gap = 26, 32, 6
    pw, pad, label_h = cols * (tw + gap), 60, 120
    W, H = pw + 2 * pad, rows * (th + gap) + 2 * pad + label_h
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for i, c in enumerate(cell):
        r, q = divmod(c, cols)
        img.paste(thumb(files[i], tw, th), (pad + q * (tw + gap), pad + r * (th + gap)))
    # 세 패널 구분선
    for k in (1, 2):
        x = pad + pw * k // 3 - gap // 2
        d.line([(x, 0), (x, H)], fill=(40, 40, 40), width=8)
    d.text((pad, H - label_h + 20), "Covers are sorted according to", fill=FG, font=font(22))
    d.text((pad, H - label_h + 50), "Global analysis  ·  SigLIP 2 embedding", fill=FG, font=font(40))
    img.save(OUT / "global_map.jpg", quality=88)
    np.save(EMB / "umap_xy.npy", xy)


def triptych(files: list[str], vecs: np.ndarray, anchor: int, axis: str) -> None:
    sim = vecs @ vecs[anchor]
    sim[anchor] = -1
    near = np.argsort(-sim)[:2]
    idx = [near[0], anchor, near[1]]
    pw, ph, gap, label_h = 400, 497, 24, 150
    W, H = 3 * pw + 4 * gap, ph + label_h + 2 * gap
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for k, i in enumerate(idx):
        x = gap + k * (pw + gap)
        img.paste(thumb(files[i], pw, ph), (x, gap))
        cap = title(files[i]) + ("" if i == anchor else f"   sim {vecs[i] @ vecs[anchor]:.3f}")
        d.text((x, gap + ph + 12), cap, fill=FG, font=font(18))
    d.text((gap, H - 70), "The 3 covers are associated based on", fill=FG, font=font(18))
    d.text((gap, H - 48), axis, fill=FG, font=font(32))
    img.save(OUT / f"triptych_{files[anchor][:4]}.jpg", quality=88)


def technique_sheets(files: list[str], tech: dict[str, dict[str, float]], top: int = 8) -> None:
    labels = list(next(iter(tech.values())))
    tw, th, gap = 180, 224, 12
    rows = []
    for lab in labels:
        ranked = sorted(files, key=lambda f: -tech[f][lab])[:top]
        rows.append((lab, ranked))
    W = 220 + top * (tw + gap)
    H = len(rows) * (th + gap + 30) + gap
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    for r, (lab, ranked) in enumerate(rows):
        y = gap + r * (th + gap + 30)
        d.text((gap, y + th // 2 - 12), lab, fill=FG, font=font(22))
        for k, f in enumerate(ranked):
            x = 220 + k * (tw + gap)
            img.paste(thumb(f, tw, th), (x, y))
            d.text((x, y + th + 4), f"{tech[f][lab]:.2f}", fill=(150, 150, 150), font=font(16))
    img.save(OUT / "technique_top.jpg", quality=85)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="siglip2-base-patch16-384")
    ap.add_argument("--anchors", nargs="*", default=["0002", "0004", "0005", "0010"])
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(EMB / f"{a.model}.npz")
    files, vecs = list(d["files"]), d["vectors"]
    tech = json.loads((EMB / f"{a.model}_technique.json").read_text())

    global_map(files, vecs)
    for s in a.anchors:
        hit = [i for i, f in enumerate(files) if f.startswith(s)]
        if hit:
            triptych(files, vecs, hit[0], "Overall look  ·  SigLIP 2")
    technique_sheets(files, tech)
    print(f"{len(files)} covers -> {OUT}")


if __name__ == "__main__":
    main()
