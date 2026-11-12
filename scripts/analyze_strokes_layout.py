"""획(stroke)과 레이아웃 분석. 이미 처리한 커버는 건너뛴다.

획
  Informative Drawings(lineart, controlnet_aux)로 선화를 뽑고, 텍스트 다각형과 크라이테리온 틀을 가린
  그림 부분에서 선의 양, 굵기, 방향, 결(coherence)을 잰다. 글자 획은 텍스트 배치 축이 따로 다룬다.
레이아웃
  커버를 40×50 칸으로 나눠 칸마다 text(글자) / figure(그림) / empty(빈 공간) / frame(틀) 로 나눈다.
  figure = 바탕색과 다르거나 선화 밀도가 있는 곳. 바탕색은 가장자리(틀 제외)의 중앙값.

입력: posters/, data/regions/combined/, data/analysis/visual/ (틀 위치)
출력
  data/analysis/strokes_layout/<stem>.json   지표와 비교 벡터
  web/strokes/<stem>.png                     선화 240×298 (커버에서 나온 이미지라 공개하지 않는다)

사용 (.venv-ml)
  .venv-ml/bin/python scripts/analyze_strokes_layout.py
  .venv-ml/bin/python scripts/analyze_strokes_layout.py --force --limit 10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
REGIONS = ROOT / "data" / "regions" / "combined"
VISUAL = ROOT / "data" / "analysis" / "visual"
OUT = ROOT / "data" / "analysis" / "strokes_layout"
WEB_STROKES = ROOT / "web" / "strokes"

W, H = 400, 497            # 분석 크기 (커버 비율 1288:1600)
GW, GH = 40, 50            # 레이아웃 칸
INK = 0.22                 # 선화 값이 이보다 크면 잉크
BG_DIST = 18.0             # Lab 거리 이보다 크면 바탕과 다른 색
WIDTH_BINS = [0, 2.1, 2.9, 4, 6, 1e9]   # 획 굵기(px, W=400 기준)


def masks(stem: str) -> tuple[np.ndarray, np.ndarray, list]:
    """(text, frame) 마스크 (H, W) bool, frame rects."""
    t = Image.new("L", (W, H), 0)
    rj = REGIONS / f"{stem}.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        d = ImageDraw.Draw(t)
        for x in r["regions"]:
            if len(x["poly"]) >= 3:
                d.polygon([(px / r["width"] * W, py / r["height"] * H) for px, py in x["poly"]], fill=255)
    f = Image.new("L", (W, H), 0)
    rects = []
    vj = VISUAL / f"{stem}.json"
    if vj.exists():
        rects = json.loads(vj.read_text())["frame"].get("rects") or []
        d = ImageDraw.Draw(f)
        for x0, y0, x1, y1 in rects:
            d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=255)
    return np.asarray(t) > 0, np.asarray(f) > 0, rects


def stroke_stats(line: np.ndarray, art: np.ndarray) -> dict:
    """line: 선화 0–1 (H, W). art: 그림 부분 마스크."""
    vis = art.sum()
    ink = (line > INK) & art
    out = {"ink_share": round(float(ink.sum() / max(vis, 1)), 4),
           "intensity": round(float(line[ink].mean()) if ink.any() else 0.0, 4)}
    # 굵기: 스켈레톤 위의 거리 변환 × 2
    dt = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sk = skeletonize(ink)
    widths = 2 * dt[sk]
    wh = np.histogram(widths, bins=WIDTH_BINS)[0].astype(float) if widths.size else np.zeros(len(WIDTH_BINS) - 1)
    out["width_median"] = round(float(np.median(widths)) if widths.size else 0.0, 2)
    out["skeleton_len"] = int(sk.sum())
    out["width_mean"] = round(float(ink.sum() / max(sk.sum(), 1)), 3)   # 잉크 면적 ÷ 선 길이
    # 방향과 결: 구조 텐서
    g = cv2.GaussianBlur(line.astype(np.float32), (0, 0), 1.0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    jxx, jyy, jxy = (cv2.GaussianBlur(a, (0, 0), 3.0) for a in (gx * gx, gy * gy, gx * gy))
    lam_d = np.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2)
    coh = lam_d / (jxx + jyy + 1e-6)
    theta = (0.5 * np.arctan2(2 * jxy, jxx - jyy) + np.pi / 2) % np.pi   # 선 방향 (0=수평)
    sel = ink & (coh > 0)
    oh = np.histogram(theta[sel], bins=12, range=(0, np.pi), weights=coh[sel])[0] if sel.any() else np.zeros(12)
    out["coherence"] = round(float(coh[ink].mean()) if ink.any() else 0.0, 4)
    hv = oh.sum() or 1
    out["horizontal_share"] = round(float((oh[0] + oh[11] + oh[1] * 0.5 + oh[10] * 0.5) / hv), 4)
    out["vertical_share"] = round(float((oh[5] + oh[6] + oh[4] * 0.5 + oh[7] * 0.5) / hv), 4)
    out["vectors"] = {"orientation": (oh / hv).round(5).tolist(),
                      "width": (wh / (wh.sum() or 1)).round(5).tolist()}
    return out


def layout(rgb: np.ndarray, line: np.ndarray, text: np.ndarray, frame: np.ndarray) -> dict:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[..., 0] *= 100 / 255
    lab[..., 1:] -= 128
    ring = np.zeros(text.shape, bool)
    b = 8
    ring[:b, :] = ring[-b:, :] = ring[:, :b] = ring[:, -b:] = True
    ring &= ~frame & ~text
    bg = np.median(lab[ring], axis=0) if ring.sum() > 50 else np.median(lab.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(lab - bg, axis=-1)
    dens = cv2.GaussianBlur((line > INK).astype(np.float32), (0, 0), 4)
    content = (cv2.GaussianBlur(dist, (0, 0), 2) > BG_DIST) | (dens > 0.08)
    content = cv2.morphologyEx(content.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
    figure = content & ~text & ~frame
    empty = ~content & ~text & ~frame

    def cells(m: np.ndarray) -> np.ndarray:
        return cv2.resize(m.astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)

    ct, cf, ce, cc = cells(text), cells(figure), cells(empty), cells(frame)
    stack = np.stack([ce, cf, ct, cc])        # 칸마다 가장 큰 쪽. 글자는 조금 우대(얇아서 면적이 작다)
    stack[2] *= 1.6
    grid = "".join("eftc"[k] for k in stack.argmax(0).ravel())
    n = text.size

    def centroid(m: np.ndarray) -> list | None:
        ys, xs = np.nonzero(m)
        return [round(float(xs.mean() / W), 4), round(float(ys.mean() / H), 4)] if xs.size else None

    weight = figure | text
    tc = centroid(text)
    return {
        "shares": {k: round(float(m.sum() / n), 4) for k, m in
                   (("text", text & ~frame), ("figure", figure), ("empty", empty), ("frame", frame))},
        "background_lab": [round(float(v), 1) for v in bg],
        "figure_centroid": centroid(figure),
        "text_centroid": tc,
        "title_band": None if tc is None else ("top" if tc[1] < 0.36 else "bottom" if tc[1] > 0.64 else "middle"),
        "weight_centroid": centroid(weight),
        "lr_balance": round(float(1 - abs(weight[:, : W // 2].sum() - weight[:, W // 2:].sum()) / max(weight.sum(), 1)), 4),
        "grid": grid,                                  # GW×GH, 행 우선. e/f/t/c
        "vectors": {"text": ct[::5, ::4].round(3).ravel().tolist(),       # 10×10 로 줄인 점유
                    "figure": cf[::5, ::4].round(3).ravel().tolist()},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    WEB_STROKES.mkdir(parents=True, exist_ok=True)
    todo = [p for p in sorted(POSTERS.glob("*.jpg")) if a.force or not (OUT / f"{p.stem}.json").exists()][: a.limit]
    print(f"{len(todo)} to analyze", flush=True)
    if not todo:
        return

    import torch
    from controlnet_aux import LineartDetector
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    det = LineartDetector.from_pretrained("lllyasviel/Annotators").to(dev)

    t0 = time.time()
    for n, p in enumerate(todo, 1):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
        except Exception as e:  # 받는 중인 파일
            print(f"skip {p.name}: {e}")
            continue
        small = im.resize((W, H), Image.LANCZOS)
        ln = det(im, detect_resolution=640, image_resolution=640).convert("L").resize((W, H), Image.BILINEAR)
        line = np.asarray(ln, dtype=np.float32) / 255
        text, frame, rects = masks(p.stem)
        res = {"file": p.name, "size": [W, H], "frame_rects": rects,
               "strokes": stroke_stats(line, ~text & ~frame),
               "layout": layout(np.asarray(small), line, text, frame)}
        (OUT / f"{p.stem}.json").write_text(json.dumps(res, ensure_ascii=False))
        ln.resize((240, 298), Image.LANCZOS).save(WEB_STROKES / f"{p.stem}.png", optimize=True)
        if n % 50 == 0 or n == len(todo):
            s, l = res["strokes"], res["layout"]["shares"]
            print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s  {p.stem[:28]:28s} ink {s['ink_share']:.3f} "
                  f"w {s['width_median']:.1f} coh {s['coherence']:.2f} | text {l['text']:.2f} fig {l['figure']:.2f} "
                  f"empty {l['empty']:.2f}", flush=True)


if __name__ == "__main__":
    main()
