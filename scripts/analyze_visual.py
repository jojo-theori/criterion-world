"""색, 대비, 구도선 분석. 커버별 JSON 을 data/analysis/visual/ 에 저장한다. 이미 처리한 커버는 건너뛴다.

두 버전을 함께 계산한다.
  full : 커버 전체
  art  : 텍스트 영역(data/regions/combined 의 모든 다각형)과 크라이테리온 띠·로고를 가린 그림 부분

크라이테리온 틀(frame)
  브랜드 글자(CRITERION, COLLECTION)가 어디서 읽혔는지로 템플릿을 판정하고, 템플릿의 고정 위치를 통째로 가린다.
  로고는 검출이 들쭉날쭉해서(10장 중 2장) 검출 결과 대신 템플릿 위치를 쓴다.
    modern-strip : 왼쪽 세로 띠 + 왼쪽 위 C 로고 (현행 블루레이 디자인)
    dvd-topband  : 위쪽 가로 띠 (옛 DVD 디자인)
    unknown      : 브랜드 글자 다각형만 넓혀서 가린다

지표 (사람이 읽는 값)
  color    : Lab k-means 팔레트(6색, 비율순), Hasler–Süsstrunk colorfulness, 채도 통계, color_mode
             (grayscale: 채도 p95 < 12 / single-hue: 유채색 픽셀의 색상이 한 계열에 몰림(hue_concentration > 0.9) / color)
  contrast : RMS contrast(L/100 의 표준편차), 명도 백분위, 어두운·밝은 픽셀 비율, 텍스트 영역별 글자 대 바탕 대비(WCAG 비)
  lines    : OpenCV LSD 선분(대각선 길이의 2% 이상). 방향 비율(수평 ±15°, 수직 ±15°, 나머지 대각), 선 밀도
  saliency : Spectral Residual saliency 의 무게중심, 퍼짐, 3분할선·교차점·중심까지 거리
  기타     : 에지 밀도(Canny), 좌우 대칭도

비교용 벡터 (vectors, 각각 합 1 로 정규화)
  color    : Lab 결합 히스토그램 L 4 × a 6 × b 6 = 144
  contrast : 명도 히스토그램 16
  lines    : 길이 가중 방향 히스토그램 12 (0–180°, 15° 간격)
  saliency : 8 × 10 격자 (가로 × 세로, 행 우선)

웹에서 그릴 흔적
  lines.segments : 긴 순서로 최대 300개, [x1, y1, x2, y2] 정규화 좌표(0–1, 원점 왼쪽 위)
  palette        : hex 와 비율

사용
  .venv/bin/python scripts/analyze_visual.py            # 새 커버만
  .venv/bin/python scripts/analyze_visual.py --force    # 전부 다시
  .venv/bin/python scripts/analyze_visual.py --render   # 확인용 이미지도 out/visual/ 에 그린다
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
REGIONS = ROOT / "data" / "regions" / "combined"
OUT = ROOT / "data" / "analysis" / "visual"
RENDER = ROOT / "out" / "visual"

WORK_LONG = 800          # 분석 해상도 (긴 변)
DILATE = 0.006           # 텍스트 다각형을 긴 변 대비 이만큼 넓혀서 가린다 (글자 가장자리 번짐)
K = 6                    # 팔레트 색 수
MIN_LINE = 0.02          # 대각선 대비 최소 선분 길이
MAX_SEGMENTS = 300

# 템플릿 고정 위치 (정규화 x0, y0, x1, y1). 10장에서 잰 값에 여백을 조금 더했다.
FRAMES = {
    "modern-strip": [(0.0, 0.535, 0.075, 0.93), (0.035, 0.025, 0.195, 0.165)],
    "dvd-topband": [(0.0, 0.0, 1.0, 0.075)],
}
BRAND_WORDS = ("criterion", "collection")


# ---------------------------------------------------------------- 마스크

def is_brand(text: str) -> bool:
    toks = re.findall(r"[a-z]+", text.lower())
    return any(difflib.SequenceMatcher(None, t, b).ratio() >= 0.75 for t in toks for b in BRAND_WORDS)


def bbox(poly, W, H):
    xs = [p[0] / W for p in poly]
    ys = [p[1] / H for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def detect_frame(doc: dict) -> tuple[str, list]:
    W, H = doc["width"], doc["height"]
    brand = [bbox(r["poly"], W, H) for r in doc["regions"] if r.get("text") and is_brand(r["text"])]
    if not brand:
        return "none", []
    cx = [(b[0] + b[2]) / 2 for b in brand]
    cy = [(b[1] + b[3]) / 2 for b in brand]
    if all(x < 0.07 for x in cx) and all(0.5 < y < 0.95 for y in cy):
        return "modern-strip", FRAMES["modern-strip"]
    if all(y < 0.08 for y in cy):
        return "dvd-topband", FRAMES["dvd-topband"]
    m = 0.01
    return "unknown", [(max(0, b[0] - m), max(0, b[1] - m), min(1, b[2] + m), min(1, b[3] + m)) for b in brand]


def in_rects(pt, rects) -> bool:
    return any(x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1 for x0, y0, x1, y1 in rects)


def build_masks(doc: dict, w: int, h: int):
    """text: 텍스트 다각형, frame: 템플릿 영역. 둘 다 True 가 가려지는 픽셀."""
    W, H = doc["width"], doc["height"]
    text = np.zeros((h, w), np.uint8)
    for r in doc["regions"]:
        pts = np.array([[x * w / W, y * h / H] for x, y in r["poly"]], np.int32)
        cv2.fillPoly(text, [pts], 1)
    r = max(1, int(round(DILATE * max(w, h))))
    text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    kind, rects = detect_frame(doc)
    frame = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in rects:
        frame[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))] = 1
    return text.astype(bool), frame.astype(bool), kind, rects


# ---------------------------------------------------------------- 색

def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rel_luminance(rgb01):
    lin = srgb_to_linear(np.asarray(rgb01, np.float64))
    return float(lin @ [0.2126, 0.7152, 0.0722])


def lab_to_hex(lab):
    px = np.array([[lab]], np.float32)
    rgb = cv2.cvtColor(px, cv2.COLOR_Lab2RGB)[0, 0]
    return "#" + "".join(f"{int(round(np.clip(v, 0, 1) * 255)):02x}" for v in rgb)


def palette(lab_px: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(0)
    if len(lab_px) > 40000:
        lab_px = lab_px[rng.choice(len(lab_px), 40000, replace=False)]
    k = min(K, len(lab_px))
    cv2.setRNGSeed(0)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)
    _, labels, centers = cv2.kmeans(lab_px.astype(np.float32), k, None, crit, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.ravel(), minlength=k) / len(labels)
    order = np.argsort(-counts)
    return [dict(hex=lab_to_hex(centers[i]), lab=[round(float(v), 1) for v in centers[i]],
                 share=round(float(counts[i]), 4)) for i in order]


def color_stats(rgb01: np.ndarray, lab: np.ndarray, valid: np.ndarray) -> tuple[dict, list]:
    px = rgb01[valid] * 255
    rg = px[:, 0] - px[:, 1]
    yb = 0.5 * (px[:, 0] + px[:, 1]) - px[:, 2]
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    L, a, b = (lab[..., i][valid] for i in range(3))
    chroma = np.hypot(a, b)
    hue = np.arctan2(b, a)
    chromatic = chroma > 8
    if chromatic.sum() > 50:
        concentration = float(np.hypot(np.cos(hue[chromatic]).mean(), np.sin(hue[chromatic]).mean()))
        mean_hue = float(np.degrees(np.arctan2(np.sin(hue[chromatic]).mean(), np.cos(hue[chromatic]).mean())) % 360)
    else:
        concentration, mean_hue = 0.0, None
    p95 = float(np.percentile(chroma, 95))
    mode = "grayscale" if p95 < 12 else "single-hue" if concentration > 0.9 else "color"

    hist, _ = np.histogramdd(np.stack([L, a, b], 1),
                             bins=[np.linspace(0, 100, 5), [-128, -30, -10, 0, 10, 30, 128], [-128, -30, -10, 0, 10, 30, 128]])
    vec = (hist.ravel() / max(1, hist.sum())).tolist()
    return dict(
        palette=palette(np.stack([L, a, b], 1)),
        colorfulness=round(float(colorfulness), 2),
        chroma_mean=round(float(chroma.mean()), 2),
        chroma_p95=round(p95, 2),
        chromatic_share=round(float(chromatic.mean()), 4),
        hue_mean_deg=None if mean_hue is None else round(mean_hue, 1),
        hue_concentration=round(concentration, 3),
        color_mode=mode,
    ), vec


# ---------------------------------------------------------------- 대비

def contrast_stats(lab: np.ndarray, valid: np.ndarray) -> tuple[dict, list]:
    L = lab[..., 0][valid]
    p = np.percentile(L, [5, 25, 50, 75, 95])
    hist, _ = np.histogram(L, bins=16, range=(0, 100))
    return dict(
        rms_contrast=round(float((L / 100).std()), 4),
        lightness_mean=round(float(L.mean()), 2),
        lightness_p5_p25_p50_p75_p95=[round(float(v), 1) for v in p],
        dynamic_range=round(float(p[4] - p[0]), 1),
        dark_share=round(float((L < 20).mean()), 4),
        light_share=round(float((L > 80).mean()), 4),
    ), (hist / max(1, hist.sum())).tolist()


def text_contrast(doc: dict, rgb01: np.ndarray, lab: np.ndarray, frame_rects: list) -> list[dict]:
    """텍스트 영역 하나마다 안쪽 픽셀을 Otsu 로 두 무리로 나눠 글자(면적이 작은 쪽)와 바탕의 대비를 잰다.
    크라이테리온 틀 안의 영역은 뺀다."""
    W, H = doc["width"], doc["height"]
    h, w = lab.shape[:2]
    out = []
    for r in doc["regions"]:
        x0, y0, x1, y1 = bbox(r["poly"], W, H)
        if in_rects(((x0 + x1) / 2, (y0 + y1) / 2), frame_rects):
            continue
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.array([[x * w / W, y * h / H] for x, y in r["poly"]], np.int32)], 1)
        m = m.astype(bool)
        if m.sum() < 30:
            continue
        L8 = np.clip(lab[..., 0][m] * 2.55, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(L8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        hi = L8 > t
        if hi.all() or (~hi).all():
            continue
        groups = sorted([hi, ~hi], key=lambda g: g.sum())
        glyph, ground = (rgb01[m][g].mean(0) for g in groups)
        la, lb = rel_luminance(glyph), rel_luminance(ground)
        ratio = (max(la, lb) + 0.05) / (min(la, lb) + 0.05)
        out.append(dict(
            text=r.get("text"), source=r["source"],
            area=round(float(m.mean()), 5),
            glyph_hex="#" + "".join(f"{int(v * 255):02x}" for v in glyph),
            ground_hex="#" + "".join(f"{int(v * 255):02x}" for v in ground),
            glyph_lighter=bool(la > lb),
            wcag_ratio=round(float(ratio), 2),
        ))
    return out


# ---------------------------------------------------------------- 구도선

def line_stats(gray: np.ndarray, valid: np.ndarray) -> tuple[dict, list]:
    h, w = gray.shape
    diag = float(np.hypot(w, h))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    found = lsd.detect(gray)[0]
    segs = [] if found is None else found.reshape(-1, 4)
    keep = []
    for x1, y1, x2, y2 in segs:
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < MIN_LINE * diag:
            continue
        t = np.linspace(0, 1, 9)
        xs = np.clip((x1 + (x2 - x1) * t).astype(int), 0, w - 1)
        ys = np.clip((y1 + (y2 - y1) * t).astype(int), 0, h - 1)
        if valid[ys, xs].mean() < 2 / 3:          # 선분의 1/3 넘게 가려진 영역에 걸치면 버린다
            continue
        angle = float(np.degrees(np.arctan2(-(y2 - y1), x2 - x1)) % 180)
        keep.append((length, angle, (x1, y1, x2, y2)))

    lengths = np.array([k[0] for k in keep]) if keep else np.zeros(0)
    angles = np.array([k[1] for k in keep]) if keep else np.zeros(0)
    total = float(lengths.sum())
    horiz = (angles < 15) | (angles >= 165)
    vert = (angles >= 75) & (angles < 105)
    hist, _ = np.histogram(angles, bins=12, range=(0, 180), weights=lengths)
    share = lambda m: round(float(lengths[m].sum() / total), 4) if total else 0.0
    keep.sort(key=lambda k: -k[0])
    stats = dict(
        count=len(keep),
        density=round(total / diag / max(valid.mean(), 1e-6), 3),   # 보이는 면적 기준 대각선 몇 개 분량인지
        horizontal_share=share(horiz),
        vertical_share=share(vert),
        diagonal_share=share(~horiz & ~vert),
        segments=[[round(float(x1 / w), 4), round(float(y1 / h), 4), round(float(x2 / w), 4), round(float(y2 / h), 4)]
                  for _, _, (x1, y1, x2, y2) in keep[:MAX_SEGMENTS]],
    )
    return stats, (hist / total).tolist() if total else [0.0] * 12


def saliency_stats(bgr: np.ndarray, valid: np.ndarray) -> tuple[dict, list, np.ndarray]:
    """가려진 영역이 있으면 먼저 inpaint 로 메운 뒤 계산한다. 원본에서 계산하고 가리면 글자의 번짐이 마스크 둘레에 남는다.
    Spectral Residual 은 내부에서 64px 로 줄여 계산하므로 256px 에서 메워도 충분하다."""
    h, w = valid.shape
    src = bgr
    if not valid.all():
        s = 256 / max(h, w)
        small = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        hole = cv2.resize((~valid).astype(np.uint8), small.shape[1::-1], interpolation=cv2.INTER_NEAREST)
        src = cv2.inpaint(small, cv2.dilate(hole, np.ones((3, 3), np.uint8)), 5, cv2.INPAINT_TELEA)
    ok, sal = cv2.saliency.StaticSaliencySpectralResidual_create().computeSaliency(src)
    sal = cv2.resize(sal.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    sal = cv2.GaussianBlur(sal, (0, 0), max(w, h) / 100)
    sal = sal * valid
    sal = sal / (sal.max() or 1)
    wgt = sal ** 2
    ys, xs = np.mgrid[0:h, 0:w]
    xs, ys = (xs + 0.5) / w, (ys + 0.5) / h
    s = wgt.sum() or 1
    cx, cy = float((wgt * xs).sum() / s), float((wgt * ys).sum() / s)
    spread = float(np.sqrt((wgt * ((xs - cx) ** 2 + (ys - cy) ** 2)).sum() / s))
    thirds = (1 / 3, 2 / 3)
    grid = cv2.resize(sal, (8, 10), interpolation=cv2.INTER_AREA)
    stats = dict(
        centroid=[round(cx, 4), round(cy, 4)],
        spread=round(spread, 4),
        thirds_line_dist=round(min(min(abs(cx - t) for t in thirds), min(abs(cy - t) for t in thirds)), 4),
        thirds_point_dist=round(min(np.hypot(cx - a, cy - b) for a in thirds for b in thirds), 4),
        center_dist=round(float(np.hypot(cx - 0.5, cy - 0.5)), 4),
    )
    return stats, (grid.ravel() / (grid.sum() or 1)).tolist(), sal


def misc_stats(gray: np.ndarray, valid: np.ndarray) -> dict:
    med = float(np.median(gray[valid]))
    edges = cv2.Canny(gray, int(max(0, 0.66 * med)), int(min(255, 1.33 * med) or 50)) > 0
    g = gray.astype(np.float32) / 255
    both = valid & valid[:, ::-1]
    sym = 1 - float(np.abs(g - g[:, ::-1])[both].mean()) if both.any() else None
    return dict(edge_density=round(float(edges[valid].mean()), 4),
                lr_symmetry=None if sym is None else round(sym, 4))


# ---------------------------------------------------------------- 실행

def round_vec(v):
    return [round(float(x), 5) for x in v]


def analyze_version(doc, rgb01, lab, gray, bgr, valid, frame_rects, with_text: bool):
    color, vc = color_stats(rgb01, lab, valid)
    contrast, vk = contrast_stats(lab, valid)
    if with_text:
        regions = text_contrast(doc, rgb01, lab, frame_rects)
        ratios = np.array([r["wcag_ratio"] for r in regions])
        areas = np.array([r["area"] for r in regions])
        contrast["text"] = dict(
            regions=regions,
            weighted_wcag_ratio=round(float((ratios * areas).sum() / areas.sum()), 2) if len(regions) else None,
            min_wcag_ratio=round(float(ratios.min()), 2) if len(regions) else None,
        )
    lines, vl = line_stats(gray, valid)
    sal, vs, salmap = saliency_stats(bgr, valid)
    res = dict(visible_share=round(float(valid.mean()), 4), color=color, contrast=contrast, lines=lines,
               saliency=sal, **misc_stats(gray, valid),
               vectors=dict(color=round_vec(vc), contrast=round_vec(vk), lines=round_vec(vl), saliency=round_vec(vs)))
    return res, salmap


def analyze(img_path: Path, doc: dict):
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    H0, W0 = bgr.shape[:2]
    s = WORK_LONG / max(H0, W0)
    bgr = cv2.resize(bgr, (round(W0 * s), round(H0 * s)), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    rgb01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    lab = cv2.cvtColor(rgb01, cv2.COLOR_RGB2Lab)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    text_m, frame_m, kind, rects = build_masks(doc, w, h)
    full, sal_full = analyze_version(doc, rgb01, lab, gray, bgr, np.ones((h, w), bool), rects, True)
    art_valid = ~(text_m | frame_m)
    art, sal_art = analyze_version(doc, rgb01, lab, gray, bgr, art_valid, rects, False)
    out = dict(file=img_path.name, width=W0, height=H0, work_size=[w, h],
               frame=dict(kind=kind, rects=[[round(v, 4) for v in r] for r in rects]),
               masked=dict(text_share=round(float(text_m.mean()), 4), frame_share=round(float(frame_m.mean()), 4)),
               full=full, art=art)
    return out, dict(bgr=bgr, text=text_m, frame=frame_m, sal_full=sal_full, sal_art=sal_art)


# ---------------------------------------------------------------- 확인용 이미지

def draw_panel(bgr, res, valid, salmap):
    h, w = bgr.shape[:2]
    img = bgr.copy()
    if valid is not None:
        hatch = img.copy()
        hatch[~valid] = (hatch[~valid] * 0.25 + np.array([255, 0, 255]) * 0.35).astype(np.uint8)
        img = hatch
    for t in (1 / 3, 2 / 3):
        cv2.line(img, (int(t * w), 0), (int(t * w), h), (200, 200, 200), 1)
        cv2.line(img, (0, int(t * h)), (w, int(t * h)), (200, 200, 200), 1)
    for x1, y1, x2, y2 in res["lines"]["segments"]:
        a = np.degrees(np.arctan2(-(y2 - y1), x2 - x1)) % 180
        col = (0, 255, 255) if a < 15 or a >= 165 else (255, 255, 0) if 75 <= a < 105 else (0, 140, 255)
        cv2.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), col, 2)
    cx, cy = res["saliency"]["centroid"]
    cv2.circle(img, (int(cx * w), int(cy * h)), 12, (0, 0, 255), 3)
    heat = cv2.applyColorMap((salmap * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    bar = np.zeros((40, w, 3), np.uint8)
    x = 0
    for c in res["color"]["palette"]:
        cw = int(round(c["share"] * w))
        rgb = [int(c["hex"][i:i + 2], 16) for i in (1, 3, 5)]
        bar[:, x:x + cw] = rgb[::-1]
        x += cw
    return np.vstack([np.hstack([img, heat]), np.hstack([bar, np.zeros_like(bar)])])


def render(name: str, data: dict, maps: dict) -> None:
    RENDER.mkdir(parents=True, exist_ok=True)
    valid = ~(maps["text"] | maps["frame"])
    top = draw_panel(maps["bgr"], data["full"], None, maps["sal_full"])
    bot = draw_panel(maps["bgr"], data["art"], valid, maps["sal_art"])
    sheet = np.vstack([top, np.full((8, top.shape[1], 3), 255, np.uint8), bot])
    sheet = cv2.resize(sheet, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(RENDER / f"{name}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def main() -> None:
    force = "--force" in sys.argv
    do_render = "--render" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    done = skipped = missing = 0
    for img in sorted(POSTERS.glob("*.jpg")):
        dest = OUT / f"{img.stem}.json"
        reg = REGIONS / f"{img.stem}.json"
        if not reg.exists():
            print(f"  텍스트 영역 없음, 건너뜀: {img.name} (run_text_regions.py 먼저)")
            missing += 1
            continue
        if dest.exists() and not force and not (do_render and not (RENDER / f"{img.stem}.jpg").exists()):
            skipped += 1
            continue
        t = time.time()
        data, maps = analyze(img, json.loads(reg.read_text()))
        data["elapsed_ms"] = int((time.time() - t) * 1000)
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        if do_render:
            render(img.stem, data, maps)
        a = data["art"]
        print(f"  {img.stem:32} {data['frame']['kind']:13} visible {a['visible_share']:.2f}  "
              f"{a['color']['color_mode']:9} cf {a['color']['colorfulness']:5.1f}  rms {a['contrast']['rms_contrast']:.3f}  "
              f"H/V/D {a['lines']['horizontal_share']:.2f}/{a['lines']['vertical_share']:.2f}/{a['lines']['diagonal_share']:.2f}  "
              f"{data['elapsed_ms']}ms")
        done += 1
    print(f"analyzed {done}, skipped {skipped}, no regions {missing} -> {OUT}")


if __name__ == "__main__":
    main()
