"""구조 성질 점수: 대칭 · 기울기 · 중심 집중. Lab 의 Structure 지도용(2026-11-18 사용자: 의도적으로 대칭인 것, 기울어진 것,
중심 집중된 것이 명확하게 모이게).

비슷한 것끼리 모으는 임베딩 대신 성질마다 전용 측정기로 0–1 점수를 재고, 뚜렷한 커버만 그 성질의 무리로 보낸다.
입력은 글자 지운 그림(out/art, embed_more.py 가 만든다). 지운 자리는 재지 않는다.

  대칭      윤곽 세기 지도를 좌우로 뒤집어 겹침(min 합 / max 합). 축을 좌우로 옮겨 가장 잘 맞는 자리를 찾는다.
            윤곽이 거의 없는 커버(빈 배경)는 점수를 깎는다.
  기울기    긴 선분의 방향 분포에서 대각선 구간(12–78°)에 몰린 봉우리의 집중도, 기운 글자 영역, 가장 큰 얼굴의 기울기 중 가장 큰 것.
  중심 집중 주제 덩어리 하나가 가운데 있고 왼쪽·오른쪽·위에 여백이 고르게 남은 정도. 주제는 두 가지로 찾아 큰 점수를 쓴다:
            전경 분리 모델(rembg 의 isnet-general-use, 마스크는 out/fg 에 캐시)과, 단색 배경이면 배경색과 다른 부분.
            아래쪽은 여백을 요구하지 않는다(상반신은 아래 가장자리에 닿는다).

출력
  data/analysis/structure_scores.json   {파일: {sym, axis, tilt, angle, tilt_by, center, cx, cy, size, group}}
  data/analysis/embed/structure-map.npz (files, xy, vectors)  Lab 지도 좌표(점수에서 직접 만든 배치)
  out/structure/top_<성질>.jpg          성질마다 점수 상위 커버 시트(정확도 확인용)

사용 (.venv-ml)
  .venv-ml/bin/python scripts/structure_scores.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

import embed_structure as ES

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
ART = ROOT / "out" / "art"
OUT = ROOT / "out" / "structure"
FG = ROOT / "out" / "fg"
GW, GH = 96, 120                  # 재는 격자
# 이 점수부터 그 성질의 무리. 순서 = 지도에서 도는 순서(위에서 시계 방향)
THR = {"sym": 0.5, "center": 0.5, "space": 0.5, "frame": 0.5, "grid": 0.5, "parallel": 0.5, "tilt": 0.5,
       "vup": 0.5, "vdown": 0.5, "vanish": 0.5, "circle": 0.7, "scw": 0.5, "sccw": 0.5, "spiral": 0.5}
NAMES = {"sym": "Symmetric", "center": "Centered", "space": "Negative space", "frame": "Frame in frame", "grid": "Grid",
         "parallel": "Parallel lines", "tilt": "Diagonal", "vup": "Vanishing up", "vdown": "Vanishing down", "vanish": "Vanishing point",
         "circle": "Circular", "scw": "Spiral clockwise", "sccw": "Spiral counterclockwise", "spiral": "Spiral"}
# 2차 선별: 묘사 글(Sonnet, captions_sonnet_v3.json 의 Composition·Visual idea 줄)에 그 성질을 가리키는 낱말이 있는가
WORDS = {"sym": ["symmetr", "mirror"], "center": ["centered", "dead center", "in the center", "central"],
         "space": ["negative space", "empty", "blank", "bare ", "expanse", "void"],
         "frame": ["frame within", "framed by", "framed in", "inside a frame", "border", "window", "doorway", "inset", "picture frame",
                   "oval frame", "vignette"],
         "grid": ["grid", "panels", "tiles", "tiled", "mosaic", "checkerboard", "contact sheet", "rows and columns"],
         "parallel": ["stripe", "bars", "slats", "parallel", "bands", "blinds", "rows of"],
         "tilt": ["diagonal", "tilted", "slant", "canted", "angled"],
         "vanish": ["vanishing", "converg", "reced", "perspective", "corridor", "tunnel", "hallway", "railroad", "tracks", "road "],
         "circle": ["circle", "circular", "concentric", "ring", "disc", "round", "target", "sphere", "orb", "medallion", "oval"],
         "spiral": ["spiral"], "swirl": ["swirl", "vortex"]}
RESCUE = ["sym", "center", "space", "frame", "grid", "tilt", "circle"]       # 측정이 믿을 만한 성질: 글이 맞장구치면 애매한 점수(.25+)를 올려 준다
NEED = ["parallel", "vanish", "spiral"]                                          # 측정에 잡음이 많은 성질: 글이 맞장구칠 때만 인정
SUB = {"sym": "axis", "center": "size", "space": "space_share", "frame": "frame_w", "grid": "grid_n", "parallel": "par_angle",
       "tilt": "angle", "vup": "vx", "vdown": "vx", "vanish": "vy", "scw": "ccx", "sccw": "ccx", "circle": "ccx", "spiral": "spiral_turn"}


def symmetry(E: np.ndarray, V: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """(점수, 축의 x 자리 0–1). E = 윤곽 세기(0–1), V = 잴 수 있는 자리, T = 흐린 밝기.
    윤곽 겹침만 보면 촘촘한 질감(풀밭, 폐허)이 높게 나온다: 밝기 배치도 거울과 맞아야 한다(상관)."""
    best, axis = 0.0, 0.5
    for k in range(-14, 15, 2):                       # 뒤집은 그림을 k 칸 옮김 → 축 = (GW-1+k)/2
        F, FV, FT = np.roll(E[:, ::-1], k, 1), np.roll(V[:, ::-1], k, 1), np.roll(T[:, ::-1], k, 1)
        ok = V & FV
        if k > 0:
            ok[:, :k] = False
        elif k < 0:
            ok[:, k:] = False
        if ok.mean() < 0.55:
            continue
        a, b = E[ok], F[ok]
        den = np.maximum(a, b).sum()
        s = np.minimum(a, b).sum() / (den + 1e-6)
        ta, tb = T[ok] - T[ok].mean(), FT[ok] - FT[ok].mean()
        r = float((ta * tb).sum() / (math.sqrt((ta ** 2).sum() * (tb ** 2).sum()) + 1e-6))
        s *= max(0.0, r) ** 1.5
        s *= min(1.0, den / ok.sum() / 0.05)          # 윤곽이 거의 없으면 깎는다
        s *= 1 - 0.012 * abs(k)                       # 가운데 축을 조금 더 친다
        if s > best:
            best, axis = s, (GW - 1 + k) / 2 / (GW - 1)
    return float(best), float(axis)


def line_tilt(gray: np.ndarray, er: np.ndarray) -> tuple[float, float]:
    """(집중도 0–1, 봉우리 각도 0–180°). 화면 좌표라 45° 는 오른쪽 위로 오르는 선"""
    h, w = gray.shape
    diag = math.hypot(w, h)
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    hist = np.zeros(36)
    if found is not None:
        for x1, y1, x2, y2 in found.reshape(-1, 4):
            L = math.hypot(x2 - x1, y2 - y1)
            if L < 0.05 * diag:
                continue
            t = np.linspace(0, 1, 9)
            xs = np.clip((x1 + (x2 - x1) * t).astype(int), 0, w - 1)
            ys = np.clip((y1 + (y2 - y1) * t).astype(int), 0, h - 1)
            if er[ys, xs].mean() > 1 / 3:
                continue
            ang = math.degrees(math.atan2(-(y2 - y1), x2 - x1)) % 180
            hist[int(ang // 5) % 36] += L
    total = hist.sum()
    if total < 0.8 * diag:
        return 0.0, 0.0
    sm = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))
    centers = np.arange(36) * 5 + 2.5
    dg = ((centers > 12) & (centers < 78)) | ((centers > 102) & (centers < 168))
    k = int(np.argmax(np.where(dg, sm, -1)))
    conc = sum(hist[(k + d) % 36] for d in (-1, 0, 1)) / total
    return float(conc * min(1.0, total / (1.6 * diag))), float(centers[k])


def text_tilt(stem: str) -> tuple[float, float]:
    """(기운 글자의 정도 0–1, 각도). 길쭉한 글자 영역의 긴 축이 가로·세로에서 8° 넘게 벗어난 것"""
    p = ES.REGIONS / f"{stem}.json"
    if not p.exists():
        return 0.0, 0.0
    d = json.loads(p.read_text())
    W, H = d["width"], d["height"]
    used = [r for r, u in zip([x for x in d["regions"] if len(x["poly"]) >= 3], ES.text_regions(stem)) if u[4]]
    tot = tilted = 0.0
    ang_at = 0.0
    for r in used:
        (_, _), (rw, rh), a = cv2.minAreaRect(np.array(r["poly"], np.float32))
        if min(rw, rh) < 1:
            continue
        area = rw * rh / (W * H)
        tot += area
        if max(rw, rh) / min(rw, rh) < 2.2:
            continue
        long_ang = (a if rw >= rh else a + 90) % 180         # 화면에서 시계 방향
        dev = min(long_ang % 90, 90 - long_ang % 90)
        if dev >= 8:
            tilted += area * min(1.0, dev / 15)
            ang_at = (180 - long_ang) % 180
    if tot < 0.01:
        return 0.0, 0.0
    return float(min(1.0, tilted / tot) * min(1.0, tilted / 0.02)), float(ang_at)


def face_tilt(faces: list, w: int, h: int) -> tuple[float, float]:
    big = max(faces, key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), default=None)
    if not big or (big["box"][2] - big["box"][0]) * (big["box"][3] - big["box"][1]) < 0.03 or len(big.get("eyes") or []) < 2:
        return 0.0, 0.0
    (ax, ay), (bx, by) = sorted(big["eyes"])
    roll = math.degrees(math.atan2(-(by - ay) * h, (bx - ax) * w))        # + = 오른쪽 눈이 위
    return float(np.clip((abs(roll) - 10) / 25, 0, 1)), float(roll % 180)


def centered(bgr: np.ndarray, er: np.ndarray) -> tuple[float, float, float, float]:
    """(점수, 주제 중심 x, y, 주제 넓이). 단색 배경 위에 가운데 모인 덩어리"""
    lab = cv2.cvtColor(cv2.resize(bgr, (GW, GH), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2LAB).astype(np.float32)
    V = ~cv2.resize(er.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.mgrid[0:GH, 0:GW]
    u, v = (xs + 0.5) / GW, (ys + 0.5) / GH
    ring = ((u < 0.1) | (u > 0.9) | (v < 0.08) | (v > 0.92)) & V
    if ring.sum() < 200:
        return 0.0, 0.5, 0.5, 0.0
    bg = np.median(lab[ring], 0)
    D = np.linalg.norm(lab - bg, axis=-1) * np.array(1.0)
    uniform = float(np.clip(1 - (np.percentile(D[ring], 80) - 8) / 22, 0, 1))        # 가장자리 띠의 색이 고른가
    M = np.clip((D - 14) / 30, 0, 1) * V
    tot = M.sum()
    size = float((M > 0.4).mean())
    if tot < 1 or size < 0.02:
        return 0.0, 0.5, 0.5, size
    cx, cy = float((M * u).sum() / tot), float((M * v).sum() / tot)
    inside = float(M[((u - 0.5) / 0.34) ** 2 + ((v - 0.5) / 0.36) ** 2 <= 1].sum() / tot)
    edge = float(M[ring].sum() / tot)
    s = uniform * np.clip((inside - 0.45) / 0.4, 0, 1) * np.clip(1 - math.hypot(cx - 0.5, (cy - 0.5) * 0.7) / 0.13, 0, 1) \
        * np.clip(1 - edge / 0.06, 0, 1) * np.clip((0.6 - size) / 0.2, 0, 1)
    return float(s), cx, cy, size


def segments(gray: np.ndarray, er: np.ndarray) -> np.ndarray:
    """긴 선분 [u1, v1, u2, v2, 길이, 각도°]. 좌표·길이는 그림 높이를 1 로(가로는 0–w/h). 지운 자리와 맨 가장자리 선은 뺀다."""
    h, w = gray.shape
    found = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD).detect(gray)[0]
    out = []
    if found is not None:
        for x1, y1, x2, y2 in found.reshape(-1, 4):
            L = math.hypot(x2 - x1, y2 - y1) / h
            if L < 0.06:
                continue
            t = np.linspace(0, 1, 9)
            xs = np.clip((x1 + (x2 - x1) * t).astype(int), 0, w - 1)
            ys = np.clip((y1 + (y2 - y1) * t).astype(int), 0, h - 1)
            if er[ys, xs].mean() > 1 / 3:
                continue
            m = 0.025 * h
            if (max(x1, x2) < m or min(x1, x2) > w - m) or (max(y1, y2) < m or min(y1, y2) > h - m):
                continue
            out.append([x1 / h, y1 / h, x2 / h, y2 / h, L, math.degrees(math.atan2(-(y2 - y1), x2 - x1)) % 180])
    return np.array(out, np.float64).reshape(-1, 6)


def parallel(sg: np.ndarray) -> tuple[float, float]:
    """(점수, 각도). 같은 방향의 긴 선이 여러 줄 떨어져 놓인 정도(줄무늬, 반복되는 선)"""
    if len(sg) < 4 or sg[:, 4].sum() < 1.0:
        return 0.0, 0.0
    hist = np.zeros(36)
    np.add.at(hist, (sg[:, 5] // 5).astype(int) % 36, sg[:, 4])
    sm = hist + np.roll(hist, 1) + np.roll(hist, -1)
    k = int(sm.argmax())
    ang = k * 5 + 2.5
    d = np.abs((sg[:, 5] - ang + 90) % 180 - 90)
    sel = sg[d <= 7.5]
    conc = sel[:, 4].sum() / sg[:, 4].sum()
    t = math.radians(ang)
    mx, my = (sel[:, 0] + sel[:, 2]) / 2, (sel[:, 1] + sel[:, 3]) / 2
    rho = mx * math.sin(t) + my * math.cos(t)                 # 선에 수직인 방향의 자리
    acc = {}
    for r, L in zip(rho, sel[:, 4]):
        acc[int(r // 0.04)] = acc.get(int(r // 0.04), 0) + L
    rows = sum(1 for v in acc.values() if v >= 0.12)
    return float(np.clip((rows - 3) / 4, 0, 1) * np.clip((conc - 0.35) / 0.35, 0, 1)), ang


def vanishing(sg: np.ndarray, a: float) -> tuple[float, float, float]:
    """(점수, 소실점 x 0–1, y 0–1). 방향이 서로 다른 선 여럿이 한 점으로 모이는 정도. 점은 그림 밖(위·아래 반 장)까지."""
    sg = sg[np.argsort(-sg[:, 4])][:70]
    n = len(sg)
    if n < 5:
        return 0.0, 0.5, 0.5
    p1, p2 = sg[:, :2], sg[:, 2:4]
    mid, d = (p1 + p2) / 2, (p2 - p1) / sg[:, 4:5]
    total = max(sg[:, 4].sum(), 1.5)
    best = (0.0, 0.5 * a, 0.5)
    for i in range(n):
        for j in range(i + 1, n):
            cr = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
            if abs(cr) < 0.17:                                  # 10° 보다 가까운 방향은 건너뛴다
                continue
            ti = ((mid[j, 0] - mid[i, 0]) * d[j, 1] - (mid[j, 1] - mid[i, 1]) * d[j, 0]) / cr
            P = mid[i] + ti * d[i]
            if not (-0.6 * a <= P[0] <= 1.6 * a and -0.6 <= P[1] <= 1.6):
                continue
            v = P - mid
            dist = np.linalg.norm(v, axis=1)
            err = np.abs(v[:, 0] * d[:, 1] - v[:, 1] * d[:, 0]) / (dist + 1e-9)       # 선분 방향과 점 방향 사이 각의 sin
            ok = (err < 0.045) & (dist > sg[:, 4] / 2 + 0.06)                         # 2.6° 안, 점이 선분 위에 있지 않음
            if ok.sum() < 5:
                continue
            angs = np.sort(sg[ok, 5])
            spread = 180 - np.max(np.diff(np.concatenate([angs, [angs[0] + 180]])))   # 모인 선들의 방향 폭
            if spread < 25:
                continue
            sc = sg[ok, 4].sum() / total * min(1.0, spread / 50)
            if sc > best[0]:
                best = (sc, P[0], P[1])
    return float(np.clip((best[0] - 0.2) / 0.35, 0, 1)), float(best[1] / a), float(best[2])


def grid_frame(sg: np.ndarray, a: float) -> tuple[float, int, float, list]:
    """(그리드 점수, 칸 나누는 선 수, 프레임 점수, 프레임 상자 [x0, y0, x1, y1]). 가로·세로 선분을 자리별로 모아 긴 선을 찾는다.
    그리드 = 그림을 가로지르는 가로선과 세로선이 여럿. 프레임 속 프레임 = 가장자리에서 들어온 네 변이 사각형을 이룸."""
    H = sg[(sg[:, 5] < 5) | (sg[:, 5] > 175)]
    Vt = sg[np.abs(sg[:, 5] - 90) < 5]

    def lines(S, pos_i, a_i, b_i, span):          # 같은 자리(2% 칸)의 선분을 이어 붙임 → [자리, 시작, 끝, 덮은 길이]
        acc = {}
        for r in S:
            k = int(round((r[pos_i[0]] + r[pos_i[1]]) / 2 / 0.02))
            lo, hi = sorted((r[a_i], r[b_i]))
            e = acc.setdefault(k, [k * 0.02, lo, hi, 0.0])
            e[1], e[2], e[3] = min(e[1], lo), max(e[2], hi), e[3] + hi - lo
        return [e for e in acc.values() if e[3] >= 0.25 * span]
    hl, vl = lines(H, (1, 3), 0, 2, a), lines(Vt, (0, 2), 1, 3, 1.0)
    nh = sum(1 for e in hl if e[3] >= 0.6 * a and 0.1 < e[0] < 0.9)
    nv = sum(1 for e in vl if e[3] >= 0.6 and 0.16 * a < e[0] < 0.9 * a)
    grid = float(np.clip((nh + nv - 2) / 3, 0, 1)) if nh >= 1 and nv >= 1 else 0.0
    best, box = 0.0, [0, 0, 1, 1]
    tops, bots = [e for e in hl if 0.05 < e[0] < 0.42], [e for e in hl if 0.58 < e[0] < 0.95]
    lefts, rights = [e for e in vl if 0.13 * a < e[0] < 0.42 * a], [e for e in vl if 0.58 * a < e[0] < 0.95 * a]
    for t in tops:
        for b in bots:
            for l in lefts:
                for r in rights:
                    wd, ht = r[0] - l[0], b[0] - t[0]
                    cov = min(min(t[3], wd) / wd, min(b[3], wd) / wd, min(l[3], ht) / ht, min(r[3], ht) / ht)
                    # 변의 끝이 모서리 근처에서 만나야 한다(그림을 가로지르는 선은 프레임이 아니다)
                    fit = all(abs(e[1] - lo) < 0.12 and abs(e[2] - hi) < 0.12 for e, lo, hi in
                              ((t, l[0], r[0]), (b, l[0], r[0]), (l, t[0], b[0]), (r, t[0], b[0])))
                    if fit and cov > best:
                        best, box = cov, [l[0] / a, t[0], r[0] / a, b[0]]
    return grid, nh + nv, float(np.clip((best - 0.4) / 0.4, 0, 1)), [round(float(x), 3) for x in box]


def circular(g: np.ndarray, V: np.ndarray, a: float) -> tuple[float, float, float, float, float]:
    """(원형 점수, 중심 x, y, 스파이럴 점수, 도는 방향). 윤곽이 한 중심을 둘러 도는 정도(동심원, 원, 소용돌이).
    스파이럴 = 그 중심의 로그 극좌표에서 윤곽이 한쪽으로 기운 정도. 방향 + = 바깥으로 갈수록 시계 방향."""
    gs = cv2.GaussianBlur(cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA), (0, 0), 1.0)
    gx, gy = cv2.Sobel(gs, cv2.CV_32F, 1, 0), cv2.Sobel(gs, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    strong = (mag > max(6.0, np.percentile(mag, 75))) & V
    if strong.sum() < 300:
        return 0.0, 0.5, 0.5, 0.0, 0.0
    ys, xs = np.nonzero(strong)
    u, v = (xs + 0.5) / GW * a, (ys + 0.5) / GH
    nx, ny, wgt = gx[ys, xs] / mag[ys, xs], gy[ys, xs] / mag[ys, xs], np.minimum(mag[ys, xs], 60)
    best = (0.0, 0.5, 0.5)
    for cy in np.arange(0.25, 0.76, 0.1):
        for cxn in np.arange(0.3, 0.76, 0.1):
            du, dv = u - cxn * a, v - cy
            r = np.hypot(du, dv)
            m = (r > 0.07) & (r < 0.42)
            if m.sum() < 200:
                continue
            al = ((nx[m] * du[m] + ny[m] * dv[m]) / r[m]) ** 2          # 1 = 윤곽이 중심을 둘러 돈다, 0 = 중심에서 뻗어 나간다
            sec = (np.arctan2(dv[m], du[m]) // (math.pi / 4)).astype(int) % 8
            ws = wgt[m]
            share = np.bincount(sec, ws, 8) / ws.sum()
            secal = np.bincount(sec, ws * al, 8) / (np.bincount(sec, ws, 8) + 1e-9)
            cov = ((share > 0.05) & (secal > 0.62)).sum() / 8
            sc = float((ws * al).sum() / ws.sum()) * cov
            if sc > best[0]:
                best = (sc, cxn, cy)
    score = float(np.clip((best[0] - 0.42) / 0.25, 0, 1))
    spiral, turn = 0.0, 0.0
    if score > 0.2:
        H0, W0 = g.shape
        pol = cv2.warpPolar(g, (96, 180), (best[1] * W0, best[2] * H0), 0.42 * H0, cv2.WARP_POLAR_LOG)     # 줄 = 각도, 칸 = log 반지름
        pol = cv2.GaussianBlur(pol[:, 30:], (0, 0), 1.5)
        px, py = cv2.Sobel(pol, cv2.CV_32F, 1, 0), cv2.Sobel(pol, cv2.CV_32F, 0, 1)
        jxx, jyy, jxy = (px * px).sum(), (py * py).sum(), (px * py).sum()
        coh = math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-9)
        th = math.degrees(0.5 * math.atan2(2 * jxy, jxx - jyy))            # 기울기의 주된 방향. 0 = 동심원, ±90 = 방사선
        if 8 < abs(th) < 55:
            spiral, turn = float(score * np.clip((coh - 0.15) / 0.3, 0, 1) * np.clip((abs(th) - 8) / 12, 0, 1)), th
    return score, float(best[1]), float(best[2]), spiral, float(turn)


def negative_space(E: np.ndarray, V: np.ndarray, fg: np.ndarray) -> tuple[float, float]:
    """(점수, 빈 면적의 몫). 윤곽이 없는 가장 큰 한 덩어리가 그림의 대부분이고 주제는 작다."""
    flat = ((E < 0.05) & V).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(flat, connectivity=4)
    if n < 2:
        return 0.0, 0.0
    share = float(stats[1:, cv2.CC_STAT_AREA].max() / max(V.sum(), 1))
    subject = float(((fg > 0.5) & V).sum() / max(V.sum(), 1))
    return float(np.clip((share - 0.5) / 0.3, 0, 1) * np.clip((0.4 - subject) / 0.15, 0, 1)), share


_FG = None


def fg_mask(stem: str) -> np.ndarray:
    """전경(주제) 마스크 0–1, GW×GH. 글자 지운 그림에 전경 분리 모델을 돌린 것. out/fg 에 캐시(공개하지 않는다)."""
    global _FG
    out = FG / f"{stem}.png"
    if out.exists():
        return cv2.imread(str(out), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
    from PIL import Image
    from rembg import new_session, remove
    if _FG is None:
        _FG = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    m = np.asarray(remove(Image.open(ART / f"{stem}.jpg").convert("RGB"), session=_FG, only_mask=True).convert("L"))
    m = cv2.resize(m, (GW, GH), interpolation=cv2.INTER_AREA)
    FG.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), m)
    return m.astype(np.float32) / 255


def centered_fg(M: np.ndarray, V: np.ndarray) -> tuple[float, float, float, float]:
    """(점수, 주제 중심 x, y, 주제 넓이). 전경 마스크에서: 덩어리 하나, 가운데, 왼쪽·오른쪽·위 여백"""
    M = M * V
    B = (M > 0.5).astype(np.uint8)
    size = float(B.mean())
    if size < 0.03:
        return 0.0, 0.5, 0.5, size
    n, lab, stats, _ = cv2.connectedComponentsWithStats(B, connectivity=8)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    one = stats[big, cv2.CC_STAT_AREA] / B.sum()                 # 가장 큰 덩어리가 차지하는 몫
    x, y, w, h = (stats[big, k] for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    left, right, top = x / GW - 0.1, 1 - (x + w) / GW, y / GH   # 왼쪽은 지운 세로 띠 자리(약 0.1)를 빼고 잰다
    ys, xs = np.nonzero(lab == big)
    cx, cy = float((xs.mean() + 0.5) / GW), float((ys.mean() + 0.5) / GH)
    s = np.clip((one - 0.6) / 0.3, 0, 1) * np.clip(min(left, right, top) / 0.1, 0, 1) * np.clip(1 - abs(left - right) / 0.22, 0, 1) \
        * np.clip(1 - abs(cx - 0.55) / 0.1, 0, 1) * np.clip((0.55 - size) / 0.15, 0, 1) * np.clip((0.9 - cy) / 0.15, 0, 1)
    return float(s), cx, cy, size


def measure(stem: str, faces: list) -> dict:
    bgr = cv2.imread(str(ART / f"{stem}.jpg"))
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    er = ES.erased_mask(stem, w, h)
    g = cv2.GaussianBlur(gray, (0, 0), 2.0).astype(np.float32)
    E = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    E[er] = 0
    E = cv2.GaussianBlur(cv2.resize(E, (GW, GH), interpolation=cv2.INTER_AREA), (0, 0), 1.2)
    E = np.clip(E / 40, 0, 1)
    V = ~cv2.resize(er.astype(np.uint8), (GW, GH), interpolation=cv2.INTER_NEAREST).astype(bool)
    T = cv2.GaussianBlur(cv2.resize(g, (GW, GH), interpolation=cv2.INTER_AREA), (0, 0), 2.0)
    T = T - T.mean(1, keepdims=True)            # 위아래 밝기 변화(하늘·땅, 가로 띠)는 어떤 그림이든 거울과 맞는다: 줄마다 평균을 뺀다
    sym, axis = symmetry(E, V, T)
    lt, la = line_tilt(gray, er)
    tt, ta = text_tilt(stem)
    ft, fa = face_tilt(faces, w, h)
    cands = [(float(np.clip((lt - 0.2) / 0.35, 0, 1)), la, "lines"), (tt, ta, "text"), (ft, fa, "face")]
    tilt, ang, by = max(cands)
    fg = fg_mask(stem)
    cen, cx, cy, size = max(centered(bgr, er), centered_fg(fg, V))
    a = w / h
    sg = segments(gray, er)
    par, pang = parallel(sg)
    van, vx, vy = vanishing(sg, a)
    grid, gn, frame, fbox = grid_frame(sg, a)
    circ, ccx, ccy, spiral, turn = circular(g, V, a)
    space, sshare = negative_space(E, V, fg)
    return {"parallel": round(par, 4), "par_angle": round(pang, 1), "vanish": round(van, 4), "vx": round(vx, 3), "vy": round(vy, 3),
            "grid": round(grid, 4), "grid_n": gn, "frame": round(frame, 4), "frame_box": fbox, "frame_w": round(fbox[2] - fbox[0], 3),
            "circle": round(circ if spiral < THR["spiral"] else 0.0, 4), "ccx": round(ccx, 3), "ccy": round(ccy, 3),
            "spiral": round(spiral, 4), "spiral_turn": round(turn, 1), "space": round(space, 4), "space_share": round(sshare, 3),
            "sym_raw": round(sym, 4), "axis": round(axis, 3), "tilt": round(tilt, 4), "angle": round(ang, 1), "tilt_by": by,
            "tilt_lines": round(lt, 4), "tilt_text": round(tt, 4), "tilt_face": round(ft, 4),
            "center": round(cen, 4), "cx": round(cx, 3), "cy": round(cy, 3), "size": round(size, 3)}
    return {"sym_raw": round(sym, 4), "axis": round(axis, 3), "tilt": round(tilt, 4), "angle": round(ang, 1), "tilt_by": by,
            "tilt_lines": round(lt, 4), "tilt_text": round(tt, 4), "tilt_face": round(ft, 4),
            "center": round(cen, 4), "cx": round(cx, 3), "cy": round(cy, 3), "size": round(size, 3)}


def layout(files: list[str], S: dict) -> np.ndarray:
    """점수에서 직접 만든 자리. 성질마다 한 방향으로 뻗은 무리(THR 순서대로 위에서 시계 방향).
    커버는 가장 뚜렷한 성질의 무리로 간다. 뚜렷할수록 바깥, 무리 안 옆자리는 세부 값 순(SUB).
    어느 것도 뚜렷하지 않은 커버는 가운데 원판에, 조금이라도 기운 쪽으로."""
    keys = list(THR)
    dirs = {k: np.array([math.sin(t), -math.cos(t)]) for k, t in zip(keys, np.arange(len(keys)) * 2 * math.pi / len(keys))}
    rng = np.random.default_rng(0)
    xy = np.zeros((len(files), 2))
    W = np.array([[max(0.0, (S[f][k] - THR[k]) / (1 - THR[k])) if S[f][k] >= THR[k] else -1 for k in keys] for f in files])
    grouped = W.max(1) >= 0
    for j, k in enumerate(keys):
        idx = [i for i in range(len(files)) if grouped[i] and W[i].argmax() == j]
        order = sorted(idx, key=lambda i: S[files[i]][SUB[k]])
        d = dirs[k]
        side = np.array([-d[1], d[0]])
        width = min(0.5, 0.06 * math.sqrt(len(order)) + 0.08)          # 편수가 많은 무리일수록 옆으로 넓게
        for r, i in enumerate(order):
            xy[i] = d * (0.95 + 0.55 * W[i].max()) + side * ((r + 0.5) / len(order) - 0.5) * 2 * width
    rest = np.where(~grouped)[0]
    lean = np.array([sum((S[files[i]][k] / THR[k]) ** 2 * dirs[k] for k in keys) for i in rest])
    strength = np.linalg.norm(lean, axis=1)
    for r, m in enumerate(np.argsort(strength)):      # 약한 것부터 안쪽: 넓이가 고르게 차도록 반지름 = √순위
        i = rest[m]
        ang = math.atan2(lean[m][1], lean[m][0]) + rng.normal(0, 0.5) if strength[m] > 0.02 else rng.uniform(0, 2 * math.pi)
        rad = 0.6 * math.sqrt((r + 0.5) / len(rest))
        xy[i] = [rad * math.cos(ang), rad * math.sin(ang)]
    xy[:, 1] *= 0.8                                   # 세로로 긴 타일이라 세로를 조금 줄인다
    return xy


def sheet(name: str, files: list[str], per_row: int = 12) -> None:
    from PIL import Image
    tw, th = 96, 120
    W = Image.new("RGB", (per_row * (tw + 4), math.ceil(len(files) / per_row) * (th + 4)), "black")
    for k, f in enumerate(files):
        p = ROOT / "web" / "thumbs" / f
        im = Image.open(p if p.exists() else ROOT / "posters" / f).convert("RGB").resize((tw, th))
        W.paste(im, ((k % per_row) * (tw + 4), (k // per_row) * (th + 4)))
    OUT.mkdir(parents=True, exist_ok=True)
    W.save(OUT / f"top_{name}.jpg", quality=82)


def main() -> None:
    files = sorted(p.name for p in ES.POSTERS.glob("*.jpg"))
    faces = json.loads((A / "faces.json").read_text()) if (A / "faces.json").exists() else {}
    raw_f = OUT / "raw.json"                       # 1차 측정은 4분쯤 걸린다: 캐시해 두고 문턱·2차 선별만 바꿀 때는 다시 재지 않는다(--fresh 로 다시)
    S = json.loads(raw_f.read_text()) if raw_f.exists() and "--fresh" not in sys.argv else {}
    for k, f in enumerate(files):
        if f not in S:
            S[f] = measure(Path(f).stem, faces.get(f, []))
            if k % 300 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw_f.write_text(json.dumps(S))
    cap = json.loads((A / "captions_sonnet_v3.json").read_text()) if (A / "captions_sonnet_v3.json").exists() else {}
    for f in files:
        text = " ".join(str(cap.get(f, {}).get(k, "")) for k in ("Composition", "Visual idea")).lower()
        said = {k for k, ws in WORDS.items() if any(w in text for w in ws)}
        m = S[f]
        if "vanish" not in m:                                 # 옛 캐시(위·아래를 따로 적던 것)
            m["vanish"] = max(m.pop("vup", 0), m.pop("vdown", 0))
        m["said"] = sorted(said)
        m["geo"] = {k: m.get(k, 0.0) for k in THR if k != "sym"}
        for k in RESCUE:
            if k != "sym" and k in said and m[k] >= 0.25:
                m[k] = max(m[k], 0.6)
        m["parallel"] = m["parallel"] if "parallel" in said else 0.0
        # 소실점: 글이 말한 커버만. 위로 모이는지 아래로 모이는지는 잰 점(vy)이 믿을 만하지 않아 한 무리로 둔다
        m["vanish"] = max(m["vanish"], 0.6) if "vanish" in said else 0.0
        # 스파이럴: 측정으로는 못 잡는다(로그 극좌표 기울기는 전부 0.11 이하). 글이 spiral 이라 했거나, swirl·vortex + 원형 측정 .3 이상
        if "spiral" in said or ("swirl" in said and m["geo"]["circle"] >= 0.3):
            m["spiral"], m["circle"] = max(m["spiral"], 0.6 + 0.4 * m["geo"]["circle"]), 0.0
        else:
            m["spiral"] = 0.0
    # 3차 선별: 후보 커버를 Sonnet 이 그림으로 보고 판정한 것(out/structure/judge/out_*.json, 2026-11-18). 판정이 있으면 그 커버의
    # 소실점·원형·스파이럴은 판정을 따른다. 소실점은 모이는 자리(위·아래·그 밖), 스파이럴은 안에서 밖으로 도는 방향.
    judge = {}
    for jf in sorted((OUT / "judge").glob("out_*.json")):
        for f, v in json.loads(jf.read_text()).items():      # 한 커버가 소실점·스파이럴 두 묶음에 다 있을 수 있다: 합친다
            e = judge.setdefault(f, {})
            note = " / ".join(x for x in (e.get("note"), v.get("note")) if x)
            e.update(v)
            e["note"] = note
    if judge:
        (A / "structure_judge.json").write_text(json.dumps(judge, ensure_ascii=False, indent=0))
    for f in files:
        m, j = S[f], judge.get(f, {})
        m.update(vup=0.0, vdown=0.0, scw=0.0, sccw=0.0)
        if not judge:
            continue
        if "vanish" in j:
            v = {"yes": 0.9, "weak": 0.3}.get(j["vanish"], 0.0)
            w = j.get("where") or ""
            m["vup"], m["vdown"], m["vanish"] = (v, 0.0, 0.0) if "up" in w else (0.0, v, 0.0) if "down" in w else (0.0, 0.0, v)
        else:
            m["vanish"] = 0.0                                 # 글만으로 올라온 것은 판정 대상이었다: 판정이 없으면 인정하지 않는다
        if "shape" in j:
            sp = 0.9 if j["shape"] == "spiral" else 0.0
            t = j.get("turn")
            m["scw"], m["sccw"], m["spiral"] = (sp, 0.0, 0.0) if t == "cw" else (0.0, sp, 0.0) if t == "ccw" else (0.0, 0.0, sp)
            m["circle"] = max(m["geo"]["circle"], 0.8) if j["shape"] in ("concentric", "circle") else 0.0
        else:
            m["spiral"] = 0.0
        m["judge"] = j
    # 대칭 원점수는 커버마다 바닥이 있다(아무 그림이나 조금은 겹친다): 분포의 중앙값을 0, 상위 3% 를 1 로
    raw = np.array([S[f]["sym_raw"] for f in files])
    lo, hi = np.percentile(raw, 70), np.percentile(raw, 98)
    for f in files:
        S[f]["sym"] = round(float(np.clip((S[f]["sym_raw"] - lo) / (hi - lo), 0, 1)), 4)
        S[f]["geo"]["sym"] = S[f]["sym"]
        if "sym" in S[f]["said"] and S[f]["sym"] >= 0.25:
            S[f]["sym"] = max(S[f]["sym"], 0.6)
        strong = {k: S[f][k] for k in THR if S[f][k] >= THR[k]}
        S[f]["group"] = NAMES[max(strong, key=strong.get)] if strong else None
    (A / "structure_scores.json").write_text(json.dumps(S, ensure_ascii=False, indent=0))
    np.savez(ES.EMB / "structure-map.npz", files=np.array(files), xy=layout(files, S).astype(np.float32))
    for k in THR:
        top = sorted(files, key=lambda f: -S[f][k])
        sheet(k, top[:48])
        print(f"{k:9s} ≥ thr {sum(1 for f in files if S[f][k] >= THR[k]):4d}   top {S[top[0]][k]:.2f}   48th {S[top[47]][k]:.2f}")
    from collections import Counter
    print(Counter(S[f]["group"] for f in files))


if __name__ == "__main__":
    main()
