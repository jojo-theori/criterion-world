"""내용(무엇이 그려졌나)을 지우고 글자 배치만으로 재는 거리. Lab 지도용(2026-11-18 사용자: Composition·Type 이 아쉽다).

Typography(Lab 지도 이름)  → data/analysis/embed/type-layout.npz (files, dist)
   커버의 글자 상자 집합끼리의 거리(LTSim 식). 상자 = (가운데 x, y, 폭, 높이), 무게 = √넓이.
   두 커버의 상자를 최적 수송(여럿 대 여럿)으로 맞춘 비용 + 글자 양 차이.
   여러 줄 제목이 한 덩어리로 잡힌 커버와 줄마다 잡힌 커버도 가깝게 나온다(일대일 매칭은 그렇지 않다).
   크라이테리온 틀(세로 띠, 맨 위 띠, 로고) 글자는 뺀다. 글자가 없는 커버는 서로 거리 0.

Structure 지도는 scripts/structure_scores.py 가 만든다(이 파일의 erased_mask, text_regions 를 쓴다).
구조 이미지를 DINOv2 로 임베딩하던 첫 시도는 기존 지도보다 나아지지 않아 뺐다(2026-11-18).

사용 (.venv-ml)
  .venv-ml/bin/python scripts/embed_structure.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
REGIONS = ROOT / "data" / "regions" / "combined"
VISUAL = ROOT / "data" / "analysis" / "visual"
EMB = ROOT / "data" / "analysis" / "embed"
ART = ROOT / "out" / "art"
SIZE_W = 0.5          # 상자 크기 차이의 무게(위치 차이 1 에 대해)
AMOUNT_W = 0.6        # 글자 양(√넓이 합) 차이의 무게


# ---------- Type layout ----------
def text_boxes(stem: str) -> np.ndarray:
    """글자 상자 (cx, cy, w, h) 비율. 크라이테리온 틀 글자는 뺀다(build_tropes.text_boxes 와 같은 규칙 + 틀 상자 안)."""
    return np.array([[(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0] for x0, y0, x1, y1, used in text_regions(stem) if used],
                    np.float32).reshape(-1, 4)


def text_regions(stem: str) -> list[list[float]]:
    """찾은 글자 영역 전부 [x0, y0, x1, y1, 쓰임]. 쓰임 0 = 크라이테리온 틀 글자나 너무 작은 것(Type layout 에서 뺀 것)."""
    p = REGIONS / f"{stem}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    W, H = d["width"], d["height"]
    vj = VISUAL / f"{stem}.json"
    rects = json.loads(vj.read_text()).get("frame", {}).get("rects") or [] if vj.exists() else []
    out = []
    for r in d["regions"]:
        if len(r["poly"]) < 3:
            continue
        xs, ys = [q[0] / W for q in r["poly"]], [q[1] / H for q in r["poly"]]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        frame = (x0 < 0.2 and h > 3 * w and w < 0.08) or (y0 < 0.06 and h < 0.03) \
            or any(a <= cx <= c and b <= cy <= e for a, b, c, e in rects) or w * h < 1e-4
        out.append([x0, y0, x1, y1, 0 if frame else 1])
    return out


def type_layout(files: list[str]) -> None:
    import ot
    B = [text_boxes(Path(f).stem) for f in files]
    mass = [np.sqrt(b[:, 2] * b[:, 3]).astype(np.float64) for b in B]
    amount = np.array([m.sum() for m in mass])
    P = [m / m.sum() if len(m) else m for m in mass]
    n = len(files)
    D = np.zeros((n, n), np.float32)
    t0 = time.time()
    for i in range(n):
        bi = B[i]
        for j in range(i + 1, n):
            bj = B[j]
            d = AMOUNT_W * abs(amount[i] - amount[j])
            if len(bi) and len(bj):
                pos = np.linalg.norm(bi[:, None, :2] - bj[None, :, :2], axis=-1)
                size = np.abs(bi[:, None, 2:] - bj[None, :, 2:]).sum(-1)
                M = (pos + SIZE_W * size).astype(np.float64)
                d += M[0, 0] if M.size == 1 else ot.emd2(P[i], P[j], M)
            D[i, j] = D[j, i] = d
        if i % 200 == 0:
            print(f"  type layout {i}/{n} {time.time() - t0:.0f}s", flush=True)
    np.savez(EMB / "type-layout.npz", files=np.array(files), dist=D)
    empty = sum(1 for b in B if not len(b))
    print(f"type-layout: {n} covers, {empty} without text, median boxes {int(np.median([len(b) for b in B]))}, "
          f"median distance {np.median(D[np.triu_indices(n, 1)]):.3f}")


def erased_mask(stem: str, w: int, h: int) -> np.ndarray:
    """out/art 에서 지운 자리(글자 영역 + 틀 상자 + 왼쪽 세로 띠 자리)를 넉넉히 넓힌 것."""
    import cv2
    m = np.zeros((h, w), np.uint8)
    rj, vj = REGIONS / f"{stem}.json", VISUAL / f"{stem}.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        for x in r["regions"]:
            if len(x["poly"]) >= 3:
                cv2.fillPoly(m, [np.array([[a * w / r["width"], b * h / r["height"]] for a, b in x["poly"]], np.int32)], 1)
    if vj.exists():
        for x0, y0, x1, y1 in json.loads(vj.read_text()).get("frame", {}).get("rects") or []:
            m[int(y0 * h):int(y1 * h) + 1, int(x0 * w):int(x1 * w) + 1] = 1
    m[:, :int(0.085 * w)] = 1
    k = max(3, int(0.03 * max(w, h))) | 1
    return cv2.dilate(m, np.ones((k, k), np.uint8)).astype(bool)



def main() -> None:
    files = sorted(p.name for p in POSTERS.glob("*.jpg"))
    EMB.mkdir(parents=True, exist_ok=True)
    type_layout(files)


if __name__ == "__main__":
    main()
