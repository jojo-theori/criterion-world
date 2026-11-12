"""웹 뷰어(web/index.html)가 읽을 데이터와 썸네일을 만든다. 웹은 이 결과를 읽기만 한다.

  web/data.js          window.META, window.COVERS (file:// 로 열 수 있게 JSON 대신 JS)
  web/trace/<stem>.js  커버별 흔적: 선분, 텍스트 다각형, 무게중심. 세 폭 보기에서 필요할 때 불러온다
  web/thumbs/*.jpg     112x140 썸네일
  web/large/*.jpg      480x596 (세 폭 보기용)
  thumbs, large 는 커버 이미지라 공개 레포·서버에 올리지 않는다.

입력
  posters/, data/criterion.sqlite
  data/analysis/visual/ (analyze_visual.py)        색·대비·구도선 축. 없으면 그 축만 빠진다
  data/regions/combined/ (run_text_regions.py)     텍스트 배치 축
  data/analysis/embed/<모델>.npz (embed_siglip.py) 전체 인상 축, 유사도 지도
  data/analysis/objects/owlv2.json (detect_objects.py) 물체 축
  data/analysis/strokes_layout/ (analyze_strokes_layout.py) 획, 레이아웃 축. 선화는 web/strokes/

사용 (.venv-ml)
  .venv-ml/bin/python scripts/build_web.py
"""

from __future__ import annotations

import base64
import colorsys
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, str(Path(__file__).resolve().parent))
from row_names import shorten_all  # noqa: E402  같은 scripts/ 폴더

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
VISUAL = ROOT / "data" / "analysis" / "visual"
REGIONS = ROOT / "data" / "regions" / "combined"
EMBED = ROOT / "data" / "analysis" / "embed" / "siglip2-base-patch16-384.npz"
OBJECTS = ROOT / "data" / "analysis" / "objects" / "owlv2.json"
STROKES = ROOT / "data" / "analysis" / "strokes_layout"
OBJ_MIN_SCORE = 0.2     # 축 벡터와 흔적에 쓰는 최소 점수
OBJ_MAX_COVERED = 0.5   # 박스 면적 중 틀·텍스트에 덮인 비율이 이보다 크면 뺀다(C 로고 = moon 같은 오검출)
WEB = ROOT / "web"
SIZES = {"thumbs": (112, 140), "large": (480, 596)}
GRAY_SAT = 0.12    # 시각 분석이 없는 커버의 무채색 판정 기준
NEIGHBORS = 6
TEXT_GRID = (16, 20)
MAX_LINES = 160


def palette(im: Image.Image, k: int = 6):
    """median cut 팔레트. (hex, 비율, hls) 를 비율 순으로."""
    q = im.resize((80, 100), Image.BILINEAR).quantize(colors=k, method=Image.Quantize.MEDIANCUT)
    pal = q.getpalette()[: 3 * k]
    counts = sorted(q.getcolors(), reverse=True)
    total = sum(c for c, _ in counts)
    out = []
    for c, i in counts:
        r, g, b = pal[3 * i: 3 * i + 3]
        out.append((f"#{r:02x}{g:02x}{b:02x}", c / total, colorsys.rgb_to_hls(r / 255, g / 255, b / 255)))
    return out


def representative(pal):
    """대표색: 면적 × 채도 가중이 가장 큰 색. 무채색이면 가장 넓은 색."""
    chroma = sum(p * s * (1 - abs(2 * l - 1)) for _, p, (_, l, s) in pal)
    gray = chroma < GRAY_SAT * 0.5
    key = (lambda x: x[1]) if gray else (lambda x: x[1] * (0.15 + x[2][2] * (1 - abs(2 * x[2][1] - 1))))
    hx, _, (h, _, _) = max(pal, key=key)
    return hx, h, gray


def grid_assign(xy: np.ndarray, fill: float = 1.25, aspect: float = 1.6) -> tuple[list[list[int]], int, int]:
    """0–1 좌표를 격자 칸에 일대일 배정(헝가리안). fill = 칸 수 / 점 수."""
    n = len(xy)
    cols = int(np.ceil(np.sqrt(n * aspect * fill)))
    rows = int(np.ceil(n * fill / cols))
    gx, gy = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    _, cell = linear_sum_assignment(((xy[:, None, :] - grid[None]) ** 2).sum(-1))
    return [[int(c % cols), int(c // cols)] for c in cell], cols, rows


def text_vector(regions: dict, frame_rects: list) -> np.ndarray:
    """텍스트 다각형 점유 격자. 크라이테리온 틀(세로 띠, 로고)은 모든 커버에 공통이라 뺀다."""
    gw, gh = TEXT_GRID
    W, H = regions["width"], regions["height"]
    im = Image.new("L", (gw * 4, gh * 4), 0)
    d = ImageDraw.Draw(im)
    for r in regions["regions"]:
        pts = [(x / W * gw * 4, y / H * gh * 4) for x, y in r["poly"]]
        if len(pts) >= 3:
            d.polygon(pts, fill=255)
    for x0, y0, x1, y1 in frame_rects:
        d.rectangle([x0 * gw * 4, y0 * gh * 4, x1 * gw * 4, y1 * gh * 4], fill=0)
    return np.asarray(im.resize((gw, gh), Image.BOX), dtype=np.float32).ravel() / 255


def cover_mask(regions: dict | None, frame_rects: list, w: int = 64, h: int = 80) -> np.ndarray:
    """틀과 텍스트 다각형을 칠한 마스크 (h, w), 0–1."""
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    if regions:
        W, H = regions["width"], regions["height"]
        for r in regions["regions"]:
            if len(r["poly"]) >= 3:
                d.polygon([(x / W * w, y / H * h) for x, y in r["poly"]], fill=255)
    for x0, y0, x1, y1 in frame_rects:
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], fill=255)
    return np.asarray(im, dtype=np.float32) / 255


def clean_objects(dets: list, mask: np.ndarray) -> list:
    h, w = mask.shape
    out = []
    for d in dets:
        if d["score"] < OBJ_MIN_SCORE:
            continue
        x0, y0, x1, y1 = d["box"]
        c0, c1 = int(x0 * w), max(int(x0 * w) + 1, int(np.ceil(x1 * w)))
        r0, r1 = int(y0 * h), max(int(y0 * h) + 1, int(np.ceil(y1 * h)))
        if mask[r0:r1, c0:c1].mean() > OBJ_MAX_COVERED:
            continue
        out.append(d)
    return out


def simplify(poly: list, W: int, H: int, k: int = 14) -> list:
    step = max(1, len(poly) // k)
    return [[round(x / W, 4), round(y / H, 4)] for x, y in poly[::step]]


def to_rings(pos: dict[int, list[int]]) -> list[list[list[int]]]:
    """격자 좌표 {i: [x, y]} → 줄 목록. 줄 = [[i, x], ...] (x 순). 빈 줄은 뺀다.
    세 폭 보기는 이것을 토러스로 쓴다: 좌우는 줄 안에서, 위아래는 줄 사이에서 순환."""
    rows: dict[int, list[list[int]]] = {}
    for i, (x, y) in pos.items():
        rows.setdefault(y, []).append([i, x])
    return [sorted(rows[y], key=lambda e: e[1]) for y in sorted(rows)]


def tour(vecs: dict[int, np.ndarray], kind: str) -> list[int]:
    """모든 커버를 한 줄로 잇는 순서(가까운 것을 차례로 잇는 탐욕 경로 + 2-opt 몇 번).
    세 폭 보기의 양옆 = 이 순서의 이전/다음. 최근접 둘을 쓰면 A↔B 를 오가는 고리가 생긴다."""
    if not vecs:
        return []
    ids = list(vecs)
    X = np.stack([vecs[i] for i in ids]).astype(np.float32)
    if kind == "hel":
        X = np.sqrt(np.clip(X, 0, None))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    D = 1 - X @ X.T
    n = len(ids)
    start = int(np.argmax(D.sum(1)))            # 가장 외딴 커버에서 시작
    seen = np.zeros(n, bool); seen[start] = True
    path = [start]
    for _ in range(n - 1):
        d = D[path[-1]].copy(); d[seen] = np.inf
        j = int(np.argmin(d)); seen[j] = True; path.append(j)
    # 2-opt: 교차한 두 구간을 뒤집어 긴 점프를 줄인다 (짧게 몇 바퀴만)
    P = np.array(path)
    for _ in range(3):
        improved = False
        for a in range(1, n - 2):
            b = np.arange(a + 1, n - 1)
            gain = D[P[a - 1], P[a]] + D[P[b], P[b + 1]] - D[P[a - 1], P[b]] - D[P[a], P[b + 1]]
            k = int(np.argmax(gain))
            if gain[k] > 1e-6:
                P[a:b[k] + 1] = P[a:b[k] + 1][::-1].copy(); improved = True
        if not improved:
            break
    return [ids[k] for k in P]


def neighbors(vecs: dict[int, np.ndarray], kind: str) -> dict[int, list[int]]:
    """kind: cos (그대로 코사인), hel (히스토그램 → Hellinger, 제곱근 후 코사인)."""
    if not vecs:
        return {}
    ids = list(vecs)
    X = np.stack([vecs[i] for i in ids]).astype(np.float32)
    if kind == "hel":
        X = np.sqrt(np.clip(X, 0, None))
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    top = np.argsort(-S, axis=1)[:, :NEIGHBORS]
    return {ids[r]: [ids[j] for j in top[r]] for r in range(len(ids))}


ATLAS_TILE = (80, 100)
TILE_AS = {"type": "text", "composition": "lines", "medium": "strokes"}   # 범주 축의 흔적은 이전 분석의 그림을 쓴다
ATLAS_COLS = 32
FONT = "/System/Library/Fonts/Helvetica.ttc"


def trace_tile(axis: str, it: dict, tr: dict, big: Image.Image | None) -> Image.Image:
    """격자 타일용 흔적 한 장. 세 폭 보기 캔버스(drawTrace)와 같은 문법을 작게."""
    from PIL import ImageFilter, ImageFont, ImageOps
    W, H = ATLAS_TILE
    S = 3                                    # 크게 그려서 줄이면 선이 부드럽다
    w, h = W * S, H * S
    im = Image.new("RGB", (w, h), "black")
    d = ImageDraw.Draw(im)
    if axis == "cover" and big is not None:          # 원본 썸네일 (격자 캔버스가 쓴다)
        return big.resize((W, H), Image.LANCZOS)
    if axis in ("spine", "year", "country", "runtime", "aspect"):
        txt = str({"spine": it["spine"], "year": it["year"], "country": it.get("country", ""),
                   "runtime": f"{it['rt']}′" if it.get("rt") else "",
                   "aspect": (it.get("ar") or [""])[0].split(" ")[0].split(",")[0].split("/")[0]}[axis])
        f = ImageFont.truetype(FONT, int(h * (0.34 if len(txt) <= 3 else 0.28 if len(txt) <= 4 else min(0.2, 1.7 / max(len(txt), 1)))))
        d.text((w / 2, h / 2), txt, font=f, fill=(235, 235, 235), anchor="mm")
    elif axis == "color":
        y = 0.0
        for c, r in it["palette"]:
            d.rectangle([0, int(y * h), w, int((y + r) * h) + 1], fill=c)
            y += r
    elif axis == "look" and big is not None:
        im = big.resize((12, 15), Image.BILINEAR).resize((w, h), Image.NEAREST)
    elif axis == "contrast" and big is not None:
        g = ImageOps.autocontrast(big.convert("L"), cutoff=2).resize((w, h)).filter(ImageFilter.GaussianBlur(18))
        im = g.convert("RGB")
    elif axis == "lines" and tr.get("lines"):
        im = Image.new("L", (w, h), 242)
        d = ImageDraw.Draw(im)
        for k, (x0, y0, x1, y1) in enumerate(tr["lines"]):
            d.line([x0 * w, y0 * h, x1 * w, y1 * h], fill=0, width=9 if k < 20 else 5 if k < 60 else 3)
        im = im.filter(ImageFilter.GaussianBlur(3)).convert("RGB")
    elif axis == "text" and tr.get("text"):
        for p in tr["text"]:
            if len(p) >= 3:
                d.polygon([(x * w, y * h) for x, y in p], fill=(242, 242, 242))
    elif axis == "strokes":
        sp = WEB / "strokes" / f"{Path(it['file']).stem}.png"
        if sp.exists():
            im = ImageOps.invert(Image.open(sp).convert("L").resize((w, h))).convert("RGB")
    elif axis == "layout" and tr.get("layout"):
        col = {"e": (0, 0, 0), "f": (110, 110, 110), "t": (242, 242, 242), "c": (38, 38, 38)}
        g = tr["layout"]
        small = Image.new("RGB", (40, 50))
        small.putdata([col[ch] for ch in g])
        im = small.resize((w, h), Image.NEAREST)
    elif axis == "people" and tr.get("people"):
        for x0, y0, x1, y1, face in sorted(tr["people"], key=lambda b: b[4]):
            d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], fill=(235, 235, 235) if face else (40, 40, 40),
                        outline=(235, 235, 235), width=4)
    elif axis == "objects" and tr.get("objects"):
        for o in reversed(tr["objects"][:10]):
            x0, y0, x1, y1 = o["box"]
            d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], fill=(28, 28, 28), outline=(235, 235, 235), width=4)
    return im.resize((W, H), Image.LANCZOS)


def build_atlases(items: list[dict], traces: list[dict]) -> dict:
    """축마다 흔적 썸네일을 한 장에 모은다(web/atlas/<축>.jpg). 타일 i 의 자리는 (i % cols, i // cols)."""
    out = WEB / "atlas"
    out.mkdir(exist_ok=True)
    W, H = ATLAS_TILE
    rows = (len(items) + ATLAS_COLS - 1) // ATLAS_COLS
    axes = ["cover", "spine", "year", "country", "runtime", "aspect", "color", "people", "type", "composition", "medium"]
    sheets = {a: Image.new("RGB", (ATLAS_COLS * W, rows * H), "black") for a in axes}
    for i, (it, tr) in enumerate(zip(items, traces)):
        with Image.open(WEB / "large" / it["file"]) as big:
            big = big.convert("RGB")
            for a in axes:
                sheets[a].paste(trace_tile(TILE_AS.get(a, a), it, tr, big), ((i % ATLAS_COLS) * W, (i // ATLAS_COLS) * H))
    for a, sh in sheets.items():
        jp = out / f"{a}.jpg"
        sh.save(jp, quality=88 if a == "cover" else 85)
        # WebGL 은 file:// 이미지를 텍스처로 쓰지 못한다(교차 출처). data URL 을 담은 JS 로도 둔다
        b64 = base64.b64encode(jp.read_bytes()).decode()
        (out / f"{a}.js").write_text(f'(window.ATLAS=window.ATLAS||{{}})[{json.dumps(a)}]="data:image/jpeg;base64,{b64}";\n')
        # 배포(http)용: WebP 가 jpg 의 약 절반. 원본 커버는 반 크기를 먼저 보여주고 큰 것으로 바꾼다
        sh.save(out / f"{a}.webp", "WEBP", quality=80 if a == "cover" else 75, method=6)
        if a == "cover":
            sh.resize((sh.width // 2, sh.height // 2), Image.LANCZOS).save(out / "cover_s.webp", "WEBP", quality=72, method=6)
    print(f"  atlas {len(axes)} axes, {ATLAS_COLS}x{rows} tiles of {W}x{H} -> {out}", flush=True)
    return {"cols": ATLAS_COLS, "rows": rows, "tile": [W, H]}


EXTRA = ROOT / "data" / "raw" / "wikidata_extra.json"
CMETA = ROOT / "data" / "raw" / "criterion_meta.json"


def film_extra(d: dict, title: str) -> dict:
    """Wikidata 추가 정보를 뷰어용 짧은 키로. 없으면 키를 두지 않는다."""
    out: dict = {}
    if d.get("runtime"):
        rt = [r for r in d["runtime"] if 5 <= r <= 1000]
        if rt:
            out["rt"] = max(rt)                     # 판본이 여럿이면 긴 쪽(감독판·복원판)
    if d.get("aspect"):
        out["ar"] = d["aspect"][:2]
    if d.get("color"):
        c = {"black-and-white": "B&W", "color": "Color"}
        v = list(dict.fromkeys(c.get(x, x) for x in d["color"]))
        out["clr"] = " / ".join(v[:2])
    for k, n in (("language", 2), ("dop", 2), ("music", 2), ("writer", 3), ("cast", 4), ("genre", 3)):
        if d.get(k):
            out[k] = d[k][:n]
    ot = [t for t in d.get("original_title", []) if t.casefold() != (title or "").casefold()]
    if ot:
        out["orig"] = ot[0]
    return out


HUES = [(15, "Red"), (40, "Orange"), (65, "Yellow"), (160, "Green"), (200, "Cyan"), (255, "Blue"), (290, "Purple"),
        (335, "Pink"), (360, "Red")]


def color_name(f: dict) -> str:
    if f["gray"]:
        return "Black" if f["lum"] < .25 else "White" if f["lum"] > .75 else "Gray"
    deg = f["hue"] * 360
    name = next(n for d, n in HUES if deg < d)
    if name in ("Orange", "Red") and f["lum"] < .3:
        return "Brown" if name == "Orange" else "Dark red"
    return name


ROW_LEN = 36   # 순회 줄 하나의 목표 길이


def tiers(vals: dict[int, float], words: list[str]) -> dict[int, str]:
    """커버별 값을 전체 분포의 삼분위로 말로 바꾼다."""
    v = np.array(list(vals.values()), dtype=float)
    q = np.nanquantile(v, [1 / 3, 2 / 3])
    return {i: words[0] if x < q[0] else words[2] if x > q[1] else words[1] for i, x in vals.items()}


def item_categories(axis: str, ids: list[int], feat: dict[int, dict], X: np.ndarray | None) -> tuple[dict[int, str], list[str]]:
    """축마다 커버 하나에 이름 붙일 수 있는 범주 하나. (범주, 범주 순서)"""
    F = {i: feat.get(i, {}) for i in ids}
    g = lambda k: {i: F[i].get(k, np.nan) if F[i].get(k) is not None else np.nan for i in ids}  # noqa: E731
    if axis == "color":
        order = [n for _, n in HUES[:-1]] + ["Brown", "Dark red", "Black", "Gray", "White"]
        return {i: color_name(F[i]) for i in ids}, order
    if axis == "contrast":
        a = tiers(g("light"), ["Dark", "Mid-tone", "Light"])
        b = tiers(g("rms"), ["soft", "medium contrast", "high contrast"])
        return {i: f"{a[i]}, {b[i]}" for i in ids}, [f"{x}, {y}" for x in ("Dark", "Mid-tone", "Light")
                                                      for y in ("soft", "medium contrast", "high contrast")]
    if axis == "lines":
        d = tiers(g("dens"), ["Few lines", "Some lines", "Many lines"])
        def ori(f):
            sh = {"horizontal": f.get("lh") or 0, "vertical": f.get("lv") or 0, "diagonal": f.get("ld") or 0}
            return max(sh, key=sh.get)
        return {i: f"{d[i]}, mostly {ori(F[i])}" for i in ids}, [f"{x}, mostly {y}" for y in ("horizontal", "vertical", "diagonal")
                                                                 for x in ("Few lines", "Some lines", "Many lines")]
    if axis == "text":
        amt = tiers(g("tshare"), ["little type", "some type", "lots of type"])
        where = {"top": "Title on top", "middle": "Title in the middle", "bottom": "Title at the bottom"}
        return ({i: f"{where.get(F[i].get('band'), 'No title')}, {amt[i]}" for i in ids},
                [f"{w}, {x}" for w in [*where.values(), "No title"] for x in ("little type", "some type", "lots of type")])
    if axis == "strokes":
        a = tiers(g("ink"), ["Sparse", "Moderate", "Dense"])
        b = tiers(g("width"), ["fine strokes", "medium strokes", "bold strokes"])
        return {i: f"{a[i]}, {b[i]}" for i in ids}, [f"{x}, {y}" for x in ("Sparse", "Moderate", "Dense")
                                                     for y in ("fine strokes", "medium strokes", "bold strokes")]
    if axis == "layout":
        z = {}
        for k in ("tshare", "fig", "empty"):
            v = np.array([F[i].get(k) or 0 for i in ids], dtype=float)
            z[k] = dict(zip(ids, (v - v.mean()) / (v.std() + 1e-9)))
        word = {"tshare": "Type-led", "fig": "Image-filled", "empty": "Open space"}
        band = {"top": "title on top", "middle": "title in the middle", "bottom": "title at the bottom"}
        cats = {}
        for i in ids:
            w = word[max(z, key=lambda k: z[k][i])]
            cats[i] = f"{w}, {band.get(F[i].get('band'), 'no title')}"
        return cats, [f"{w}, {b}" for w in word.values() for b in [*band.values(), "no title"]]
    if axis == "objects":
        from collections import Counter
        df = Counter(o for i in ids for o in set(F[i].get("objs", [])))
        common = {o for o, n in df.items() if n > 0.2 * len(ids)}     # person, face 같은 흔한 것은 뒤로
        def pick(i, allowed=None):
            objs = F[i].get("objs", [])
            pref = [o for o in objs if o not in common] + [o for o in objs if o in common]
            pref = [o for o in pref if allowed is None or o in allowed]
            return pref[0] if pref else None
        cats = {i: pick(i) for i in ids}
        for _ in range(3):   # 너무 작은 범주는 그 커버의 다음 물체로 옮긴다
            size = Counter(cats.values())
            big = {c for c, n in size.items() if c and n >= 8}
            cats = {i: c if c in big else (pick(i, big) or "Various") for i, c in cats.items()}
        size = Counter(cats.values())
        return cats, [c for c, _ in size.most_common() if c != "Various"] + ["Various"]
    if axis == "look" and X is not None:   # 의미 지도: 묶음마다 많은 시대 · 나라 · 물체
        from collections import Counter
        from sklearn.cluster import KMeans
        k = max(8, len(ids) // ROW_LEN)
        lab = KMeans(k, n_init=4, random_state=0).fit_predict(X / np.linalg.norm(X, axis=1, keepdims=True))
        def facets(f):
            out = []
            if f.get("year"):
                out.append(("dec", f"{f['year'] // 10 * 10}s"))
            if f.get("country"):
                out.append(("cty", f["country"]))
            if f.get("genre"):
                out.append(("gen", f["genre"]))
            out += [("obj", o) for o in f.get("objs", [])[:3]]
            return out
        base = Counter(x for i in ids for x in facets(F[i]))
        # 묶음마다 두드러진 특징(시대·나라·장르·물체) 둘. 이미 쓴 이름이면 다음 조합으로
        from itertools import combinations
        KIND = ["dec", "cty", "gen", "obj"]
        names, used = {}, set()
        for c in sorted(range(k), key=lambda c: -(lab == c).sum()):
            mem = [i for i, l in zip(ids, lab) if l == c]
            cnt = Counter(x for i in mem for x in facets(F[i]))
            score = sorted(((n / len(mem)) * np.log((n / len(mem)) / (base[x] / len(ids)) + 1e-9), x)
                           for x, n in cnt.items() if n >= 2)[::-1][:8]
            cands = [x for _, x in score]
            name = None
            for a_, b_ in combinations(cands, 2):
                if a_[0] == b_[0]:
                    continue
                nm = " · ".join(v for _, v in sorted([a_, b_], key=lambda x: KIND.index(x[0])))
                if nm not in used:
                    name = nm
                    break
            names[c] = name or (" · ".join(v for _, v in cands[:3]) or "Mixed")
            used.add(names[c])
        return {i: names[l] for i, l in zip(ids, lab)}, list(dict.fromkeys(names[c] for c in range(k)))
    return {i: "" for i in ids}, [""]


# 한 범주가 여러 줄로 나뉠 때 줄마다 붙일 두 번째 기준: (지표, 낮음→높음 낱말, 이름 만드는 법)
SPLIT = {
    "color": ("lum", None, None),   # 색 이름별 낱말은 COLOR_WORDS
    "contrast": ("chroma", ["grayscale", "near-gray", "faded", "muted", "subdued", "toned", "colorful", "saturated", "vivid"], "{c}, {w}"),
    "lines": ("sym", ["asymmetric", "lopsided", "skewed", "off-balance", "loosely balanced", "near-symmetric", "balanced", "symmetric", "mirrored"], "{c}, {w}"),
    "text": ("tx", ["flush left", "far left", "left", "left of center", "centered", "right of center", "right", "far right", "flush right"], "{c}, {w}"),
    "strokes": ("coh", ["scribbled", "messy", "rough", "loose", "steady", "even", "orderly", "precise", "rigid"], "{c}, {w}"),
    "layout": ("fy", ["image at the top", "image high", "image upper", "image above center", "image centered", "image below center", "image lower", "image low", "image at the bottom"], "{c}, {w}"),
}
COLOR_WORDS = {   # 어두움 → 밝음
    "Black": ["Jet black", "Ink black", "Onyx", "Black", "Soft black", "Charcoal", "Smoke"],
    "Gray": ["Graphite", "Dark gray", "Slate", "Gray", "Ash", "Silver", "Light gray"],
    "White": ["Bone", "Stone", "Off-white", "Cream", "Ivory", "White", "Paper white"],
}
TONE = ["Deepest", "Deep", "Dark", "Mid", "Light", "Pale", "Palest"]
ROMAN = ["", " II", " III", " IV", " V", " VI", " VII", " VIII", " IX", " X"]


def spread(words: list[str], n: int) -> list[str]:
    """낱말 목록에서 n 개를 고르게(양 끝 포함) 뽑는다."""
    if n == 1:
        return [""]
    return [words[round(k * (len(words) - 1) / (n - 1))] for k in range(n)]


def category_rings(axis: str, cats: dict[int, str], order: list[str], u: dict[int, tuple[float, float]],
                   feat: dict[int, dict]) -> tuple[list, list[str]]:
    """같은 범주를 줄로 묶는다. 큰 범주는 두 번째 기준(SPLIT, 없으면 지도 세로) 순으로 ROW_LEN 안팎씩 나누고
    그 기준을 이름에 붙여 줄 이름이 모두 다르게 한다. 줄 안은 지도 가로(u x) 순.
    줄의 x(위아래 이동 때 가까운 칸 찾기)는 지도 가로 위치 0–100."""
    from collections import Counter
    rows, names = [], []
    rank = {c: n for n, c in enumerate(order)}
    groups: dict[str, list[int]] = {}
    for i, c in cats.items():
        groups.setdefault(c, []).append(i)
    small = [c for c, m in groups.items() if len(m) < 8 and c != "Various"]   # 너무 짧은 줄은 좌우가 금방 돈다
    for c in small:
        groups.setdefault("Various", []).extend(groups.pop(c))
    key, words, fmt = SPLIT.get(axis, (None, None, None))
    objs_used: set[str] = set()
    for c in sorted(groups, key=lambda c: (rank.get(c, len(rank)), c)):
        n = max(1, round(len(groups[c]) / ROW_LEN))
        if key:
            mem = sorted(groups[c], key=lambda i: (feat.get(i, {}).get(key) is None, feat.get(i, {}).get(key) or 0))
        else:
            mem = sorted(groups[c], key=lambda i: u[i][1])
        parts = [p.tolist() for p in np.array_split(np.array(mem), n)]
        if axis == "color":
            ws = spread(COLOR_WORDS[c], n) if c in COLOR_WORDS else [f"{w} {c.lower()}" if w else c for w in spread(TONE, n)]
            labels = [w or c for w in ws]
        elif words and n > 1:
            labels = [fmt.format(c=c, w=w) for w in spread(words, n)]
        elif axis == "look" and n > 1:   # 큰 묶음은 줄마다 그 줄에서 가장 많은 물체·장르로 구분
            labels, taken = [], set()
            for part in parts:
                cnt = Counter(v for i in part for v in [feat.get(i, {}).get("genre"), *feat.get(i, {}).get("objs", [])[:3]]
                              if v and v not in c and v not in taken)
                v = next((v for v, _ in cnt.most_common()), None)
                if v:
                    taken.add(v)
                labels.append(f"{c} · {v}" if v else c)
        elif axis == "objects" and n > 1:   # 같은 물체 줄은 그 줄에 함께 많이 나온 다른 물체로 구분
            labels = []
            for part in parts:
                cnt = Counter(o for i in part for o in feat.get(i, {}).get("objs", []) if o != c and o not in objs_used)
                o = next((o for o, _ in cnt.most_common()), None)
                if o:
                    objs_used.add(o)
                labels.append(f"{c} · {o}" if o else c)
        else:
            labels = [c] * n
        for part, lab in zip(parts, labels):
            row = sorted(part, key=lambda i: u[i][0])
            rows.append([[int(i), int(round(u[i][0] * 100))] for i in row])
            names.append(lab)
    # 그래도 겹치는 이름은 차례로 II, III … (마지막 수단)
    seen: Counter = Counter()
    out = []
    for nm in names:
        seen[nm] += 1
        out.append(nm + ROMAN[min(seen[nm] - 1, len(ROMAN) - 1)] if names.count(nm) > 1 else nm)
    return rows, out


# ---------- 범주 축(2026-11-12): 나라 · 상영 시간 · 화면비 · People · Type · Composition · Medium ----------
TROPES = ROOT / "data" / "analysis" / "tropes.json"
FACES = ROOT / "data" / "analysis" / "faces.json"
PERSON_LABELS = {"person", "silhouette of a person", "couple embracing", "child"}
TROPE_ORDER = {   # build_tropes.py FAMILIES 와 같은 순서
    "people": ["Crowd", "Couple", "Talking heads", "Big face", "Lone figure", "Figures", "No one"],
    "type": ["Type only", "Big centered title", "Big title", "Quiet type", "Title on top", "Title in the middle", "Title at the bottom"],
    "composition": ["Grid / collage", "Symmetric", "Off-center subject", "Silhouette", "Busy frame", "Close crop",
                    "Single object", "Open space", "Diagonal", "Wide scene"],
    "medium": ["Photo", "Illustration", "Painting", "Graphic", "Typographic"],
}
TROPE_SUB = {"people": "medium", "type": "medium", "composition": "medium", "medium": "people"}   # 큰 범주를 나눌 두 번째 말
COUNTRY_FIX = {"Austrailia": "Australia", "Angolia": "Angola", "Bengladesh": "Bangladesh", "Republic of Macedonia": "North Macedonia"}
REGION = {   # 나라 → 지역(줄 묶음). 순서가 줄 순서
    "North America": ["United States", "Canada"],
    "Latin America": ["Mexico", "Cuba", "Brazil", "Argentina", "Guatemala"],
    "British Isles": ["United Kingdom", "Ireland"],
    "France": ["France"],
    "Western Europe": ["Belgium", "Switzerland", "Austria", "Germany", "West Germany", "Portugal", "Spain"],
    "Italy & Greece": ["Italy", "Greece"],
    "Nordic": ["Sweden", "Denmark", "Norway", "Finland"],
    "Eastern Europe": ["Soviet Union", "Ukraine", "Latvia", "Kazakhstan", "Poland", "Czechoslovakia", "Hungary", "Romania",
                       "Yugoslavia", "North Macedonia"],
    "Middle East & North Africa": ["Iran", "Turkey", "Egypt", "Morocco", "Algeria"],
    "Africa": ["Senegal", "Angola", "Nigeria", "Ethiopia", "Cameroon", "Lesotho", "Guinea-Bissau"],
    "South Asia": ["India", "Bangladesh"],
    "East Asia": ["Japan", "Hong Kong", "Taiwan", "China", "South Korea"],
    "Southeast Asia & Oceania": ["Philippines", "Thailand", "Indonesia", "Australia", "New Zealand"],
}
COUNTRY_REGION = {c: r for r, cs in REGION.items() for c in cs}


def first_country(s: str | None) -> str:
    c = (s or "").split("|")[0].split(",")[0].strip()
    return COUNTRY_FIX.get(c, c)


def aspect_value(ar: list | None) -> float | None:
    import re
    m = re.match(r"\s*(\d\.\d+)", (ar or [""])[0] or "")
    return float(m.group(1)) if m else None


ASPECT_BUCKETS = [(1.30, "1.19:1"), (1.355, "1.33:1"), (1.46, "1.37:1"), (1.70, "1.66:1"), (1.80, "1.78:1"),
                  (1.95, "1.85:1"), (9.0, "2:1 and wider")]


def years_label(items: list[dict], part: list[int]) -> str:
    ys = sorted(items[i]["year"] for i in part if items[i].get("year"))
    if not ys:
        return ""
    a, b = ys[0], ys[-1]
    return str(a) if a == b else f"{a}–{str(b)[2:]}" if a // 100 == b // 100 else f"{a}–{b}"


def group_rings(groups: list[tuple[str, list[int]]], label, order_in_row, extra=None) -> tuple[list, list[str]]:
    """[(범주, 정렬된 커버)] → 줄. 범주는 ROW_LEN 안팎으로 고르게 나누고 label(범주, 줄 커버, 나눈 수) 로 이름을 짓는다.
    줄 안은 order_in_row(i) 순, 줄의 x 는 0–100 (위아래 이동 때 가까운 칸)."""
    from collections import Counter
    rows, names, parts_all = [], [], []
    for c, mem in groups:
        if not mem:
            continue
        n = max(1, round(len(mem) / ROW_LEN))
        for part in np.array_split(np.array(mem), n):
            part = [int(i) for i in part]
            names.append(label(c, part, n))
            parts_all.append(part)
            row = sorted(part, key=order_in_row)
            m = max(1, len(row) - 1)
            rows.append([[i, round(k / m * 100)] for k, i in enumerate(row)])
    if extra:   # 겹치는 이름에는 extra(줄 커버)를 붙인다(예: 연도 범위)
        names = [f"{nm}, {extra(pt)}" if names.count(nm) > 1 else nm for nm, pt in zip(names, parts_all)]
    seen: Counter = Counter()
    out = []
    for nm in names:
        seen[nm] += 1
        out.append(nm + ROMAN[min(seen[nm] - 1, len(ROMAN) - 1)] if names.count(nm) > 1 else nm)
    return rows, out


def rings_grid(rows: list) -> tuple[dict[int, list[int]], list[int]]:
    """격자 = 줄을 그대로 쌓은 것(한 줄 = 한 범주 조각). 세 폭 보기의 위아래와 격자의 위아래가 같다."""
    pos = {i: [c, r] for r, row in enumerate(rows) for c, (i, _) in enumerate(row)}
    return pos, [max(len(r) for r in rows), len(rows)]


def ring_neighbors(rows: list) -> dict[int, list[int]]:
    nb = {}
    for row in rows:
        ids = [i for i, _ in row]
        for k, i in enumerate(ids):
            nb[i] = [ids[(k + d) % len(ids)] for d in (-1, 1, -2, 2)]
    return nb


def category_axes(items: list[dict], tropes: dict, u_look: dict[int, tuple[float, float]]) -> dict[str, tuple[list, list[str]]]:
    from collections import Counter
    out: dict[str, tuple[list, list[str]]] = {}
    ux = lambda i: u_look.get(i, (0.5, 0.5))[0]
    by_year = lambda i: (items[i].get("year") or 0, items[i]["spine"])

    # 나라: 지역 → 나라(편수 많은 순) → 연도. 줄 이름은 나라 하나면 '나라, 연도', 여럿이면 '지역: 나라, 나라'
    groups = []
    for reg, cs in [*REGION.items(), ("Elsewhere", [])]:
        mem = [i for i, it in enumerate(items) if COUNTRY_REGION.get(first_country(it["countries"]), "Elsewhere") == reg]
        cnt = Counter(first_country(items[i]["countries"]) for i in mem)
        rank = {c: k for k, (c, _) in enumerate(cnt.most_common())}
        mem.sort(key=lambda i: (rank[first_country(items[i]["countries"])], by_year(i)))
        # 큰 나라(한 줄 이상)는 따로, 작은 나라는 지역으로 모은다
        big = [c for c, v in cnt.items() if v >= ROW_LEN * 0.75]
        for c in sorted(big, key=lambda c: rank[c]):
            groups.append((c, [i for i in mem if first_country(items[i]["countries"]) == c]))
        rest = [i for i in mem if first_country(items[i]["countries"]) not in big]
        if rest and len(rest) < 8 and big:      # 너무 짧은 줄은 같은 지역의 마지막 큰 나라 줄에 붙인다
            c, m = groups[-1]
            groups[-1] = (c, m + rest)
        elif rest:
            groups.append((reg, rest))

    def country_label(c, part, n):
        cs = list(dict.fromkeys(first_country(items[i]["countries"]) for i in part))
        if len(cs) == 1:
            return f"{cs[0]}, {years_label(items, part)}" if n > 1 else cs[0]
        if len(cs) <= 2 and cs[0] == c:          # 큰 나라 줄에 작은 이웃이 붙은 경우
            return f"{c} & {cs[1]}, {years_label(items, part)}" if n > 1 else f"{c} & {cs[1]}"
        reg = COUNTRY_REGION.get(cs[0], "Elsewhere")    # 여러 나라 줄은 지역 이름만(세로 줄 이름이 길어지지 않게)
        return f"{reg}, {years_label(items, part)}" if n > 1 else reg
    out["country"] = group_rings(groups, country_label, by_year)

    # 상영 시간: 짧은 것부터, 줄 이름은 그 줄의 범위
    have = sorted((i for i, it in enumerate(items) if it.get("rt")), key=lambda i: (items[i]["rt"], by_year(i)))
    def rt_label(c, part, n):
        v = sorted(items[i]["rt"] for i in part)
        return f"{v[0]} min" if v[0] == v[-1] else f"{v[0]}–{v[-1]} min"
    out["runtime"] = group_rings([("rt", have)], rt_label, lambda i: (items[i]["rt"], by_year(i)))

    # 화면비: 표준 비율 묶음 → 연도. 나뉜 줄은 연도 범위로 구분
    buckets: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        v = aspect_value(it.get("ar"))
        if v:
            buckets.setdefault(next(lab for lim, lab in ASPECT_BUCKETS if v < lim), []).append(i)
    groups = [(lab, sorted(buckets[lab], key=by_year)) for _, lab in ASPECT_BUCKETS if lab in buckets]
    out["aspect"] = group_rings(groups, lambda c, part, n: f"{c}, {years_label(items, part)}" if n > 1 else c, by_year)

    # People · Type · Composition · Medium (build_tropes.py). 큰 범주는 두 번째 말(TROPE_SUB)로 나눠 이름에 붙인다
    tr = {i: tropes[it["file"]] for i, it in enumerate(items) if it["file"] in tropes}
    for fam, order in TROPE_ORDER.items():
        sub = TROPE_SUB[fam]
        sub_rank = {c: k for k, c in enumerate(TROPE_ORDER[sub])}
        groups = []
        for c in order:
            mem = sorted((i for i in tr if tr[i][fam] == c), key=lambda i: (sub_rank.get(tr[i][sub], 99), by_year(i)))
            groups.append((c, mem))
        def lab(c, part, n, sub=sub):
            if n == 1:
                return c
            top = Counter(tr[i][sub] for i in part).most_common(1)[0][0]
            return f"{c} · {top}"
        out[fam] = group_rings(groups, lab, ux, lambda pt: years_label(items, pt))
    return out



# ---------- 자유 배치(2026-11-12): 격자를 직사각형으로 채우지 않고 축마다 모양을 그린 뒤 가까운 칸에 배정한다 ----------
CELL_X, CELL_Y = 1.18, 1.42     # 칸 하나의 화면 크기(타일 너비 1 기준, 간격 포함). 모양을 화면 비율로 맞출 때 쓴다

LAND = ROOT / "data" / "raw" / "ne_110m_land.geojson"   # Natural Earth(public domain), 지도 배경
LONLAT = {
    "United States": (-98, 39), "Canada": (-92, 57), "Mexico": (-102, 23), "Cuba": (-79, 21.5), "Guatemala": (-90.5, 15.5),
    "Brazil": (-51, -10), "Argentina": (-64, -34), "United Kingdom": (-2, 54), "Ireland": (-8, 53.3), "France": (2.5, 46.5),
    "Belgium": (4.5, 50.6), "Switzerland": (8.2, 46.8), "Austria": (14.5, 47.5), "Germany": (10.4, 51), "West Germany": (8, 50.5),
    "Portugal": (-8, 39.5), "Spain": (-3.7, 40.3), "Italy": (12.5, 42.8), "Greece": (22, 39), "Sweden": (16, 62),
    "Denmark": (10, 56), "Norway": (9, 61), "Finland": (26, 63), "Soviet Union": (37.6, 55.7), "Ukraine": (31, 49),
    "Latvia": (24.6, 56.9), "Kazakhstan": (67, 48), "Poland": (19.4, 52), "Czechoslovakia": (16.5, 49.7), "Hungary": (19.5, 47.2),
    "Romania": (25, 45.9), "Yugoslavia": (20.5, 44), "North Macedonia": (21.7, 41.6), "Iran": (53, 32.5), "Turkey": (35, 39),
    "Egypt": (30, 26.8), "Morocco": (-6, 32), "Algeria": (3, 28), "Senegal": (-14.5, 14.5), "Angola": (17.9, -12.3),
    "Nigeria": (8, 9.5), "Ethiopia": (39.5, 9), "Cameroon": (12.4, 5.7), "Lesotho": (28.2, -29.5), "Guinea-Bissau": (-15, 12),
    "India": (79, 22), "Bangladesh": (90.3, 23.7), "Japan": (138, 36.5), "Hong Kong": (114.2, 22.3), "Taiwan": (121, 23.7),
    "China": (104, 35), "South Korea": (127.8, 36.3), "Philippines": (122, 12.5), "Thailand": (101, 15), "Indonesia": (113, -1),
    "Australia": (134, -25), "New Zealand": (172, -41),
}


def sunflower(n: int, scale: float = 1.0) -> np.ndarray:
    """원반 안에 고르게 n 점(화면 단위, 넓이 ≈ n 칸). 0 번이 가운데."""
    k = np.arange(n) + 0.5
    r = np.sqrt(k / np.pi) * np.sqrt(CELL_X * CELL_Y) * scale
    a = k * np.pi * (3 - np.sqrt(5))
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def pack(ids: list[int], target: np.ndarray, margin: int = 2) -> tuple[dict[int, list[int]], list[int]]:
    """화면 단위 목표 좌표 → 겹치지 않는 정수 칸(목표에 가장 가깝게, 헝가리안). 빈칸이 남아 모양이 보인다."""
    tc = np.stack([target[:, 0] / CELL_X, target[:, 1] / CELL_Y], 1)
    lo = np.floor(tc.min(0)) - margin
    hi = np.ceil(tc.max(0)) + margin
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    while len(cells) < len(ids) * 1.05:      # 칸이 모자라면 넓힌다
        lo -= 2; hi += 2
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        cells = np.stack([gx.ravel(), gy.ravel()], 1)
    d = ((tc[:, None, 0] - cells[None, :, 0]) * CELL_X) ** 2 + ((tc[:, None, 1] - cells[None, :, 1]) * CELL_Y) ** 2
    _, col = linear_sum_assignment(d)
    got = cells[col].astype(int)
    mn = got.min(0)
    got -= mn
    pos = {i: [int(x), int(y)] for i, (x, y) in zip(ids, got)}
    size = [int(got[:, 0].max()) + 1, int(got[:, 1].max()) + 1]
    pack.offset = mn                          # 이름표 좌표를 같은 칸 좌표로 옮길 때 쓴다
    return pos, size


def blob_labels(pos: dict[int, list[int]], groups: dict[str, list[int]], min_n: int = 1) -> list:
    """덩어리마다 이름표: 덩어리 맨 위 칸의 가운데 위."""
    out = []
    for name, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):   # 큰 덩어리부터, 겹치는 이름표는 뺀다
        mem = [i for i in mem if i in pos]
        if len(mem) < min_n:
            continue
        xs = [pos[i][0] for i in mem]; ys = [pos[i][1] for i in mem]
        top = min(ys)
        cx = sum(xs) / len(xs)
        w = len(name) * 1.0 + 1
        if any(abs(cx + 0.5 - o[0]) < (w + len(o[2]) * 1.0 + 1) / 2 and abs(top - 0.35 - o[1]) < 2.2 for o in out):
            continue
        out.append([round(cx + 0.5, 2), top - 0.35, name, "center"])
    return out


# ---------- 범주 그림(픽토그램): 범주의 뜻을 그린 모양을 칸으로 채운다 ----------
def _pict(fam: str, c: str):
    """(가로:세로 비율, 그리는 함수(d, W, H)). 흰색으로 칠한 곳에 커버가 놓인다."""
    def ell(d, W, H, x0, y0, x1, y1, fill=255):
        d.ellipse([x0 * W, y0 * H, x1 * W, y1 * H], fill=fill)
    def rect(d, W, H, x0, y0, x1, y1, fill=255):
        d.rectangle([x0 * W, y0 * H, x1 * W, y1 * H], fill=fill)
    def frame(d, W, H, t=0.16):
        rect(d, W, H, 0, 0, 1, 1); rect(d, W, H, t, t * 0.8, 1 - t, 1 - t * 0.8, 0)
    def figure(d, W, H, cx, top, s):
        ell(d, W, H, cx - 0.09 * s, top, cx + 0.09 * s, top + 0.16 * s * W / H * 0 + 0.13 * s)
        d.polygon([((cx - 0.17 * s) * W, (top + 0.95 * s) * H), ((cx - 0.12 * s) * W, (top + 0.17 * s) * H),
                   ((cx + 0.12 * s) * W, (top + 0.17 * s) * H), ((cx + 0.17 * s) * W, (top + 0.95 * s) * H)], fill=255)
    P = {
        ("people", "Crowd"): (1.4, lambda d, W, H: [ell(d, W, H, x - .11, y - .15, x + .11, y + .15)
                                                   for y, xs in ((.2, (.15, .38, .62, .85)), (.5, (.27, .5, .73)), (.8, (.15, .38, .62, .85)))
                                                   for x in xs]),
        ("people", "Couple"): (1.5, lambda d, W, H: (ell(d, W, H, 0, 0, .62, 1), ell(d, W, H, .38, 0, 1, 1))),
        ("people", "Talking heads"): (2.6, lambda d, W, H: [ell(d, W, H, x, .05, x + .3, .95) for x in (0, .35, .7)]),
        ("people", "Big face"): (0.85, lambda d, W, H: (ell(d, W, H, 0, 0, 1, 1), ell(d, W, H, .25, .3, .38, .42, 0),
                                                       ell(d, W, H, .62, .3, .75, .42, 0),
                                                       d.chord([.27 * W, .45 * H, .73 * W, .78 * H], 20, 160, fill=0))),
        ("people", "Lone figure"): (0.5, lambda d, W, H: figure(d, W, H, .5, 0, 1)),
        ("people", "Figures"): (1.5, lambda d, W, H: [figure(d, W, H, x, .1 if k % 2 else 0, .9) for k, x in enumerate((.17, .5, .83))]),
        ("people", "No one"): (0.8, lambda d, W, H: frame(d, W, H)),
        ("type", "Type only"): (0.8, lambda d, W, H: (rect(d, W, H, 0, 0, 1, .28), rect(d, W, H, .36, .28, .64, 1))),
        ("type", "Big centered title"): (0.8, lambda d, W, H: (frame(d, W, H), rect(d, W, H, .18, .38, .82, .62))),
        ("type", "Big title"): (0.8, lambda d, W, H: (frame(d, W, H), rect(d, W, H, 0, .3, 1, .7))),
        ("type", "Quiet type"): (0.8, lambda d, W, H: (frame(d, W, H), rect(d, W, H, .35, .8, .65, .86))),
        ("type", "Title on top"): (0.8, lambda d, W, H: (frame(d, W, H), rect(d, W, H, .15, .1, .85, .3))),
        ("type", "Title in the middle"): (0.8, lambda d, W, H: (frame(d, W, H), rect(d, W, H, .15, .42, .85, .58))),
        ("type", "Title at the bottom"): (0.8, lambda d, W, H: (frame(d, W, H), rect(d, W, H, .15, .7, .85, .9))),
        ("composition", "Grid / collage"): (1.0, lambda d, W, H: [rect(d, W, H, x, y, x + .28, y + .28) for x in (0, .36, .72) for y in (0, .36, .72)]),
        ("composition", "Symmetric"): (1.3, lambda d, W, H: (d.polygon([(0, 0), (.5 * W, .5 * H), (0, H)], fill=255),
                                                            d.polygon([(W, 0), (.5 * W, .5 * H), (W, H)], fill=255),
                                                            ell(d, W, H, .38, .3, .62, .7))),
        ("composition", "Off-center subject"): (1.2, lambda d, W, H: (frame(d, W, H, .13), ell(d, W, H, .2, .22, .55, .78))),
        ("composition", "Silhouette"): (0.9, lambda d, W, H: (ell(d, W, H, .3, 0, .7, .45), ell(d, W, H, 0, .45, 1, 1.5))),
        ("composition", "Busy frame"): (1.2, lambda d, W, H: [ell(d, W, H, x, y, x + .17, y + .2) for x in np.linspace(0, .83, 6)
                                                            for y in np.linspace(0, .8, 5) if (round(x * 10) + round(y * 10)) % 3]),
        ("composition", "Close crop"): (1.0, lambda d, W, H: rect(d, W, H, 0, 0, 1, 1)),
        ("composition", "Single object"): (1.0, lambda d, W, H: (frame(d, W, H, .13), ell(d, W, H, .32, .32, .68, .68))),
        ("composition", "Open space"): (1.4, lambda d, W, H: (frame(d, W, H, .1), ell(d, W, H, .68, .6, .8, .78))),
        ("composition", "Diagonal"): (1.2, lambda d, W, H: d.polygon([(0, .75 * H), (.75 * W, 0), (W, 0), (W, .25 * H), (.25 * W, H), (0, H)], fill=255)),
        ("composition", "Wide scene"): (3.4, lambda d, W, H: (rect(d, W, H, 0, .45, 1, 1), ell(d, W, H, .62, 0, .78, .4))),
        ("medium", "Photo"): (1.5, lambda d, W, H: (rect(d, W, H, .15, 0, .38, .2), rect(d, W, H, 0, .15, 1, 1),
                                                    ell(d, W, H, .3, .3, .7, .92, 0), ell(d, W, H, .4, .45, .6, .78))),
        ("medium", "Illustration"): (2.6, lambda d, W, H: d.polygon([(0, .5 * H), (.18 * W, .15 * H), (W, .15 * H), (W, .85 * H), (.18 * W, .85 * H)], fill=255)),
        ("medium", "Painting"): (1.3, lambda d, W, H: (ell(d, W, H, 0, 0, 1, 1), ell(d, W, H, .62, .55, .8, .78, 0))),
        ("medium", "Graphic"): (2.0, lambda d, W, H: (d.polygon([(0, H), (.17 * W, 0), (.33 * W, H)], fill=255),
                                                      ell(d, W, H, .36, 0, .64, 1), rect(d, W, H, .68, .05, 1, .95))),
        ("medium", "Typographic"): (0.95, lambda d, W, H: (d.polygon([(0, H), (.38 * W, 0), (.62 * W, 0), (W, H), (.75 * W, H), (.5 * W, .2 * H), (.25 * W, H)], fill=255),
                                                           rect(d, W, H, .25, .58, .75, .74))),
    }
    return P.get((fam, c), (1.0, lambda d, W, H: ell(d, W, H, 0, 0, 1, 1)))


def pictogram_cells(fam: str, c: str, n: int) -> np.ndarray:
    """범주 그림을 칸으로 채운 n 개 자리(칸 좌표, 0 부터). 칸은 화면에서 세로가 길어(CELL_Y/CELL_X) 그 비율로 그린다."""
    aspect, draw = _pict(fam, c)
    def cells_at(s: float) -> np.ndarray:
        Wv, Hv = s * aspect, s                                       # 화면 단위
        cw, ch = int(np.ceil(Wv / CELL_X)), int(np.ceil(Hv / CELL_Y))
        K = 12
        im = Image.new("L", (int(Wv * K) + 1, int(Hv * K) + 1), 0)
        draw(ImageDraw.Draw(im), im.width - 1, im.height - 1)
        a = np.asarray(im)
        out = [(x, y) for y in range(ch) for x in range(cw)
               if a[min(a.shape[0] - 1, int((y + .5) * CELL_Y * K)), min(a.shape[1] - 1, int((x + .5) * CELL_X * K))] > 127]
        return np.array(out)
    lo, hi = 1.0, 200.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if len(cells_at(mid)) >= n:
            hi = mid
        else:
            lo = mid
    cells = cells_at(hi)
    if len(cells) > n:      # 남는 칸은 고르게 솎아 모양을 유지한다
        keep = np.round(np.linspace(0, len(cells) - 1, n)).astype(int)
        cells = cells[keep]
    return cells - cells.min(0)


def free_layouts(items: list[dict], tropes: dict, look: dict[int, np.ndarray], u_look: dict[int, tuple[float, float]]) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    n = len(items)

    # 나라: 경위도에 나라별 덩어리(오래된 영화가 가운데). 대양은 줄이고 붐비는 유럽은 넓힌다.
    # 모양이 뚜렷한 나라는 덩어리를 늘이고 돌린다(가로 배율, 세로 배율, 회전°)
    def proj(lon: float, lat: float) -> tuple[float, float]:
        if lon < -30:
            x = -14 + (lon + 30) * 0.42
        elif lon <= 45:
            x = lon * 0.95
        else:
            x = 43 + (lon - 45) * 0.5
        y = -lat * 1.0 if lat > 30 else -30 - (lat - 30) * 0.45
        return x, y
    SHAPE = {"United States": (1.55, 0.65, 0), "Japan": (0.62, 1.6, -35), "Italy": (0.6, 1.6, -35), "United Kingdom": (0.7, 1.45, 10),
             "Sweden": (0.6, 1.6, 20), "France": (1.0, 1.0, 0), "Canada": (1.8, 0.55, 0), "Mexico": (1.3, 0.75, -25),
             "Soviet Union": (1.5, 0.7, 0), "India": (0.9, 1.1, 0), "Chile": (0.4, 2.5, 0)}
    S = 1.25
    ids, tg, groups = [], [], {}
    by_c: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        by_c.setdefault(it.get("country") or "", []).append(i)
    for c, mem in by_c.items():
        lon, lat = LONLAT.get(c, (-30, -40))
        mem.sort(key=lambda i: (items[i].get("year") or 0, items[i]["spine"]))
        sx, sy, rot = SHAPE.get(c, (1, 1, 0))
        pts = sunflower(len(mem)) * np.array([sx, sy])
        th = np.radians(rot)
        pts = pts @ np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]).T
        x, y = proj(lon, lat)
        ids += mem; tg.append(pts + np.array([x * S * 1.2, y * S * 1.2]))
        groups[c or "Unknown"] = mem
    pos, size = pack(ids, np.concatenate(tg), margin=1)
    # 배경: Natural Earth 1:110m 육지 윤곽(공개 데이터)을 같은 투영으로 칸 좌표에 그린다(뷰어가 희미하게 깐다)
    path = []
    if LAND.exists():
        off = pack.offset
        for ftr in json.loads(LAND.read_text())["features"]:
            g = ftr["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                ring = poly[0]
                if max(lat for _, lat in ring) < -55:          # 남극은 뺀다
                    continue
                pts = []
                for lon, lat in ring[::2] if len(ring) > 40 else ring:
                    x, y = proj(lon, lat)
                    pts.append(f"{x * S * 1.2 / CELL_X - off[0]:.1f} {y * S * 1.2 / CELL_Y - off[1]:.1f}")
                path.append("M" + "L".join(pts) + "Z")
    out["country"] = (pos, size, blob_labels(pos, {c: m for c, m in groups.items()}, min_n=12), "".join(path))

    # 화면비: 비율마다 진짜 화면비 모양의 블록. 블록 안은 연도 순(왼쪽 위 → 오른쪽 아래), 블록은 왼쪽부터 아래를 맞춰 놓는다
    buckets: dict[str, list[int]] = {}
    ratio = {}
    for i, it in enumerate(items):
        v = aspect_value(it.get("ar"))
        if v:
            lab = next(l for lim, l in ASPECT_BUCKETS if v < lim)
            buckets.setdefault(lab, []).append(i)
            ratio.setdefault(lab, []).append(v)
    pos, labels, x0, blocks = {}, [], 0, []
    for _, lab in ASPECT_BUCKETS:
        if lab not in buckets:
            continue
        mem = sorted(buckets[lab], key=lambda i: (items[i].get("year") or 0, items[i]["spine"]))
        R = float(np.median(ratio[lab]))
        best = None
        for rows in range(1, len(mem) + 1):
            cols = int(np.ceil(len(mem) / rows))
            err = abs(np.log(cols * CELL_X / (rows * CELL_Y) / R)) + 0.02 * (cols * rows - len(mem))
            if best is None or err < best[0]:
                best = (err, rows, cols)
        blocks.append((lab, mem, best[1], best[2]))
    H = max(b[2] for b in blocks)
    for lab, mem, rows, cols in blocks:
        for k, i in enumerate(mem):
            pos[i] = [x0 + k % cols, H - rows + k // cols]
        labels.append([x0 + cols / 2, H + 0.9, f"{lab}", "center"])
        x0 += cols + 2
    out["aspect"] = (pos, [x0 - 2, H + 1], labels)

    # 상영 시간: 분포 막대(3분 칸, 막대 너비 2칸, 60분 아래와 200분 위는 양 끝 막대). 아래에서 위로, 같은 막대 안은 짧은 순
    have = [i for i, it in enumerate(items) if it.get("rt")]
    NB, BW = 2 + (200 - 60) // 3 + 1, 2
    bin_of = lambda m: 0 if m < 60 else NB - 1 if m >= 200 else 1 + (m - 60) // 3
    bars: dict[int, list[int]] = {}
    for i in sorted(have, key=lambda i: (items[i]["rt"], items[i].get("year") or 0)):
        bars.setdefault(bin_of(items[i]["rt"]), []).append(i)
    xb = lambda b: b * BW + (1 if b > 0 else 0) + (1 if b == NB - 1 else 0)      # 양 끝 막대는 한 칸 띄운다
    Hc = max((len(v) + BW - 1) // BW for v in bars.values())
    pos = {i: [xb(b) + k % BW, Hc - 1 - k // BW] for b, v in bars.items() for k, i in enumerate(v)}
    labels = [[xb(0) + 1, Hc + 0.6, "< 60′", "center"], [xb(NB - 1) + 1, Hc + 0.6, "200′ +", "center"]]
    labels += [[xb(1 + (m - 60) // 3), Hc + 0.6, f"{m}′", "center"] for m in (90, 120, 150, 180)]
    out["runtime"] = (pos, [xb(NB - 1) + BW, Hc + 1], labels)

    # 그림 범주: 범주마다 그 뜻을 그린 그림(픽토그램) 모양으로 커버를 채운다. 그림 안은 겉모습 지도에서 가까운 것끼리.
    # 그림들은 왼쪽부터 줄지어 놓고 아래에 이름을 단다
    ids_all = [i for i in range(n) if i in u_look]
    upos = {i: np.array(u_look[i]) for i in ids_all}
    for fam, order in TROPE_ORDER.items():
        groups = {c: [i for i in ids_all if tropes.get(items[i]["file"], {}).get(fam) == c] for c in order}
        groups = {c: m for c, m in groups.items() if m}
        pieces = []
        for c, m in groups.items():
            cells = pictogram_cells(fam, c, len(m))
            L = np.stack([upos[i] for i in m]); L = (L - L.mean(0)) / (L.std(0) + 1e-9)
            C = cells.astype(float); Cn = (C - C.mean(0)) / (C.std(0) + 1e-9)
            _, col = linear_sum_assignment(((L[:, None] - Cn[None]) ** 2).sum(-1))
            pieces.append((c, m, cells[col]))
        # 줄 배치: 전체 넓이에서 화면 비율(≈1.7)에 맞는 폭을 정하고 넘치면 다음 줄
        area = sum((p[2][:, 0].max() + 1) * (p[2][:, 1].max() + 1) for p in pieces)
        maxw = max(max(p[2][:, 0].max() + 1 for p in pieces), np.sqrt(area * 1.7 * CELL_Y / CELL_X) * 1.15)
        pos, labels, x, y, rowh, row = {}, [], 0, 0, 0, []
        def flush(row, y, rowh):
            for c, m, cells, x0 in row:
                h = cells[:, 1].max() + 1
                for i, (cx, cy) in zip(m, cells):
                    pos[i] = [int(x0 + cx), int(y + rowh - h + cy)]          # 줄 안에서 아래 맞춤
                labels.append([x0 + (cells[:, 0].max() + 1) / 2, y + rowh + 0.9, c, "center"])
        for c, m, cells in pieces:
            w, h = cells[:, 0].max() + 1, cells[:, 1].max() + 1
            if row and x + w > maxw:
                flush(row, y, rowh); y += rowh + 3; x, rowh, row = 0, 0, []
            row.append((c, m, cells, x)); x += w + 3; rowh = max(rowh, h)
        flush(row, y, rowh)
        size = [int(max(v[0] for v in pos.values()) + 1), int(y + rowh + 2)]
        out[fam] = (pos, size, [[float(a), float(b), c, al] for a, b, c, al in labels])
    return out


def main() -> None:
    for d in SIZES:
        (WEB / d).mkdir(parents=True, exist_ok=True)
    (WEB / "trace").mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    films = {r[0]: r[1:] for r in db.execute("select spine, title, year, directors, countries, kind, box_set, criterion_id, wikidata_qid from films")}
    extra = json.loads(EXTRA.read_text()) if EXTRA.exists() else {}
    cmeta = json.loads(CMETA.read_text()) if CMETA.exists() else {}

    emb = {}
    if EMBED.exists():
        e = np.load(EMBED)
        emb = dict(zip(e["files"], e["vectors"]))

    objects = json.loads(OBJECTS.read_text()) if OBJECTS.exists() else {}
    faces = json.loads(FACES.read_text()) if FACES.exists() else {}
    tropes = json.loads(TROPES.read_text()) if TROPES.exists() else {}
    obj_raw: dict[int, list] = {}

    items, hue, traces = [], [], []
    ax: dict[str, dict[int, np.ndarray]] = {k: {} for k in
                                            ("look", "color", "lines", "text", "contrast", "objects", "strokes", "layout")}
    stroke_scalars: dict[int, list[float]] = {}
    feat: dict[int, dict] = {}     # 줄 이름(row_names)용 지표
    for p in sorted(POSTERS.glob("*.jpg")):
        spine = int(p.name[:4])
        if spine not in films:
            continue
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                for d, size in SIZES.items():
                    t = WEB / d / p.name
                    if not t.exists():
                        im.resize(size, Image.LANCZOS).save(t, quality=82)
                pal = palette(im)
                lmean = float(np.asarray(im.resize((40, 50)).convert("L"), dtype=np.float32).mean() / 255)
        except Exception as e:  # 받는 중인 파일
            print(f"skip {p.name}: {e}")
            continue
        hx, h, gray = representative(pal)
        title, year, directors, countries, kind, box_set, cid, qid = films[spine]
        i = len(items)
        it = {"spine": spine, "file": p.name, "title": title, "year": year, "directors": directors,
              "countries": countries, "palette": [[c, round(r, 3)] for c, r, _ in pal], "color": hx}
        if kind == "box":
            it["kind"] = "box"
        if box_set:
            it["box"] = box_set
        if cid:   # criterion.com 주소: 숫자는 /films/<id>, 박스는 /boxsets/<id-slug>
            it["cid"] = cid
        it.update(film_extra(extra.get(qid or "", {}), title))
        cm = cmeta.get(str(spine), {})   # criterion.com Film Info 가 있으면 그쪽이 우선(그 판본의 실제 사양)
        if cm.get("runtime"):
            it["rt"] = cm["runtime"]
        if cm.get("aspect"):
            it["ar"] = [cm["aspect"]]
        if cm.get("color"):
            it["clr"] = cm["color"]
        if cm.get("language"):
            it["language"] = [x.strip() for x in cm["language"].split(",")][:3]
        if cm.get("formats"):
            it["fmt"] = cm["formats"]

        trace: dict = {}
        vj, rj = VISUAL / f"{p.stem}.json", REGIONS / f"{p.stem}.json"
        frame_rects: list = []
        if vj.exists():
            v = json.loads(vj.read_text())
            f = v["full"]
            gray = f["color"]["color_mode"] == "grayscale"
            ax["color"][i] = np.array(f["vectors"]["color"])
            ax["lines"][i] = np.array(f["vectors"]["lines"])
            ax["contrast"][i] = np.array(f["vectors"]["contrast"])
            frame_rects = v["frame"].get("rects") or []
            segs = sorted(f["lines"]["segments"], key=lambda s: -((s[2] - s[0]) ** 2 + (s[3] - s[1]) ** 2))
            trace["lines"] = segs[:MAX_LINES]
            trace["sal"] = f["saliency"]["centroid"]
            trace["frame"] = frame_rects
            fl = f["lines"]
            feat.setdefault(i, {}).update(light=f["contrast"]["lightness_mean"], rms=f["contrast"]["rms_contrast"],
                                          lh=fl["horizontal_share"], lv=fl["vertical_share"], ld=fl["diagonal_share"],
                                          dens=fl["density"], chroma=f["color"]["colorfulness"], sym=f["lr_symmetry"])
            it["mode"] = f["color"]["color_mode"]
            it["rms"] = f["contrast"]["rms_contrast"]
        r = json.loads(rj.read_text()) if rj.exists() else None
        if p.name in objects and (vj.exists() or r):
            dets = clean_objects(objects[p.name], cover_mask(r, frame_rects))
            if dets:
                obj_raw[i] = dets
                trace["objects"] = dets[:16]
                # 라벨별 가장 높은 점수 순으로 중복 없이
                it["objs"] = list(dict.fromkeys(d["label"] for d in dets))[:8]
        # People 흔적: 얼굴(1)과 사람(0) 상자
        ppl = [[*d["box"], 1] for d in faces.get(p.name, [])] + \
              [[*o["box"], 0] for o in objects.get(p.name, []) if o["label"] in PERSON_LABELS and o["score"] >= 0.25][:12]
        if ppl:
            trace["people"] = [[round(v, 4) for v in b[:4]] + [b[4]] for b in ppl]
        it["country"] = first_country(countries)
        if r:
            ax["text"][i] = text_vector(r, frame_rects)
            trace["text"] = [simplify(x["poly"], r["width"], r["height"]) for x in r["regions"] if len(x["poly"]) >= 3]
        sj = STROKES / f"{p.stem}.json"
        if sj.exists():
            sl = json.loads(sj.read_text())
            st, ly = sl["strokes"], sl["layout"]
            # 획: 방향·굵기 분포(제곱근) + 스칼라(나중에 표준화해 붙인다)
            ax["strokes"][i] = np.concatenate([np.sqrt(st["vectors"]["orientation"]), 0.7 * np.sqrt(st["vectors"]["width"])])
            stroke_scalars[i] = [st["ink_share"], st["coherence"], st["width_mean"], st["intensity"]]
            ax["layout"][i] = np.concatenate([ly["vectors"]["text"], ly["vectors"]["figure"]])
            trace["layout"] = ly["grid"]
            trace["strokes"] = True
            it["st"] = [st["ink_share"], st["coherence"], st["width_mean"]]
            it["ly"] = [ly["shares"]["text"], ly["shares"]["figure"], ly["shares"]["empty"]]
            tc, fc = ly.get("text_centroid") or [None, None], ly.get("figure_centroid") or [None, None]
            feat.setdefault(i, {}).update(tx=tc[0], fy=fc[1], coh=st["coherence"],
                                          band=ly["title_band"], tshare=ly["shares"]["text"], fig=ly["shares"]["figure"],
                                          empty=ly["shares"]["empty"], ink=st["ink_share"], width=st["width_mean"],
                                          shv=st["horizontal_share"], svv=st["vertical_share"])
        if p.name in emb:
            ax["look"][i] = emb[p.name]
        if trace:
            (WEB / "trace" / f"{p.stem}.js").write_text(
                f"(window.TRACE=window.TRACE||{{}})[{json.dumps(p.stem)}]={json.dumps(trace, separators=(',', ':'))};\n")
            it["trace"] = True
        it["gray"] = gray
        items.append(it)
        traces.append(trace)
        hue.append((h, lmean, gray))
        feat.setdefault(i, {}).update(hue=h, lum=lmean, gray=gray, year=year, country=(countries or "").split("|")[0].split(",")[0].strip(),
                                      objs=it.get("objs", []), genre=((it.get("genre") or [""])[0].removesuffix(" film") or None))

    # 물체 벡터: 라벨별 max(점수 × (0.5 + √면적)), 흔한 라벨(face, person)은 IDF 로 낮춘다
    if obj_raw:
        labels = sorted({d["label"] for v in obj_raw.values() for d in v})
        li = {l: k for k, l in enumerate(labels)}
        df = np.zeros(len(labels))
        for v in obj_raw.values():
            for l in {d["label"] for d in v}:
                df[li[l]] += 1
        idf = np.log((1 + len(obj_raw)) / (1 + df)) + 1
        for i, v in obj_raw.items():
            vec = np.zeros(len(labels))
            for d in v:
                x0, y0, x1, y1 = d["box"]
                vec[li[d["label"]]] = max(vec[li[d["label"]]], d["score"] * (0.5 + np.sqrt(max(0, (x1 - x0) * (y1 - y0)))))
            ax["objects"][i] = vec * idf

    # 획 스칼라(잉크 양, 결, 평균 굵기, 진하기)를 표준화해 분포 벡터 뒤에 붙인다
    if stroke_scalars:
        ids = list(stroke_scalars)
        S = np.array([stroke_scalars[i] for i in ids])
        S = (S - S.mean(0)) / (S.std(0) + 1e-9)
        for i, z in zip(ids, S):
            ax["strokes"][i] = np.concatenate([ax["strokes"][i], 0.35 * z])

    # 색 보기: 가로 = 색상환(330°에서 자름, 무채색은 오른쪽 띠), 세로 = 명도
    xy = np.array([[0.9 + 0.1 * (1 - l) if g else ((h * 360 + 30) % 360) / 360 * 0.86, 1 - l] for h, l, g in hue])
    cg, ccols, crows = grid_assign(xy)

    # 축별 지도: 그 축의 벡터 UMAP 을 격자에 맞춘다 (Poster World 의 Global analysis 를 축마다).
    # 색은 지도 대신 위의 색상 × 명도 배치를 쓴다
    import umap
    COS = ("look", "text", "objects", "strokes", "layout")
    grids: dict[str, list[int]] = {"color": [ccols, crows]}
    # 순회 격자: 줄 = 이름 붙일 수 있는 범주(색이면 Red, 대비면 Dark, high contrast …), 줄 안 = 지도에서 가까운 순
    names: dict[str, list[str]] = {}
    cc, co = item_categories("color", list(range(len(items))), feat, None)
    rings: dict[str, list] = {}
    rings["color"], names["color"] = category_rings("color", cc, co, {i: (float(xy[i][1]), float(xy[i][0])) for i in range(len(items))}, feat)
    gpos: dict[str, dict[int, list[int]]] = {"color": dict(enumerate(cg))}
    # 이전 분석 축(Look, Lines, Type, Contrast, Strokes, Layout, Objects)은 범주 축으로 바뀌었다(2026-11-12).
    # Look 지도(SigLIP UMAP)의 가로 위치만 범주 줄 안의 순서로 쓴다
    u_look: dict[int, tuple[float, float]] = {}
    if len(ax["look"]) > 20:
        ids = sorted(ax["look"])
        X = np.stack([ax["look"][i] for i in ids]).astype(np.float32)
        u = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(X)
        u = (u - u.min(0)) / (u.max(0) - u.min(0))
        u_look = {i: (float(a), float(b)) for i, (a, b) in zip(ids, u)}
    nb = {"color": neighbors(ax["color"], "hel")}
    for k, (rr, nm) in category_axes(items, tropes, u_look).items():
        rings[k], names[k] = rr, nm
        gpos[k], grids[k] = rings_grid(rr)
        nb[k] = ring_neighbors(rr)
    glabels: dict[str, list] = {}
    gbg: dict[str, str] = {}
    for k, (pos, size, labs, *bg) in free_layouts(items, tropes, ax["look"], u_look).items():
        gpos[k], grids[k], glabels[k] = pos, size, labs
        if bg and bg[0]:
            gbg[k] = bg[0]
    print("  rings " + ", ".join(f"{k} {len(r)}x~{round(sum(len(x) for x in r) / len(r))}" for k, r in rings.items()), flush=True)
    for i, it in enumerate(items):
        it["g"] = {k: gpos[k][i] for k in gpos if i in gpos[k]}
        it["nb"] = {k: nb[k][i] for k in nb if i in nb[k]}
        if i in nb["color"]:
            it["nb"]["color"] = nb["color"][i]

    atlas = build_atlases(items, traces)
    for png in sorted((WEB / "strokes").glob("*.png")):   # 획 흔적: 배포용 WebP (PNG 의 약 1/4)
        wp = png.with_suffix(".webp")
        if not wp.exists() or wp.stat().st_mtime < png.stat().st_mtime:
            with Image.open(png) as im:
                im.convert("L").save(wp, "WEBP", quality=80, method=6)

    meta = {"count": len(items), "films": len(films), "maxSpine": max(films),
            "grids": grids, "gridLabels": glabels, "gridBg": gbg, "atlas": atlas, "rings": rings, "rowNames": shorten_all(names),
            "built": __import__("datetime").date.today().isoformat(),
            "yearMin": min(f[1] for f in films.values()), "yearMax": max(f[1] for f in films.values()),
            "axes": {k: sum(len(r) for r in v) for k, v in rings.items()}}
    (WEB / "data.js").write_text("window.META = " + json.dumps(meta) + ";\nwindow.COVERS = "
                                 + json.dumps(items, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"{len(items)} covers, axes {meta['axes']} -> {WEB / 'data.js'}")
    # 첫 로딩 화면이 진행률을 계산할 크기(풀린 바이트)를 index.html 에 적는다
    import re as _re
    sizes = {f: (WEB / f).stat().st_size for f in ("data.js", "atlas/cover_s.webp") if (WEB / f).exists()}
    html = (WEB / "index.html").read_text(encoding="utf-8")
    html = _re.sub(r"/\*SIZES\*/.*?/\*END\*/", "/*SIZES*/" + json.dumps(sizes) + "/*END*/", html, count=1)
    (WEB / "index.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
