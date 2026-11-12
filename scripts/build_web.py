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
            "grids": grids, "atlas": atlas, "rings": rings, "rowNames": shorten_all(names),
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
