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
        txt = str({"spine": it["spine"], "year": it["year"], "country": it.get("cc") or it.get("country", ""),
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
    "people": ["3+", "Two", "One", "No-face", "Ambiguous", "None"],          # People v2(묘사 판정, people_v2.json)
    "type": ["Type only", "Big centered title", "Big title", "Quiet type", "Title on top", "Title in the middle", "Title at the bottom"],
    "composition": ["Grid / collage", "Symmetric", "Off-center subject", "Silhouette", "Busy frame", "Close crop",
                    "Single object", "Open space", "Diagonal", "Wide scene"],
    "medium": ["Photo", "Painting", "Drawing", "Print", "Collage", "Graphic", "Other"],   # 묘사 Medium 줄(medium_v2.json)
}
# 묘사 기반 판정·지도(2026-11-13). 파일이 있으면 tropes 의 people·medium 을 덮어쓰고, 지도는 묘사 그 줄의 임베딩으로 그린다
PEOPLE_V2 = ROOT / "data" / "analysis" / "people_v2.json"
MEDIUM_V2 = ROOT / "data" / "analysis" / "medium_v2.json"
LAB_LABELS = ROOT / "data" / "labels" / "lab.json"     # Lab(scripts/lab_server.py)에서 사람이 고친 범주. 모델 판정보다 앞선다
CAPTION_EMB = {"people": ROOT / "data" / "analysis" / "embed" / "gemma2-cap-people.npz",
               "medium": ROOT / "data" / "analysis" / "embed" / "gemma2-cap-medium.npz"}
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


OLD_CODES = {"Soviet Union": "SU", "Czechoslovakia": "CS", "Yugoslavia": "YU", "West Germany": "DE", "Taiwan": "TW"}
_CODES: dict[str, str] = {}


def country_code(name: str) -> str:
    """두 글자 국가 코드(Natural Earth 의 ISO_A2, 없으면 WB_A2). 지금은 없는 나라는 옛 ISO 코드."""
    if not _CODES and BORDERS.exists():
        for f in json.loads(BORDERS.read_text())["features"]:
            pr = f["properties"]
            code = next((pr.get(k) for k in ("ISO_A2", "ISO_A2_EH", "WB_A2") if pr.get(k) and pr.get(k) != "-99"), "")
            _CODES[pr["NAME"]] = code
        _CODES["United States"] = _CODES.get("United States of America", "US")
    return OLD_CODES.get(name) or _CODES.get(name) or name[:2].upper()


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

LAND = ROOT / "data" / "raw" / "ne_50m_land.geojson"   # Natural Earth(public domain), 지도 배경
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


# ---------- 텍스트 기울기 ----------
TEXT_TILT: dict[int, float] = {}


def text_tilt(r: dict, frame_rects: list) -> float:
    """글자 영역의 긴 변이 가로에서 벗어난 각도(0–90°)를 넓이로 가중 평균. 왼쪽 크라이테리온 띠와 틀 안 글자는 뺀다."""
    import cv2
    W, H = r["width"], r["height"]
    acc = wsum = 0.0
    for x in r["regions"]:
        P = np.array(x["poly"], np.float32)
        if len(P) < 3:
            continue
        cx, cy = P[:, 0].mean() / W, P[:, 1].mean() / H
        if cx < 0.1 or any(a <= cx <= c and b <= cy <= d for a, b, c, d in frame_rects):
            continue
        (_, _), (w, h), ang = cv2.minAreaRect(P)
        if w < h:
            ang += 90
        dev = abs(((ang + 90) % 180) - 90)          # 가로에서 벗어난 정도
        area = w * h
        acc += dev * area; wsum += area
    return float(acc / wsum) if wsum else 0.0


def deoverlap(P: np.ndarray, W: np.ndarray, H: np.ndarray, gap: float = 0.06, max_iter: int = 4000) -> np.ndarray:
    """왼쪽 위 좌표 P, 크기 W×H 인 타일들이 서로 겹치지 않을 때까지(틈 gap) 덜 움직여도 되는 방향으로 밀어낸다."""
    from scipy.spatial import cKDTree
    P = P.astype(float).copy()
    r = float(np.hypot(W.max(), H.max())) * 1.1 + gap
    for _ in range(max_iter):
        Cn = P + np.stack([W, H], 1) / 2
        pairs = cKDTree(Cn).query_pairs(r=r, output_type="ndarray")
        if not len(pairs):
            break
        a_, b_ = pairs[:, 0], pairs[:, 1]
        d = Cn[b_] - Cn[a_]
        ox = (W[a_] + W[b_]) / 2 + gap - np.abs(d[:, 0])
        oy = (H[a_] + H[b_]) / 2 + gap - np.abs(d[:, 1])
        hit = (ox > 1e-4) & (oy > 1e-4)
        if not hit.any():
            break
        a_, b_, d, ox, oy = a_[hit], b_[hit], d[hit], ox[hit], oy[hit]
        alongx = ox < oy
        mv = np.zeros_like(d)
        sx = np.where(np.abs(d[:, 0]) > 1e-9, np.sign(d[:, 0]), np.where(a_ % 2, 1, -1))
        sy = np.where(np.abs(d[:, 1]) > 1e-9, np.sign(d[:, 1]), np.where(a_ % 2, 1, -1))
        mv[alongx, 0] = sx[alongx] * ox[alongx] * 0.52
        mv[~alongx, 1] = sy[~alongx] * oy[~alongx] * 0.52
        np.add.at(P, a_, -mv); np.add.at(P, b_, mv)
    # 그래도 남은 겹침(같은 자리에서 맴도는 쌍): 뒤의 타일을 겹치지 않는 가장 가까운 자리로 옮긴다
    from scipy.spatial import cKDTree as _T
    for _ in range(200):
        Cn = P + np.stack([W, H], 1) / 2
        pairs = _T(Cn).query_pairs(r=r, output_type="ndarray")
        if not len(pairs):
            break
        d = np.abs(Cn[pairs[:, 1]] - Cn[pairs[:, 0]])
        bad = pairs[(d[:, 0] < (W[pairs[:, 0]] + W[pairs[:, 1]]) / 2 + gap - 1e-4) & (d[:, 1] < (H[pairs[:, 0]] + H[pairs[:, 1]]) / 2 + gap - 1e-4)]
        if not len(bad):
            break
        j = bad[0, 1]
        others = np.delete(np.arange(len(P)), j)
        for rad in np.arange(0.2, 40, 0.2):
            done = False
            for ang in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                q = P[j] + rad * np.array([np.cos(ang) * W[j], np.sin(ang) * H[j]])
                cq = q + np.array([W[j], H[j]]) / 2
                dd = np.abs(Cn[others] - cq)
                if not ((dd[:, 0] < (W[others] + W[j]) / 2 + gap) & (dd[:, 1] < (H[others] + H[j]) / 2 + gap)).any():
                    P[j] = q; done = True; break
            if done:
                break
    return P


def relax(ids: list[int], tg: np.ndarray) -> tuple[dict[int, list[float]], list[int]]:
    """화면 단위 목표 좌표(타일 중심) → 소수점 칸 좌표. 겹치는 타일이 하나도 없을 때까지 밀어낸다."""
    TWv, THv = 1.0, 1 / (1288 / 1600)
    n = len(ids)
    V = deoverlap(tg - np.array([TWv, THv]) / 2, np.full(n, TWv), np.full(n, THv))
    P = V / np.array([CELL_X, CELL_Y]); P -= P.min(0)
    pos = {i: [round(float(x), 3), round(float(y), 3)] for i, (x, y) in zip(ids, P)}
    return pos, [int(np.ceil(P[:, 0].max())) + 1, int(np.ceil(P[:, 1].max())) + 1]


def smooth_outline(Pc: np.ndarray, res: float = 0.25, sigma: float = 2.0, level: float = 0.3) -> tuple[str, float | None]:
    """점들(칸 좌표)의 밀도 지도에서 등고선 하나(가장 큰 것)를 뽑아 부드러운 곡선 테두리 path 로. 반환: (path, 가장 위 y)."""
    from scipy.ndimage import gaussian_filter
    from skimage import measure
    pad = 3.0
    mn = Pc.min(0) - pad; mx = Pc.max(0) + pad
    W_, H_ = int((mx[0] - mn[0]) / res) + 1, int((mx[1] - mn[1]) / res) + 1
    g = np.zeros((H_, W_))
    ix = ((Pc - mn) / res).astype(int)
    np.add.at(g, (ix[:, 1], ix[:, 0]), 1.0)
    g = gaussian_filter(g, sigma / res)
    cs = measure.find_contours(g, level * g.max())
    if not cs:
        return "", None
    cmax = max(cs, key=len)
    pts = cmax[:, ::-1] * res + mn                          # (x, y)
    pts = pts[::max(1, len(pts) // 60)]
    for _ in range(3):   # 차이킨 다듬기: 모서리를 깎아 둥글게
        q = np.roll(pts, -1, 0)
        pts = np.stack([0.75 * pts + 0.25 * q, 0.25 * pts + 0.75 * q], 1).reshape(-1, 2)
    path = "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + "Z"
    return path, float(pts[:, 1].min())


def partition_regions(pos: dict, groups: dict[str, list[int]], res: float = 0.25, sigma: float = 2.2,
                      level: float = 0.33, tol: float = 1.8) -> list[dict]:
    """클러스터 영역을 서로 겹치지 않게: 범주마다 밀도 지도를 만들고, 각 지점은 밀도가 가장 높은 범주 하나가 차지한다.
    범주의 땅(가장 큰 덩어리)을 꼭짓점이 적은 다각형(직선)으로 단순화한다. 반환 [{name, outline, all, label}]."""
    from scipy.ndimage import gaussian_filter, label as cc_label, binary_fill_holes
    groups = {k: [i for i in v if i in pos] for k, v in groups.items()}
    groups = {k: v for k, v in groups.items() if len(v) >= 8}
    if not groups:
        return []
    allp = np.array([pos[i] for v in groups.values() for i in v], float) + np.array([0.42, 0.44])
    pad = 3.0
    mn = allp.min(0) - pad; mx = allp.max(0) + pad
    W_, H_ = int((mx[0] - mn[0]) / res) + 1, int((mx[1] - mn[1]) / res) + 1
    names = list(groups)
    D = np.zeros((len(names), H_, W_))
    for k, nm in enumerate(names):
        P = np.array([pos[i] for i in groups[nm]], float) + np.array([0.42, 0.44])
        ix = ((P - mn) / res).astype(int)
        np.add.at(D[k], (ix[:, 1], ix[:, 0]), 1.0)
        D[k] = gaussian_filter(D[k], sigma / res)
    win = D.argmax(0)
    BLK = 3.0                                           # 계단 한 칸(칸 단위)
    bp = int(round(BLK / res))
    hb, wb = H_ // bp, W_ // bp
    tiles = np.array(list(pos.values()), float) + np.array([0.42, 0.44])
    tb = ((tiles - mn) / BLK).astype(int)
    out = []
    for k, nm in enumerate(names):
        m = (win == k) & (D[k] >= level * D[k].max())
        B = m[:hb * bp, :wb * bp].reshape(hb, bp, wb, bp).mean((1, 3)) >= 0.4
        lab, n = cc_label(B)
        if not n:
            continue
        B = binary_fill_holes(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1))
        # 영역 안 포스터 중 이 범주 비율(자연스럽게 모인 무리만)
        ok = (tb[:, 0] >= 0) & (tb[:, 0] < wb) & (tb[:, 1] >= 0) & (tb[:, 1] < hb)
        inside = np.zeros(len(tiles), bool); inside[ok] = B[tb[ok, 1], tb[ok, 0]]
        mem = np.array([pos[i] for i in groups[nm]], float) + np.array([0.42, 0.44])
        mb = ((mem - mn) / BLK).astype(int)
        okm = (mb[:, 0] >= 0) & (mb[:, 0] < wb) & (mb[:, 1] >= 0) & (mb[:, 1] < hb)
        n_mem = int(B[mb[okm, 1], mb[okm, 0]].sum())
        if inside.sum() == 0 or n_mem / inside.sum() < 0.45 or n_mem < 6:
            continue
        pts = rect_outline(B)
        if len(pts) < 4:
            continue
        pts = pts * BLK + mn
        path = "M" + "L".join(f"{x:.2f} {y:.2f}" for x, y in pts) + "Z"
        top = pts[:, 1].min(); cx = pts[pts[:, 1] <= top + 0.01][:, 0].mean()
        out.append({"name": nm, "outline": path, "all": groups[nm], "label": [float(cx), float(top - .4)]})
    return out


def rect_outline(B: np.ndarray) -> np.ndarray:
    """블록 격자(참 = 땅)의 바깥 경계를 수직·수평선 다각형 꼭짓점(블록 좌표 x, y)으로. 같은 방향으로 이어지는 점은 뺀다."""
    nxt = {}
    h, w = B.shape
    for i in range(h):
        for j in range(w):
            if not B[i, j]:
                continue
            if i == 0 or not B[i - 1, j]: nxt[(j, i)] = (j + 1, i)
            if j == w - 1 or not B[i, j + 1]: nxt[(j + 1, i)] = (j + 1, i + 1)
            if i == h - 1 or not B[i + 1, j]: nxt[(j + 1, i + 1)] = (j, i + 1)
            if j == 0 or not B[i, j - 1]: nxt[(j, i + 1)] = (j, i)
    if not nxt:
        return np.zeros((0, 2))
    start = min(nxt, key=lambda v: (v[1], v[0]))
    loop, v = [start], nxt[start]
    while v != start and len(loop) < 10000:
        loop.append(v); v = nxt.get(v, start)
    P = np.array(loop, float)
    keep = []
    for k in range(len(P)):
        a, b, c = P[k - 1], P[k], P[(k + 1) % len(P)]
        if not ((a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1])):
            keep.append(b)
    return np.array(keep)


def dense_region(pos: dict[int, list[int]], mem: list[int], name: str, keep: float = 0.7) -> dict | None:
    """클러스터 영역: 그 무리가 가장 빽빽한 곳을 중심으로 가까운 keep 비율의 '그 무리 커버 칸'만. 사이에 낀 다른 커버는 들지 않는다.
    라벨 자리는 영역 위쪽 가운데."""
    mem = [i for i in mem if i in pos]
    if len(mem) < 8:
        return None
    P = np.array([pos[i] for i in mem], float)
    D = np.hypot(*(P[:, None] - P[None]).transpose(2, 0, 1))
    c = P[np.argmax((D < 4).sum(1))]
    d = np.hypot(*(P - c).T)
    Q = P[d <= np.quantile(d, keep)]
    mx = Q[:, 0].mean()
    col = Q[np.abs(Q[:, 0] - mx) <= 2]                  # 가운데 열들 중 가장 위 칸 바로 위
    top = (col if len(col) else Q)[:, 1].min()
    core = [i for i, dd in zip(mem, d) if dd <= np.quantile(d, keep)]
    outline, top_ = smooth_outline(Q + np.array([0.42, 0.44]))     # 칸 중심
    return {"name": name, "outline": outline, "ids": core, "all": mem,
            "label": [float(mx + .5), float((top_ if top_ is not None else top) - .4)]}


def soft_axes(X: np.ndarray, anchor: np.ndarray, pull: float = 0.35) -> np.ndarray:
    """느슨한 축: UMAP 을 축 좌표에서 출발시켜 고차원에서 비슷한 것끼리 뭉치게 하고, 결과를 축 방향으로 돌려 세운 뒤
    축 좌표 쪽으로 pull 만큼 당긴다. 축은 대략적인 방향만 말하고, 실제 자리는 무리가 정한다."""
    import umap
    a = (anchor - anchor.mean(0)) / (anchor.std(0) + 1e-9)
    u = umap.UMAP(n_neighbors=15, min_dist=0.2, init=a * 4, n_epochs=400, random_state=0).fit_transform(X)
    u = (u - u.mean(0)) / (u.std(0) + 1e-9)
    # 직교 프로크루스테스: 회전·뒤집기만으로 u 를 축 좌표에 맞춘다
    U_, _, Vt = np.linalg.svd(u.T @ a)
    u = u @ (U_ @ Vt)
    u = (u - u.mean(0)) / (u.std(0) + 1e-9)
    return (1 - pull) * u + pull * a


# ---------- 나라: 실제 국경 모양 카토그램 ----------
BORDERS = ROOT / "data" / "raw" / "ne_50m_admin_0_countries.geojson"     # Natural Earth(public domain)
SHAPE_OF = {"United States": ["United States of America"], "Soviet Union": ["Russia"], "Czechoslovakia": ["Czechia", "Slovakia"],
            "Yugoslavia": ["Serbia", "Croatia", "Bosnia and Herz.", "Slovenia", "Montenegro", "Kosovo"]}
MAP_GROUP = {"West Germany": "Germany"}      # 같은 땅은 한 모양에 함께 담는다


def _ring_area(r: np.ndarray) -> float:
    x, y = r[:, 0], r[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def country_rings(names: list[str], feats: dict) -> list[np.ndarray]:
    """나라(들)의 외곽 고리(경위도). 본토에서 먼 해외 영토와 아주 작은 섬은 뺀다."""
    rings = []
    for nm in names:
        g = feats.get(nm)
        if not g:
            continue
        polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
        rings += [np.array(p[0], float) for p in polys]
    if not rings:
        return []
    main = max(rings, key=_ring_area)
    c0 = main.mean(0)
    big = _ring_area(main)
    keep = [r for r in rings if _ring_area(r) >= big * 0.02 and np.hypot(*(r.mean(0) - c0)) < 22]
    return keep


def country_cartogram(items: list[dict]) -> tuple:
    """나라마다 실제 국경 모양을 영화 수에 맞는 크기로 키우거나 줄여 그 안을 커버로 채우고, 실제 위치 가까이 겹치지 않게 놓는다.
    배경 = 놓인 나라 윤곽(칸 좌표 SVG path)."""
    feats = {f["properties"]["NAME"]: f["geometry"] for f in json.loads(BORDERS.read_text())["features"]}
    feats["United States of America"] = feats.get("United States of America")
    groups: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        c = it.get("country") or ""
        groups.setdefault(MAP_GROUP.get(c, c), []).append(i)

    K = 10                                             # 칸 판정용 래스터 해상도(화면 단위당 픽셀)
    shapes = []
    for c, mem in groups.items():
        rings = country_rings(SHAPE_OF.get(c, [c]), feats)
        if not rings:
            lon, lat = LONLAT.get(c, (0, -50))
            rings = [np.array([[lon + np.cos(a), lat + np.sin(a)] for a in np.linspace(0, 2 * np.pi, 24)])]
        allp = np.concatenate(rings)
        lon0, lat0 = allp.mean(0)
        cosl = np.cos(np.radians(lat0))
        loc = [np.stack([(r[:, 0] - lon0) * cosl, -(r[:, 1] - lat0)], 1) for r in rings]   # 도 단위, 가로는 위도 보정
        n = len(mem)

        def cells_at(k: float):
            pts = [r * k for r in loc]
            mn = np.min([q.min(0) for q in pts], 0); mx = np.max([q.max(0) for q in pts], 0)
            W, H = int((mx[0] - mn[0]) * K) + 2, int((mx[1] - mn[1]) * K) + 2
            im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
            for q in pts:
                d.polygon([((x - mn[0]) * K, (y - mn[1]) * K) for x, y in q], fill=255)
            a = np.asarray(im)
            # 칸 중심이 모양 안이면 그 칸을 쓴다. 칸 격자의 원점은 모양 왼쪽 위
            cw, ch = int(np.ceil((mx[0] - mn[0]) / CELL_X)) + 1, int(np.ceil((mx[1] - mn[1]) / CELL_Y)) + 1
            out = [(x, y) for y in range(ch) for x in range(cw)
                   if a[min(H - 1, int((y + .5) * CELL_Y * K)), min(W - 1, int((x + .5) * CELL_X * K))] > 127]
            return np.array(out, int).reshape(-1, 2), mn
        lo, hi = 1e-3, 50.0
        for _ in range(36):
            mid = (lo + hi) / 2
            if len(cells_at(mid)[0]) >= n:
                hi = mid
            else:
                lo = mid
        cells, mn = cells_at(hi)
        if len(cells) > n:                             # 남는 칸은 고르게 비운다
            cells = cells[np.round(np.linspace(0, len(cells) - 1, n)).astype(int)]
        # 오래된 영화가 모양 가운데
        ctr = cells.mean(0)
        order = np.argsort(np.hypot((cells[:, 0] - ctr[0]) * CELL_X, (cells[:, 1] - ctr[1]) * CELL_Y))
        mem_sorted = sorted(mem, key=lambda i: (items[i].get("year") or 0, items[i]["spine"]))
        cell_of = {i: cells[order[k]] for k, i in enumerate(mem_sorted)}
        outline = [np.stack([(q[:, 0] * hi - mn[0]) / CELL_X, (q[:, 1] * hi - mn[1]) / CELL_Y], 1) for q in loc]
        shapes.append(dict(name=c, mem=mem, cell_of=cell_of, cells=cells, outline=outline, lonlat=(lon0, lat0)))

    # 놓을 자리: 매끄러운 투영(유럽을 넓히고 대양을 줄임) 위 나라 중심. 큰 나라부터, 겹치거나 맞닿지 않는 가장 가까운 자리
    gl = np.arange(-180, 180.5, 0.5); sx = 0.42 + 0.55 * np.exp(-((gl - 15) / 38) ** 2)
    X = np.concatenate([[0], np.cumsum((sx[1:] + sx[:-1]) / 4)]); X -= np.interp(0, gl, X)
    gb = np.arange(-90, 90.5, 0.5); sy = 0.45 + 0.55 / (1 + np.exp(-(gb - 30) / 6))
    Y = np.concatenate([[0], np.cumsum((sy[1:] + sy[:-1]) / 4)]); Y -= np.interp(0, gb, Y)
    proj = lambda lon, lat: (float(np.interp(lon, gl, X)), -float(np.interp(lat, gb, Y)))
    S = 1.9                                            # 투영 단위 → 화면 단위
    G = 700
    occ = np.zeros((G, G), bool)
    pos, placed = {}, []
    for s in sorted(shapes, key=lambda s: -len(s["mem"])):
        px, py = proj(*s["lonlat"])
        cc = s["cells"]
        mid = cc.mean(0)
        tx, ty = px * S / CELL_X - mid[0] + G / 2, py * S / CELL_Y - mid[1] + G / 2
        best = None
        for r in range(0, 120):
            cand = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1) if max(abs(dx), abs(dy)) == r]
            cand.sort(key=lambda d: (d[0] * CELL_X) ** 2 + (d[1] * CELL_Y) ** 2)
            for dx, dy in cand:
                ox, oy = int(round(tx)) + dx, int(round(ty)) + dy
                xs, ys = cc[:, 0] + ox, cc[:, 1] + oy
                # 한 칸 틈: 자기 칸과 그 이웃이 비어 있어야 한다
                ok = True
                for ex, ey in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
                    if occ[ys + ey, xs + ex].any():
                        ok = False; break
                if ok:
                    best = (ox, oy); break
            if best:
                break
        ox, oy = best
        occ[cc[:, 1] + oy, cc[:, 0] + ox] = True
        for i, (x, y) in s["cell_of"].items():
            pos[i] = [int(x + ox), int(y + oy)]
        placed.append((s, ox, oy))
    xs = [v[0] for v in pos.values()]; ys = [v[1] for v in pos.values()]
    x0, y0 = min(xs) - 1, min(ys) - 1
    pos = {i: [x - x0, y - y0] for i, (x, y) in pos.items()}
    size = [max(xs) - x0 + 2, max(ys) - y0 + 2]
    path = []
    for s, ox, oy in placed:
        for q in s["outline"]:
            pts, last = [], None
            for x, y in q:
                v = (x + ox - x0, y + oy - y0)
                if last is None or abs(v[0] - last[0]) + abs(v[1] - last[1]) > 0.08:
                    pts.append(f"{v[0]:.2f} {v[1]:.2f}"); last = v
            if len(pts) >= 3:
                path.append("M" + "L".join(pts) + "Z")
    # 모든 나라: 같은 투영·배율의 실제 지도(위계가 낮게 흐리게). 작은 섬과 남극은 생략, 0.3 칸보다 가까운 점은 합친다
    world = []
    for f in json.loads(BORDERS.read_text())["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            r = np.array(poly[0], float)
            if r[:, 1].max() < -58 or _ring_area(r) < 0.8:
                continue
            pts, last = [], None
            for lon, lat in r:
                X_, Y_ = proj(lon, lat)
                v = (X_ * S / CELL_X + G / 2 - x0, Y_ * S / CELL_Y + G / 2 - y0)
                if last is None or abs(v[0] - last[0]) + abs(v[1] - last[1]) > 0.3:
                    pts.append(f"{v[0]:.1f} {v[1]:.1f}"); last = v
            if len(pts) >= 3:
                world.append("M" + "L".join(pts) + "Z")
    labels = blob_labels(pos, {s["name"]: s["mem"] for s, _, _ in placed}, min_n=12)
    return pos, size, labels, {"world": "".join(world), "films": "".join(path)}


def centroid_labels(pos: dict[int, list[int]], groups: dict[str, list[int]]) -> list:
    """범주 이름표: 그 범주 커버들이 가장 많이 모인 자리(칸 좌표의 중앙값). 큰 범주부터, 겹치면 조금씩 아래로 비킨다."""
    out = []
    for name, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        mem = [i for i in mem if i in pos]
        if not mem:
            continue
        P = np.array([pos[i] for i in mem], float)
        med = np.median(P, 0)
        # 중앙값 근처의 밀집한 곳: 중앙값에서 가까운 절반의 평균
        d = np.hypot(*(P - med).T); core = P[d <= np.median(d)]
        x, y = core.mean(0) + 0.5
        for _ in range(8):
            if any(abs(x - o[0]) < (len(name) + len(o[2])) * 0.5 / 2 + 1 and abs(y - o[1]) < 1.6 for o in out):
                y += 1.8
            else:
                break
        out.append([float(round(x, 2)), float(round(y, 2)), name, "center"])
    return out


def equal_earth(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equal Earth 도법(넓이가 정확한 표준 세계지도). 입력 도, 출력은 지구 반지름 1 기준(y 는 북쪽이 +)."""
    A1, A2, A3, A4 = 1.340264, -0.081106, 0.000893, 0.003796
    lam, phi = np.radians(lon), np.radians(lat)
    th = np.arcsin(np.sqrt(3) / 2 * np.sin(phi))
    t2 = th * th; t6 = t2 ** 3
    x = 2 * np.sqrt(3) * lam * np.cos(th) / (3 * (A1 + 3 * A2 * t2 + t6 * (7 * A3 + 9 * A4 * t2)))
    y = th * (A1 + A2 * t2 + t6 * (A3 + A4 * t2))
    return x, y


def mercator(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """가장 평범한 메르카토르 도법(웹 지도와 같은 방식). 입력 도, 출력 라디안 단위(y 는 북쪽이 +)."""
    lat = np.clip(lat, -85, 85)
    return np.radians(lon), np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def country_map(items: list[dict]) -> tuple:
    """지도는 늘리거나 당기지 않는다(메르카토르, 사용자 선택). 커버를 실제 국경 안에 채우고, 나라 넓이에 비해 영화가 많으면 그 나라 타일을 작게.
    좌표 단위: 지도 폭 1000. 반환 pos[i] = [x, y](타일 왼쪽 위), size[i] = 타일 너비."""
    feats = {f["properties"]["NAME"]: f["geometry"] for f in json.loads(BORDERS.read_text())["features"]}
    WMAP = 1000.0
    PROJ = mercator
    xmax = PROJ(np.array([180.0]), np.array([0.0]))[0][0]
    ytop = PROJ(np.array([0.0]), np.array([84.0]))[1][0]
    ybot = PROJ(np.array([0.0]), np.array([-58.0]))[1][0]
    sc = WMAP / (2 * xmax)
    to_map = lambda lon, lat: ((PROJ(lon, lat)[0] + xmax) * sc, (ytop - PROJ(lon, lat)[1]) * sc)
    HMAP = (ytop - ybot) * sc
    TA = 1 / (1288 / 1600)                       # 타일 세로/가로
    GAPX, GAPY = 1.18, 1.18 * 1.42 / 1.18 * (1288 / 1600) * TA / TA   # 칸 간격(타일 너비 대비)

    groups: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        c = it.get("country") or ""
        groups.setdefault(MAP_GROUP.get(c, c), []).append(i)
    pos, size, labels = {}, {}, []
    cty_path, cty_label = {}, {}
    for c, mem in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        rings = country_rings(SHAPE_OF.get(c, [c]), feats)
        n = len(mem)
        polys = [np.stack(to_map(r[:, 0], r[:, 1]), 1) for r in rings] if rings else []
        if polys:
            allp = np.concatenate(polys)
            mn, mx = allp.min(0), allp.max(0)
            K = 8
            W, H = int((mx[0] - mn[0]) * K) + 3, int((mx[1] - mn[1]) * K) + 3
            im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
            for q in polys:
                d.polygon([((x - mn[0]) * K + 1, (y - mn[1]) * K + 1) for x, y in q], fill=255)
            mask = np.asarray(im) > 127

            def lattice(s):   # 타일 너비 s 일 때 나라 안에 들어가는 자리(칸 중심이 국경 안)
                sx, sy = s * 1.18, s * TA + s * 0.18
                xs = np.arange(mn[0], mx[0], sx); ys = np.arange(mn[1], mx[1], sy)
                out = []
                for yy in ys:
                    for xx in xs:
                        cx, cy = xx + s / 2, yy + s * TA / 2
                        if mask[min(H - 1, int((cy - mn[1]) * K + 1)), min(W - 1, int((cx - mn[0]) * K + 1))]:
                            out.append((xx, yy))
                return np.array(out).reshape(-1, 2)
            lo, hi = 0.05, 60.0
            for _ in range(32):
                mid = (lo + hi) / 2
                if len(lattice(mid)) >= n:
                    lo = mid
                else:
                    hi = mid
            s = min(lo, 4.5)                             # 땅이 넓어 영화가 드문 나라도 타일이 너무 커지지 않게
            pts = lattice(s)
            if len(pts) > n * 1.15 or n <= 3:
                # 땅에 비해 영화가 적은 나라: 격자 앞쪽부터 고르면 구석에 몰린다. 나라 안을 촘촘한 점으로 채우고
                # 영화 수만큼 가장 고르게 퍼진 자리(k-평균 중심, 국경 안 점으로 맞춤)를 쓴다. 1 편이면 나라 한가운데
                yy, xx = np.nonzero(mask)
                S_ = np.stack([(xx - 1) / K + mn[0], (yy - 1) / K + mn[1]], 1)
                if len(S_) > 8000:
                    S_ = S_[np.random.default_rng(0).choice(len(S_), 8000, replace=False)]
                if n == 1:
                    cen = S_.mean(0, keepdims=True)
                else:
                    from sklearn.cluster import KMeans
                    cen = KMeans(n_clusters=n, n_init=4, random_state=0).fit(S_).cluster_centers_
                cen = S_[np.argmin(((cen[:, None] - S_[None]) ** 2).sum(-1), 1)]     # 국경 안 점으로
                if n > 1:   # 이웃과 겹치지 않는 가장 큰 타일
                    dx = np.abs(cen[:, None, 0] - cen[None, :, 0]); dy = np.abs(cen[:, None, 1] - cen[None, :, 1])
                    lim = np.maximum(dx / 1.18, dy / (TA + 0.18)); np.fill_diagonal(lim, np.inf)
                    s = min(4.5, float(lim.min()))
                pts = cen - np.array([s / 2, s * TA / 2])
        else:
            pts, s = np.zeros((0, 2)), 0
        SMIN = 1.1
        if len(pts) < n or s < SMIN:   # 너무 작은 나라: 최소 크기로 그 자리에 모은다(국경을 조금 넘어도)
            s = max(s, SMIN)
            lon0, lat0 = (np.concatenate(rings).mean(0) if rings else LONLAT.get(c, (0, -50)))
            cx, cy = to_map(np.array([lon0]), np.array([lat0])); cx, cy = cx[0], cy[0]
            pts = sunflower(n) * s * 1.18 / np.sqrt(CELL_X * CELL_Y) + np.array([cx - s / 2, cy - s * TA / 2])
        if len(pts) > n:
            pts = pts[np.round(np.linspace(0, len(pts) - 1, n)).astype(int)]
        ctr = pts.mean(0)
        order = np.argsort(np.hypot(*(pts - ctr).T))
        mem_sorted = sorted(mem, key=lambda i: (items[i].get("year") or 0, items[i]["spine"]))
        for k, i in enumerate(mem_sorted):
            pos[i] = [round(float(pts[order[k]][0]), 2), round(float(pts[order[k]][1]), 2)]
            size[i] = round(float(s), 3)
        # 클러스터 라벨 자리: 나라(또는 그 나라 커버 무리) 위쪽 가운데. 화면에는 호버할 때만
        tops = [p_[:, 1].min() for p_ in polys] if polys else []
        cty_label[c] = [round(float(pts[:, 0].mean() + s / 2), 2), round(float(min([pts[:, 1].min(), *tops]) - 1.5), 2)]
        cty_path[c] = "".join("M" + "L".join(f"{x:.1f} {y:.1f}" for x, y in q[::max(1, len(q) // 400)]) + "Z" for q in polys)
        if not polys:   # 국경 데이터가 없는 나라: 커버 무리를 감싸는 작은 네모
            x0_, y0_ = pts.min(0) - s * .4; x1_, y1_ = pts.max(0) + s * 1.4
            cty_path[c] = f"M{x0_:.1f} {y0_:.1f}H{x1_:.1f}V{y1_:.1f}H{x0_:.1f}Z"
    # 나라끼리 겹친 타일(국경 근처, 최소 크기로 모은 작은 나라)도 풀어 준다(크기는 나라마다 다르다)
    keys = list(pos)
    Pc = deoverlap(np.array([pos[i] for i in keys]), np.array([size[i] for i in keys]),
                   np.array([size[i] * TA for i in keys]), gap=0.12)
    pos = {i: [round(float(x), 2), round(float(y), 2)] for i, (x, y) in zip(keys, Pc)}
    # 모든 나라(흐리게)
    world = []
    for f in json.loads(BORDERS.read_text())["features"]:
        g = f["geometry"]
        for poly in ([g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]):
            r = np.array(poly[0], float)
            if r[:, 1].max() < -58 or _ring_area(r) < 0.5:
                continue
            X, Y = to_map(r[:, 0], r[:, 1])
            pts_, last = [], None
            for x, y in zip(X, Y):
                if last is None or abs(x - last[0]) + abs(y - last[1]) > 0.6:
                    pts_.append(f"{x:.1f} {y:.1f}"); last = (x, y)
            if len(pts_) >= 3:
                world.append("M" + "L".join(pts_) + "Z")
    return pos, size, {"free": True, "w": WMAP, "h": round(HMAP, 1)}, [], {"world": "".join(world), "countries": cty_path, "labels": cty_label}


def free_layouts(items: list[dict], tropes: dict, look: dict[int, np.ndarray], u_look: dict[int, tuple[float, float]], ax: dict) -> dict[str, tuple]:
    out: dict[str, tuple] = {}
    n = len(items)

    out["country"] = country_map(items)

    # 화면비: 비율마다 진짜 화면비 모양의 블록. 블록 안은 연도 순(왼쪽 위 → 오른쪽 아래), 블록은 왼쪽부터 아래를 맞춰 놓는다
    buckets: dict[str, list[int]] = {}
    ratio = {}
    for i, it in enumerate(items):
        v = aspect_value(it.get("ar"))
        if v:
            lab = next(l for lim, l in ASPECT_BUCKETS if v < lim)
            buckets.setdefault(lab, []).append(i)
            ratio.setdefault(lab, []).append(v)
        else:
            buckets.setdefault("Unknown", []).append(i)
            ratio.setdefault("Unknown", []).append(1.5)
    # 블록(액자)의 바깥 비율이 실제 화면비와 정확히 같도록 칸 간격을 정수가 아닌 값으로 조율한다.
    # 남는 자리는 마지막 줄에 몰지 않고 고르게 흩어 액자 모양을 지킨다. 액자는 배경에 가는 선으로 그린다
    TW_, TH_ = 1 / CELL_X, (CELL_Y - (CELL_X - 1)) / CELL_Y      # 타일이 칸에서 차지하는 비율(가로, 세로)
    PAD = 0.45
    pos, labels, x0, blocks = {}, [], 0.0, []
    for lab in [l for _, l in ASPECT_BUCKETS] + ["Unknown"]:
        if lab not in buckets:
            continue
        mem = sorted(buckets[lab], key=lambda i: (items[i].get("year") or 0, items[i]["spine"]))
        n_ = len(mem)
        R = float(np.median(ratio[lab]))
        a_, b_ = TW_ + 2 * PAD, TH_ + 2 * PAD
        best = None
        for rows in range(1, n_ + 1):
            for cols in (int(np.ceil(n_ / rows)), int(np.ceil(n_ / rows)) + 1):
                if cols * rows < n_ or cols * rows > n_ * 1.3 + 2:
                    continue
                sx, sy = 1.0, 1.0
                if cols > 1:
                    sx = (R * ((rows - 1) + b_) * CELL_Y / CELL_X - a_) / (cols - 1)
                if cols == 1 or sx < 1:
                    sx = 1.0
                    sy = (((cols - 1) + a_) * CELL_X / (R * CELL_Y) - b_) / max(rows - 1, 1)
                    if sy < 1:
                        continue
                score = np.log(max(sx, sy)) + 0.6 * (cols * rows - n_) / n_
                if best is None or score < best[0]:
                    best = (score, rows, cols, sx, sy)
        if best is None:
            best = (0, 1, n_, 1.0, 1.0)
        blocks.append((lab, mem, *best[1:]))
    Hmax = max((r_ - 1) * sy_ + b_ for _, _, r_, _, _, sy_ in blocks)
    frames = []
    for lab, mem, rows, cols, sx, sy in blocks:
        slots = np.round(np.linspace(0, rows * cols - 1, len(mem))).astype(int)
        hb = (rows - 1) * sy + TH_ + 2 * PAD
        top = Hmax - hb + PAD
        for k, i in zip(slots, mem):
            pos[i] = [round(x0 + PAD + (k % cols) * sx, 3), round(top + (k // cols) * sy, 3)]
        fx0, fy0 = x0, Hmax - hb
        fx1, fy1 = x0 + (cols - 1) * sx + TW_ + 2 * PAD, Hmax
        if lab != "Unknown":
            frames.append(f"M{fx0:.2f} {fy0:.2f}L{fx1:.2f} {fy0:.2f}L{fx1:.2f} {fy1:.2f}L{fx0:.2f} {fy1:.2f}Z")
        labels.append([round((fx0 + fx1) / 2, 2), round(Hmax + 0.9, 2), f"{lab}", "center"])
        x0 = fx1 + 1.6
    out["aspect"] = (pos, [int(np.ceil(x0)), int(np.ceil(Hmax + 1.5))], labels, {"world": "", "films": "".join(frames)})

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
    # 상영 시간을 모르는 커버도 자리를(축을 바꿀 때 제자리에서 사라지거나 생기지 않게): 오른쪽 끝 '?' 막대
    unk = sorted((i for i, it in enumerate(items) if not it.get("rt")), key=lambda i: (items[i].get("year") or 0, items[i]["spine"]))
    UW, ux0 = 4, xb(NB - 1) + BW + 2
    for k, i in enumerate(unk):
        pos[i] = [ux0 + k % UW, Hc - 1 - k // UW]
    if unk:
        labels.append([ux0 + UW / 2, Hc + 0.6, "?", "center"])
    out["runtime"] = (pos, [ux0 + UW if unk else xb(NB - 1) + BW, Hc + 1], labels)

    # 그림 범주: 축마다 그 축의 모델 신호(검출기 수치 + SigLIP 문장 점수 + 분포 벡터)를 모은 벡터를 UMAP 으로 펼친 연속 지도.
    # 가까이 놓인 커버는 그 축에서 실제로 비슷하다. 범주(판정 결과)는 지도 위 그 범주가 모인 자리에 이름표로만 붙는다
    import umap
    def pca(M: dict[int, np.ndarray], ids: list[int], k: int) -> np.ndarray:
        X = np.stack([M[i] if i in M else np.zeros(len(next(iter(M.values())))) for i in ids]).astype(np.float32)
        X = X - X.mean(0)
        u, s_, _ = np.linalg.svd(X, full_matrices=False)
        return u[:, :k] * s_[:k]
    ids = [i for i in range(n) if items[i]["file"] in tropes]
    F = [tropes[items[i]["file"]]["f"] for i in ids]
    Z = [tropes[items[i]["file"]].get("medium_scores", {}) for i in ids]
    col = lambda key, fn=lambda v: v: np.array([fn(float(f.get(key, 0) or 0)) for f in F])
    zc = lambda key: np.array([z.get(key, 0.0) for z in Z])
    sq, lg = np.sqrt, np.log1p
    spec = {
        "people": ([col("faces", lg), col("faces", lg), col("faces", lg),          # 사람 수에 세 배 무게
                    col("face_max", sq), col("face_sum", sq), col("big_faces"), col("hand"), zc("body_part"),
                    col("faces_touch"), col("embrace"),                                # 커플(맞닿은 두 얼굴, 포옹)
                    np.array([np.log1p(f.get("faces", 0) or 0) * (1 - min(1, 8 * (f.get("face_max", 0) or 0))) for f in F]),   # 작은 얼굴이 아주 많음
                    col("face_y_spread"), col("person_max", sq), col("persons", lg), col("subj_cx"), col("subj_area", sq),
                    zc("face_close"), zc("people"),
                    col("p_person", sq), col("p_face", sq), col("p_limbs", sq), col("p_torso", sq)], []),   # 사람 부위 분할
        "type": ([col("text_area", sq), col("title_h"), col("title_cx"), col("title_cy"), col("title_w"), col("text_share"),
                  col("figure_share")], [("text", 10)]),
        "composition": ([col("long_h", lg), col("long_v", lg), col("img_sym"), col("lr_sym"), col("mirror_iou"), col("sal_cx"), col("sal_cy"),
                         col("sal_spread"), col("diag"), col("empty_share"), col("figure_share"), col("subj_cx"), col("subj_area", sq),
                         *[zc(k) for k in ("Close crop", "Wide scene", "Silhouette", "Busy frame", "Collage", "Centered subject", "Off-center subject")]],
                        [("lines", 8), ("dino_art", 10)]),            # 글자를 지운 그림의 생김새(DINOv2)
        "medium": ([*[zc(k) for k in ("Photo", "Illustration", "Painting", "Graphic", "Typographic")], col("ink"), col("gray")],
                   [("strokes", 8), ("dino", 8), ("siglip_art", 6)]),
    }
    def caption_map(fam: str):
        """묘사 그 줄의 임베딩(EmbeddingGemma 2)을 UMAP 으로 펼친 지도. 파일이 없으면 None"""
        p = CAPTION_EMB.get(fam)
        if not p or not p.exists():
            return None
        d = np.load(p, allow_pickle=True)
        idx = {str(f): k for k, f in enumerate(d["files"])}
        if not all(items[i]["file"] in idx for i in ids):
            return None
        V = d["vectors"].astype(np.float32)[[idx[items[i]["file"]] for i in ids]]
        u = umap.UMAP(n_neighbors=20, min_dist=0.35, metric="cosine", random_state=0).fit_transform(V)
        u = (u - u.mean(0)) / (u.std(0) + 1e-9)
        span = u.max(0) - u.min(0)
        u = u / span * np.array([1.5, 1.0])
        return u * np.sqrt(len(ids) * CELL_X * CELL_Y * 1.6 / 1.5)

    for fam, (cols, vecs) in spec.items():
        if fam == "medium" and (cu := caption_map("medium")) is not None:
            pos, size = relax(ids, cu)
            groups = {c: [i for i in ids if tropes[items[i]["file"]].get(fam) == c] for c in TROPE_ORDER[fam] if c != "Other"}
            out[fam] = (pos, size, [], "", {"regions": partition_regions(pos, groups)})
            continue
        X = np.stack(cols, 1)
        for name, k in vecs:
            if ax.get(name):
                X = np.concatenate([X, pca(ax[name], ids, k)], 1)
        X = (X - X.mean(0)) / (X.std(0) + 1e-9)
        X = np.clip(X, -3, 3)
        u = umap.UMAP(n_neighbors=20, min_dist=0.35, random_state=0).fit_transform(X)
        u = (u - u.mean(0)) / (u.std(0) + 1e-9)
        # 화면 비율(가로로 조금 긴)에 맞춰 늘이고, 넓이 ≈ 커버 수 × 1.6 칸
        span = u.max(0) - u.min(0)
        u = u / span * np.array([1.5, 1.0])
        area = len(ids) * CELL_X * CELL_Y * 1.6
        u = u * np.sqrt(area / 1.5)
        pos, size = relax(ids, u)
        groups = {c: [i for i in ids if tropes[items[i]["file"]].get(fam) == c] for c in TROPE_ORDER[fam]}
        regs = partition_regions(pos, groups)
        out[fam] = (pos, size, [], "", {"regions": regs})

    cu = caption_map("people")
    if cu is not None:
        # People v2: 묘사의 People 줄 임베딩 지도. 사람 없음 → 애매함·얼굴 없음 → 한 명 → 두 명 → 여럿이 이어진다
        pos, size = relax(ids, cu)
        pcat = lambda i: tropes[items[i]["file"]].get("people")
        regions = partition_regions(pos, {c: [i for i in ids if pcat(i) == c] for c in TROPE_ORDER["people"]})
        out["people"] = (pos, size, [], "", {"regions": regions})
    Wv = np.sqrt(len(ids) * CELL_X * CELL_Y * 1.9 * 1.7); Hv = Wv / 1.7
    if cu is None:
        # (묘사가 없을 때) People 은 뜻이 분명한 두 좌표로: 가로 = 사람 수(로그), 세로 = 사람이 화면을 차지하는 비율(위로 클수록).
        # 같은 값끼리는 겉모습 지도 순으로 조금 흩어 둔다. 딱 두 사람이 적당히 큰 자리에 Couple 영역(배경 네모)을 둔다
        cnt = np.array([max(f.get("faces", 0) or 0, f.get("persons", 0) or 0) for f in F], float)
        occ = np.array([min(1.0, max(f.get("person_max", 0) or 0, 2.5 * (f.get("face_max", 0) or 0), 2 * (f.get("face_sum", 0) or 0))) for f in F])
        # 검출기가 놓친 꽉 찬 얼굴·사람: 판정이 사람 쪽이면 최소 한 명, SigLIP 클로즈업 점수만큼 크게
        cat = [tropes[items[i]["file"]].get("people") for i in ids]
        zf = np.array([z.get("face_close", 0.0) for z in Z])
        cnt = np.where((cnt == 0) & np.isin(cat, ["Big face", "Lone figure", "Figures", "Couple", "Talking heads", "Crowd"]), 1, cnt)
        occ = np.where(np.array(cat) == "Big face", np.maximum(occ, np.clip(0.35 + 0.4 * zf, 0.35, 1.0)), occ)
        # 사람이 고른 사람 수(data/labels/people.json → tropes 의 human)가 있으면 그 수를 쓴다
        HUMAN_N = {"none": 0, "animal": 0, "some": 0.6, "one": 1, "two": 2}
        hl = [tropes[items[i]["file"]].get("human") for i in ids]
        cnt = np.array([HUMAN_N[h] if h in HUMAN_N else (max(c, 5) if h == "many" else c) for h, c in zip(hl, cnt)], float)
        XMAX = np.log1p(max(cnt.max(), 12))                 # 12 명에서 자르지 않는다
        xn = np.log1p(cnt) / XMAX
        yn = np.sqrt(occ)
        jit = np.array([u_look.get(i, (0.5, 0.5)) for i in ids]); jit = (jit - jit.mean(0)) / (jit.std(0) + 1e-9)
        # 느슨한 축: 사람 수·비율은 방향만, 실제 자리는 얼굴·사람 신호 전체 + 겉모습이 비슷한 것끼리
        Xp = np.stack(spec["people"][0], 1)
        Xp = np.concatenate([Xp, 0.6 * pca(ax.get("siglip_art") or look, ids, 6)], 1)   # 겉모습은 글자를 지운 그림으로
        Xp = np.clip((Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9), -3, 3)
        P2 = soft_axes(Xp, np.stack([xn, yn], 1), pull=0.5)   # 사람 수·크기 쪽으로 반쯤 붙잡는다
        span = P2.max(0) - P2.min(0)
        Wv = np.sqrt(len(ids) * CELL_X * CELL_Y * 1.9 * 1.7); Hv = Wv / 1.7
        tg = np.stack([(P2[:, 0] - P2[:, 0].min()) / span[0] * Wv, (1 - (P2[:, 1] - P2[:, 1].min()) / span[1]) * Hv], 1)
        pos, size = relax(ids, tg)
        labels = []   # 축이 느슨해 눈금은 두지 않는다(방향만 축 이름으로)
        cp = np.array([pos[i] for i in ids if tropes[items[i]["file"]].get("people") == "Couple"], float)
        pcat = lambda i: tropes[items[i]["file"]].get("people")
        regions = partition_regions(pos, {"No people": [i for i in ids if pcat(i) == "No one"],
                                          "One big face": [i for i in ids if pcat(i) == "Big face"],
                                          "Couples": [i for i in ids if pcat(i) == "Couple"],
                                          "Crowd": [i for i in ids if pcat(i) == "Crowd"],
                                          "Some human": [i for i in ids if pcat(i) == "Some human"]})   # Couple 은 따로 두지 않는다(사용자: 억지스럽다)
        out["people"] = (pos, size, labels, "", {"axes": ["More people", "Larger in the frame"], "regions": regions})

    # Type: 느슨한 축(가로 = 제목이 아래로, 세로 = 글자가 클수록 위) + 기울기·글자 분포·겉모습으로 뭉친다
    deg = np.array([TEXT_TILT.get(i, 0.0) for i in ids])
    diag = np.clip(np.minimum(deg - 8, 82 - deg) / 20, 0, 1)        # 15–75° 쯤이 1(대각선 글자)
    vert = np.clip((deg - 70) / 15, 0, 1)                           # 세로쓰기
    big = np.sqrt(col("title_h") + 0.5 * col("text_area"))
    Xt = np.stack([*spec["type"][0], diag, diag, vert, big, big, col("title_cx"), col("title_cx")], 1)   # 대각선·크기·좌우 위치에 두 배 무게
    Xt = np.concatenate([Xt, pca(ax["text"], ids, 10), 0.5 * pca(look, ids, 6)], 1)
    Xt = np.clip((Xt - Xt.mean(0)) / (Xt.std(0) + 1e-9), -3, 3)
    tx_ = col("title_cx"); ty_ = 1 - col("title_cy")          # 격자를 포스터 한 장처럼: 제목이 놓인 자리
    P2 = soft_axes(Xt, np.stack([tx_, ty_], 1), pull=0.5)
    span = P2.max(0) - P2.min(0)
    tg = np.stack([(P2[:, 0] - P2[:, 0].min()) / span[0] * Wv, (1 - (P2[:, 1] - P2[:, 1].min()) / span[1]) * Hv], 1)
    pos, size = relax(ids, tg)
    # 클러스터 라벨은 제목이 놓인 자리로: 왼쪽·오른쪽, 가운데면 위·가운데·아래. 큰 가운데 제목·조용한 글자·글자만은 그대로
    tcat = lambda i: tropes[items[i]["file"]].get("type")
    keep_ = ("Type only", "Quiet type", "Big centered title")
    cxv = dict(zip(ids, col("title_cx"))); cyv = dict(zip(ids, col("title_cy")))
    groups = {c: [i for i in ids if tcat(i) == c] for c in keep_}
    rest = [i for i in ids if tcat(i) not in keep_]
    groups["Left title"] = [i for i in rest if cxv[i] < 0.38]
    groups["Right title"] = [i for i in rest if cxv[i] > 0.62]
    mid = [i for i in rest if 0.38 <= cxv[i] <= 0.62]
    groups["Title on top"] = [i for i in mid if cyv[i] < 0.38]
    groups["Title in the middle"] = [i for i in mid if 0.38 <= cyv[i] <= 0.62]
    groups["Title at the bottom"] = [i for i in mid if cyv[i] > 0.62]
    groups["Diagonal type"] = [i for i, d in zip(ids, diag) if d >= .8]
    groups["Vertical type"] = [i for i, v in zip(ids, vert) if v >= .8]
    tregions = partition_regions(pos, groups)
    out["type"] = (pos, size, [], "", {"axes": ["Title to the right", "Title higher"], "regions": tregions})
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
    if PEOPLE_V2.exists():
        pv2 = json.loads(PEOPLE_V2.read_text())
        for f, r in pv2.items():
            if f in tropes:
                tropes[f]["people_old"] = tropes[f].get("people")
                tropes[f]["people"] = r["cat"]
    if MEDIUM_V2.exists():
        for f, c in json.loads(MEDIUM_V2.read_text()).items():
            if f in tropes:
                tropes[f]["medium"] = c
    if LAB_LABELS.exists():
        for scheme, d in json.loads(LAB_LABELS.read_text()).get("labels", {}).items():
            fam = {"People": "people", "Medium": "medium", "Composition": "composition", "Type": "type"}.get(scheme)
            for f, c in (d.items() if fam else []):
                if f in tropes and c in TROPE_ORDER[fam]:
                    tropes[f][fam] = c
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
        it["cc"] = country_code(it["country"])
        if r:
            TEXT_TILT[i] = text_tilt(r, frame_rects)
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
    gextra: dict[str, dict] = {}
    gsize: dict[str, dict[int, float]] = {}
    # 더 많은 시각 신호(embed_more.py): 글자를 지운 그림의 SigLIP·DINOv2, 원본 DINOv2
    idx = {it["file"]: i for i, it in enumerate(items)}
    for key, fn in (("siglip_art", "siglip2-base-patch16-384-art.npz"), ("dino", "dinov2-base.npz"), ("dino_art", "dinov2-base-art.npz")):
        pth = EMBED.parent / fn
        if pth.exists():
            d_ = np.load(pth, allow_pickle=True)
            ax[key] = {idx[str(f_)]: v_ for f_, v_ in zip(d_["files"], d_["vectors"]) if str(f_) in idx}
    for k, res in free_layouts(items, tropes, ax["look"], u_look, ax).items():
        if len(res) == 5 and isinstance(res[2], dict) and res[2].get("free"):   # 연속 좌표(지도): pos, 타일 크기, 판 크기
            pos, tsize, grid, labs, bgd = res
            gpos[k], grids[k], glabels[k], gsize[k] = pos, grid, labs, tsize
            bg = [bgd]
        else:
            pos, size, labs, *bg = res
            gpos[k], grids[k], glabels[k] = pos, size, labs
        if len(bg) > 1 and bg[1]:
            gextra[k] = bg[1]
        bg = bg[:1]
        if bg and bg[0]:   # 배경 그림은 따로 둔다(그 축을 열 때만 받는다)
            (WEB / "map").mkdir(exist_ok=True)
            (WEB / "map" / f"{k}.js").write_text(f"(window.GRIDBG=window.GRIDBG||{{}})[{json.dumps(k)}]={json.dumps(bg[0])};\n")
            gbg[k] = len(bg[0])
    print("  rings " + ", ".join(f"{k} {len(r)}x~{round(sum(len(x) for x in r) / len(r))}" for k, r in rings.items()), flush=True)
    for i, it in enumerate(items):
        it["g"] = {k: gpos[k][i] for k in gpos if i in gpos[k]}
        gw = {k: gsize[k][i] for k in gsize if i in gsize[k]}
        if gw:
            it["gw"] = gw
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
            "grids": grids, "gridLabels": glabels, "gridBg": gbg, "gridExtra": gextra, "atlas": atlas, "rings": rings, "rowNames": shorten_all(names),
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
