"""CCW Lab: 축·배치·범주를 사이트에 넣기 전에 실험하는 로컬 전용 지도(out/lab/index.html, 공개하지 않음)

무엇으로 배치하나(Arrange by): 임베딩마다 UMAP 2D. 묘사 글(전체, 한 줄씩)과 그림(Gemma, SigLIP, Gemma 글자 지운 그림, 그림+묘사),
  형태만(Structure = 대칭·기울기·중심 집중 점수로 만든 자리, structure_scores.py / Typography = 글자 상자 배치 거리 행렬, embed_structure.py).
무엇으로 색칠하나(Color by): People(people_v2) · Medium(medium_v2) · Composition·Type(사이트 판정, tropes.json).
배치: Exact = UMAP 좌표 그대로(겹침 허용) / Grid = 빈칸을 조금 둔 격자에 헝가리안 배정(겹치지 않음).
그림: canvas 에 사이트 아틀라스(web/atlas/cover_s.webp, 40×50)로 그리고, 크게 보이면 web/thumbs(112×140) → web/large(480) 로 바꾼다.
색: Colored by 의 None 이면 범주 색 테두리 없이 커버만. 다이얼(Pull toward): 색칠한 범주 쪽으로 지도를 당긴 반지도 UMAP(target_weight 0–1, 서버 /api/steer 가 계산, 출처 = 내 라벨만 / 모든 판정).
조작: 마우스 휠·핀치 = 커서 기준 확대, 끌기·두 손가락 = 이동, 0 = 전체, 커버 클릭 = 그 임베딩에서 가장 가까운 12장 강조.
탐지(Detected, 스페이스): 커버를 어둡게 하고 컴퓨터가 찾은 것을 그 위에 그린다 — 글자 영역(흰색, 틀 글자는 테두리만), 얼굴(노랑),
  물체(파랑, OWLv2 점수 .3 이상), 긴 선분(초록). 바닥의 이름을 눌러 층을 끄고 켠다. Structure 지도에서는 잰 것을 그린다 —
  그 커버의 무리에서 잰 자리를 무리 색으로(대칭축, 기운 방향, 주제 타원, 프레임 상자, 평행선 방향, 원의 중심).
편집(Edit): 커버를 끌어 옮기고, 커버 위에서 숫자 키로 범주를 고친다(0 = 모델 판정으로). Export 로 JSON 도 받을 수 있다.
UMAP 결과는 out/lab/umap_cache.npz 에 둔다(임베딩 파일이 바뀌면 다시 계산).
사용: .venv-ml/bin/python scripts/build_lab.py, 그다음 python3 scripts/lab_server.py → http://localhost:8777 (이 Mac 에서만 열린다)
편집·라벨은 서버가 data/labels/lab.json 에 저장한다(Label 화면과 같은 파일). build_web.py 가 그 라벨을 판정보다 앞세운다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
EMB = A / "embed"
OUT = ROOT / "out" / "lab"
TW, TH, GAP = 40, 50, 4
SLACK = 1.12
K_NB = 12

# (이름, 묶음, 임베딩 파일, 기본 색칠, 설명)
MAPS = [
    ("Description", "text", "gemma2-cap-all.npz", "People", "All eight lines of the description"),
    ("People", "text", "gemma2-cap-people.npz", "People", "People line of the description"),
    ("Composition", "text", "gemma2-cap-composition.npz", "Composition", "Composition line of the description"),
    ("Medium", "text", "gemma2-cap-medium.npz", "Medium", "Medium line of the description"),
    ("Type", "text", "gemma2-cap-typography.npz", "Type", "Typography line of the description"),
    ("Color", "text", "gemma2-cap-color.npz", "Medium", "Color and light line of the description"),
    ("Picture", "image", "gemma2-plain.npz", "Medium", "The cover image (EmbeddingGemma 2)"),
    ("Picture, no text", "image", "gemma2-art.npz", "Composition", "The cover with its lettering erased (EmbeddingGemma 2)"),
    ("SigLIP", "image", "siglip2-base-patch16-384.npz", "Medium", "The cover image (SigLIP 2)"),
    ("Picture + description", "image", "gemma2-cap-img.npz", "People", "Description text and image together (EmbeddingGemma 2)"),
    # 형태만: Structure = 성질 점수에서 직접 만든 자리(scripts/structure_scores.py), Typography = 글자 배치 거리(scripts/embed_structure.py)
    ("Structure", "form", "structure-map.npz", "Structure", "Symmetry, tilt and centering, each measured on its own. Only clear cases join a group"),
    ("Typography", "form", "type-layout.npz", "Type", "Where the lettering sits and how big it is (optimal transport between text boxes)"),
]
PAL = ["#ffffff", "#f5d000", "#ff3b30", "#2f8cff", "#34c759", "#bf5af2", "#ff9500", "#64d2ff", "#ff2d92", "#a2845e"]
SCHEMES = {
    "People": {"None": "#6e6e6e", "No-face": "#2f8cff", "One": "#ffffff", "Two": "#f5d000", "3+": "#ff3b30", "Ambiguous": "#bf5af2"},
    "Medium": dict(zip(["Photo", "Painting", "Drawing", "Print", "Collage", "Graphic", "Other"], PAL[:6] + ["#6e6e6e"])),
    "Composition": dict(zip(["Grid / collage", "Symmetric", "Off-center subject", "Silhouette", "Busy frame", "Close crop",
                             "Single object", "Open space", "Diagonal", "Wide scene"], PAL)),
    "Structure": dict(zip(["Symmetric", "Centered", "Negative space", "Frame in frame", "Grid", "Parallel lines", "Diagonal",
                           "Vanishing up", "Vanishing down", "Vanishing point", "Circular", "Spiral clockwise",
                           "Spiral counterclockwise", "Spiral"],
                          ["#ffffff", "#f5d000", "#ff3b30", "#2f8cff", "#34c759", "#bf5af2", "#ff9500", "#64d2ff", "#30b0c7", "#5e8fa8",
                           "#ff2d92", "#a2845e", "#d4b483", "#8e8e93"])),
    "Type": dict(zip(["Type only", "Big centered title", "Big title", "Quiet type", "Title on top", "Title in the middle",
                      "Title at the bottom"], PAL[:7])),
}
LINE = {"People": "People", "Medium": "Medium", "Composition": "Composition", "Type": "Typography", "Structure": "Composition"}


def structure_scores() -> dict:
    p = A / "structure_scores.json"
    return json.loads(p.read_text()) if p.exists() else {}


def grid(xy: np.ndarray) -> tuple[np.ndarray, int, int]:
    n = len(xy)
    cols = math.ceil(math.sqrt(n * SLACK * TH / TW))
    rows = math.ceil(n * SLACK / cols)
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    cells = np.stack([gx.ravel() / (cols - 1), gy.ravel() / (rows - 1)], 1)
    cost = ((xy[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    out = np.zeros((n, 2), int)
    out[r] = np.stack([gx.ravel()[c], gy.ravel()[c]], 1)
    return out, cols, rows


def detections(files: list[str]) -> list[dict]:
    """커버마다 컴퓨터가 찾은 것(좌표는 천분율 정수): t 글자 영역 [x0,y0,x1,y1,쓰임], f 얼굴, o 물체 [x0,y0,x1,y1,이름], l 긴 선분"""
    import embed_structure as ES
    faces = json.loads((A / "faces.json").read_text()) if (A / "faces.json").exists() else {}
    sc = structure_scores()
    op = A / "objects" / "owlv2.json"
    objs = json.loads(op.read_text()) if op.exists() else {}
    q = lambda v: [int(round(max(0.0, min(1.0, x)) * 1000)) for x in v]
    out = []
    for n in files:
        stem = Path(n).stem
        vj = A / "visual" / f"{stem}.json"
        segs = json.loads(vj.read_text()).get("art", {}).get("lines", {}).get("segments", [])[:24] if vj.exists() else []
        ob = sorted((o for o in objs.get(n, []) if o["score"] >= 0.3), key=lambda o: -o["score"])[:6]
        m = sc.get(n)
        out.append({"t": [q(r[:4]) + [r[4]] for r in ES.text_regions(stem)], "f": [q(f["box"]) for f in faces.get(n, [])],
                    "o": [q(o["box"]) + [o["label"]] for o in ob], "l": [q(s) for s in segs],
                    "s": struct_det(m) if m else None})
    return out


def struct_det(m: dict) -> dict:
    """Structure 지도의 탐지 표시용: g 무리, p 그 무리에서 잰 자리(천분율·도), sc 점수가 20 넘는 성질 [이름, 백분율]"""
    import structure_scores as SS
    pm = lambda *v: [round(x * 1000) for x in v]
    g = m.get("group")
    p = {"Symmetric": lambda: pm(m["axis"]), "Diagonal": lambda: [round(m["angle"])], "Centered": lambda: pm(m["cx"], m["cy"], m["size"]),
         "Frame in frame": lambda: pm(*m["frame_box"]), "Parallel lines": lambda: [round(m["par_angle"])],

         "Circular": lambda: pm(m["ccx"], m["ccy"])}.get(g, lambda: [])()
    sc = sorted(([SS.NAMES[k], round(m[k] * 100)] for k in SS.THR if m[k] >= 0.2), key=lambda x: -x[1])
    return {"g": g, "p": p, "sc": sc, "n": (m.get("judge") or {}).get("note") or ""}


def lab_files() -> list[str]:
    """Lab 커버 순서(사이트 data.js 순서 중 묘사가 있는 것). 지도 배열의 행 순서와 같다."""
    data = (ROOT / "web" / "data.js").read_text().split("\n")
    covers = json.loads(data[1].split("=", 1)[1].strip().rstrip(";"))
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    return [c["file"] for c in covers if c["file"] in cap]


def model_cats(files: list[str]) -> dict[str, list]:
    """범주 체계마다 모델 판정(커버 순서대로)"""
    pv2 = json.loads((A / "people_v2.json").read_text())
    mv2 = json.loads((A / "medium_v2.json").read_text())
    tropes = json.loads((A / "tropes.json").read_text())
    sc = structure_scores()                           # 한 번만 읽는다(커버마다 읽으면 다이얼 한 번에 1,344번 파싱)
    return {"People": [pv2[n]["cat"] for n in files], "Medium": [mv2.get(n, "Other") for n in files],
            "Composition": [tropes.get(n, {}).get("composition") for n in files], "Type": [tropes.get(n, {}).get("type") for n in files],
            "Structure": [sc.get(n, {}).get("group") for n in files]}


def human_cats(files: list[str], scheme: str, lab: dict) -> list:
    """사람이 고른 범주만(없으면 None): Lab 라벨, People 은 옛 사람 라벨(people_v2 의 human)도"""
    mine = lab.get(scheme, {})
    pv2 = json.loads((A / "people_v2.json").read_text()) if scheme == "People" else {}
    return [mine.get(n) or (pv2.get(n, {}).get("human") if scheme == "People" else None) for n in files]


def map_space(fname: str, files: list[str]) -> tuple[np.ndarray, str] | None:
    """지도 하나의 공간: (정규화한 벡터, "cosine") 또는 (거리 행렬, "precomputed"). 커버가 빠져 있으면 None."""
    d = np.load(EMB / fname, allow_pickle=True)
    idx = {str(n): i for i, n in enumerate(d["files"])}
    if not all(n in idx for n in files):
        return None
    o = [idx[n] for n in files]
    if "xy" in d:                                     # 자리가 이미 정해진 지도(UMAP 없이 그대로)
        return d["xy"].astype(np.float32)[o], "euclidean"
    if "dist" in d:
        return d["dist"].astype(np.float32)[np.ix_(o, o)], "precomputed"
    V = d["vectors"].astype(np.float32)[o]
    return V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9), "cosine"


def steer_layout(V: np.ndarray, base: np.ndarray, cats: list, weight: float, metric: str = "cosine") -> np.ndarray:
    """라벨 쪽으로 당긴 UMAP(반지도: 라벨 없는 커버는 -1). weight 0 = 임베딩만, 1 = 라벨만.
    기준 지도(base, 0–1)에서 출발해 방향·자리가 크게 바뀌지 않게 한다."""
    import umap
    names = sorted({c for c in cats if c})
    y = np.array([names.index(c) if c else -1 for c in cats])
    init = (base - base.mean(0)) / (base.std(0) + 1e-9) * 3
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric, random_state=0, init=init.astype(np.float32),
                   target_metric="categorical", target_weight=float(weight)).fit_transform(V, y)
    # 라벨 쪽으로 당기면 범주 무리가 멀리 떨어져 나가 본체가 작아진다: 가로·세로마다 빈 틈을 폭의 5% 까지만 남긴다(순서·이웃은 그대로)
    for k in range(2):
        v = xy[:, k]; o = np.argsort(v); d = np.diff(v[o])
        d = np.minimum(d, 0.05 * (v.max() - v.min()))
        v2 = np.empty_like(v); v2[o] = np.concatenate([[0], np.cumsum(d)]); xy[:, k] = v2
    return (xy - xy.min(0)) / (xy.max(0) - xy.min(0) + 1e-9)


def main() -> None:
    import umap
    data = (ROOT / "web" / "data.js").read_text().split("\n")
    covers = json.loads(data[1].split("=", 1)[1].strip().rstrip(";"))
    atlas = {c["file"]: k for k, c in enumerate(covers)}
    meta = {c["file"]: c for c in covers}
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    pv2 = json.loads((A / "people_v2.json").read_text())
    mv2 = json.loads((A / "medium_v2.json").read_text())
    tropes = json.loads((A / "tropes.json").read_text())
    files = [c["file"] for c in covers if c["file"] in cap]
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT / "umap_cache.npz"
    cache = dict(np.load(cache_f, allow_pickle=True)) if cache_f.exists() else {}
    maps = []
    for name, group, f, scheme, desc in MAPS:
        p = EMB / f
        if not p.exists():
            continue
        sp = map_space(f, files)
        if sp is None:
            continue
        V, metric = sp
        key = f"{f}|{p.stat().st_mtime_ns}|{len(files)}"
        if metric == "euclidean":
            xy = V
        elif key in cache:
            xy = cache[key]
        else:
            xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric=metric, random_state=0).fit_transform(V)
            cache[key] = xy
        q = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
        g, cols, rows = grid(q)
        S = -V.copy() if metric == "precomputed" else -np.linalg.norm(V[:, None] - V[None], axis=-1) if metric == "euclidean" else V @ V.T
        np.fill_diagonal(S, -9)
        nb = np.argsort(-S, axis=1)[:, :K_NB]
        maps.append({"name": name, "group": group, "scheme": scheme, "desc": desc, "cols": cols, "rows": rows,
                     "xy": np.round(q, 4).tolist(), "pos": g.tolist(), "nb": nb.tolist()})
        print(name, flush=True)
    np.savez(cache_f, **cache)
    items = []
    sc = structure_scores()
    for n in files:
        c, m, t = cap[n], meta[n], tropes.get(n, {})
        items.append({"f": n, "t": m.get("title") or n[5:-4], "y": m.get("year"), "d": m.get("directors") or "", "a": atlas[n],
                      "h": pv2[n].get("human"),
                      "k": {"People": pv2[n]["cat"], "Medium": mv2.get(n, "Other"), "Composition": t.get("composition"), "Type": t.get("type"),
                            "Structure": sc.get(n, {}).get("group")},
                      "l": {s: c[LINE[s]] for s in SCHEMES}})
    used = {it["k"]["Structure"] for it in items}
    SCHEMES["Structure"] = {k: v for k, v in SCHEMES["Structure"].items() if k in used}
    D = {"items": items, "maps": maps, "schemes": SCHEMES, "tw": TW, "th": TH, "gap": GAP, "det": detections(files),
         "atlas": {"src": "/atlas/cover_s.webp", "cols": 32, "tw": 40, "th": 50}, "thumbs": "/thumbs/", "large": "/large/"}
    (OUT / "index.html").write_text(PAGE.replace("/*DATA*/", json.dumps(D, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print("->", OUT / "index.html")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CCW Lab</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
:root{--t1:clamp(10px,9.6px + .095vw,11.5px);--t2:clamp(11px,10.6px + .095vw,12.5px);--t3:clamp(12px,11.6px + .095vw,13.5px);
  --t6:clamp(24px,20.3px + .95vw,36px);--fg:#ebebeb;--dim:#8a8a8a;--out:cubic-bezier(.22,1,.36,1)}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#000;color:var(--fg);font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;overflow:hidden;touch-action:none}
button{font:inherit}
header{position:fixed;top:0;left:0;right:0;height:56px;background:linear-gradient(#000 70%,#0000);display:flex;align-items:center;gap:14px;padding:0 32px;z-index:4}
.group{position:relative;display:flex;gap:2px;background:#111;border-radius:999px;padding:4px}
.group button{position:relative;z-index:1;border:0;background:transparent;color:var(--dim);font-size:var(--t3);height:28px;padding:0 14px;
  border-radius:999px;cursor:pointer;white-space:nowrap;transition:color .3s var(--out)}
.group button:not(.on):hover{color:var(--fg)}.group button.on{color:#000}
.group .sep{width:1px;margin:6px 4px;background:#333}
.pill{position:absolute;top:4px;bottom:4px;left:0;width:0;border-radius:999px;background:var(--fg);transition:transform .45s var(--out),width .45s var(--out);pointer-events:none}
.sp{flex:1}
.tog{border:0;background:#111;color:var(--dim);height:36px;padding:0 16px;border-radius:999px;cursor:pointer;font-size:var(--t3);transition:background .3s var(--out),color .3s var(--out)}
.tog:hover{color:var(--fg)}a.tog{display:inline-flex;align-items:center;text-decoration:none}.tog.on{background:var(--fg);color:#000}
.count{font-size:var(--t2);color:var(--dim)}
canvas{position:fixed;inset:0;width:100%;height:100%;cursor:grab;display:block}
canvas.drag{cursor:grabbing}
.foot{position:fixed;left:32px;bottom:28px;z-index:3;pointer-events:none;max-width:calc(100vw - 64px)}
.foot .k{font-size:var(--t2);color:var(--dim)}
.foot h1{font-weight:400;font-size:var(--t6);margin:4px 0 2px;letter-spacing:-.01em}
.foot .desc{font-size:var(--t2);color:var(--dim);margin-bottom:14px}
.foot .row{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;pointer-events:auto}
.foot .group button{font-size:var(--t2);height:24px;padding:0 12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:12px;pointer-events:auto}
.legend button{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:6px;font-size:var(--t2);color:var(--fg);transition:opacity .3s}
.legend button.off{opacity:.3}.legend i{width:8px;height:8px;border-radius:50%;display:inline-block}
.legend em{font-style:normal;color:var(--dim)}
#tip{position:fixed;z-index:5;max-width:340px;pointer-events:none;font-size:var(--t2);line-height:1.45;color:#cfcfcf;background:#000d;padding:10px 12px;border-radius:10px;opacity:0;transition:opacity .15s}
#tip b{display:block;color:#fff;font-weight:500;font-size:var(--t3)}#tip em{font-style:normal;color:var(--dim);display:block;margin-bottom:4px}
.steer{display:flex;align-items:center;gap:10px;margin-top:10px;pointer-events:auto;font-size:var(--t2);color:var(--dim)}
.steer input{width:180px;accent-color:#ebebeb}.steer b{color:var(--fg);font-weight:400;min-width:2.6em}
.steer .group button{font-size:var(--t2);height:24px;padding:0 12px}.steer .st{min-width:9em}
.detrow{display:none;margin-top:10px;pointer-events:auto}body.det .detrow{display:flex;align-items:center;gap:10px}
.detrow .legend{margin-top:0}.detrow .legend i{border-radius:2px}
.zoom{position:fixed;right:32px;bottom:28px;z-index:3;display:flex;gap:2px;background:#111;border-radius:999px;padding:4px}
.zoom button{border:0;background:transparent;color:var(--dim);width:28px;height:28px;border-radius:50%;cursor:pointer;font-size:16px}
.zoom button:hover{color:var(--fg)}
</style></head><body>
<header><div class="group" id="maps"></div><span class="sp"></span><span class="count" id="count"></span>
<button class="tog" id="det">Detected</button><a class="tog" href="/label">Label</a><button class="tog" id="grid">Grid</button><button class="tog" id="edit">Edit</button><button class="tog" id="export">Export</button><button class="tog" id="reset">Reset edits</button></header>
<canvas id="cv"></canvas>
<div class="foot"><div class="k">Arranged by</div><h1 id="title"></h1><div class="desc" id="desc"></div>
  <div class="row"><span class="k">Colored by</span><div class="group" id="schemes"></div></div><div class="legend" id="legend"></div>
  <div class="steer" id="steer"><span class="k">Pull toward</span><span id="stName"></span><input type="range" id="stW" min="0" max="1" step="0.1" value="0"><b id="stV">0</b>
    <div class="group" id="stSrc"></div><span class="st" id="stMsg"></span></div>
  <div class="detrow"><span class="k">Detected</span><div class="legend" id="detLegend"></div></div></div>
<div class="zoom"><button id="zin" aria-label="Zoom in">+</button><button id="zout" aria-label="Zoom out">−</button></div>
<div id="tip"></div>
<script>
const D=/*DATA*/;
const N=D.items.length,cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const CW=D.tw+D.gap,CH=D.th+D.gap;
let mapI=Math.max(0,D.maps.findIndex(m=>m.name==='People')),scheme=D.maps[mapI].scheme,exact=true,edit=false,focusCat=null,sel=null,hover=-1;
// 편집 내용(서버가 data/labels/lab.json 에 저장): labels[scheme][file]=범주, moves[map|layout][file]=[x,y]
let E={labels:{},moves:{}},saveT=0,colorOn=true;   // colorOn: 범주 색 테두리(끄면 커버만)
const save=()=>{clearTimeout(saveT);saveT=setTimeout(()=>fetch('/api/edits',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(E)}),300)};
const catOf=i=>{const it=D.items[i],o=(E.labels[scheme]||{})[it.f];return o||it.k[scheme]};
// ---------- 이미지: 아틀라스(저화질) → thumbs → large ----------
const atlas=new Image();atlas.src=D.atlas.src;atlas.onload=()=>dirty=true;
const mid={},big={};let loading=0;
function want(store,base,i,ext){const f=D.items[i].f;if(store[f]||loading>12)return;const im=new Image();store[f]=im;loading++;
  im.onload=im.onerror=()=>{loading--;im.ok=im.complete&&im.naturalWidth>0;dirty=true};im.src=base+(ext?f.replace(/\.jpg$/,ext):f)}
// ---------- 탐지: 컴퓨터가 찾은 것을 커버 위에 ----------
let detOn=false;
const LAYERS=[['t','Text','#ffffff'],['f','Faces','#f5d000'],['o','Objects','#2f8cff'],['l','Lines','#34c759']],layerOn={t:true,f:true,o:true,l:true};
const structView=()=>detOn&&D.maps[mapI].name==='Structure';
// Structure 지도: 잰 것만 — 대칭축(세로선), 기운 방향(가운데를 지나는 선), 가운데 모인 주제(타원). 뚜렷한 것만 그린다
function drawStruct(i,x,y,w,h,a){const m=D.det[i].s;if(!m||!m.g)return;const p=m.p,g=m.g,X=v=>x+v*w/1000,Y=v=>y+v*h/1000,mx=x+w/2,my=y+h/2;
  ctx.save();ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();ctx.globalAlpha=a;ctx.lineWidth=Math.max(1.5,Math.min(3,h/50));ctx.strokeStyle=ctx.fillStyle=D.schemes.Structure[g]||'#fff';ctx.beginPath();
  const ray=(t,L,ox=0,oy=0)=>{ctx.moveTo(mx+ox-Math.cos(t)*L,my+oy+Math.sin(t)*L);ctx.lineTo(mx+ox+Math.cos(t)*L,my+oy-Math.sin(t)*L)};
  if(g==='Symmetric'){ctx.moveTo(X(p[0]),y+h*.04);ctx.lineTo(X(p[0]),y+h*.96)}
  else if(g==='Diagonal')ray(p[0]*Math.PI/180,Math.min(w,h)*.46);
  else if(g==='Parallel lines'){const t=p[0]*Math.PI/180;for(const k of[-1,0,1])ray(t,Math.min(w,h)*.42,Math.sin(t)*k*w*.2,Math.cos(t)*k*w*.2)}
  else if(g==='Centered'){const r=Math.sqrt(p[2]/1000/Math.PI);ctx.ellipse(X(p[0]),Y(p[1]),Math.max(3,r*w*1.25),Math.max(3,r*h),0,0,7)}
  else if(g==='Frame in frame')ctx.rect(X(p[0]),Y(p[1]),(p[2]-p[0])*w/1000,(p[3]-p[1])*h/1000);
  else if(g==='Circular'){ctx.arc(X(p[0]),Y(p[1]),w*.3,0,7);ctx.moveTo(X(p[0])+w*.12,Y(p[1]));ctx.arc(X(p[0]),Y(p[1]),w*.12,0,7)}
  ctx.stroke();ctx.restore();ctx.globalAlpha=a}
function drawDet(i,x,y,w,h,a){const d=D.det[i],X=v=>x+v*w/1000,Y=v=>y+v*h/1000,lw=Math.max(1,Math.min(2,h/90));
  if(layerOn.l&&h>=40){ctx.globalAlpha=a*.9;ctx.strokeStyle='#34c759';ctx.lineWidth=lw;ctx.beginPath();for(const l of d.l){ctx.moveTo(X(l[0]),Y(l[1]));ctx.lineTo(X(l[2]),Y(l[3]))}ctx.stroke()}
  if(layerOn.t){ctx.fillStyle=ctx.strokeStyle='#fff';ctx.lineWidth=lw;for(const t of d.t){const bx=X(t[0]),by=Y(t[1]),bw=(t[2]-t[0])*w/1000,bh=(t[3]-t[1])*h/1000;
    if(t[4]){ctx.globalAlpha=a*.85;ctx.fillRect(bx,by,bw,bh)}else if(h>=40){ctx.globalAlpha=a*.6;ctx.strokeRect(bx,by,bw,bh)}}}
  if(layerOn.f){ctx.globalAlpha=a*.8;ctx.fillStyle='#f5d000';for(const f of d.f)ctx.fillRect(X(f[0]),Y(f[1]),(f[2]-f[0])*w/1000,(f[3]-f[1])*h/1000)}
  if(layerOn.o&&h>=40){ctx.globalAlpha=a;ctx.strokeStyle='#2f8cff';ctx.lineWidth=lw;for(const o of d.o){ctx.strokeRect(X(o[0]),Y(o[1]),(o[2]-o[0])*w/1000,(o[3]-o[1])*h/1000);
    if(h>=220){ctx.fillStyle='#2f8cff';ctx.font='11px "Helvetica Neue",Helvetica,Arial,sans-serif';ctx.fillText(o[4],X(o[0])+3,Y(o[1])+12)}}}
  ctx.globalAlpha=a}
function detLegend(){const el=document.getElementById('detLegend');el.innerHTML='';
  document.querySelector('.detrow').style.visibility=structView()?'hidden':'';if(structView())return;
  for(const[k,n,c]of LAYERS){const b=document.createElement('button');b.className=layerOn[k]?'':'off';b.innerHTML=`<i style="background:${c}"></i>${n}`;
    b.onclick=()=>{layerOn[k]=!layerOn[k];detLegend();dirty=true};el.appendChild(b)}}
function setDet(v){detOn=v;document.body.classList.toggle('det',v);document.getElementById('det').classList.toggle('on',v);detLegend();dirty=true}
// ---------- 자리: 지금 자리(cx,cy)가 목표(gx,gy)로 부드럽게 ----------
const cx=new Float32Array(N),cy=new Float32Array(N),gx=new Float32Array(N),gy=new Float32Array(N);
let stW=0,stSrc='human',stReq=0;const steered={};
const stKey=()=>`${D.maps[mapI].name}|${scheme}|${stSrc}|${stW}`;
const curMap=()=>{const m=D.maps[mapI],o=stW>0&&steered[stKey()];return o?{...m,xy:o.xy,pos:o.pos,cols:o.cols,rows:o.rows}:m};
const moveKey=()=>D.maps[mapI].name+(exact?'|exact':'|grid')+(stW>0?`|${scheme}:${stSrc}:${stW}`:'');
function targets(){const m=curMap(),W=(m.cols-1)*CW,H=(m.rows-1)*CH,mv=E.moves[moveKey()]||{};
  for(let i=0;i<N;i++){const o=mv[D.items[i].f];if(o){gx[i]=o[0];gy[i]=o[1];continue}
    if(exact){gx[i]=m.xy[i][0]*W;gy[i]=m.xy[i][1]*H}else{gx[i]=m.pos[i][0]*CW;gy[i]=m.pos[i][1]*CH}}}
// ---------- 화면 변환 ----------
let s=1,tx=0,ty=0,S=1,TX=0,TY=0,vx=0,vy=0,dirty=true,dpr=1;
function worldSize(){const m=curMap();return[m.cols*CW,m.rows*CH]}
const fitS=()=>{const[w,h]=worldSize();return Math.min(innerWidth/(w+120),(innerHeight-260)/(h+40))};
function fit(now){const[w,h]=worldSize();S=fitS();TX=(innerWidth-w*S)/2;TY=70+(innerHeight-260-h*S)/2;if(now){s=S;tx=TX;ty=TY}}
function zoomAt(f,x,y){const ns=Math.max(fitS()*.7,Math.min(14,S*f));TX=x-(x-TX)*ns/S;TY=y-(y-TY)*ns/S;S=ns}
function resize(){dpr=devicePixelRatio||1;cv.width=innerWidth*dpr;cv.height=innerHeight*dpr;dirty=true}
// ---------- 그리기 ----------
function draw(){ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,innerWidth,innerHeight);
  const w=D.tw*s,h=D.th*s,col=D.schemes[scheme],nbs=sel!==null?new Set(D.maps[mapI].nb[sel]):null;
  const useMid=h>=56,useBig=h>=150,settled=Math.abs(S-s)<.01*S,sv=structView();
  const order=[];for(let i=0;i<N;i++)if(i!==hover)order.push(i);if(hover>=0)order.push(hover);
  for(const i of order){const x=cx[i]*s+tx,y=cy[i]*s+ty;if(x>innerWidth||y>innerHeight||x+w<0||y+h<0)continue;
    const c=catOf(i),f=D.items[i].f;
    let a=1;if(focusCat&&c!==focusCat)a=.1;if(nbs&&i!==sel&&!nbs.has(i))a=.1;ctx.globalAlpha=a;
    let src=null;if(useBig){if(settled)want(big,D.large,i);if(big[f]&&big[f].ok)src=big[f]}
    if(!src&&useMid){if(settled)want(mid,D.thumbs,i);if(mid[f]&&mid[f].ok)src=mid[f]}
    {if(detOn){ctx.fillStyle='#000';ctx.fillRect(x,y,w,h);ctx.globalAlpha=a*.28}
      if(src)ctx.drawImage(src,x,y,w,h);
      else if(atlas.complete){const k=D.items[i].a,ac=D.atlas.cols;ctx.drawImage(atlas,(k%ac)*D.atlas.tw,Math.floor(k/ac)*D.atlas.th,D.atlas.tw,D.atlas.th,x,y,w,h)}
      if(sv)drawStruct(i,x,y,w,h,a);else if(detOn)drawDet(i,x,y,w,h,a)}
    if(colorOn&&col[c]){ctx.strokeStyle=col[c];ctx.lineWidth=Math.max(1,Math.min(2.5,h/22));ctx.strokeRect(x+.5,y+.5,w-1,h-1)}
    if(i===sel||(nbs&&nbs.has(i))){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(x-2,y-2,w+4,h+4)}
    const edited=(E.labels[scheme]||{})[f]||(E.moves[moveKey()]||{})[f];
    if(edited&&a>.5){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x+w-3,y+3,Math.max(2,Math.min(4,h/14)),0,7);ctx.fill()}}
  ctx.globalAlpha=1}
function tick(){let moving=false;
  if(!drag){TX+=vx;TY+=vy;vx*=.92;vy*=.92;if(Math.abs(vx)<.05)vx=0;if(Math.abs(vy)<.05)vy=0;if(vx||vy)moving=true}
  const k=.22;if(Math.abs(S-s)>1e-4||Math.abs(TX-tx)>.05||Math.abs(TY-ty)>.05){s+=(S-s)*k;tx+=(TX-tx)*k;ty+=(TY-ty)*k;moving=true}
  for(let i=0;i<N;i++){const dx=gx[i]-cx[i],dy=gy[i]-cy[i];if(dx*dx+dy*dy>.01){cx[i]+=dx*.14;cy[i]+=dy*.14;moving=true}}
  if(moving||dirty){draw();dirty=false}requestAnimationFrame(tick)}
// ---------- 맞히기: 위에 그려진 것부터 ----------
function hit(px,py){const w=D.tw*s,h=D.th*s;if(hover>=0){const x=cx[hover]*s+tx,y=cy[hover]*s+ty;if(px>=x&&px<=x+w&&py>=y&&py<=y+h)return hover}
  for(let i=N-1;i>=0;i--){const x=cx[i]*s+tx,y=cy[i]*s+ty;if(px>=x&&px<=x+w&&py>=y&&py<=y+h)return i}return -1}
function detTip(i){if(structView()){const m=D.det[i].s;return m&&m.sc.length?`<em style="margin:6px 0 0">${m.sc.map(v=>v[0]+' '+v[1]).join(' · ')}${m.n?'<br>Seen: '+m.n:''}</em>`:''}
  const d=D.det[i],used=d.t.filter(t=>t[4]).length,names=[...new Set(d.o.map(o=>o[4]))];
  return `<em style="margin:6px 0 0">Text ${used}${d.t.length>used?' + '+(d.t.length-used)+' frame':''} · Faces ${d.f.length} · Lines ${d.l.length}${names.length?'<br>'+names.join(', '):''}</em>`}
function showTip(i,e){if(i<0){tip.style.opacity=0;return}const it=D.items[i],c=catOf(i),o=(E.labels[scheme]||{})[it.f];
  tip.innerHTML=`<b>${it.t}</b><em>${it.y||''}${it.d?' · '+it.d:''}<br>${c||'—'}${o?' (edited, model: '+it.k[scheme]+')':''}${scheme==='People'&&it.h&&it.h!==c?' · my label '+it.h:''}</em>${it.l[scheme]}${detOn?detTip(i):''}`;
  tip.style.opacity=1;const r=tip.getBoundingClientRect();tip.style.left=Math.min(e.clientX+16,innerWidth-r.width-12)+'px';tip.style.top=Math.min(e.clientY+16,innerHeight-r.height-12)+'px'}
// ---------- 입력 ----------
const isMouseWheel=e=>e.deltaMode!==0||(e.deltaX===0&&Number.isInteger(e.deltaY)&&Math.abs(e.deltaY)>=50);
cv.addEventListener('wheel',e=>{e.preventDefault();
  if(e.ctrlKey||e.metaKey)zoomAt(Math.exp(-Math.max(-40,Math.min(40,e.deltaY))*.01),e.clientX,e.clientY);
  else if(isMouseWheel(e)){const d=e.deltaMode===1?e.deltaY*33:e.deltaY;zoomAt(Math.exp(-Math.sign(d)*Math.min(Math.abs(d),240)*.0035),e.clientX,e.clientY)}
  else{TX-=e.deltaX;TY-=e.deltaY}},{passive:false});
let drag=null;
cv.addEventListener('pointerdown',e=>{const i=hit(e.clientX,e.clientY);vx=vy=0;
  drag={x:e.clientX,y:e.clientY,lx:e.clientX,ly:e.clientY,t:performance.now(),moved:false,item:edit&&i>=0?i:-1,down:i};
  cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{
  if(drag){const dx=e.clientX-drag.lx,dy=e.clientY-drag.ly;if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)drag.moved=true;
    if(drag.moved){cv.classList.add('drag');tip.style.opacity=0;
      if(drag.item>=0){const i=drag.item;gx[i]+=dx/s;gy[i]+=dy/s;cx[i]=gx[i];cy[i]=gy[i];hover=i;dirty=true}
      else{TX+=dx;TY+=dy;tx+=dx;ty+=dy;vx=dx;vy=dy;dirty=true}}
    drag.lx=e.clientX;drag.ly=e.clientY;drag.t=performance.now();return}
  const i=hit(e.clientX,e.clientY);if(i!==hover){hover=i;dirty=true}showTip(i,e)});
cv.addEventListener('pointerup',e=>{cv.classList.remove('drag');
  if(drag){if(drag.item>=0&&drag.moved){const i=drag.item,k=moveKey();(E.moves[k]=E.moves[k]||{})[D.items[i].f]=[gx[i],gy[i]];save();vx=vy=0;updateCount()}
    else if(!drag.moved){sel=drag.down>=0&&drag.down!==sel?drag.down:null;dirty=true}
    else if(performance.now()-drag.t>60)vx=vy=0;
    if(drag.item>=0)vx=vy=0}
  drag=null});
cv.addEventListener('pointerleave',()=>{hover=-1;tip.style.opacity=0;dirty=true});
let g0=1;addEventListener('gesturestart',e=>{e.preventDefault();g0=1});addEventListener('gesturechange',e=>{e.preventDefault();zoomAt(e.scale/g0,e.clientX,e.clientY);g0=e.scale});
addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey)return;
  if(e.key==='0'&&!edit){fit();return}if(e.key==='Escape'){sel=null;focusCat=null;legend();fit();return}
  if(e.key==='='||e.key==='+')zoomAt(1.6,innerWidth/2,innerHeight/2);if(e.key==='-')zoomAt(1/1.6,innerWidth/2,innerHeight/2);
  // 편집: 커버 위에서 숫자 키 = 그 번호 범주로 고침, 0 = 모델 판정으로 되돌림
  if(edit&&colorOn&&hover>=0&&/^[0-9]$/.test(e.key)){const cats=Object.keys(D.schemes[scheme]),f=D.items[hover].f,L=(E.labels[scheme]=E.labels[scheme]||{});
    const n=+e.key;if(n===0)delete L[f];else if(cats[n-1])L[f]=cats[n-1]===D.items[hover].k[scheme]?undefined:cats[n-1];if(L[f]===undefined)delete L[f];
    save();legend();updateCount();dirty=true;showTip(hover,{clientX:parseFloat(tip.style.left)-16,clientY:parseFloat(tip.style.top)-16})}});
addEventListener('resize',()=>{resize();fit()});
document.getElementById('zin').onclick=()=>zoomAt(1.8,innerWidth/2,innerHeight/2);
document.getElementById('zout').onclick=()=>zoomAt(1/1.8,innerWidth/2,innerHeight/2);
// ---------- 머리줄·바닥 ----------
function segment(el,labels,onIdx,cur,seps=[]){el.innerHTML='<span class="pill"></span>';labels.forEach((l,k)=>{if(seps.includes(k)){const s=document.createElement('span');s.className='sep';el.appendChild(s)}
  const b=document.createElement('button');b.textContent=l;b.onclick=()=>{onIdx(k);mark(el,k)};el.appendChild(b)});requestAnimationFrame(()=>mark(el,cur))}
function mark(el,k){const bs=[...el.querySelectorAll('button')];bs.forEach((b,j)=>b.classList.toggle('on',j===k));const b=bs[k],p=el.querySelector('.pill');
  if(b){p.style.width=b.offsetWidth+'px';p.style.transform=`translateX(${b.offsetLeft}px)`}}
const gstart=D.maps.map((m,k)=>k&&m.group!==D.maps[k-1].group?k:0).filter(Boolean);
segment(document.getElementById('maps'),D.maps.map(m=>m.name),k=>{mapI=k;sel=null;if(detOn)detLegend();if(colorOn)setScheme(D.maps[k].scheme);if(stW>0)steer();targets();updateFoot();updateCount();fit()},mapI,gstart);
const SCH=[...Object.keys(D.schemes),'None'];
// Colored by 의 None = 색 테두리 없이 커버만(범례·다이얼·범주 고치기 숨김). 다시 고르면 그 범주로
function setScheme(n){colorOn=n!=='None';document.getElementById('legend').style.display=document.getElementById('steer').style.display=colorOn?'':'none';
  mark(document.getElementById('schemes'),SCH.indexOf(n));if(!colorOn){focusCat=null;dirty=true;return}
  scheme=n;focusCat=null;legend();dirty=true;
  document.getElementById('stName').textContent=n;if(stW>0)steer()}
// 다이얼: 색칠한 범주 쪽으로 지도를 당긴다(서버가 반지도 UMAP 을 계산, 한 번 계산한 것은 저장)
async function steer(){const key=stKey(),msg=document.getElementById('stMsg');
  if(stW===0||steered[key]){msg.textContent=stW>0?steered[key].note:'';targets();return}
  const id=++stReq;msg.textContent='computing…';
  try{const r=await(await fetch(`/api/steer?map=${encodeURIComponent(D.maps[mapI].name)}&scheme=${scheme}&src=${stSrc}&w=${stW}`)).json();
    if(r.error){msg.textContent=r.error;return}
    steered[key]={...r,note:`${r.labeled} labeled covers`};if(id===stReq&&key===stKey()){msg.textContent=steered[key].note;targets();fit()}}
  catch(e){msg.textContent='failed'}}
const stIn=document.getElementById('stW');
stIn.oninput=()=>document.getElementById('stV').textContent=(+stIn.value).toFixed(1);
stIn.onchange=()=>{stW=+(+stIn.value).toFixed(1);steer()};
segment(document.getElementById('stSrc'),['My labels','All judgments'],k=>{stSrc=k?'model':'human';steer()},0);
segment(document.getElementById('schemes'),SCH,k=>setScheme(SCH[k]),SCH.indexOf(scheme));
function legend(){const el=document.getElementById('legend'),cnt={};for(let i=0;i<N;i++){const c=catOf(i);cnt[c]=(cnt[c]||0)+1}el.innerHTML='';
  Object.entries(D.schemes[scheme]).forEach(([c,col],k)=>{const b=document.createElement('button');b.innerHTML=`<i style="background:${col}"></i>${edit?(k+1)+' ':''}${c} <em>${cnt[c]||0}</em>`;
    b.classList.toggle('off',!!focusCat&&focusCat!==c);b.onclick=()=>{focusCat=focusCat===c?null:c;legend();dirty=true};el.appendChild(b)})}
function updateFoot(){const m=D.maps[mapI];document.getElementById('title').textContent=m.name;document.getElementById('desc').textContent=m.desc+(exact?'':' · grid')}
function updateCount(){const nl=Object.values(E.labels).reduce((a,o)=>a+Object.keys(o).length,0),nm=Object.values(E.moves).reduce((a,o)=>a+Object.keys(o).length,0);
  document.getElementById('count').textContent=`${N} covers`+(nl||nm?` · ${nl} relabeled · ${nm} moved`:'')}
document.getElementById('det').onclick=()=>setDet(!detOn);
addEventListener('keydown',e=>{if(e.code==='Space'&&!e.metaKey&&!e.ctrlKey&&e.target.tagName!=='INPUT'){e.preventDefault();setDet(!detOn)}});
document.getElementById('grid').onclick=e=>{exact=!exact;e.target.classList.toggle('on',!exact);targets();updateFoot()};
document.getElementById('edit').onclick=e=>{edit=!edit;e.target.classList.toggle('on',edit);legend()};
document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify({exported:new Date().toISOString(),...E},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='ccw-lab-edits.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('reset').onclick=()=>{delete E.moves[moveKey()];if(E.labels[scheme])delete E.labels[scheme];save();targets();legend();updateCount();dirty=true};
function start(){resize();targets();for(let i=0;i<N;i++){cx[i]=gx[i];cy[i]=gy[i]}updateFoot();updateCount();setScheme(scheme);fit(true);tick()}
fetch('/api/edits').then(r=>r.json()).then(d=>{E={labels:d.labels||{},moves:d.moves||{}};start()}).catch(start);
</script></body></html>"""


if __name__ == "__main__":
    main()
