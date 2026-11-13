"""범주 하나를 글로 정하고, 그 범주에 '가장 가까운' 커버를 모델마다 순서대로 모아 나란히 본다(정답 라벨이 없는 범주의 육안 시험).

Composition: Diagonal · Grid · Circle · Symmetric · Frame within frame
Medium:      Photography · Painting · Drawing · Graphic design · Collage · Print

모델
  Gemma plain / art / composition(또는 medium) 방향: 이미지 임베딩과 설명 문장(task: classification)의 코사인 순
  SigLIP 2: 같은 문장("this is …")의 코사인 순
  Caption: Sonnet 묘사(embed_captions.py) 전체 글 / 그 축의 줄만 / 글 + 이미지, Gemma 상세 설명과의 코사인 순
  규칙 신호(있는 범주만): Diagonal = 그림 부분 구도선의 대각선 비율, Grid = 길게 뻗은 가로·세로 선 수, Circle = OWLv2 'circle' 점수×넓이, Symmetric = 글자 지운 윤곽 지도의 좌우 거울 상관(build_tropes img_sym)

출력: out/gemma/query.html(범주마다 모델별 상위 40장)
사용: .venv-ml/bin/python scripts/query_axes.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "analysis" / "embed"
A = ROOT / "data" / "analysis"
OUT = ROOT / "out" / "gemma"
POSTERS = ROOT / "posters"
TOP = 40
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

QUERIES = {
    "composition": {
        "Diagonal": "a strong diagonal composition, the main lines and figures tilted across the frame",
        "Grid": "a grid of equal panels or many repeated images arranged in rows and columns",
        "Circle": "a large circle or round shape dominating the composition",
        "Symmetric": "a perfectly symmetrical composition, mirrored left and right",
        "Frame within frame": "a frame within a frame, seen through a window, doorway or border",
    },
    "medium": {
        "Photography": "a photograph",
        "Painting": "a painting with visible brushstrokes, oil or watercolor",
        "Drawing": "a pen, pencil or ink drawing",
        "Graphic design": "flat graphic design made of simple geometric shapes and solid colors",
        "Collage": "a collage of cut-out photographs and paper",
        "Print": "a woodcut, linocut or screen print",
    },
}
# 상세 설명: 무엇이 보이는지, 어떤 모양인지, 무엇이 아닌지까지. 같은 범주의 짧은 설명과 나란히 비교한다
DETAILED = {
    "composition": {
        "Diagonal": "A poster built on a strong diagonal. The main subject, a figure, a road, a beam of light or a block of color, "
                    "runs from one corner toward the opposite corner, so the whole image feels tilted and in motion. "
                    "Not a calm upright layout with horizontal and vertical lines.",
        "Grid": "A poster divided into a grid: several separate pictures or panels of similar size placed side by side in rows "
                "and columns, like a contact sheet, a comic strip or a wall of photo booth portraits. Clear straight borders "
                "between the panels. Not a single continuous image.",
        "Circle": "A poster whose composition is organized around one big circle or round shape: a disc, a sun, a moon, a wheel, "
                  "a round porthole, a ring or a circular frame that takes up a large part of the poster and draws the eye to its center. "
                  "Not small round details scattered around.",
        "Symmetric": "A strictly symmetrical poster: the left half is a mirror image of the right half around a vertical center line, "
                     "with the subject placed exactly in the middle, facing the viewer, balanced and formal like a coat of arms "
                     "or a temple facade. Not an off-center or asymmetric layout.",
        "Frame within frame": "A poster where the main image is seen through another frame inside the picture: a window, a doorway, "
                              "a mirror, a keyhole, a screen or a painted border, so there is an inner picture nested inside the outer one.",
    },
    "medium": {
        "Photography": "A poster made from a real photograph or a film still shot with a camera: realistic light, real skin, real places "
                       "and photographic grain. Not drawn, not painted, not an illustration.",
        "Painting": "A poster made from a painting: oil, acrylic, gouache or watercolor, with visible brushstrokes, blended paint, "
                    "painterly texture and colors mixed by hand. Not a photograph and not a flat graphic.",
        "Drawing": "A poster made from a hand drawing: pen and ink, pencil, charcoal or crayon lines on paper, with hatching, "
                   "outlines and sketchy marks. Mostly line work rather than painted color areas or photographs.",
        "Graphic design": "A modernist graphic design poster made of flat simple shapes, solid colors and clean geometry, like a "
                          "Saul Bass or Swiss style poster. No photographic detail, no brushstrokes, few colors and lots of abstraction.",
        "Collage": "A poster made as a collage: several cut-out photographs, paper scraps and images of different scale pasted "
                   "together, with visible edges, overlaps and a cut-and-paste look.",
        "Print": "A poster made as a relief or screen print: woodcut, linocut or silkscreen, with bold carved lines, limited flat "
                 "inks, rough printed texture and slight misregistration.",
    },
}


def load(f: str):
    p = EMB / f
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    V = d["vectors"].astype(np.float32)
    return [str(x) for x in d["files"]], V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def gemma_text(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("google/embeddinggemma-2", device=DEV, config_kwargs={"vision_config": None, "audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32})
    return m.encode(["task: classification | query: " + t for t in texts], normalize_embeddings=True)


def siglip_text(texts: list[str]) -> np.ndarray:
    from transformers import AutoModel, AutoProcessor
    mid = "google/siglip2-base-patch16-384"
    model = AutoModel.from_pretrained(mid).eval(); proc = AutoProcessor.from_pretrained(mid)
    with torch.no_grad():
        t = proc(text=[f"this is {x}." for x in texts], padding="max_length", max_length=64, return_tensors="pt")
        o = model.get_text_features(**t)
        e = o if isinstance(o, torch.Tensor) else o.pooler_output
    e = torch.nn.functional.normalize(e, dim=-1).numpy()
    return e


def rules(files: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    vis = {}
    for n in files:
        p = A / "visual" / f"{n[:-4]}.json"
        vis[n] = json.loads(p.read_text()) if p.exists() else None
    out["Diagonal"] = np.array([v["art"]["lines"]["diagonal_share"] if v else 0 for v in (vis[n] for n in files)])
    def grid(v):
        if not v:
            return 0
        segs = v["full"]["lines"]["segments"]
        h = sum(1 for x0, y0, x1, y1 in segs if abs(y1 - y0) < 0.01 and abs(x1 - x0) > 0.4)
        vv = sum(1 for x0, y0, x1, y1 in segs if abs(x1 - x0) < 0.01 and abs(y1 - y0) > 0.4 and min(x0, x1) > 0.1)
        return min(h, 6) + min(vv, 6) + 2 * min(h, vv)
    out["Grid"] = np.array([grid(vis[n]) for n in files], float)
    tr = json.loads((A / "tropes.json").read_text()) if (A / "tropes.json").exists() else {}
    out["Symmetric"] = np.array([tr.get(n, {}).get("f", {}).get("img_sym", 0) for n in files])   # 글자 지운 윤곽 지도의 좌우 거울 상관
    obj = json.loads((A / "objects" / "owlv2.json").read_text())
    out["Circle"] = np.array([max((o["score"] * (o["box"][2] - o["box"][0]) * (o["box"][3] - o["box"][1])
                                   for o in obj.get(n, []) if o["label"] == "circle"), default=0) for n in files])
    return out


def thumb(n: str) -> str:
    t = OUT / "thumbs" / n
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        Image.open(POSTERS / n).convert("RGB").resize((112, 140), Image.LANCZOS).save(t, quality=80)
    return f"thumbs/{html.escape(n)}"


def main() -> None:
    all_q = [(fam, c, t) for fam, qs in QUERIES.items() for c, t in qs.items()]
    keys = [(f, c) for f, c, _ in all_q]
    G = dict(zip(keys, gemma_text([t for *_, t in all_q])))
    GD = dict(zip(keys, gemma_text([DETAILED[f][c] for f, c in keys])))
    S = dict(zip(keys, siglip_text([t for *_, t in all_q])))
    emb = {k: load(f) for k, f in {"Gemma plain": "gemma2-plain.npz", "Gemma art": "gemma2-art.npz",
                                     "Gemma composition": "gemma2-composition.npz", "Gemma medium": "gemma2-medium.npz",
                                     "SigLIP 2": "siglip2-base-patch16-384.npz",
                                     "Caption": "gemma2-cap-all.npz", "Caption composition": "gemma2-cap-composition.npz",
                                     "Caption medium": "gemma2-cap-medium.npz", "Caption + image": "gemma2-cap-img.npz"}.items()}
    files = emb["Gemma plain"][0]
    R = rules(files)
    out, nav = [], []
    for fam, c, text in all_q:
        anchor = f"{fam}-{c}".replace(" ", "-").replace("/", "")
        nav.append(f'<a href="#{anchor}">{c}</a>')
        out.append(f'<h2 id="{anchor}">{c} <span>{fam}</span></h2><p>Short: "{html.escape(text)}"<br>Detailed: "{html.escape(DETAILED[fam][c])}"</p>')
        lines = []
        for k, q, label in [("Gemma plain", G[(fam, c)], "Gemma plain · short"), ("Gemma plain", GD[(fam, c)], "Gemma plain · detailed"),
                            ("Gemma art", GD[(fam, c)], "Gemma art · detailed"), (f"Gemma {fam}", GD[(fam, c)], f"Gemma {fam} · detailed"),
                            ("SigLIP 2", S[(fam, c)], "SigLIP 2 · short"),
                            ("Caption", GD[(fam, c)], "Caption (text) · detailed"),
                            (f"Caption {fam}", GD[(fam, c)], f"Caption {fam} line · detailed"),
                            ("Caption + image", GD[(fam, c)], "Caption + image · detailed")]:
            e = emb.get(k)
            if not e:
                continue
            fs, V = e
            top = np.argsort(-(V @ q))[:TOP]
            lines.append((label, [fs[i] for i in top]))
        if c in R:
            lines.append(("Rule signal", [files[i] for i in np.argsort(-R[c])[:TOP]]))
        # 모델마다 40장을 8×5 격자 한 묶음으로, 묶음은 옆으로 나란히
        blocks = "".join(f'<div class="b"><div class="m">{k}</div><div class="g">{"".join(f"<img src={chr(34)}{thumb(n)}{chr(34)}>" for n in ns)}</div></div>'
                         for k, ns in lines)
        out.append(f'<div class="bs">{blocks}</div>')
    css = ('body{margin:0;padding:24px 32px 80px;background:#000;color:#ebebeb;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}'
           'nav{position:sticky;top:0;padding:12px 0;background:#000;z-index:2;display:flex;flex-wrap:wrap;gap:16px;font-size:13px}'
           'nav a{color:#8a8a8a;text-decoration:none}nav a:hover{color:#ebebeb}'
           'h2{font-weight:400;font-size:24px;margin:36px 0 8px}p{color:#8a8a8a;font-size:12px;max-width:900px;line-height:1.5}'
           'h2 span{font-size:13px;color:#8a8a8a;margin-left:8px}'
           '.bs{display:flex;flex-wrap:wrap;gap:28px 32px}.m{font-size:12px;color:#8a8a8a;margin-bottom:6px}'
           '.g{display:grid;grid-template-columns:repeat(8,44px);gap:3px}.g img{width:44px;height:55px;object-fit:cover}')
    (OUT / "query.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Category queries</title>'
                                    f'<style>{css}</style></head><body><nav>{"".join(nav)}</nav>{"".join(out)}</body></html>', encoding="utf-8")
    print("->", OUT / "query.html", "variants:", [k for k, v in emb.items() if v])


if __name__ == "__main__":
    main()
