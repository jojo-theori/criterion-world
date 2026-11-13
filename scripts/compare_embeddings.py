"""임베딩끼리 묶음(클러스터)과 이웃을 나란히 비교하는 확인 페이지를 만든다: out/gemma/index.html

1. 이웃 비교: 같은 커버 16장에 대해 모델마다 가장 가까운 6장
2. 클러스터: 모델마다 KMeans(k=24). 묶음마다 중심에 가까운 20장. EmbeddingGemma 2 는 글과 같은 공간이라
   이름 후보 문장(NAMES)과의 유사도로 묶음 이름을 붙인다
3. 범주 제로샷(EmbeddingGemma 2): build_tropes 의 묶음(People · Type · Composition · Medium)을 글 설명으로 바꿔
   커버마다 가장 가까운 설명 하나를 고른다(사람이 쓰는 범주를 글로 직접 정의하는 방식의 시험)

사용: .venv-ml/bin/python scripts/compare_embeddings.py
"""

from __future__ import annotations

import html
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "analysis" / "embed"
POSTERS = ROOT / "posters"
OUT = ROOT / "out" / "gemma"
MODELS = {
    "SigLIP 2": "siglip2-base-patch16-384.npz",
    "Gemma plain": "gemma2-plain.npz",
    "Gemma art (text erased)": "gemma2-art.npz",
    "Gemma composition": "gemma2-composition.npz",
    "Gemma people": "gemma2-people.npz",
    "Gemma type": "gemma2-type.npz",
    "Gemma medium": "gemma2-medium.npz",
}
SAMPLE_SPINES = [2, 26, 62, 84, 140, 217, 300, 368, 465, 541, 711, 782, 822, 1025, 1162, 1306]
NAMES = [
    "a close-up portrait of a face", "two faces together", "a couple embracing", "a crowd of people", "a lone figure in a vast landscape",
    "a silhouette", "a black and white photograph", "a film still", "a hand-drawn illustration", "a painting", "a watercolor",
    "a woodcut or linocut print", "a collage", "a minimal graphic design with flat shapes", "a poster made of large typography",
    "small text on a plain background", "a symmetrical composition", "a diagonal composition", "a single object in the center",
    "lots of empty space", "a grid of images", "red and black", "pastel colors", "a dark night scene", "a bright colorful pop art poster",
    "a landscape", "a city street", "the sea", "animals", "a gun or weapon", "a car", "flowers", "a cartoon", "an abstract pattern",
    "Japanese art style", "a vintage 1950s movie poster style", "a monster or horror imagery", "a dancer", "children",
]
TROPES = {
    "people": {
        "Crowd": "a crowd of many people", "Couple": "a romantic couple, two people close together",
        "Talking heads": "several faces side by side", "Big face": "one large face filling the poster",
        "Lone figure": "a small lone figure in a large space", "Figures": "people shown full body",
        "No one": "no people at all, only objects, shapes or landscape",
    },
    "type": {
        "Type only": "a poster made only of typography", "Big centered title": "a huge title in the middle",
        "Quiet type": "very small discreet text", "Title on top": "the title at the top",
        "Title at the bottom": "the title at the bottom", "Handwritten": "hand-lettered or handwritten title",
    },
    "composition": {
        "Symmetric": "a perfectly symmetrical composition", "Single object": "a single object centered on a plain background",
        "Open space": "lots of empty negative space", "Diagonal": "strong diagonal lines and dynamic angles",
        "Grid / collage": "a grid or collage of several images", "Full-bleed image": "a photograph filling the whole frame",
        "Frame within frame": "a frame within a frame, a window or doorway",
    },
    "medium": {
        "Photo": "a photograph", "Film still": "a film still from the movie", "Illustration": "a pen and ink illustration",
        "Painting": "a painting", "Graphic": "flat graphic design with simple shapes", "Typographic": "typographic design",
        "Collage": "a cut-paper collage", "Print": "a woodcut or screen print",
    },
}


def load(f: str) -> tuple[list[str], np.ndarray] | None:
    p = EMB / f
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    V = d["vectors"].astype(np.float32)
    return [str(x) for x in d["files"]], V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)


def thumb(n: str) -> str:
    t = OUT / "thumbs" / n
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        Image.open(POSTERS / n).convert("RGB").resize((112, 140), Image.LANCZOS).save(t, quality=80)
    return f"thumbs/{html.escape(n)}"


def text_vectors(texts: list[str], prefix: str = "task: clustering | query: ") -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer
    global _M
    if "_M" not in globals():
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        _M = SentenceTransformer("google/embeddinggemma-2", device=dev,
                                 config_kwargs={"vision_config": None, "audio_config": None},
                                 model_kwargs={"torch_dtype": torch.float32})
    return _M.encode([prefix + t for t in texts], normalize_embeddings=True)


def main() -> None:
    db = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    title = {s: t for s, t in db.execute("select spine, title from films")}
    img = lambda n: f'<img loading="lazy" src="{thumb(n)}" title="#{int(n[:4])} {html.escape(title.get(int(n[:4]), n))}">'
    loaded = {k: v for k, v in ((k, load(f)) for k, f in MODELS.items()) if v}
    names_vec = text_vectors(NAMES)
    out = []

    # 1. 이웃
    out.append('<h2 id="nb">Nearest neighbors</h2><p>Left: the cover. Each row: 6 nearest in that model.</p>')
    for sp in SAMPLE_SPINES:
        out.append(f"<h3>#{sp} {html.escape(title.get(sp, ''))}</h3>")
        for k, (files, V) in loaded.items():
            idx = {n: i for i, n in enumerate(files)}
            q = next((n for n in files if int(n[:4]) == sp), None)
            if q is None:
                continue
            s = V @ V[idx[q]]
            nb = [files[j] for j in np.argsort(-s)[1:7]]
            out.append(f'<div class="r"><span class="m">{k}</span>{img(q)}<span class="sep"></span>{"".join(img(n) for n in nb)}</div>')

    # 2. 클러스터
    for k, (files, V) in loaded.items():
        km = KMeans(24, n_init=4, random_state=0).fit(V)
        out.append(f'<h2 id="c-{k.split()[0]}-{k.split()[-1]}">Clusters: {k}</h2>')
        order = np.argsort(-np.bincount(km.labels_))
        for c in order:
            mem = np.where(km.labels_ == c)[0]
            cen = km.cluster_centers_[c] / np.linalg.norm(km.cluster_centers_[c])
            top = mem[np.argsort(-(V[mem] @ cen))][:20]
            label = ""
            if k.startswith("Gemma"):
                sc = names_vec @ cen
                label = " · ".join(NAMES[j] for j in np.argsort(-sc)[:2])
            out.append(f'<h3>{len(mem)} covers <span>{html.escape(label)}</span></h3><div class="g">{"".join(img(files[j]) for j in top)}</div>')

    # 3. 범주 제로샷(Gemma plain 과 art)
    for k in ("Gemma plain", "Gemma art (text erased)"):
        if k not in loaded:
            continue
        files, V = loaded[k]
        out.append(f'<h2 id="z-{k.split()[1]}">Tropes by text description: {k}</h2>')
        for fam, cats in TROPES.items():
            T = text_vectors(list(cats.values()))
            S = V @ T.T
            S = (S - S.mean(0)) / (S.std(0) + 1e-9)          # 문장마다 기본 점수가 달라 커버 전체에 대해 표준화
            lab = S.argmax(1)
            out.append(f"<h3 class='fam'>{fam.capitalize()}</h3>")
            for j, c in enumerate(cats):
                mem = np.where(lab == j)[0]
                top = mem[np.argsort(-S[mem, j])][:24]
                out.append(f'<h3>{c} <span>{len(mem)} · "{html.escape(cats[c])}"</span></h3><div class="g">{"".join(img(files[i]) for i in top)}</div>')

    nav = '<a href="#nb">Neighbors</a>' + "".join(f'<a href="#c-{k.split()[0]}-{k.split()[-1]}">{k}</a>' for k in loaded) + \
          '<a href="#z-plain">Tropes (plain)</a><a href="#z-art">Tropes (art)</a>'
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Embedding comparison</title>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><style>'
            'body{margin:0;padding:24px 32px 80px;background:#000;color:#ebebeb;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}'
            'nav{position:sticky;top:0;padding:12px 0;background:#000;z-index:2;display:flex;flex-wrap:wrap;gap:16px;font-size:13px}'
            'nav a{color:#8a8a8a;text-decoration:none}nav a:hover{color:#ebebeb}p{color:#8a8a8a;font-size:13px}'
            'h2{font-weight:400;font-size:28px;margin:44px 0 8px}h3{font-weight:400;font-size:14px;margin:18px 0 6px}'
            'h3 span{color:#8a8a8a;margin-left:8px}h3.fam{font-size:20px;margin-top:30px}'
            '.g,.r{display:flex;flex-wrap:wrap;gap:4px;align-items:center}.r{margin:3px 0}'
            '.g img,.r img{width:56px;height:70px;object-fit:cover}.m{width:170px;font-size:12px;color:#8a8a8a}.sep{width:14px}'
            f'</style></head><body><nav>{nav}</nav>{"".join(out)}</body></html>')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "index.html").write_text(page, encoding="utf-8")
    print("->", OUT / "index.html", "models:", list(loaded))


if __name__ == "__main__":
    main()
