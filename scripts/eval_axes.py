"""사람이 정한 범주(정답)를 임베딩이 얼마나 잘 모으는지 잰다. "어떤 평면에서 볼지"의 시험.

정답: data/labels/people.json(사람 라벨, People 축) — none · one · two · many · some · animal
     'one' 중 얼굴이 크게 차지하는 것(One big face)은 YuNet face_max ≥ 0.035 로 따로 본다.

임베딩마다 세 가지로 잰다(모두 라벨 545편 안에서)
  probe   선형 분류기(로지스틱 회귀, 5겹 교차검증): 그 공간에 범주를 가르는 '평면'이 있는가. 균형 정확도
  seed    범주마다 무작위 예시 10장의 평균(프로토타입)에서 가까운 순으로 모았을 때 상위 30장 중 정답 비율(precision@30)
          예시를 고르는 운을 줄이려고 20번 평균
  text    (EmbeddingGemma 2 만) 범주 설명 문장 하나로 모았을 때 precision@30

출력: 표를 출력하고 out/gemma/eval.html 에 범주별 상위 30장(seed 방식, 정답 테두리)을 모델마다 나란히
사용: .venv-ml/bin/python scripts/eval_axes.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parent.parent
EMB = ROOT / "data" / "analysis" / "embed"
OUT = ROOT / "out" / "gemma"
POSTERS = ROOT / "posters"
MODELS = {
    "SigLIP 2": "siglip2-base-patch16-384.npz",
    "SigLIP 2 art": "siglip2-base-patch16-384-art.npz",
    "DINOv2": "dinov2-base.npz",
    "DINOv2 art": "dinov2-base-art.npz",
    "Gemma plain": "gemma2-plain.npz",
    "Gemma art": "gemma2-art.npz",
    "Gemma people": "gemma2-people.npz",
    "Gemma composition": "gemma2-composition.npz",
    "Gemma medium": "gemma2-medium.npz",
}
TEXT = {   # 범주 설명(Gemma 글 공간)
    "none": "no people at all, only objects, shapes, landscape or typography",
    "one": "a single person",
    "one big face": "one large human face filling the poster, a close-up portrait",
    "two": "two people together",
    "many": "a crowd or a group of many people",
    "some": "only part of a human body, hands or legs, no face",
    "animal": "an animal",
}


def load(f: str):
    p = EMB / f
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    V = d["vectors"].astype(np.float32)
    return {str(n): v / (np.linalg.norm(v) + 1e-9) for n, v in zip(d["files"], V)}


def thumb(n: str) -> str:
    t = OUT / "thumbs" / n
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        Image.open(POSTERS / n).convert("RGB").resize((112, 140), Image.LANCZOS).save(t, quality=80)
    return f"thumbs/{html.escape(n)}"


def main() -> None:
    lab = {k: v["label"] for k, v in json.loads((ROOT / "data" / "labels" / "people.json").read_text()).items()}
    faces = json.loads((ROOT / "data" / "analysis" / "faces.json").read_text())
    fmax = {n: max(((b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1]) for b in faces.get(n, [])), default=0) for n in lab}
    cats = {c: {n for n, l in lab.items() if l == c} for c in ["none", "one", "two", "many", "some", "animal"]}
    cats["one big face"] = {n for n in cats["one"] if fmax[n] >= 0.035}
    names = sorted(lab)
    y = np.array([lab[n] for n in names])
    rng = np.random.default_rng(0)
    text_vec = None
    rows, page = [], []
    for mname, f in MODELS.items():
        E = load(f)
        if E is None or not all(n in E for n in names):
            continue
        X = np.stack([E[n] for n in names])
        pred = cross_val_predict(LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"), X, y,
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0))
        probe = balanced_accuracy_score(y, pred)
        seed, txt, tops = {}, {}, {}
        for c, mem in cats.items():
            mem_l = sorted(mem)
            mi = np.array([names.index(n) for n in mem_l])
            ps = []
            for r in range(20):
                s = rng.choice(mi, size=min(10, len(mi) - 1), replace=False)
                proto = X[s].mean(0)
                sc = X @ proto
                sc[s] = -9                                    # 예시 자신은 빼고
                top = np.argsort(-sc)[:30]
                ps.append(np.mean([names[i] in mem for i in top]))
                if r == 0:
                    tops[c] = [names[i] for i in top]
            seed[c] = float(np.mean(ps))
            if mname.startswith("Gemma"):
                if text_vec is None:
                    import torch
                    from sentence_transformers import SentenceTransformer
                    m = SentenceTransformer("google/embeddinggemma-2", device="mps" if torch.backends.mps.is_available() else "cpu",
                                            config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
                    text_vec = dict(zip(TEXT, m.encode(["task: classification | query: " + t for t in TEXT.values()], normalize_embeddings=True)))
                sc = X @ text_vec[c]
                txt[c] = float(np.mean([names[i] in mem for i in np.argsort(-sc)[:30]]))
        rows.append((mname, probe, seed, txt))
        page.append((mname, tops))

    base = {c: len(m) / len(names) for c, m in cats.items()}
    cs = list(cats)
    print(f"\n{'model':20s} probe  | seed p@30: " + " ".join(f"{c[:8]:>8s}" for c in cs))
    print(f"{'(chance)':20s}   -    | " + " ".join(f"{base[c]:8.2f}" for c in cs))
    for m, p, s, t in rows:
        print(f"{m:20s} {p:.2f}  | " + " ".join(f"{s[c]:8.2f}" for c in cs))
        if t:
            print(f"{'  text query':20s}   -    | " + " ".join(f"{t[c]:8.2f}" for c in cs))

    # 확인 페이지
    out = []
    for c in cs:
        out.append(f'<h2>{html.escape(c)} <span>{len(cats[c])} labeled</span></h2>')
        for m, tops in page:
            imgs = "".join(f'<img class="{"ok" if n in cats[c] else "no"}" src="{thumb(n)}">' for n in tops[c])
            out.append(f'<div class="r"><span class="m">{m}</span>{imgs}</div>')
    css = ('body{margin:0;padding:24px 32px 80px;background:#000;color:#ebebeb;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}'
           'h2{font-weight:400;font-size:24px;margin:36px 0 8px}h2 span{font-size:13px;color:#8a8a8a;margin-left:8px}'
           '.r{display:flex;flex-wrap:wrap;gap:3px;align-items:center;margin:3px 0}.m{width:150px;font-size:12px;color:#8a8a8a}'
           '.r img{width:40px;height:50px;object-fit:cover;box-sizing:border-box}.r img.no{opacity:.3}')
    (OUT / "eval.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Axis eval</title><style>{css}</style></head><body>'
                                   f'<p style="color:#8a8a8a;font-size:13px">Top 30 gathered from 10 random examples of each human-labeled category. Dim = wrong.</p>{"".join(out)}</body></html>', encoding="utf-8")
    print("->", OUT / "eval.html")


if __name__ == "__main__":
    main()
