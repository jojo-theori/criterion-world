"""Sonnet 묘사 문장을 임베딩에 넣었을 때 People 축을 얼마나 잘 가르는지, 이미지만 넣었을 때와 비교한다.

표본: out/gemma/people_sample120.json (사람 라벨 범주마다 20장, animal 은 있는 만큼)
묘사: data/analysis/captions_sonnet_v3.json

비교하는 입력
  SigLIP 2 / Gemma 이미지        기존 임베딩(표본만 잘라 씀)
  Gemma 묘사 전체                묘사 8줄을 글로만
  Gemma 묘사(People 줄 빼고)     나머지 줄만으로도 사람 정보가 실리는가
  Gemma People 줄만
  Gemma 묘사 + 이미지            묘사 글 뒤에 이미지를 붙여 한 벡터로
지표: probe(5겹 교차검증 균형 정확도), text p@10(범주 설명 문장으로 모은 상위 10장의 정답 비율), seed p@10(예시 5장 평균에서 상위 10장)
사용: .venv-ml/bin/python scripts/eval_captions.py [--all]   (--all: 사람 라벨 545편 전부, p@10 대신 p@30)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
EMB = A / "embed"
FIELDS = ["People", "Composition", "Medium", "Typography", "Color and light", "Story it suggests", "Visual idea", "Feels like"]
QUERIES = {
    "none": "no people at all, only objects, shapes, landscape or typography",
    "one": "a single person",
    "two": "two people together",
    "many": "a crowd or a group of many people",
    "some": "only part of a human body, hands or legs, no face",
    "animal": "an animal",
}
K = 30 if "--all" in sys.argv else 10


def text_of(c: dict, fields: list[str]) -> str:
    return " ".join(f"{f}: {c[f]}" for f in fields if c.get(f))


def load(f: str, names: list[str]) -> np.ndarray:
    d = np.load(EMB / f, allow_pickle=True)
    idx = {str(n): i for i, n in enumerate(d["files"])}
    V = d["vectors"].astype(np.float32)[[idx[n] for n in names]]
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def score(X: np.ndarray, y: np.ndarray, Q: np.ndarray | None, rng) -> dict:
    pred = cross_val_predict(LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"), X, y,
                             cv=StratifiedKFold(5, shuffle=True, random_state=0))
    out = {"probe": float(balanced_accuracy_score(y, pred)), "text": {}, "seed": {}}
    for i, c in enumerate(QUERIES):
        mi = np.where(y == c)[0]
        if Q is not None:
            out["text"][c] = float(np.mean(y[np.argsort(-(X @ Q[i]))[:K]] == c))
        ps = []
        for _ in range(20):
            s = rng.choice(mi, 5, replace=False)
            sc = X @ X[s].mean(0); sc[s] = -9
            ps.append(np.mean(y[np.argsort(-sc)[:K]] == c))
        out["seed"][c] = float(np.mean(ps))
    return out


def main() -> None:
    from sentence_transformers import SentenceTransformer
    lab = {k: v["label"] for k, v in json.loads((ROOT / "data" / "labels" / "people.json").read_text()).items()}
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    pool = sorted(lab) if "--all" in sys.argv else json.loads((ROOT / "out" / "gemma" / "people_sample120.json").read_text())
    names = [n for n in pool if n in cap]
    y = np.array([lab[n] for n in names])
    print(len(names), "covers with captions")
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    m = SentenceTransformer("google/embeddinggemma-2", device=dev, config_kwargs={"audio_config": None},
                            model_kwargs={"torch_dtype": torch.float32})
    Q = m.encode(["task: classification | query: " + t for t in QUERIES.values()], normalize_embeddings=True)
    doc = lambda t: "title: none | text: " + t
    rest = [f for f in FIELDS if f != "People"]
    X = {
        "SigLIP 2 image": (load("siglip2-base-patch16-384.npz", names), None),
        "Gemma image": (load("gemma2-plain.npz", names), Q),
        "Gemma caption (all)": (m.encode([doc(text_of(cap[n], FIELDS)) for n in names], normalize_embeddings=True), Q),
        "Gemma caption (no People line)": (m.encode([doc(text_of(cap[n], rest)) for n in names], normalize_embeddings=True), Q),
        "Gemma People line only": (m.encode([doc(text_of(cap[n], ["People"])) for n in names], normalize_embeddings=True), Q),
        "Gemma caption + image": (m.encode([{"text": doc(text_of(cap[n], FIELDS)) + " <|image|>", "image": str(ROOT / "posters" / n)}
                                            for n in names], normalize_embeddings=True, batch_size=8), Q),
    }
    rng = np.random.default_rng(0)
    res = {k: score(V, y, q, rng) for k, (V, q) in X.items()}
    (ROOT / "out" / "gemma" / ("eval_captions_all.json" if "--all" in sys.argv else "eval_captions.json")).write_text(json.dumps(res, indent=1))
    cs = list(QUERIES)
    print(f"\n{'input':32s} probe | text p@{K}: " + " ".join(f"{c:>6s}" for c in cs) + "  mean | seed mean")
    for k, r in res.items():
        t = " ".join(f"{r['text'].get(c, float('nan')):6.2f}" for c in cs)
        tm = np.mean(list(r["text"].values())) if r["text"] else float("nan")
        print(f"{k:32s} {r['probe']:.2f}  | {t}  {tm:.2f} | {np.mean(list(r['seed'].values())):.2f}")


if __name__ == "__main__":
    main()
