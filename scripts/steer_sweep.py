"""EmbeddingGemma 2 방향 문장(이미지 앞에 붙이는 글)을 여러 개 시험한다. People 사람 라벨 545편만 계산해 빠르게 비교.

방향 문장마다
  probe  선형 분류기 5겹 교차검증 균형 정확도(그 공간에 사람 수를 가르는 평면이 있는가)
  text   범주 설명 문장 하나로 모은 상위 30장의 정답 비율(precision@30), 범주별
입력 이미지는 원본(plain)과 글자 지운 그림(art) 두 가지로.

결과: out/gemma/steer_sweep.json, 표 출력. 임베딩 캐시: data/analysis/embed/sweep/<hash>.npz
사용: .venv-ml/bin/python scripts/steer_sweep.py
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
ART = ROOT / "out" / "art"
CACHE = ROOT / "data" / "analysis" / "embed" / "sweep"
OUT = ROOT / "out" / "gemma" / "steer_sweep.json"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

# 시험할 방향 문장. None = 이미지만
STEERS = {
    "none": None,
    "composition (기존)": "the visual composition and layout of this poster",
    "people (기존)": "the people, faces and figures in this poster",
    "count people": "how many people appear in this poster and how large they are",
    "human presence": "whether this poster shows a person, a couple, a crowd, only body parts, or nobody at all",
    "framing": "how the human subject is framed: close-up face, half body, full body, tiny figure in a landscape, or no person",
    "describe": "describe this poster",
    "scene": "the scene depicted in this poster",
    "subject": "the main subject of this poster",
}
QUERIES = {
    "none": "no people at all, only objects, shapes, landscape or typography",
    "one": "a single person",
    "two": "two people together",
    "many": "a crowd or a group of many people",
    "some": "only part of a human body, hands or legs, no face",
    "animal": "an animal",
}


def main() -> None:
    from sentence_transformers import SentenceTransformer
    lab = {k: v["label"] for k, v in json.loads((ROOT / "data" / "labels" / "people.json").read_text()).items()}
    names = sorted(lab)
    y = np.array([lab[n] for n in names])
    model = SentenceTransformer("google/embeddinggemma-2", device=DEV, config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.float32})
    Q = model.encode(["task: classification | query: " + t for t in QUERIES.values()], normalize_embeddings=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for img_kind in ("plain", "art"):
        for sname, steer in STEERS.items():
            key = f"{img_kind} · {sname}"
            h = hashlib.md5(f"{img_kind}|{steer}".encode()).hexdigest()[:10]
            f = CACHE / f"{h}.npz"
            if f.exists():
                X = np.load(f)["v"]
            else:
                t0 = time.time()
                paths = [str((ART if img_kind == "art" else POSTERS) / n) for n in names]
                items = [{"image": p} if steer is None else {"text": f"task: clustering | query: {steer} <|image|>", "image": p} for p in paths]
                X = model.encode(items, normalize_embeddings=True, batch_size=16)
                np.savez(f, v=X)
                print(f"  embedded {key} in {time.time() - t0:.0f}s", flush=True)
            pred = cross_val_predict(LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced"), X, y,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=0))
            probe = float(balanced_accuracy_score(y, pred))
            text = {}
            for (c, _), q in zip(QUERIES.items(), Q):
                top = np.argsort(-(X @ q))[:30]
                text[c] = float(np.mean(y[top] == c))
            res[key] = {"steer": steer, "probe": probe, "text": text, "text_mean": float(np.mean(list(text.values())))}
            OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            print(f"{key:34s} probe {probe:.2f} | text mean {res[key]['text_mean']:.2f} | " +
                  " ".join(f"{c} {v:.2f}" for c, v in text.items()), flush=True)


if __name__ == "__main__":
    main()
