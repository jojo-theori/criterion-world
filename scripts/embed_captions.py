"""Sonnet 커버 묘사(data/analysis/captions_sonnet_v3.json)를 EmbeddingGemma 2 로 임베딩한다.

출력(data/analysis/embed/, files·vectors 형식은 다른 임베딩과 같다)
  gemma2-cap-all.npz          묘사 여덟 줄 전체(글만)
  gemma2-cap-<줄>.npz         한 줄만(people, composition, medium, typography, color)
  gemma2-cap-img.npz          묘사 전체 글 + 커버 이미지를 한 입력으로
글은 문서 형식("title: none | text: …"), 범주 문장은 질의 형식(task: classification | query: …)으로 맞춘다.
사용: .venv-ml/bin/python scripts/embed_captions.py [--no-image]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
EMB = A / "embed"
FIELDS = ["People", "Composition", "Medium", "Typography", "Color and light", "Story it suggests", "Visual idea", "Feels like"]
LINES = {"people": "People", "composition": "Composition", "medium": "Medium", "typography": "Typography", "color": "Color and light"}


def doc(c: dict, fields: list[str]) -> str:
    return "title: none | text: " + " ".join(f"{f}: {c[f]}" for f in fields if c.get(f))


def main() -> None:
    from sentence_transformers import SentenceTransformer
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    files = sorted(cap)
    m = SentenceTransformer("google/embeddinggemma-2", device="mps" if torch.backends.mps.is_available() else "cpu",
                            config_kwargs={"audio_config": None}, model_kwargs={"torch_dtype": torch.float32})
    jobs = {"cap-all": [doc(cap[n], FIELDS) for n in files]}
    jobs.update({f"cap-{k}": [doc(cap[n], [f]) for n in files] for k, f in LINES.items()})
    if "--no-image" not in sys.argv:
        jobs["cap-img"] = [{"text": doc(cap[n], FIELDS) + " <|image|>", "image": str(ROOT / "posters" / n)} for n in files]
    for k, items in jobs.items():
        t0 = time.time()
        V = m.encode(items, normalize_embeddings=True, batch_size=8 if k == "cap-img" else 32)
        np.savez(EMB / f"gemma2-{k}.npz", files=np.array(files), vectors=V.astype(np.float32))
        print(f"{k}: {V.shape} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
