"""EmbeddingGemma 2(google/embeddinggemma-2) 로 커버 임베딩을 만든다. 이미지와 글이 한 공간(768d)에 들어간다.

변형(variant)
  plain        커버 그대로
  art          글자 영역(data/regions/combined)과 왼쪽 띠를 주변 색으로 지운 커버(제목 글자에 끌려 의미로 묶이는 것을 줄인다)
  composition  글로 방향을 준다: "task: clustering | query: the visual composition and layout of this poster <|image|>"
  people       … the people, faces and figures in this poster …
  type         … the typography and lettering of this poster …
  medium       … the artistic medium and technique of this poster …
  described    분석 결과 묘사 문장(data/analysis/descriptions.json, scripts/describe_covers.py) + 이미지

모델 카드 권장: float16 금지(NaN). MPS 에서는 float32. 오디오 인코더는 싣지 않는다.

출력: data/analysis/embed/gemma2-<variant>.npz  (files, vectors: L2 정규화 768d)
사용
  .venv-ml/bin/python scripts/embed_gemma.py plain art composition   # 새 커버만
  .venv-ml/bin/python scripts/embed_gemma.py --force plain
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
REGIONS = ROOT / "data" / "regions" / "combined"
OUT = ROOT / "data" / "analysis" / "embed"
ART = ROOT / "out" / "art"          # 글자 지운 커버(확인용, 레포에 올리지 않는 out/)
MODEL = "google/embeddinggemma-2"
STEER = {
    "composition": "the visual composition and layout of this poster",
    "people": "the people, faces and figures in this poster",
    "type": "the typography and lettering of this poster",
    "medium": "the artistic medium and technique of this poster",
}


def art_image(p: Path) -> Path:
    """글자 다각형과 왼쪽 띠를 지운다: 그 자리를 크게 흐린 커버로 채운다(검은 칸을 남기지 않는다)."""
    out = ART / p.name
    if out.exists():
        return out
    im = Image.open(p).convert("RGB")
    w, h = im.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    rj = REGIONS / f"{p.stem}.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        sx, sy = w / r["width"], h / r["height"]
        for reg in r["regions"]:
            d.polygon([(x * sx, y * sy) for x, y in reg["poly"]], fill=255)
    mask = mask.resize((w // 4, h // 4)).filter(ImageFilter.MaxFilter(3)).resize((w, h))
    # 큰 흐림은 1/8 크기에서 하고 키운다(원본 크기에서 하면 장당 4초 넘게 걸렸다)
    fill = im.resize((w // 8, h // 8), Image.BILINEAR).filter(ImageFilter.GaussianBlur(max(w, h) / 200)).resize((w, h), Image.BILINEAR)
    im.paste(fill, (0, 0), mask)
    ART.mkdir(parents=True, exist_ok=True)
    im.save(out, quality=90)
    return out


_DESC: dict | None = None


def item(variant: str, p: Path) -> dict:
    global _DESC
    if variant == "described":
        if _DESC is None:
            _DESC = json.loads((ROOT / "data" / "analysis" / "descriptions.json").read_text())
        return {"text": f"task: clustering | query: {_DESC.get(p.name, '')} <|image|>", "image": str(p)}
    if variant == "plain":
        return {"image": str(p)}
    if variant == "art":
        return {"image": str(art_image(p))}
    return {"text": f"task: clustering | query: {STEER[variant]} <|image|>", "image": str(p)}


def main() -> None:
    from sentence_transformers import SentenceTransformer
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    variants = args or ["plain"]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=dev, config_kwargs={"audio_config": None},
                                model_kwargs={"torch_dtype": torch.float32})
    paths = sorted(POSTERS.glob("*.jpg"))
    OUT.mkdir(parents=True, exist_ok=True)
    for v in variants:
        f = OUT / f"gemma2-{v}.npz"
        have: dict[str, np.ndarray] = {}
        if f.exists() and not force:
            d = np.load(f)
            have = dict(zip([str(x) for x in d["files"]], d["vectors"]))
        todo = [p for p in paths if p.name not in have]
        print(f"{v}: {len(todo)} covers", flush=True)
        t0 = time.time()
        for k in range(0, len(todo), 16):
            batch = todo[k:k + 16]
            e = model.encode([item(v, p) for p in batch], normalize_embeddings=True, batch_size=16)
            have.update({p.name: x for p, x in zip(batch, e)})
            if (k // 16) % 10 == 0:
                print(f"  [{k + len(batch)}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
        names = sorted(have)
        V = np.stack([have[n] for n in names]).astype(np.float32)
        assert not np.isnan(V).any(), "NaN in embeddings"
        np.savez(f, files=np.array(names), vectors=V)
        print(f"  -> {f} {V.shape}", flush=True)


if __name__ == "__main__":
    main()
