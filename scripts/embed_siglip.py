"""SigLIP 2 로 커버 임베딩을 뽑고 기법 제로샷 점수를 매긴다. 이미 처리한 커버는 건너뛴다.

결과
  data/analysis/embed/<모델 이름>.npz   files, vectors (L2 정규화된 이미지 임베딩)
  data/analysis/embed/<모델 이름>_technique.json   커버별 기법 라벨 점수

사용 (.venv-ml, Python 3.12)
  .venv-ml/bin/python scripts/embed_siglip.py
  .venv-ml/bin/python scripts/embed_siglip.py --model google/siglip2-so400m-patch16-384
  .venv-ml/bin/python scripts/embed_siglip.py --force
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
OUT = ROOT / "data" / "analysis" / "embed"

# 기법 제로샷 라벨. 프롬프트는 SigLIP 학습 문장 형식("this is a photo of ...")에 맞춘다.
TECHNIQUES = {
    "photograph": "a photograph",
    "film still": "a still frame from a film",
    "woodcut": "a woodcut print",
    "linocut": "a linocut print",
    "screen print": "a screen print poster with flat colors",
    "watercolor": "a watercolor painting",
    "oil painting": "an oil painting",
    "ink drawing": "a pen and ink drawing",
    "pencil drawing": "a pencil sketch",
    "collage": "a collage",
    "digital illustration": "a digital illustration",
    "vector graphic": "a flat vector graphic",
    "typographic": "a typographic design made only of letters",
    "photomontage": "a photomontage",
    "comic": "a comic book illustration",
}


def device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def feat(out) -> torch.Tensor:
    """transformers 5 는 get_*_features 가 출력 객체를 돌려준다."""
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def load_images(paths: list[Path]) -> tuple[list[Path], list[Image.Image]]:
    ok, imgs = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                imgs.append(im.convert("RGB"))
            ok.append(p)
        except Exception as e:  # 받는 중인 파일 등
            print(f"skip {p.name}: {e}")
    return ok, imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/siglip2-base-patch16-384")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    name = a.model.split("/")[-1]
    npz = OUT / f"{name}.npz"
    tech_json = OUT / f"{name}_technique.json"

    files: list[str] = []
    vecs = np.zeros((0, 0), dtype=np.float32)
    tech: dict[str, dict[str, float]] = {}
    if npz.exists() and not a.force:
        d = np.load(npz)
        files, vecs = list(d["files"]), d["vectors"]
        if tech_json.exists():
            tech = json.loads(tech_json.read_text())

    todo = [p for p in sorted(POSTERS.glob("*.jpg")) if p.name not in set(files)]
    print(f"{len(todo)} to embed, {len(files)} done, model {a.model}", flush=True)
    if not todo:
        return

    dev = device()
    model = AutoModel.from_pretrained(a.model, dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    proc = AutoProcessor.from_pretrained(a.model)
    dtype = next(model.parameters()).dtype

    labels = list(TECHNIQUES)
    with torch.no_grad():
        t = proc(text=[f"this is {TECHNIQUES[k]}." for k in labels], padding="max_length", max_length=64,
                 return_tensors="pt").to(dev)
        tf = feat(model.get_text_features(**t))
        tf = tf / tf.norm(dim=-1, keepdim=True)
        scale, bias = model.logit_scale.exp(), model.logit_bias

    new_files, new_vecs = [], []
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        ok, imgs = load_images(todo[i:i + a.batch])
        if not imgs:
            continue
        with torch.no_grad():
            x = proc(images=imgs, return_tensors="pt").to(dev)
            f = feat(model.get_image_features(pixel_values=x["pixel_values"].to(dtype)))
            f = f / f.norm(dim=-1, keepdim=True)
            # SigLIP 은 sigmoid 라 라벨마다 독립 확률
            prob = torch.sigmoid(f @ tf.T * scale + bias).float().cpu().numpy()
        for p, v, pr in zip(ok, f.float().cpu().numpy(), prob):
            new_files.append(p.name)
            new_vecs.append(v)
            tech[p.name] = {k: round(float(s), 4) for k, s in zip(labels, pr)}
        print(f"[{i + len(ok)}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)

    nv = np.stack(new_vecs).astype(np.float32)
    vecs = nv if vecs.size == 0 else np.concatenate([vecs, nv])
    files = files + new_files
    np.savez(npz, files=np.array(files), vectors=vecs)
    tech_json.write_text(json.dumps(tech, ensure_ascii=False, indent=0))
    print(f"saved {len(files)} -> {npz}")


if __name__ == "__main__":
    main()
