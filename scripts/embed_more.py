"""더 많은 시각 신호(2026-11-12 사용자: "전반적으로 데이터가 부족하다").

1) 그림만(art): 글자 영역(data/regions/combined)과 크라이테리온 틀(visual frame.rects)을 주변 색으로 메워 지운 커버.
   SigLIP 이 제목 글자를 읽어 '의미'로 묶는 영향을 빼려고.
2) SigLIP 2 임베딩(그림만)          → data/analysis/embed/siglip2-base-patch16-384-art.npz
3) DINOv2 임베딩(원본, 그림만)       → data/analysis/embed/dinov2-base.npz, dinov2-base-art.npz
   의미보다 모양·구도·질감에 민감한 자기지도 모델.
4) 사람 부위 분할(SegFormer, 옷·신체 18 부위) → data/analysis/human_parse.json
   {파일: {person, face, hair, arms, legs, torso, ...: 화면 비율}}. 손·다리·몸통만 나온 커버(Some human)를 잡으려고.

모두 이미 처리한 커버는 건너뛴다. 그림만 이미지는 out/art/ 에 캐시(공개하지 않는다).

사용 (.venv-ml)
  .venv-ml/bin/python scripts/embed_more.py            # 전부
  .venv-ml/bin/python scripts/embed_more.py --only parse
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
REGIONS = ROOT / "data" / "regions" / "combined"
VISUAL = ROOT / "data" / "analysis" / "visual"
EMB = ROOT / "data" / "analysis" / "embed"
ART = ROOT / "out" / "art"
PARSE = ROOT / "data" / "analysis" / "human_parse.json"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def art_image(p: Path) -> Image.Image:
    """글자·틀을 지운 그림. 512px 로 줄여 OpenCV inpaint(주변 색으로 메움)."""
    import cv2
    out = ART / (p.stem + ".jpg")
    if out.exists():
        return Image.open(out).convert("RGB")
    im = Image.open(p).convert("RGB")
    W, H = im.size
    s = 512 / max(W, H)
    small = im.resize((int(W * s), int(H * s)), Image.LANCZOS)
    mask = Image.new("L", small.size, 0)
    d = ImageDraw.Draw(mask)
    rj = REGIONS / f"{p.stem}.json"
    if rj.exists():
        r = json.loads(rj.read_text())
        k = small.width / r["width"]
        for x in r["regions"]:
            if len(x["poly"]) >= 3:
                d.polygon([(a * k, b * k) for a, b in x["poly"]], fill=255)
    vj = VISUAL / f"{p.stem}.json"
    if vj.exists():
        for x0, y0, x1, y1 in json.loads(vj.read_text()).get("frame", {}).get("rects") or []:
            d.rectangle([x0 * small.width, y0 * small.height, x1 * small.width, y1 * small.height], fill=255)
    m = cv2.dilate(np.asarray(mask), np.ones((7, 7), np.uint8))
    a = cv2.inpaint(cv2.cvtColor(np.asarray(small), cv2.COLOR_RGB2BGR), m, 9, cv2.INPAINT_TELEA)
    res = Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB))
    ART.mkdir(parents=True, exist_ok=True)
    res.save(out, quality=90)
    return res


def load(path: Path) -> tuple[list[str], dict[str, np.ndarray]]:
    if path.exists():
        d = np.load(path, allow_pickle=True)
        return list(map(str, d["files"])), dict(zip(map(str, d["files"]), d["vectors"]))
    return [], {}


def save(path: Path, vecs: dict[str, np.ndarray]) -> None:
    files = sorted(vecs)
    np.savez(path, files=np.array(files), vectors=np.stack([vecs[f] for f in files]).astype(np.float32))


def run_embed(name: str, files: list[Path], make, encode, batch: int = 16) -> None:
    path = EMB / f"{name}.npz"
    _, vecs = load(path)
    todo = [p for p in files if p.name not in vecs]
    print(f"{name}: {len(todo)} to do, {len(vecs)} done", flush=True)
    t0 = time.time()
    for k in range(0, len(todo), batch):
        chunk = todo[k:k + batch]
        ims = [make(p) for p in chunk]
        with torch.no_grad():
            v = encode(ims)
        for p, x in zip(chunk, v):
            vecs[p.name] = x
        if (k // batch) % 10 == 0:
            print(f"  {name} {k + len(chunk)}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
            save(path, vecs)
    save(path, vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["siglip", "dino", "parse"])
    args = ap.parse_args()
    files = sorted(POSTERS.glob("*.jpg"))
    EMB.mkdir(parents=True, exist_ok=True)
    full = lambda p: Image.open(p).convert("RGB")

    if args.only in (None, "siglip"):
        from transformers import AutoModel, AutoProcessor
        mid = "google/siglip2-base-patch16-384"
        model = AutoModel.from_pretrained(mid).to(DEV).eval(); proc = AutoProcessor.from_pretrained(mid)

        def enc(ims):
            o = model.get_image_features(**proc(images=ims, return_tensors="pt").to(DEV))
            o = o if isinstance(o, torch.Tensor) else o.pooler_output
            return torch.nn.functional.normalize(o, dim=-1).cpu().numpy()
        run_embed("siglip2-base-patch16-384-art", files, art_image, enc)
        del model

    if args.only in (None, "dino"):
        from transformers import AutoImageProcessor, AutoModel
        mid = "facebook/dinov2-base"
        model = AutoModel.from_pretrained(mid).to(DEV).eval(); proc = AutoImageProcessor.from_pretrained(mid)

        def enc(ims):
            h = model(**proc(images=ims, return_tensors="pt").to(DEV)).last_hidden_state
            v = torch.cat([h[:, 0], h[:, 1:].mean(1)], -1)          # CLS + 패치 평균
            return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
        run_embed("dinov2-base", files, full, enc)
        run_embed("dinov2-base-art", files, art_image, enc)
        del model

    if args.only in (None, "parse"):
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
        mid = "mattmdjaga/segformer_b2_clothes"
        model = AutoModelForSemanticSegmentation.from_pretrained(mid).to(DEV).eval(); proc = SegformerImageProcessor.from_pretrained(mid)
        names = model.config.id2label
        GROUP = {"face": ["Face"], "hair": ["Hair"], "arms": ["Left-arm", "Right-arm"], "legs": ["Left-leg", "Right-leg"],
                 "torso": ["Upper-clothes", "Dress", "Belt", "Scarf"], "lower": ["Pants", "Skirt"],
                 "shoes": ["Left-shoe", "Right-shoe"], "hat": ["Hat", "Sunglasses"]}
        res = json.loads(PARSE.read_text()) if PARSE.exists() else {}
        todo = [p for p in files if p.name not in res]
        print(f"parse: {len(todo)} to do", flush=True)
        t0 = time.time()
        for k, p in enumerate(todo):
            im = full(p); im.thumbnail((512, 512))
            with torch.no_grad():
                lg = model(**proc(images=im, return_tensors="pt").to(DEV)).logits
            lab = lg.argmax(1)[0].cpu().numpy()
            tot = lab.size
            share = {nm: float(np.isin(lab, [i for i, n in names.items() if n in ls]).sum() / tot) for nm, ls in GROUP.items()}
            share["person"] = float((lab != 0).sum() / tot)
            res[p.name] = {kk: round(v, 4) for kk, v in share.items()}
            if k % 100 == 0:
                print(f"  parse {k}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
                PARSE.write_text(json.dumps(res))
        PARSE.write_text(json.dumps(res))
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
