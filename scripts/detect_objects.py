"""OWLv2 오픈 어휘 검출로 커버 속 물체를 찾는다. 이미 처리한 커버는 건너뛴다.

결과: data/analysis/objects/owlv2.json
  {파일 이름: [{"label": "sword", "score": 0.41, "box": [x0, y0, x1, y1]}, ...]}   box 는 0–1, 원점 왼쪽 위

어휘는 영화 커버에 자주 나오는 것 위주로 골랐다(VOCAB). 바꾸면 --force 로 다시 돌린다.

사용 (.venv-ml)
  .venv-ml/bin/python scripts/detect_objects.py
  .venv-ml/bin/python scripts/detect_objects.py --limit 10 --force   # 시험
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision.ops import nms
from transformers import Owlv2ForObjectDetection, Owlv2Processor

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
OUT = ROOT / "data" / "analysis" / "objects" / "owlv2.json"
MODEL = "google/owlv2-base-patch16-ensemble"

VOCAB = [
    # 사람과 몸
    "person", "face", "eye", "hand", "lips", "silhouette of a person", "skull", "child", "couple embracing",
    # 무기와 폭력
    "gun", "knife", "sword", "blood",
    # 탈것
    "car", "train", "airplane", "boat", "ship", "bicycle", "motorcycle", "horse",
    # 동물
    "dog", "cat", "bird", "fish", "insect",
    # 자연과 하늘
    "tree", "flower", "mountain", "sea", "moon", "sun", "cloud", "star", "fire", "smoke",
    # 건물과 공간
    "house", "building", "city skyline", "window", "door", "staircase", "road", "bridge", "fence",
    # 물건
    "hat", "glasses", "mask", "umbrella", "cigarette", "telephone", "camera", "mirror", "clock",
    "bottle", "glass of wine", "book", "envelope", "television", "guitar", "microphone", "chair", "bed",
    "flag", "cross", "crown", "candle", "rope", "money",
    # 그래픽 요소
    "circle", "geometric shapes", "film strip", "map",
]
THRESHOLD = 0.15
NMS_IOU = 0.4
MAX_PER_COVER = 30


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {} if a.force or not OUT.exists() else json.loads(OUT.read_text())
    todo = [p for p in sorted(POSTERS.glob("*.jpg")) if p.name not in done][: a.limit]
    print(f"{len(todo)} to detect, {len(done)} done, {len(VOCAB)} labels", flush=True)
    if not todo:
        return

    dev = device()
    proc = Owlv2Processor.from_pretrained(MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(MODEL).to(dev).eval()
    queries = [f"a photo of a {v}" for v in VOCAB]
    t0 = time.time()
    for i in range(0, len(todo), a.batch):
        paths, imgs = [], []
        for p in todo[i:i + a.batch]:
            try:
                with Image.open(p) as im:
                    imgs.append(im.convert("RGB"))
                paths.append(p)
            except Exception as e:  # 받는 중인 파일
                print(f"skip {p.name}: {e}")
        if not imgs:
            continue
        with torch.no_grad():
            x = proc(text=[queries] * len(imgs), images=imgs, return_tensors="pt").to(dev)
            out = model(**x)
        # OWLv2 는 이미지를 정사각형으로 패딩한다. 긴 변 기준 크기를 넘겨야 원본 좌표로 돌아온다
        sizes = torch.tensor([[max(im.size)] * 2 for im in imgs], device=dev)
        res = proc.post_process_grounded_object_detection(out, threshold=THRESHOLD, target_sizes=sizes)
        for p, im, r in zip(paths, imgs, res):
            W, H = im.size
            boxes, scores, labels = r["boxes"].float().cpu(), r["scores"].float().cpu(), r["labels"].cpu()
            keep = []
            for lab in labels.unique():   # 라벨별 NMS
                idx = (labels == lab).nonzero().squeeze(1)
                keep += idx[nms(boxes[idx], scores[idx], NMS_IOU)].tolist()
            keep.sort(key=lambda k: -scores[k].item())
            dets = []
            for k in keep[:MAX_PER_COVER]:
                x0, y0, x1, y1 = boxes[k].tolist()
                dets.append({"label": VOCAB[labels[k].item()], "score": round(scores[k].item(), 3),
                             "box": [round(max(0, x0) / W, 4), round(max(0, y0) / H, 4),
                                     round(min(W, x1) / W, 4), round(min(H, y1) / H, 4)]})
            done[p.name] = dets
        n = i + len(paths)
        if n % 40 < a.batch or n == len(todo):
            print(f"[{n}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
            OUT.write_text(json.dumps(done, ensure_ascii=False))
    OUT.write_text(json.dumps(done, ensure_ascii=False))
    print(f"saved {len(done)} -> {OUT}")


if __name__ == "__main__":
    main()
