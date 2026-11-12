"""커버의 얼굴을 찾는다(OpenCV YuNet). People 축(Talking heads, Big face, Lone figure …)의 재료.

좌표와 크기는 모두 커버 크기 대비 비율이라 해상도와 무관하다. 작은 얼굴을 놓치지 않게 긴 변을 960px 로 키워 찾는다.
YuNet 은 사진 얼굴에 강하고 그림 얼굴에는 약하다. 그림 얼굴은 build_tropes.py 가 OWLv2 의 face 상자와 합쳐 보완한다.

출력: data/analysis/faces.json  {파일 이름: [{"box": [x0, y0, x1, y1], "score": s, "eyes": [[x, y], [x, y]]}, ...]}

사용
  .venv-ml/bin/python scripts/detect_faces.py          # 새 커버만
  .venv-ml/bin/python scripts/detect_faces.py --force  # 전부 다시
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
OUT = ROOT / "data" / "analysis" / "faces.json"
MODEL = ROOT / "data" / "models" / "face_detection_yunet_2023mar.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
LONG = 960          # 찾을 때 긴 변(px)
SCORE = 0.7         # 이 점수 미만은 버린다


def detector(w: int, h: int) -> cv2.FaceDetectorYN:
    if not MODEL.exists():
        MODEL.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL)
    return cv2.FaceDetectorYN.create(str(MODEL), "", (w, h), SCORE, 0.3, 5000)


def faces(path: Path) -> list[dict]:
    im = cv2.imread(str(path))
    h0, w0 = im.shape[:2]
    k = LONG / max(w0, h0)
    im = cv2.resize(im, (round(w0 * k), round(h0 * k)), interpolation=cv2.INTER_CUBIC)
    h, w = im.shape[:2]
    det = detector(w, h)
    _, found = det.detect(im)
    out = []
    for f in ([] if found is None else found):
        x, y, fw, fh = f[:4]
        x0, y0, x1, y1 = max(0, x) / w, max(0, y) / h, min(w, x + fw) / w, min(h, y + fh) / h
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            continue
        out.append({"box": [round(float(v), 4) for v in (x0, y0, x1, y1)], "score": round(float(f[14]), 3),
                    "eyes": [[round(float(f[4]) / w, 4), round(float(f[5]) / h, 4)], [round(float(f[6]) / w, 4), round(float(f[7]) / h, 4)]]})
    return sorted(out, key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))


def main() -> None:
    force = "--force" in sys.argv
    data = {} if force or not OUT.exists() else json.loads(OUT.read_text())
    todo = [p for p in sorted(POSTERS.glob("*.jpg")) if p.name not in data]
    print(f"{len(todo)} covers", flush=True)
    for n, p in enumerate(todo, 1):
        data[p.name] = faces(p)
        if n % 200 == 0:
            print(f"  {n}/{len(todo)}", flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    cnt = np.array([len(v) for v in data.values()])
    print(f"faces per cover: 0={int((cnt == 0).sum())} 1={int((cnt == 1).sum())} 2-4={int(((cnt >= 2) & (cnt <= 4)).sum())} 5+={int((cnt >= 5).sum())}")
    print("->", OUT)


if __name__ == "__main__":
    main()
