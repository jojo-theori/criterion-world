"""PaddleOCR 텍스트 영역 엔진.

변형(variant)
  ocr-<side>   : 검출 + 인식 파이프라인, 사각형. 긴 변을 <side>px 로 줄여 검출한다 (full 은 원본 크기)
  det-<side>   : 검출만, 다각형(poly). 휘어진 글자 모양을 따라간다

커버마다 <outDir>/<stem>.json 을 쓰고, 이미 있으면 건너뛴다(--force 로 다시).

사용
  .venv/bin/python scripts/engines/paddleocr_engine.py <inDir> <outDir> --variant ocr-960
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def det_params(side: str) -> dict:
    if side == "full":
        return dict(limit_side_len=64, limit_type="min")      # 줄이지 않음
    return dict(limit_side_len=int(side), limit_type="max")


def build(variant: str):
    kind, side = variant.split("-", 1)
    p = det_params(side)
    if kind == "ocr":
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                        use_textline_orientation=True, lang="en",
                        text_det_limit_side_len=p["limit_side_len"], text_det_limit_type=p["limit_type"])

        def run(path: str) -> list[dict]:
            r = ocr.predict(path)[0]
            return [dict(source="recognize", text=t, confidence=float(s), poly=poly.tolist())
                    for t, s, poly in zip(r["rec_texts"], r["rec_scores"], r["rec_polys"])]
        return run

    if kind == "det":
        from paddleocr import TextDetection
        det = TextDetection(model_name="PP-OCRv6_medium_det", **p)
        det.paddlex_predictor.post_op.box_type = "poly"

        def run(path: str) -> list[dict]:
            r = det.predict(path)[0]
            return [dict(source="detect", confidence=float(s), poly=[list(map(float, pt)) for pt in poly])
                    for poly, s in zip(r["dt_polys"], r["dt_scores"])]
        return run

    raise ValueError(variant)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("in_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--variant", required=True)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in Path(a.in_dir).iterdir() if p.suffix.lower() in IMAGE_EXT)
    todo = [p for p in files if a.force or not (out / f"{p.stem}.json").exists()]
    if not todo:
        print(f"paddle {a.variant}: 0 processed, {len(files)} skipped")
        return

    from PIL import Image
    run = build(a.variant)
    for p in todo:
        w, h = Image.open(p).size
        t0 = time.time()
        regions = run(str(p))
        doc = dict(file=p.name, engine=f"paddle-{a.variant}", width=w, height=h,
                   elapsed_ms=int((time.time() - t0) * 1000), regions=regions)
        (out / f"{p.stem}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    print(f"paddle {a.variant}: {len(todo)} processed, {len(files) - len(todo)} skipped -> {out}")


if __name__ == "__main__":
    sys.exit(main())
