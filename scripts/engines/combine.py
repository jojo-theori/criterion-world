"""엔진 결과를 합친다: 모양은 shape 엔진(다각형), 글자는 text 엔진들.

  - text 엔진이 여럿이면 앞의 것이 우선이고, 뒤 엔진의 단어는 앞 단어와 겹치지 않을 때만 더한다
  - shape 다각형마다 중심이 그 안에 들어오는 단어를 붙인다 (단어 박스의 읽기 방향으로 정렬)
  - 어떤 다각형에도 안 들어간 text 단어는 자기 박스로 따로 남긴다 (shape 가 놓친 글자)
  - 글자가 하나도 안 붙은 다각형은 source="detect" 로 남는다 (로고, 무늬, 못 읽은 레터링 후보)

사용
  .venv/bin/python scripts/engines/combine.py <shape_dir> <out_dir> <text_dir> [<text_dir> ...]
  예) combine.py data/regions/paddle-det-640 data/regions/combined data/regions/vision@s50 data/regions/vision
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def norm_polys(doc: dict) -> None:
    w, h = doc["width"], doc["height"]
    for r in doc["regions"]:
        r["npoly"] = [(x / w, y / h) for x, y in r["poly"]]


def center(p):
    return sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p)


def inside(pt, poly, size=400) -> bool:
    img = Image.new("1", (size, size))
    ImageDraw.Draw(img).polygon([(x * size, y * size) for x, y in poly], fill=1)
    x, y = int(pt[0] * size), int(pt[1] * size)
    return 0 <= x < size and 0 <= y < size and bool(img.getpixel((x, y)))


def reading_order(word_polys: list) -> list[int]:
    """단어 박스의 읽기 방향(왼쪽 위 → 오른쪽 위 꼭짓점)을 평균내고, 그 방향으로 단어 중심을 투영한 순서.
    세로로 돌아간 글자나 기울어진 글자도 실제 읽는 순서가 된다."""
    v = np.mean([np.subtract(p[1], p[0]) for p in word_polys], axis=0)
    v = v / (np.linalg.norm(v) or 1)
    proj = [np.dot(center(p), v) for p in word_polys]
    return sorted(range(len(word_polys)), key=lambda i: proj[i])


def merge_words(texts: list[dict]) -> list[dict]:
    words = []
    for t in texts:
        norm_polys(t)
        for r in t["regions"]:
            if r.get("text") and not any(inside(center(r["npoly"]), w["npoly"]) for w in words):
                words.append(r)
    return words


def combine(shape: dict, texts: list[dict]) -> dict:
    norm_polys(shape)
    W, H = shape["width"], shape["height"]
    words = merge_words(texts)
    used = set()
    out = []
    for r in shape["regions"]:
        mine = [i for i, t in enumerate(words) if i not in used and inside(center(t["npoly"]), r["npoly"])]
        used.update(mine)
        if len(mine) > 1:
            mine = [mine[k] for k in reading_order([words[i]["npoly"] for i in mine])]
        reg = dict(source="recognize" if mine else "detect", confidence=r.get("confidence", 0), poly=r["poly"])
        if mine:
            reg["text"] = " ".join(words[i]["text"] for i in mine)
        out.append(reg)
    for i, t in enumerate(words):
        if i not in used:
            out.append(dict(source="recognize", text=t["text"], confidence=t.get("confidence", 0),
                            poly=[(x * W, y * H) for x, y in t["npoly"]]))
    return dict(file=shape["file"], engine="combined", width=W, height=H,
                elapsed_ms=shape.get("elapsed_ms", 0) + sum(t.get("elapsed_ms", 0) for t in texts), regions=out)


def main() -> None:
    shape_dir, out_dir, *text_dirs = map(Path, sys.argv[1:])
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(shape_dir.glob("*.json")):
        texts = [json.loads((d / f.name).read_text()) for d in text_dirs if (d / f.name).exists()]
        if not texts:
            continue
        doc = combine(json.loads(f.read_text()), texts)
        (out_dir / f.name).write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        n += 1
    print(f"combined {n} -> {out_dir}")


if __name__ == "__main__":
    main()
