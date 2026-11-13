"""분석 결과를 커버마다 짧은 영어 묘사 문장으로 바꾼다. EmbeddingGemma 2 에 이미지와 함께 넣는 '설명 입력'의 재료.

재료(모두 이미 있는 분석): 얼굴(faces.json), 물체(OWLv2), build_tropes 의 지표(f: 글자·제목·대칭·대각선·여백·그림 비율),
시각 분석의 팔레트·흑백. 판정 결과(범주 이름)는 넣지 않는다 — 판정 규칙이 그대로 새는 것을 막으려고 관찰값만 적는다.

출력: data/analysis/descriptions.json  {파일: "Two faces, the largest covers 6% of the poster. ..."}
사용: .venv-ml/bin/python scripts/describe_covers.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
WORDS = {0: "No", 1: "One", 2: "Two", 3: "Three", 4: "Four"}


def pct(x: float) -> str:
    return f"{round(100 * x)}%"


def describe(n: str, f: dict, objs: list, vis: dict | None) -> str:
    s = []
    k = f["faces"]
    if k == 0:
        s.append("No visible faces.")
    else:
        s.append(f"{WORDS.get(k, 'Many' if k >= 5 else str(k))} face{'s' if k != 1 else ''}, the largest covers {pct(f['face_max'])} of the poster.")
    if f["persons"]:
        s.append(f"Human figures cover up to {pct(f['person_max'])} of the poster.")
    t = f.get("title_h", 0)
    if f["text_area"] < 0.03:
        s.append("Almost no text.")
    else:
        size = "very large" if t > 0.3 else "large" if t > 0.15 else "medium" if t > 0.08 else "small"
        row = "top" if f["title_cy"] < 0.33 else "bottom" if f["title_cy"] > 0.66 else "middle"
        col = "left" if f["title_cx"] < 0.38 else "right" if f["title_cx"] > 0.62 else "center"
        s.append(f"The biggest lettering is {size}, placed at the {row} {col}. Text covers {pct(f['text_area'])} of the poster.")
    shapes = []
    if f["img_sym"] > 0.5:
        shapes.append("left-right symmetrical")
    if f["diag"] > 0.6:
        shapes.append("dominated by diagonal lines")
    if f["long_h"] + f["long_v"] >= 4:
        shapes.append("divided by long straight lines into panels")
    if f["empty_share"] > 0.5:
        shapes.append("mostly empty background")
    if f["figure_share"] > 0.75:
        shapes.append("the picture fills the whole frame")
    if shapes:
        s.append("The image is " + ", ".join(shapes) + ".")
    if vis:
        c = vis["full"]["color"]
        if c["color_mode"] == "grayscale":
            s.append("Black and white.")
        else:
            s.append(f"Colorfulness {round(c['colorfulness'])}.")
    labels = []
    for o in sorted(objs, key=lambda o: -o["score"]):
        if o["score"] >= 0.25 and o["label"] not in labels and o["label"] not in ("face", "eye", "lips"):
            labels.append(o["label"])
    if labels:
        s.append("Visible objects: " + ", ".join(labels[:5]) + ".")
    return " ".join(s)


def main() -> None:
    tr = json.loads((A / "tropes.json").read_text())
    obj = json.loads((A / "objects" / "owlv2.json").read_text())
    out = {}
    for n, r in tr.items():
        vp = A / "visual" / f"{n[:-4]}.json"
        out[n] = describe(n, r["f"], obj.get(n, []), json.loads(vp.read_text()) if vp.exists() else None)
    (A / "descriptions.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    for n in list(out)[:3]:
        print(n, "->", out[n])
    print(len(out), "->", A / "descriptions.json")


if __name__ == "__main__":
    main()
