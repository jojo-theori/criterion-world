"""사람 라벨과 묘사 판정이 다른 커버를 모아 본다. 둘 다 People v2 범주(data/analysis/people_v2.json, scripts/people_v2.py).

어느 쪽이 맞는지 사람이 다시 보는 용도: out/gemma/people_review.html
칸마다 커버, 사람 라벨 → 묘사 라벨, 묘사의 People 줄. 기준이 바뀌어 생긴 차이(내 One → 뒷모습 No-face)와 Ambiguous 는 뒤로.
사용: python3 scripts/review_people.py
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
OUT = ROOT / "out" / "gemma"
ORDER = ["None", "No-face", "One", "Two", "3+", "Ambiguous"]


def thumb(n: str) -> str:
    t = OUT / "thumbs_l" / n
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        Image.open(ROOT / "posters" / n).convert("RGB").resize((240, 298), Image.LANCZOS).save(t, quality=85)
    return f"thumbs_l/{html.escape(n)}"


def main() -> None:
    v2 = json.loads((A / "people_v2.json").read_text())
    human = {k: v["human"] for k, v in v2.items() if "human" in v}
    cap = {k: {"label": v["cat"], "ambiguous": v["cat"] == "Ambiguous" or (v["human"] == "One" and v["cat"] == "No-face"),
               "reason": v["reason"]} for k, v in v2.items() if "human" in v}
    text = json.loads((A / "captions_sonnet_v3.json").read_text())
    diff = [n for n in human if n in cap and human[n] != cap[n]["label"]]
    diff.sort(key=lambda n: (cap[n]["ambiguous"], ORDER.index(human[n]), ORDER.index(cap[n]["label"]), n))
    cells = "".join(
        f'<figure><img src="{thumb(n)}"><figcaption><b>{human[n]} → {cap[n]["label"]}</b>'
        f'{" · " + cap[n]["reason"] if cap[n]["reason"] else ""}<span class="t">{html.escape(n[5:-4].replace("-", " "))}</span>'
        f'<span>{html.escape(text[n]["People"])}</span></figcaption></figure>' for n in diff)
    css = ('body{margin:0;padding:24px 32px 80px;background:#000;color:#ebebeb;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}'
           'p{color:#8a8a8a;font-size:13px}.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:36px 28px}'
           'figure{margin:0}img{width:100%;aspect-ratio:240/298;object-fit:cover}figcaption{font-size:12px;line-height:1.45;color:#bdbdbd}'
           'figcaption b{display:block;font-weight:500;color:#fff;font-size:14px;margin:8px 0 2px}figcaption span{display:block}'
           '.t{color:#8a8a8a;margin-bottom:4px}')
    (OUT / "people_review.html").write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>People review</title><style>{css}</style></head><body>'
        f'<p>{len(diff)} covers where the human label and the caption label differ (human → caption).</p><div class="g">{cells}</div></body></html>',
        encoding="utf-8")
    print(len(diff), "->", OUT / "people_review.html")


if __name__ == "__main__":
    main()
