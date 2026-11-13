"""Medium 범주 v2: 묘사의 Medium 줄에서 가장 먼저 나오는 매체 낱말로 고른다(토큰 없음, 대략적인 판정).

범주: Photo · Painting · Drawing · Print · Collage · Graphic · Other 
출력: data/analysis/medium_v2.json {파일: 범주}
사용: python3 scripts/medium_v2.py   (.venv-ml 아니어도 된다)
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
MED = [("Collage", r"collage|cut-out|cut out|pasted|montage|photomontage"),
       ("Print", r"woodcut|woodblock|linocut|screen[ -]?print|silkscreen|etching|engraving|lithograph|risograph|letterpress|block print"),
       ("Drawing", r"pencil|pen and ink|pen-and-ink|ink line|ink drawing|ink and brush|ink-brush|brush and ink|\bink\b|charcoal|cross-?hatch|hatching|sketch|line drawing|crayon"),
       ("Painting", r"paint|brushstroke|brush stroke|gouache|water ?colou?r|oil|acrylic|airbrush"),
       ("Photo", r"photograph|photo|film grain|film still"), ("Graphic", r"flat|vector|solid ink|geometric|digital|graphic")]


def earliest(text: str) -> str:
    best, pos = "Other", 10 ** 9
    for name, pat in MED:
        m = re.search(pat, text, re.I)
        if m and m.start() < pos:
            best, pos = name, m.start()
    return best


def main() -> None:
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    out = {n: earliest(c["Medium"]) for n, c in sorted(cap.items())}
    (A / "medium_v2.json").write_text(json.dumps(out, indent=0))
    print(collections.Counter(out.values()))


if __name__ == "__main__":
    main()
