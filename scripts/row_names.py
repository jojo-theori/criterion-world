"""세 폭 보기의 줄 이름을 짧게 한다(build_web.py 가 META.rowNames 를 쓰기 직전에 쓴다).

아래 축 이름(Lines, Strokes, Contrast …)이 이미 말하는 낱말과 'mostly' 같은 군더더기를 뺀다.
줄인 이름이 같은 축 안에서 겹치면 그 이름들은 원래대로 둔다(줄 이름은 모두 달라야 한다).
"""

from __future__ import annotations

import re

RULES: dict[str, list[tuple[str, str]]] = {
    "lines": [(r"\b(Few|Some|Many) lines\b", r"\1"), (r"\bmostly (horizontal|vertical|diagonal)\b", r"\1")],
    "text": [(r"\bTitle on top\b", "Top title"), (r"\bTitle in the middle\b", "Middle title"),
             (r"\bTitle at the bottom\b", "Bottom title"),
             (r"\blittle type\b", "light"), (r"\bsome type\b", "some"), (r"\blots of type\b", "heavy")],
    "contrast": [(r"\b(medium|high) contrast\b", r"\1")],
    "strokes": [(r"\b(fine|medium|bold) strokes\b", r"\1")],
    "layout": [(r"\btitle on top\b", "title top"), (r"\btitle in the middle\b", "title middle"),
               (r"\btitle at the bottom\b", "title bottom"),
               (r"\bimage at the top\b", "image top"), (r"\bimage at the bottom\b", "image bottom"),
               (r"\bimage centered\b", "image center"),
               (r"\bimage above center\b", "image above"), (r"\bimage below center\b", "image below")],
    "type": [(r"\bTitle on top\b", "Top title"), (r"\bTitle in the middle\b", "Middle title"),
             (r"\bTitle at the bottom\b", "Bottom title"), (r"\bBig centered title\b", "Big centered")],
    "composition": [(r"\bOff-center subject\b", "Off-center"), (r"\bGrid / collage\b", "Collage")],
    "medium": [(r"\bTalking heads\b", "Two or more"), (r"\bBig centered title\b", "Big centered")],
    "look": [(r"\bsilhouette of a person\b", "silhouette"), (r"\bcouple embracing\b", "embrace")],
    "objects": [(r"\bsilhouette of a person\b", "silhouette"), (r"\bcouple embracing\b", "embrace")],
}


def shorten(axis: str, name: str) -> str:
    for a, b in RULES.get(axis, []):
        name = re.sub(a, b, name)
    return name


def shorten_all(names: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {}
    for axis, rows in names.items():
        short = [shorten(axis, r) for r in rows]
        dup = {s for s in short if short.count(s) > 1}
        out[axis] = [r if s in dup else s for r, s in zip(rows, short)]
    return out
