"""텍스트 영역 엔진들을 같은 기준으로 비교한다.

정답은 손으로 그리지 않는다. 커버 파일명 앞의 spine 번호로 data/films.csv 를 찾아
  - title    : 제목 단어
  - director : 감독 이름 단어
  - brand    : CRITERION, COLLECTION
을 기대 단어로 삼는다. 커버가 늘어도 DB만 있으면 정답이 따라온다.

지표
  read     : 글자를 읽는 엔진이 기대 단어를 읽어낸 비율 (철자 유사도 0.75 이상이면 인정)
  located  : 기대 단어의 위치를 덮은 비율. 위치는 '어느 엔진이든 그 단어를 읽어낸 박스'로 정한다(합의 정답).
             아무 엔진도 못 읽은 단어는 위치를 몰라 분모에서 빠진다(unlocatable).
  extra    : 어떤 기대 단어 위치와도 겹치지 않는 영역 수. 오탐이거나 DB 밖 글자(간판, 서명 등).
  ms       : 커버당 평균 처리 시간

사용
  .venv/bin/python scripts/evaluate.py data/regions/vision data/regions/paddle-ocr-960 ...
  (인자가 없으면 data/regions 아래 전부)
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
MASK_W = 300              # 겹침 계산용 마스크 너비 (좌표는 0-1 로 정규화되어 해상도와 무관)
READ_SIM = 0.75
COVER_MIN = 0.5            # 기대 단어 박스의 이 비율 이상을 덮어야 located


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", s)


def words(s: str) -> list[str]:
    return [w for w in (norm(x) for x in re.split(r"[\s\-:,/]+", s)) if w]


def load_films() -> dict[int, dict]:
    with open(ROOT / "data" / "films.csv", encoding="utf-8") as f:
        return {int(r["spine"]): r for r in csv.DictReader(f)}


def expected(film: dict) -> list[tuple[str, str]]:
    out = [("title", w) for w in words(film["title"]) if len(w) >= 3 or w.isdigit()]
    out += [("director", w) for w in words(film["directors"]) if len(w) >= 3]
    out += [("brand", "criterion"), ("brand", "collection")]
    return list(dict.fromkeys(out))


def similar(a: str, b: str) -> bool:
    return a == b or SequenceMatcher(None, a, b).ratio() >= READ_SIM or (len(a) >= 5 and a in b)


def mask(polys: list, aspect: float) -> np.ndarray:
    """polys 는 0-1 정규화 좌표. aspect = 높이/너비"""
    w, h = MASK_W, int(MASK_W * aspect)
    img = Image.new("1", (w, h))
    d = ImageDraw.Draw(img)
    for p in polys:
        pts = [(x * w, y * h) for x, y in p]
        if len(pts) >= 3:
            d.polygon(pts, fill=1)
    return np.array(img, dtype=bool)


def load_engine(d: Path) -> dict[str, dict]:
    docs = {}
    for p in sorted(d.glob("*.json")):
        doc = json.loads(p.read_text())
        w, h = doc["width"], doc["height"]
        for r in doc["regions"]:
            r["poly"] = [(x / w, y / h) for x, y in r["poly"]]
        docs[p.stem] = doc
    return docs


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] or sorted(p for p in (ROOT / "data" / "regions").iterdir() if p.is_dir())
    engines = {d.name: load_engine(d) for d in dirs}
    films = load_films()
    stems = sorted(set().union(*[set(e) for e in engines.values()]))

    # 1) 엔진별로 읽어낸 단어와 그 박스 → 기대 단어의 합의 위치
    truth: dict[str, list] = {}            # stem -> [(group, word, [poly...])]
    for stem in stems:
        spine = int(re.match(r"\d+", stem).group())
        exp = expected(films[spine])
        locs = {w: [] for _, w in exp}
        # 축소본(이름에 @ 가 붙은 엔진)은 정답 위치를 만드는 데 쓰지 않는다
        sources = [e for n, e in engines.items() if "@" not in n] or list(engines.values())
        for e in sources:
            doc = e.get(stem)
            if not doc:
                continue
            for r in doc["regions"]:
                for tok in words(r.get("text") or ""):
                    for _, w in exp:
                        if similar(w, tok):
                            locs[w].append(r["poly"])
        truth[stem] = [(g, w, locs[w]) for g, w in exp]

    # 2) 엔진별 점수
    groups = ("title", "director", "brand")
    summary, per_cover = {}, {}
    for name, e in engines.items():
        agg = {g: [0, 0, 0, 0] for g in groups}      # read, read_total, located, locatable
        extra = ms = n = 0
        for stem in stems:
            doc = e.get(stem)
            if not doc:
                continue
            n += 1
            ms += doc.get("elapsed_ms", 0)
            aspect = doc["height"] / doc["width"]
            toks = [t for r in doc["regions"] for t in words(r.get("text") or "")]
            has_text = any(r.get("text") for r in doc["regions"])
            eng_mask = mask([r["poly"] for r in doc["regions"]], aspect)
            truth_mask = np.zeros_like(eng_mask)
            row = []
            for g, word, polys in truth[stem]:
                hit_read = any(similar(word, t) for t in toks)
                if has_text:
                    agg[g][0] += hit_read
                    agg[g][1] += 1
                if polys:
                    m = mask(polys, aspect)
                    truth_mask |= m
                    cov = (eng_mask & m).sum() / max(m.sum(), 1)
                    agg[g][2] += cov >= COVER_MIN
                    agg[g][3] += 1
                    row.append((word, hit_read, cov >= COVER_MIN))
                else:
                    row.append((word, hit_read, None))
            for r in doc["regions"]:
                m = mask([r["poly"]], aspect)
                if m.sum() and (m & truth_mask).sum() / m.sum() < 0.2:
                    extra += 1
            per_cover.setdefault(stem, {})[name] = row
        summary[name] = dict(agg=agg, extra=extra, ms=ms / max(n, 1), n=n)

    # 3) 출력
    pct = lambda a, b: f"{100 * a / b:5.1f}%" if b else "    -"
    print(f"{'engine':22} {'read title':>10} {'read dir':>9} {'read brand':>10} | "
          f"{'loc title':>9} {'loc dir':>8} {'loc brand':>9} | {'extra':>5} {'ms':>6}")
    for name, s in summary.items():
        a = s["agg"]
        print(f"{name:22} {pct(*a['title'][:2]):>10} {pct(*a['director'][:2]):>9} {pct(*a['brand'][:2]):>10} | "
              f"{pct(*a['title'][2:]):>9} {pct(*a['director'][2:]):>8} {pct(*a['brand'][2:]):>9} | "
              f"{s['extra']:5d} {s['ms']:6.0f}")

    unloc = [(stem, w) for stem, t in truth.items() for g, w, p in t if not p]
    print(f"\n아무 엔진도 못 읽은 기대 단어 {len(unloc)}개: " + ", ".join(f"{s[:2]}:{w}" for s, w in unloc))

    out = ROOT / "out" / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=int))
    lines = ["# 커버별 결과", "", "R = 읽음, L = 위치 덮음, ? = 위치 모름", ""]
    for stem, by_engine in per_cover.items():
        lines.append(f"## {stem}")
        for name, row in by_engine.items():
            cells = " ".join(f"{w}[{'R' if r else '-'}{'L' if l else ('?' if l is None else '-')}]" for w, r, l in row)
            lines.append(f"- {name}: {cells}")
        lines.append("")
    (out / "per_cover.md").write_text("\n".join(lines))
    print(f"\n자세한 결과: {out / 'per_cover.md'}")


if __name__ == "__main__":
    main()
