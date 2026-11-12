"""커버 수집 체크리스트를 만든다: data/cover_checklist.csv

열: spine, title, year, directors, filename(저장할 이름), have(posters/ 에 있으면 yes)
같은 영화당 최신판 커버 하나만 모은다. 파일 이름은 <spine 네 자리>_<제목 slug>.jpg

사용
  .venv/bin/python scripts/cover_checklist.py
"""

from __future__ import annotations

import csv
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def slug(title: str) -> str:
    s = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "-", s).strip("-")


def filename(spine: int, title: str) -> str:
    return f"{spine:04d}_{slug(title)}.jpg"


def main() -> None:
    have = {int(m.group()) for p in (ROOT / "posters").glob("*.jpg") if (m := re.match(r"\d+", p.name))}
    with open(ROOT / "data" / "films.csv", encoding="utf-8") as f:
        films = list(csv.DictReader(f))
    out = ROOT / "data" / "cover_checklist.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["spine", "title", "year", "directors", "filename", "have"])
        for r in films:
            spine = int(r["spine"])
            w.writerow([spine, r["title"], r["year"], r["directors"], filename(spine, r["title"]),
                        "yes" if spine in have else ""])
    print(f"{out}  ({len(have)}/{len(films)} 확보)")


if __name__ == "__main__":
    main()
