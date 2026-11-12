"""박스 세트 쪽 spine 번호의 커버를 posters_box/ 에 받는다. 이미 있는 커버는 건너뛴다.

결정 전 시험 수집(2026-11-11). DB(films)에는 넣지 않고, 메인 파이프라인(posters/)과 섞지 않는다.
대상: build_film_db 가 박스 세트로 걸러낸 spine 중 Wikidata 에 Criterion ID 가 있는 것.
경로와 요청 간격은 fetch_covers.py 와 같다(그 함수를 그대로 쓴다).

기록: data/raw/box_cover_sources.csv (spine, criterion_id, title, box_set, snapshot, image_url, width, height, status)
      image_url 이 같은 행끼리는 커버 하나를 나눠 쓰는 패키지다(예: Gates of Heaven / Vernon, Florida).

사용
  .venv/bin/python scripts/fetch_box_covers.py
"""

from __future__ import annotations

import csv
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_film_db as db  # noqa: E402
from cover_checklist import filename  # noqa: E402
from fetch_covers import WORKERS, fetch_one  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "posters_box"
LOG = ROOT / "data" / "raw" / "box_cover_sources.csv"
FIELDS = ["spine", "criterion_id", "title", "box_set", "snapshot", "image_url", "width", "height", "status"]


def box_names() -> dict[int, str]:
    """film spine -> 박스 세트 이름 (Wikipedia 박스 세트 표)."""
    tables = pd.read_html(io.StringIO((db.RAW / "wikipedia_list.html").read_text(encoding="utf-8")))
    names: dict[int, str] = {}
    for t in (tables[1], tables[2]):
        if "Name of box set" not in t:
            continue
        for _, r in t.iterrows():
            for col in ("Spine number (of film)", "Spine"):
                if col in t:
                    s = db.parse_spine(r[col])
                    if s is not None and s not in names:
                        names[s] = db.clean(r["Name of box set"])
    return names


def targets() -> list[tuple[int, str, str]]:
    _, box = db.read_wikipedia()
    box |= set(db.BOX_SET_ONLY)
    seen, out = set(), []
    with open(db.RAW / "wikidata_criterion.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r["spine"].isdigit() or not r["criterionId"]:
                continue
            sp = int(r["spine"])
            key = (sp, r["criterionId"])
            if sp in box and key not in seen:
                seen.add(key)
                out.append((sp, r["title"], r["criterionId"]))
    return sorted(out)


def main() -> None:
    DEST.mkdir(exist_ok=True)
    names = box_names()
    todo = [(sp, t, c, DEST / filename(sp, t)) for sp, t, c in targets()]
    skip = sum(1 for *_, d in todo if d.exists())
    todo = [x for x in todo if not x[3].exists()]
    print(f"{len(todo)} to fetch, {skip} already in posters_box/, workers {WORKERS}", flush=True)
    new_log = not LOG.exists() or LOG.stat().st_size == 0
    ok = fail = 0
    with open(LOG, "a", newline="", encoding="utf-8") as log_f, ThreadPoolExecutor(WORKERS) as pool:
        log = csv.DictWriter(log_f, FIELDS)
        if new_log:
            log.writeheader()
        futs = {pool.submit(fetch_one, *x): x for x in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            spine, title, *_ = futs[fut]
            row = fut.result()
            row.update(title=title, box_set=names.get(spine, ""))
            log.writerow(row)
            log_f.flush()
            if row["status"] == "ok":
                ok += 1
                print(f"[{n}/{len(todo)}] {spine:4d} ok   {title}", flush=True)
            else:
                fail += 1
                print(f"[{n}/{len(todo)}] {spine:4d} FAIL {row['status']}  {title}", flush=True)
    print(f"ok {ok}, fail {fail}, skipped {skip} -> {DEST}")


if __name__ == "__main__":
    main()
