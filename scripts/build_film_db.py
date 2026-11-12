"""크라이테리온 컬렉션 메타데이터 DB를 만든다. 단독 발매작 + 박스 세트(2026-11-12 사용자 결정으로 포함).

출처
  - Wikipedia "List of films in the Criterion Collection" (CC BY-SA): 단독 발매 표가 기준 목록
  - Wikidata (CC0): spine 번호(P12279)로 연결해 Wikidata/IMDb/TMDB/Criterion ID 를 붙이고 빈 값을 채운다

박스 세트도 넣는다(kind, box_set 열).
  - kind=film, box_set=<이름>: 박스 안에서 자기 spine 번호를 받은 영화 (Wikipedia 박스 세트 표의 'Spine number (of film)')
  - kind=box: 박스 자체 (표의 'Spine'). 제목 = 박스 이름, 연도 = 수록작 중 가장 이른 해
    criterion_id 는 "box:<id-slug>" 형식. Internet Archive 의 criterion.com/boxsets/ 주소 목록(data/raw/boxsets_cdx.txt)과 이름으로 맞춘다
  - spine 번호 없이 박스에만 든 영화(두 번째 박스 표)는 넣지 않는다

결과
  - data/criterion.sqlite  (films 테이블)
  - data/films.csv

사용
  .venv/bin/python scripts/build_film_db.py            # data/raw 의 캐시 사용
  .venv/bin/python scripts/build_film_db.py --refresh  # 두 출처를 다시 받아온다
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sqlite3
import sys
import unicodedata
import urllib.parse
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
UA = "criterion-world-research/0.1 (personal non-commercial project)"
WIKIPEDIA_PAGE = "List_of_films_in_the_Criterion_Collection"

# Wikipedia 박스 세트 표에 spine 번호 없이 실려 있어 자동으로 걸러지지 않는 박스 세트 수록작
BOX_SET_ONLY = {
    188,  # Love on the Run (The Adventures of Antoine Doinel)
}


def fetch(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": UA}
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
        return r.read()


def refresh_sources() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    query = (ROOT / "scripts" / "wikidata_query.rq").read_text()
    url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query})
    (RAW / "wikidata_criterion.csv").write_bytes(fetch(url, "text/csv"))
    url = ("https://en.wikipedia.org/w/api.php?action=parse&prop=text&format=json&formatversion=2&page="
           + WIKIPEDIA_PAGE)
    html = json.loads(fetch(url))["parse"]["text"]
    (RAW / "wikipedia_list.html").write_text(html, encoding="utf-8")


# ---------- 정리 함수 ----------

FOOTNOTE = re.compile(r"\[[^\]]*\]")


def clean(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return re.sub(r"\s+", " ", FOOTNOTE.sub("", str(v))).strip()


def parse_spine(v) -> int | None:
    m = re.match(r"\d+", clean(v).split(".")[0])
    return int(m.group()) if m else None


def parse_year(v) -> int | None:
    m = re.search(r"\d{4}", clean(v))
    return int(m.group()) if m else None


def norm_title(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"^(the|a|an|la|le|les|l')\s+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm_title(a), norm_title(b)).ratio()


# ---------- 출처별 읽기 ----------

def read_wikipedia() -> tuple[list[dict], set[int]]:
    """(단독 발매작 목록, 박스 세트에 쓰인 spine 번호)"""
    tables = pd.read_html(io.StringIO((RAW / "wikipedia_list.html").read_text(encoding="utf-8")))
    films = []
    for _, r in tables[0].iterrows():
        title = clean(r["Film"])
        spine = parse_spine(r["Spine"])
        if title.endswith("*") or spine is None:   # LaserDisc 전용: spine 체계가 달라 제외
            continue
        films.append(dict(spine=spine, title=title.replace("†", "").strip(),
                          year=parse_year(r["Film Release Year"]), directors=clean(r["Director"]),
                          countries=clean(r["Country"]), laserdisc_also="†" in title))

    box_spines = set()
    for t in (tables[1], tables[2]):
        for col in ("Spine", "Spine number (of film)"):
            if col in t:
                box_spines |= {parse_spine(v) for v in t[col] if parse_spine(v) is not None}
    box_spines -= {f["spine"] for f in films}   # 단독으로도 나온 영화는 남긴다
    return films, box_spines


def read_boxes() -> tuple[list[dict], list[dict]]:
    """(박스 수록 영화 행, 박스 자체 행). Wikipedia 박스 세트 표(tables[1])."""
    t = pd.read_html(io.StringIO((RAW / "wikipedia_list.html").read_text(encoding="utf-8")))[1]
    films, boxes = [], {}
    for _, r in t.iterrows():
        name = clean(r["Name of box set"])
        title = clean(r["Included films"]).replace("†", "").rstrip("*").strip()
        year, director, country = parse_year(r["Year"]), clean(r["Director"]), clean(r.get("Country", ""))
        bs = parse_spine(r["Spine"])
        if bs is not None:
            b = boxes.setdefault(bs, dict(name=name, years=[], directors=[], countries=[]))
            if year: b["years"].append(year)
            for lst, v in ((b["directors"], director), (b["countries"], country)):
                for x in (y.strip() for y in v.split(",")):
                    if x and x not in lst: lst.append(x)
        fs = parse_spine(r["Spine number (of film)"])
        if fs is not None:
            films.append(dict(spine=fs, title=title, year=year, directors=director, countries=country,
                              laserdisc_also=False, box_set=name, kind="film"))
    packs = [dict(spine=s, title=b["name"], year=min(b["years"]) if b["years"] else None,
                  directors=", ".join(b["directors"][:6]), countries=", ".join(b["countries"]),
                  laserdisc_also=False, box_set=b["name"], kind="box") for s, b in boxes.items()]
    return films, packs


def boxset_ids() -> dict[str, str]:
    """boxsets_cdx.txt 의 criterion.com/boxsets/<id>-<slug> → {slug: "id-slug"} (같은 id 면 가장 긴 slug)."""
    path = RAW / "boxsets_cdx.txt"
    if not path.exists():
        return {}
    best: dict[str, str] = {}
    for m in re.finditer(r"boxsets/(\d+)-([a-z0-9-]+)", path.read_text()):
        i, sl = m.group(1), m.group(2)
        if len(sl) > len(best.get(i, "")):
            best[i] = sl
    return {sl: f"{i}-{sl}" for i, sl in best.items()}


NUMS = {str(i): w for i, w in enumerate(["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"])}


def small_numbers(slug_: str) -> set[str]:
    """이름 속 1–9 (숫자든 영어 단어든). 'Volume Two' 와 'Volumes One and Two', 'No. 6' 과 'No. 1' 을 가른다."""
    toks = slug_.split("-")
    return {NUMS.get(t, t) for t in toks if t in NUMS or t in NUMS.values()} - {"zero"}


def match_boxset(name: str, ids: dict[str, str]) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()).strip("-")
    s2 = s.replace("-s-", "s-")
    best, score = "", 0.0
    for sl, full in ids.items():
        if small_numbers(s) != small_numbers(sl):
            continue
        r = max(SequenceMatcher(None, s, sl).ratio(), SequenceMatcher(None, s2, sl).ratio())
        if r > score:
            best, score = full, r
    return f"box:{best}" if score >= 0.82 else ""


def read_wikidata() -> dict[int, dict]:
    out = {}
    with open(RAW / "wikidata_criterion.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            spine = parse_spine(r["spine"])
            if spine is not None:
                out[spine] = dict(qid=r["film"].rsplit("/", 1)[-1], title=r["title"], year=parse_year(r["pubYear"]),
                                  directors=r["directors"].replace("|", ", "),
                                  countries=r["countries"].replace("|", ", "),
                                  imdb=r["imdb"], tmdb=r["tmdb"], criterion_id=r["criterionId"])
    return out


# ---------- 합치기 ----------

def same_director(a: str, b: str) -> bool:
    last = lambda s: {n.strip().split(" ")[-1].lower() for n in s.split(",") if n.strip()}
    return bool(last(a) & last(b))


def attach(row: dict, d: dict) -> None:
    """Wikipedia 행에 Wikidata 값을 붙인다. Wikipedia 값이 우선이고 빈 값만 채운다."""
    row.update(year_wikidata=d["year"], wikidata_qid=d["qid"], imdb=d["imdb"], tmdb=d["tmdb"],
               criterion_id=d["criterion_id"], sources="wikipedia+wikidata")
    row["year"] = row["year"] or d["year"]
    row["directors"] = row["directors"] or d["directors"]
    row["countries"] = row["countries"] or d["countries"]


def merge(wp: list[dict], wd: dict[int, dict], box_spines: set[int]) -> tuple[list[dict], list[str]]:
    notes = []
    by_spine: dict[int, list[int]] = {}
    for i, r in enumerate(wp):
        by_spine.setdefault(r["spine"], []).append(i)

    # 같은 spine 의 Wikipedia 행 중 제목이 가장 비슷한 것에 Wikidata 를 붙인다
    match: dict[int, int] = {}
    for spine, d in wd.items():
        cands = by_spine.get(spine, [])
        if not cands:
            continue
        best = max(cands, key=lambda i: title_similarity(wp[i]["title"], d["title"]))
        if len(cands) == 1 or title_similarity(wp[best]["title"], d["title"]) >= 0.5:
            match[best] = spine

    rows = []
    for i, r in enumerate(wp):
        row = dict(r, year_wikidata=None, wikidata_qid="", imdb="", tmdb="", criterion_id="", sources="wikipedia")
        if i in match:
            d = wd[match[i]]
            attach(row, d)
            if d["year"] and r["year"] and abs(d["year"] - r["year"]) > 1:
                notes.append(f"연도 불일치 spine {r['spine']} {r['title']}: wikipedia {r['year']} / wikidata {d['year']}")
        rows.append(row)

    # Wikipedia 와 짝을 못 찾은 Wikidata spine
    #   1) Wikipedia 에 같은 영화가 다른 spine 으로 적혀 있으면(연도·감독 일치) Wikidata spine 으로 바로잡는다
    #   2) 아니면 박스 세트가 아닐 때만 Wikidata 행으로 추가한다
    matched = set(match.values())
    for spine, d in wd.items():
        if spine in matched or spine in box_spines or spine in BOX_SET_ONLY:
            continue
        orphan = [r for r in rows if r["sources"] == "wikipedia" and r["year"] == d["year"]
                  and same_director(r["directors"], d["directors"])
                  and (not d["title"] or title_similarity(r["title"], d["title"]) >= 0.85)]
        if len(orphan) == 1:
            r = orphan[0]
            notes.append(f"spine 수정 {r['title']}: wikipedia {r['spine']} -> wikidata {spine}")
            r["spine"] = spine
            attach(r, d)
        elif spine not in by_spine:
            rows.append(dict(spine=spine, title=d["title"] or d["qid"], year=d["year"], year_wikidata=d["year"],
                             directors=d["directors"], countries=d["countries"], laserdisc_also=False,
                             wikidata_qid=d["qid"], imdb=d["imdb"], tmdb=d["tmdb"],
                             criterion_id=d["criterion_id"], sources="wikidata"))

    for r in rows:
        r.setdefault("kind", "film"); r.setdefault("box_set", "")

    # 박스 세트: 수록 영화(자기 spine), 박스 자체
    films, packs = read_boxes()
    have = {r["spine"] for r in rows}
    ids = boxset_ids()
    for b in films + packs:
        if b["spine"] in have:
            continue
        row = dict(b, year_wikidata=None, wikidata_qid="", imdb="", tmdb="", criterion_id="", sources="wikipedia-box")
        if b["spine"] in wd:
            attach(row, wd[b["spine"]])
            row["sources"] = "wikipedia-box+wikidata"
        if b["kind"] == "box":
            row["criterion_id"] = match_boxset(b["title"], ids) or row["criterion_id"]
            if not row["criterion_id"]:
                notes.append(f"박스 세트 주소 못 찾음: spine {b['spine']} {b['title']}")
        rows.append(row); have.add(b["spine"])
    # 박스 표에 spine 없이 실렸지만 Wikidata 에 spine 이 있는 수록작(예: 188 Love on the Run)
    for spine in sorted(BOX_SET_ONLY | box_spines):
        if spine not in have and spine in wd:
            d = wd[spine]
            rows.append(dict(spine=spine, title=d["title"] or d["qid"], year=d["year"], year_wikidata=d["year"],
                             directors=d["directors"], countries=d["countries"], laserdisc_also=False,
                             wikidata_qid=d["qid"], imdb=d["imdb"], tmdb=d["tmdb"], criterion_id=d["criterion_id"],
                             sources="wikidata-box", kind="film", box_set="?"))
            have.add(spine)

    rows.sort(key=lambda r: (r["spine"], r["title"]))
    return rows, notes


COLUMNS = ["spine", "title", "year", "year_wikidata", "directors", "countries", "laserdisc_also",
           "wikidata_qid", "imdb", "tmdb", "criterion_id", "sources", "kind", "box_set"]


def write(rows: list[dict]) -> None:
    db_path = ROOT / "data" / "criterion.sqlite"
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("""CREATE TABLE films (
        id INTEGER PRIMARY KEY, spine INTEGER NOT NULL, title TEXT NOT NULL, year INTEGER, year_wikidata INTEGER, directors TEXT,
        countries TEXT, laserdisc_also INTEGER, wikidata_qid TEXT, imdb TEXT, tmdb TEXT, criterion_id TEXT,
        sources TEXT, kind TEXT, box_set TEXT)""")
    con.execute("CREATE INDEX films_spine ON films(spine)")
    con.executemany(f"INSERT INTO films ({','.join(COLUMNS)}) VALUES ({','.join('?' * len(COLUMNS))})",
                    [[int(r[c]) if c == "laserdisc_also" else r[c] for c in COLUMNS] for r in rows])
    con.commit()
    con.close()

    with open(ROOT / "data" / "films.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows({c: r[c] for c in COLUMNS} for r in rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    if ap.parse_args().refresh or not (RAW / "wikipedia_list.html").exists() \
            or not (RAW / "wikidata_criterion.csv").exists():
        refresh_sources()

    wp, box_spines = read_wikipedia()
    rows, notes = merge(wp, read_wikidata(), box_spines)
    write(rows)

    spines = {r["spine"] for r in rows}
    top = max(spines)
    gaps = [s for s in range(1, top + 1) if s not in spines]
    kinds = {k: sum(1 for r in rows if (r["kind"], bool(r["box_set"])) == k) for k in (("film", False), ("film", True), ("box", True))}
    print(f"rows {len(rows)}  (spine 1-{top})  단독 {kinds[('film', False)]}, 박스 수록 {kinds[('film', True)]}, 박스 자체 {kinds[('box', True)]}")
    print(f"설명 안 되는 빈 번호 {len(gaps)}개 {gaps[:20]}")
    for k in ("wikipedia+wikidata", "wikipedia", "wikidata"):
        print(f"  출처 {k}: {sum(r['sources'] == k for r in rows)}")
    print(f"메모 {len(notes)}건")
    for n in notes:
        print("  " + n)


if __name__ == "__main__":
    sys.exit(main())
