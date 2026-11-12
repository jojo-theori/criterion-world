"""criterion.com 영화 페이지(Internet Archive 보관본)에서 Film Info 와 판매 형식을 뽑는다.

뽑는 것: 나라, 연도, 상영 시간(분), 흑백/컬러, 화면비, 언어(Film Info 목록), 판매 형식(4K UHD / Blu-ray / DVD).
커버를 받은 보관본 시점(cover_sources.csv, box_cover_sources.csv 의 snapshot)에 가장 가까운 것을 쓴다.
받은 HTML 은 data/raw/criterion_pages/<spine>.html 에 남겨 다시 받지 않는다(공개하지 않는다).

출력: data/raw/criterion_meta.json  {spine: {runtime, color, aspect, language, formats, ...}}

사용
  .venv/bin/python scripts/fetch_criterion_meta.py            # 없는 것만
  .venv/bin/python scripts/fetch_criterion_meta.py --parse    # 받은 HTML 만 다시 읽기
"""

from __future__ import annotations

import csv
import html as H
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_covers import PAUSE, WORKERS, get  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PAGES = RAW / "criterion_pages"
OUT = RAW / "criterion_meta.json"


def snaps() -> dict[int, str]:
    out: dict[int, str] = {}
    for f in ("cover_sources.csv", "box_cover_sources.csv"):
        p = RAW / f
        if p.exists():
            for r in csv.DictReader(open(p, encoding="utf-8")):
                if r.get("status") == "ok" and r.get("snapshot"):
                    out[int(r["spine"])] = r["snapshot"]
    return out


def page_url(cid: str) -> str:
    return f"https://www.criterion.com/boxsets/{cid[4:]}" if cid.startswith("box:") else f"https://www.criterion.com/films/{cid}"


def fetch(spine: int, cid: str, ts: str) -> str:
    dest = PAGES / f"{spine}.html"
    if dest.exists():
        return "cached"
    data = get(f"https://web.archive.org/web/{ts}id_/{page_url(cid)}")
    time.sleep(PAUSE)
    if not data or b"film-meta-list" not in data and b"purchase-option" not in data:
        return "no meta"
    dest.write_bytes(data)
    return "ok"


def text(s: str) -> str:
    return re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse(h: str) -> dict:
    d: dict = {}
    m = re.search(r'<ul class="film-meta-list">([\s\S]*?)</ul>', h)
    if m:
        lis = re.findall(r"<li([^>]*)>([\s\S]*?)</li>", m.group(1))
        for attr, body in lis:
            t = text(body)
            if not t or t.startswith("Spine"):
                continue
            if "countryOfOrigin" in attr:
                d.setdefault("country", t)
            elif "inLanguage" in attr:
                d["language"] = t
            elif "duration" in attr or re.fullmatch(r"\d+ minutes?", t):
                n = re.search(r"\d+", t)
                if n:
                    d["runtime"] = int(n.group())
            elif "aspect" in attr or re.fullmatch(r"[\d.]+:1", t):
                d["aspect"] = t
            elif re.fullmatch(r"\d{4}", t):
                d["year"] = int(t)
            elif re.search(r"black|white|color|colour", t, re.I):
                d["color"] = t.replace("&", "&")
    variants = set(re.findall(r'item_variant&quot;:&quot;([^&]+)', h)) | {text(x) for x in re.findall(
        r'<span class="item">([^<]+)</span>', h)}
    fm = set()
    for v in variants:
        if "4K" in v:
            fm.add("4K UHD")
        if re.search(r"blu", v, re.I):
            fm.add("Blu-ray")
        if "DVD" in v:
            fm.add("DVD")
    if fm:
        d["formats"] = [f for f in ("4K UHD", "Blu-ray", "DVD") if f in fm]
    return d


def main() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    rows = [(s, c) for s, c in con.execute("select spine, criterion_id from films where criterion_id != ''")]
    sn = snaps()
    if "--parse" not in sys.argv:
        todo = [(s, c, sn[s]) for s, c in rows if s in sn and not (PAGES / f"{s}.html").exists()]
        print(f"{len(todo)} pages to fetch", flush=True)
        with ThreadPoolExecutor(WORKERS) as pool:
            futs = {pool.submit(fetch, *x): x for x in todo}
            for n, f in enumerate(as_completed(futs), 1):
                s = futs[f][0]
                try:
                    st = f.result()
                except Exception as e:  # noqa: BLE001
                    st = f"fail {e}"
                if st != "ok" or n % 50 == 0:
                    print(f"[{n}/{len(todo)}] {s} {st}", flush=True)
    out = {}
    for p in sorted(PAGES.glob("*.html")):
        out[int(p.stem)] = parse(p.read_text(encoding="utf-8", errors="ignore"))
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    n = len(out)
    for k in ("runtime", "color", "aspect", "language", "formats"):
        print(f"{k:9s} {sum(1 for d in out.values() if d.get(k))}/{n}")
    print("->", OUT)


if __name__ == "__main__":
    main()
