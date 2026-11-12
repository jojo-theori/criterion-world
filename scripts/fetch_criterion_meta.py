"""criterion.com 영화 페이지(Internet Archive 보관본)에서 Film Info 와 판매 형식을 뽑는다.

뽑는 것: 나라, 연도, 상영 시간(분), 흑백/컬러, 화면비, 언어(Film Info 목록), 판매 형식(4K UHD / Blu-ray / DVD).
커버를 받은 보관본 시점(cover_sources.csv, box_cover_sources.csv 의 snapshot)에 가장 가까운 것을 쓴다.
받은 HTML 은 data/raw/criterion_pages/<spine>.html 에 남겨 다시 받지 않는다(공개하지 않는다).

출력: data/raw/criterion_meta.json  {spine: {runtime, color, aspect, language, formats, ...}}

사용
  .venv/bin/python scripts/fetch_criterion_meta.py            # 없는 것만
  .venv/bin/python scripts/fetch_criterion_meta.py --parse    # 받은 HTML 만 다시 읽기
  .venv/bin/python scripts/fetch_criterion_meta.py --fill     # 화면비가 없는 영화만 최신 보관본으로 받아 기존 결과에 합친다

--fill 은 archive.org 가 동시 요청에 Connection refused 로 한동안 막아서 한 장씩 FILL_PAUSE 간격으로 받는다.
커버 보관본 시점이 없거나 그 시점 페이지에 Film Info 가 없으면, 미래 날짜(2099)로 요청해 가장 최근 보관본을 쓴다.
"""

from __future__ import annotations

import csv
import html as H
import json
import re
import sqlite3
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_covers import PAUSE, WORKERS, get  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PAGES = RAW / "criterion_pages"
OUT = RAW / "criterion_meta.json"
FILL_PAUSE = 4.0
# 화면비 한 개 또는 여러 개. 판본 주석(괄호)과 구분자(, ; /)를 허용. Criterion 표기 오타 '1:37:1' 도 받는다
RATIO_LIST = re.compile(r"\d[\d.:]*:1(\s*\([^)]*\))?(\s*[,;/]\s*\d[\d.:]*:1(\s*\([^)]*\))?)*")


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
            elif "aspect" in attr or RATIO_LIST.fullmatch(t):
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


def slug_url(cid: str) -> str | None:
    if cid.startswith("box:"):
        return None
    q = urllib.parse.urlencode({"url": f"www.criterion.com/films/{cid}-", "matchType": "prefix", "filter": "statuscode:200",
                                "output": "json", "fl": "timestamp,original", "limit": "-3"})
    rows = json.loads(get(f"https://web.archive.org/cdx/search/cdx?{q}") or b"[]")
    time.sleep(FILL_PAUSE)
    return f"https://web.archive.org/web/{rows[-1][0]}id_/{rows[-1][1]}" if len(rows) > 1 else None


def fill(rows: list[tuple[int, str]], sn: dict[int, str]) -> None:
    """화면비가 없는 단독 영화만 받아 OUT 에 합친다. 이미 있는 항목은 건드리지 않는다(로컬에 페이지가 없어도 된다)."""
    out = {int(k): v for k, v in json.loads(OUT.read_text()).items()} if OUT.exists() else {}
    con = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    films = {s for (s,) in con.execute("select spine from films where kind != 'box'")}
    todo = [(s, c) for s, c in rows if s in films and not out.get(s, {}).get("aspect")]
    print(f"{len(todo)} films without aspect", flush=True)
    fails = 0
    for n, (s, cid) in enumerate(todo, 1):
        dest = PAGES / f"{s}.html"
        h = dest.read_text(encoding="utf-8", errors="ignore") if dest.exists() else ""
        if "film-meta-list" not in h:
            h = ""
            urls = [f"https://web.archive.org/web/{ts}id_/{page_url(cid)}" for ts in ([sn[s]] if s in sn else []) + ["2099"]]
            urls.append("slug")   # 짧은 주소로 보관된 적이 없으면 제목이 붙은 주소(/films/<id>-<slug>)의 최신 보관본
            for u in urls:
                ts = u.split("/")[4] if u != "slug" else "slug"
                try:
                    if u == "slug":
                        u = slug_url(cid)
                        if not u:
                            break
                    data = get(u) or b""
                except Exception as e:  # noqa: BLE001
                    print(f"  {s} {ts} fail {e}", flush=True)
                    data = b""
                    if not re.search(r"HTTP Error 40[34]", str(e)):   # 그 페이지만 막힌 것(403·404)은 차단으로 세지 않는다
                        fails += 1
                time.sleep(FILL_PAUSE)
                if b"film-meta-list" in data:
                    dest.write_bytes(data)
                    h = data.decode("utf-8", errors="ignore")
                    fails = 0
                    break
            if fails >= 5:
                print("  연달아 실패해 멈춘다. 나중에 다시 실행하면 이어서 받는다", flush=True)
                break
        if h:
            out[s] = {**out.get(s, {}), **parse(h)}
        print(f"[{n}/{len(todo)}] {s} {out.get(s, {}).get('aspect', '-')}", flush=True)
    OUT.write_text(json.dumps({str(k): out[k] for k in sorted(out)}, ensure_ascii=False))
    print(f"aspect {sum(1 for d in out.values() if d.get('aspect'))}/{len(out)} -> {OUT}")


def main() -> None:
    PAGES.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    rows = [(s, c) for s, c in con.execute("select spine, criterion_id from films where criterion_id != ''")]
    sn = snaps()
    if "--fill" in sys.argv:
        fill(rows, sn)
        return
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
