"""커버를 받아 posters/ 에 저장한다. 이미 있는 커버는 건너뛴다.

경로
  1. Internet Archive CDX 에서 criterion.com/films/<criterion_id> 페이지의 가장 최근 정상(200) 보관본을 찾는다
  2. 보관본 HTML 의 og:image (없으면 data-product-image-url) 가 커버 이미지 주소다
  3. 이미지는 Criterion 의 S3 (s3.amazonaws.com/criterion-production) 에서 _original 크기로 받는다
criterion.com 에는 직접 요청하지 않는다.

기록: data/raw/cover_sources.csv (spine, criterion_id, snapshot, image_url, width, height, status)

사용
  .venv/bin/python scripts/fetch_covers.py                 # 없는 커버 전부
  .venv/bin/python scripts/fetch_covers.py --spines 11-20  # 범위만
  .venv/bin/python scripts/fetch_covers.py --workers 1     # 동시 요청 수 (기본 3)
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cover_checklist import filename, slug  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
LOG = ROOT / "data" / "raw" / "cover_sources.csv"
UA = "Mozilla/5.0 (personal research script)"
PAUSE = 1.5          # archive.org 요청 사이 간격(초)
WORKERS = 3          # 동시 요청 수. archive.org 는 분당 약 15회를 넘으면 429 를 준다
FIELDS = ["spine", "criterion_id", "snapshot", "image_url", "width", "height", "status"]


def get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == tries - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if k == tries - 1:
                raise
        time.sleep(10 * 2 ** k)
    raise RuntimeError("unreachable")


def snapshots(cid: str, title: str) -> list[str]:
    if cid.startswith("box:"):   # 박스 세트 자체: criterion.com/boxsets/<id>-<slug>
        return box_snapshots(cid[4:])
    """정상 보관본 "타임스탬프|주소", 최신순.
    먼저 정확한 주소(/films/<id>-<제목 slug>)로 찾는다(약 1초). 없으면 /films/<id> 로 시작하는 주소를 전부 훑는다.
    후자는 이메일 추적 파라미터가 붙은 변형 주소가 수백 개라 30초 넘게 걸린다."""
    q = urllib.parse.urlencode([
        ("url", f"criterion.com/films/{cid}-{slug(title)}"), ("output", "json"),
        ("fl", "timestamp,original"), ("filter", "statuscode:200"), ("limit", "-5"),
    ])
    rows = json.loads(get(f"https://web.archive.org/cdx/search/cdx?{q}") or b"[]")[1:]
    if rows:
        return [f"{r[0]}|{r[1]}" for r in sorted(rows, key=lambda r: r[0], reverse=True)]
    time.sleep(PAUSE)
    q = urllib.parse.urlencode([
        ("url", f"criterion.com/films/{cid}"), ("matchType", "prefix"), ("output", "json"),
        ("fl", "timestamp,original,statuscode"), ("filter", "statuscode:200"),
        ("filter", f"original:.*/films/{cid}(-[^/?]*)?/?([?].*)?$"),
    ])
    rows = json.loads(get(f"https://web.archive.org/cdx/search/cdx?{q}") or b"[]")[1:]
    rows.sort(key=lambda r: r[0], reverse=True)
    return [f"{r[0]}|{r[1]}" for r in rows]


def box_snapshots(id_slug: str) -> list[str]:
    q = urllib.parse.urlencode([
        ("url", f"criterion.com/boxsets/{id_slug}"), ("output", "json"),
        ("fl", "timestamp,original"), ("filter", "statuscode:200"), ("limit", "-5"),
    ])
    rows = json.loads(get(f"https://web.archive.org/cdx/search/cdx?{q}") or b"[]")[1:]
    if not rows:
        bid = id_slug.split("-")[0]
        time.sleep(PAUSE)
        q = urllib.parse.urlencode([
            ("url", f"criterion.com/boxsets/{bid}"), ("matchType", "prefix"), ("output", "json"),
            ("fl", "timestamp,original"), ("filter", "statuscode:200"),
            ("filter", f"original:.*/boxsets/{bid}(-[^/?]*)?/?([?].*)?$"),
        ])
        rows = json.loads(get(f"https://web.archive.org/cdx/search/cdx?{q}") or b"[]")[1:]
    rows.sort(key=lambda r: r[0], reverse=True)
    return [f"{r[0]}|{r[1]}" for r in rows]


def cover_url(html: str) -> str | None:
    for pat in (r'property="og:image"\s+content="([^"]+)"', r'og:image"\s+content="([^"]+)"',
                r'data-product-image-url="([^"]+)"'):
        m = re.search(pat, html)
        if m and "criterion-production" in m.group(1):
            return re.sub(r"_(small|medium|large|w\d+)\.jpg$", "_original.jpg", m.group(1))
    return None


def parse_spines(arg: str | None) -> set[int] | None:
    if not arg:
        return None
    out = set()
    for part in arg.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def fetch_one(spine: int, title: str, cid: str | None, dest: Path) -> dict:
    row = dict(spine=spine, criterion_id=cid or "", snapshot="", image_url="", width="", height="", status="")
    try:
        if not cid:
            raise LookupError("no criterion_id")
        snaps = snapshots(cid, title)
        time.sleep(PAUSE)
        url = None
        for snap in snaps[:3]:          # 최신 보관본에 커버 주소가 없으면 그 다음 것
            ts, orig = snap.split("|", 1)
            html = get(f"https://web.archive.org/web/{ts}id_/{orig}").decode("utf-8", "ignore")
            time.sleep(PAUSE)
            url = cover_url(html)
            if url:
                row["snapshot"] = ts
                break
        if not url:
            raise LookupError(f"no cover url ({len(snaps)} snapshots)")
        data = get(url)
        Image.open(io.BytesIO(data)).verify()
        im = Image.open(io.BytesIO(data))
        if im.format == "JPEG":
            dest.write_bytes(data)
        else:
            im.convert("RGB").save(dest, quality=95)
        row.update(image_url=url, width=im.width, height=im.height, status="ok")
    except Exception as e:  # noqa: BLE001
        row["status"] = f"fail: {type(e).__name__}: {e}"[:200]
    return row


def main() -> None:
    want = parse_spines(sys.argv[sys.argv.index("--spines") + 1] if "--spines" in sys.argv else None)
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else WORKERS
    con = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    films = con.execute("select spine, title, criterion_id from films order by spine").fetchall()
    todo = [(sp, t, c, POSTERS / filename(sp, t)) for sp, t, c in films if want is None or sp in want]
    skip = sum(1 for *_, d in todo if d.exists())
    todo = [x for x in todo if not x[3].exists()]
    print(f"{len(todo)} to fetch, {skip} already in posters/, workers {workers}", flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new_log = not LOG.exists()
    ok = fail = 0
    with open(LOG, "a", newline="", encoding="utf-8") as log_f, ThreadPoolExecutor(workers) as pool:
        log = csv.DictWriter(log_f, FIELDS)
        if new_log:
            log.writeheader()
        futs = {pool.submit(fetch_one, *x): x for x in todo}
        for n, fut in enumerate(as_completed(futs), 1):
            spine, title, *_ = futs[fut]
            row = fut.result()
            log.writerow(row)
            log_f.flush()
            if row["status"] == "ok":
                ok += 1
                print(f"[{n}/{len(todo)}] {spine:4d} ok   {row['width']}x{row['height']}  {row['snapshot'][:8]}  {title}", flush=True)
            else:
                fail += 1
                print(f"[{n}/{len(todo)}] {spine:4d} FAIL {row['status']}  {title}", flush=True)
    print(f"ok {ok}, fail {fail}, skipped {skip} -> {POSTERS}")


if __name__ == "__main__":
    main()
