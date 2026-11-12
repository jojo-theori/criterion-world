"""DB 영화(wikidata_qid 가 있는 것)의 추가 정보를 Wikidata 에서 받는다.

받는 것: 상영 시간(분), 화면비, 색(흑백/컬러), 원어, 장르, 촬영, 음악, 각본, 원제, 출연(Wikidata 순서 앞쪽).
Wikidata 는 항목마다 채워진 정도가 다르다. 없는 값은 키를 두지 않는다.

SPARQL(query.wikidata.org)은 장애 때 1분에 1번으로 막힌 적이 있어(2026-11-12),
엔티티 API(wbgetentities, 50개씩)로 받고 값 항목의 영어 이름을 한 번 더 받는다.

출력: data/raw/wikidata_extra.json  {qid: {runtime, aspect, color, language, genre, dop, music, writer, original_title, cast}}

사용
  .venv/bin/python scripts/fetch_wikidata_extra.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_film_db import RAW, ROOT, fetch  # noqa: E402

OUT = RAW / "wikidata_extra.json"
API = "https://www.wikidata.org/w/api.php?"
ITEM_PROPS = {"aspect": "P2061", "color": "P462", "language": "P364", "genre": "P136",
              "dop": "P344", "music": "P86", "writer": "P58", "cast": "P161"}
KEEP = {"cast": 4, "writer": 3, "genre": 3, "language": 2, "dop": 2, "music": 2, "aspect": 2, "color": 2}
MINUTE, HOUR, SECOND = "Q7727", "Q25235", "Q11574"


def get(ids: list[str], props: str) -> dict:
    url = API + urllib.parse.urlencode({"action": "wbgetentities", "ids": "|".join(ids), "props": props,
                                        "languages": "en", "format": "json"})
    for attempt in range(6):
        try:
            return json.loads(fetch(url))["entities"]
        except Exception as e:
            print("  retry", e, flush=True)
            time.sleep(30 * (attempt + 1))
    raise SystemExit("failed")


def values(claims: dict, prop: str) -> list:
    out = []
    for c in claims.get(prop, []):
        if c.get("rank") == "deprecated":
            continue
        dv = c["mainsnak"].get("datavalue")
        if dv:
            out.append(dv["value"])
    return out


def main() -> None:
    con = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    qids = sorted({q for (q,) in con.execute("select wikidata_qid from films where wikidata_qid != ''")})
    print(f"{len(qids)} items", flush=True)
    cache = RAW / "wikidata_claims_cache.json"     # 다시 돌릴 때 claims 를 또 받지 않는다
    raw: dict[str, dict] = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [q for q in qids if q not in raw]
    for i in range(0, len(todo), 50):
        for k, e in get(todo[i:i + 50], "claims").items():
            raw[k] = {"claims": {p: e.get("claims", {}).get(p, []) for p in [*ITEM_PROPS.values(), "P2047", "P1476"]}}
        time.sleep(1)
    cache.write_text(json.dumps(raw))
    print("claims done", flush=True)

    data: dict[str, dict] = {}
    need: set[str] = set()
    for q in qids:
        cl = raw.get(q, {}).get("claims", {})
        d: dict = {}
        for name, p in ITEM_PROPS.items():
            v = [x["id"] for x in values(cl, p) if isinstance(x, dict) and "id" in x]
            if v:
                d[name] = list(dict.fromkeys(v))[:KEEP.get(name, 3)]   # 뷰어에 보일 만큼만 (이름 요청 수를 줄인다)
                need.update(d[name])
        rt = []
        for x in values(cl, "P2047"):
            amt, unit = float(x["amount"]), x.get("unit", "").rsplit("/", 1)[-1]
            rt.append(round(amt / 60 if unit == SECOND else amt * 60 if unit == HOUR else amt))
        if rt:
            d["runtime"] = rt
        ot = [x["text"] for x in values(cl, "P1476") if isinstance(x, dict) and "text" in x]
        if ot:
            d["original_title"] = list(dict.fromkeys(ot))
        data[q] = d

    need_l = sorted(need)
    print(f"{len(need_l)} labels", flush=True)
    labels: dict[str, str] = {}
    for i in range(0, len(need_l), 50):
        for k, e in get(need_l[i:i + 50], "labels").items():
            lab = e.get("labels", {}).get("en", {}).get("value")
            if lab:
                labels[k] = lab
        time.sleep(1)
    for d in data.values():
        for name in ITEM_PROPS:
            if name in d:
                d[name] = [labels[x] for x in d[name] if x in labels]
    OUT.write_text(json.dumps(data, ensure_ascii=False))
    for name in ["runtime", "original_title", *ITEM_PROPS]:
        print(f"{name:15s} {sum(1 for d in data.values() if d.get(name))}/{len(qids)}")
    print("->", OUT)


if __name__ == "__main__":
    main()
