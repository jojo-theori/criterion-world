"""CCW Lab 서버(로컬 전용, 127.0.0.1 에만 열린다): 실험 지도 + 라벨링을 한 곳에서.

  .venv-ml/bin/python scripts/build_lab.py     # 지도 데이터가 바뀌었을 때
  python3 scripts/lab_server.py                # → http://localhost:8777

  /        Lab 지도(out/lab/index.html). 무엇으로 배치하고 무엇으로 색칠할지 바꿔 보고, Edit 로 옮기고 범주를 고친다
  /label   한 장씩 보고 키 하나로 범주를 고르는 화면(tools/label/index.html). 지금은 People v2 범주
  그림은 web/(atlas, thumbs, large)을 그대로 쓴다.

저장: data/labels/lab.json  {"labels": {축: {파일: 범주}}, "moves": {지도|배치: {파일: [x, y]}}}
      두 화면이 같은 파일을 쓴다. build_web.py 가 labels 를 모델 판정보다 앞세운다.
옛 People 라벨(data/labels/people.json, 옛 범주 none·animal·some·one·two·many)은 평가용으로 그대로 둔다.
"""

from __future__ import annotations

import json
import random
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
STORE = ROOT / "data" / "labels" / "lab.json"
LAB = ROOT / "out" / "lab" / "index.html"
LABEL_PAGE = ROOT / "tools" / "label" / "index.html"
PEOPLE = ["None", "No-face", "One", "Two", "3+", "Ambiguous"]
OLD = {"none": "None", "animal": "None", "some": "No-face", "one": "One", "two": "Two", "many": "3+"}
LOCK = threading.Lock()
STEER_LOCK = threading.Lock()
_STEER: dict = {}          # (지도, 범주, 출처, 세기, 라벨 해시) → 결과. 라벨이 바뀌면 다시 계산


def steer(q: dict) -> dict:
    """Lab 다이얼: 그 지도를 범주 라벨 쪽으로 당긴 배치(반지도 UMAP). 기준 지도에서 출발한다."""
    import hashlib
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_lab as B
    name, scheme, src, w = q["map"], q["scheme"], q.get("src", "human"), round(float(q["w"]), 2)
    spec = next((m for m in B.MAPS if m[0] == name), None)
    if not spec or scheme not in B.SCHEMES:
        return {"error": "unknown map or scheme"}
    files = B.lab_files()
    lab = load()["labels"]
    if src == "human":
        cats = B.human_cats(files, scheme, lab)
    else:
        mine = lab.get(scheme, {})
        cats = [mine.get(n) or c for n, c in zip(files, B.model_cats(files)[scheme])]
    labeled = sum(1 for c in cats if c)
    if labeled < 10:
        return {"error": f"only {labeled} labeled covers"}
    key = (name, scheme, src, w, hashlib.md5(json.dumps(cats).encode()).hexdigest())
    with STEER_LOCK:
        if key not in _STEER:
            p = B.EMB / spec[2]
            sp = B.map_space(spec[2], files) if p.exists() else None
            if sp is None:                            # 지도 파일이 없거나 빠진 커버가 있으면(build_lab 이 건너뛴 지도)
                return {"error": "map not built for these covers, run build_lab.py"}
            X, metric = sp
            cache = np.load(B.OUT / "umap_cache.npz", allow_pickle=True)
            base = X if metric == "euclidean" else cache.get(f"{spec[2]}|{p.stat().st_mtime_ns}|{len(files)}")      # 자리가 정해진 지도는 그 자리가 기준
            if base is None:
                return {"error": "run build_lab.py first"}
            base = (base - base.min(0)) / (base.max(0) - base.min(0))
            q01 = B.steer_layout(X, base, cats, w, metric)
            g, cols, rows = B.grid(q01)
            _STEER[key] = {"xy": np.round(q01, 4).tolist(), "pos": g.tolist(), "cols": cols, "rows": rows, "labeled": labeled}
    return _STEER[key]


def load() -> dict:
    d = json.loads(STORE.read_text()) if STORE.exists() else {}
    d.setdefault("labels", {})
    d.setdefault("moves", {})
    return d


def save(d: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(d, ensure_ascii=False, indent=0, sort_keys=True))


def people_queue() -> list[dict]:
    """아직 Lab 라벨이 없는 커버. 먼저 옛 사람 라벨과 묘사 판정이 엇갈린 것, 그다음 Ambiguous·No-face, 그다음 무작위"""
    v2 = json.loads((A / "people_v2.json").read_text())
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    old_p = ROOT / "data" / "labels" / "people.json"
    old = {k: OLD.get(v["label"]) for k, v in json.loads(old_p.read_text()).items()} if old_p.exists() else {}
    done = load()["labels"].get("People", {})
    rng = random.Random(7)

    def prio(n: str) -> float:
        c = v2[n]["cat"]
        if n in old and old[n] != c:
            return 0 + rng.random()
        if c in ("Ambiguous", "No-face"):
            return 1 + rng.random()
        return 2 + rng.random()
    names = sorted((n for n in v2 if n not in done), key=prio)
    return [{"file": n, "spine": int(n[:4]), "model": v2[n]["cat"], "old": old.get(n), "reason": v2[n].get("reason", ""),
             "line": cap.get(n, {}).get("People", "")} for n in names]


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "web"), **k)

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj, ensure_ascii=False).encode(), "application/json", code)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/map"):
            if not LAB.exists():
                return self._send(b"Run .venv-ml/bin/python scripts/build_lab.py first.", "text/plain; charset=utf-8", 404)
            return self._send(LAB.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/label":
            return self._send(LABEL_PAGE.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/api/edits":
            return self._json(load())
        if u.path == "/api/steer":
            try:
                return self._json(steer({k: v[0] for k, v in parse_qs(u.query).items()}))
            except Exception as e:   # 계산 실패는 화면에 이유를 보인다
                return self._json({"error": f"{type(e).__name__}: {e}"[:200]})
        if u.path == "/api/queue":
            scheme = parse_qs(u.query).get("scheme", ["People"])[0]
            d = load()["labels"].get(scheme, {})
            counts = {c: sum(1 for v in d.values() if v == c) for c in PEOPLE}
            return self._json({"queue": people_queue(), "done": len(d), "counts": counts, "cats": PEOPLE})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with LOCK:
            d = load()
            if u == "/api/edits":                       # Lab 지도: 편집 내용을 통째로
                d = {"labels": body.get("labels", {}), "moves": body.get("moves", {})}
            elif u == "/api/label" and body.get("cat") in PEOPLE:
                d["labels"].setdefault(body.get("scheme", "People"), {})[body["file"]] = body["cat"]
            elif u == "/api/undo":
                d["labels"].get(body.get("scheme", "People"), {}).pop(body.get("file"), None)
            else:
                return self._json({"error": "bad request"}, 400)
            d["t"] = int(time.time())
            save(d)
        return self._json({"ok": True, "done": len(d["labels"].get("People", {}))})


if __name__ == "__main__":
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777          # 다른 포트: python3 scripts/lab_server.py 8778
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"CCW Lab → http://localhost:{port}  (label: /label)", flush=True)
    srv.serve_forever()
