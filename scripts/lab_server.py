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
    srv = ThreadingHTTPServer(("127.0.0.1", 8777), H)
    print("CCW Lab → http://localhost:8777  (label: /label)", flush=True)
    srv.serve_forever()
