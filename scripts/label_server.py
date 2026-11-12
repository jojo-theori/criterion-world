"""사람 라벨링 서버(로컬 전용). 포스터를 한 장씩 보여 주고 키 하나로 사람 수를 고른다(2026-11-12 사용자 요청).

  .venv-ml/bin/python scripts/label_server.py        → http://localhost:8777

화면: tools/label/people.html. 커버 이미지는 web/large 를 그대로 쓴다.
답: data/labels/people.json  {파일: {"label": none|animal|some|one|two|many, "model": 모델 판정, "t": 시각}}
순서: 모델이 틀렸을 법한 커버부터(No one · Some human 판정, 얼굴 없이 사람이 있다고 본 Figures/Lone figure — 뒷모습이 많다), 그다음 무작위.
"""

from __future__ import annotations

import json
import random
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "labels" / "people.json"
TROPES = ROOT / "data" / "analysis" / "tropes.json"
PAGE = ROOT / "tools" / "label" / "people.html"
LABELS = ["none", "animal", "some", "one", "two", "many"]


def load() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def queue() -> list[dict]:
    t = json.loads(TROPES.read_text())
    done = load()
    rng = random.Random(7)

    def prio(n: str) -> float:
        v = t[n]; f = v.get("f", {}); z = v.get("medium_scores", {})
        p = v.get("people")
        if p in ("No one", "Some human"):
            return 0 + rng.random() * 0.1 - z.get("people", 0) * 0.01
        if p in ("Figures", "Lone figure") and (f.get("faces") or 0) == 0:
            return 1 + rng.random() * 0.1
        return 2 + rng.random()
    names = sorted((n for n in t if n not in done), key=prio)
    return [{"file": n, "model": t[n].get("people"), "spine": int(n[:4])} for n in names]


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(ROOT / "web"), **k)

    def log_message(self, *a):   # 조용히
        pass

    def _json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path).path
        if u in ("/", "/people"):
            b = PAGE.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        if u == "/api/queue":
            d = load()
            return self._json({"queue": queue(), "done": len(d), "counts": {l: sum(1 for v in d.values() if v["label"] == l) for l in LABELS}})
        return super().do_GET()

    def do_POST(self):
        u = urlparse(self.path).path
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        d = load()
        if u == "/api/label" and body.get("label") in LABELS:
            d[body["file"]] = {"label": body["label"], "model": body.get("model"), "t": int(time.time())}
        elif u == "/api/undo":
            d.pop(body.get("file"), None)
        else:
            return self._json({"error": "bad request"}, 400)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(d, ensure_ascii=False, indent=0))
        return self._json({"ok": True, "done": len(d)})


if __name__ == "__main__":
    srv = ThreadingHTTPServer(("127.0.0.1", 8777), H)
    print("people labeling → http://localhost:8777", flush=True)
    srv.serve_forever()
