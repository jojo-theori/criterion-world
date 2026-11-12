"""엔진별 결과를 커버마다 한 장으로 나란히 그린다.

자홍 = 읽은 영역(recognize), 청록 = 검출만 한 영역(detect)
결과: out/compare/<stem>.jpg

사용
  .venv/bin/python scripts/render_compare.py [posters 디렉터리] [엔진 디렉터리 ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
PANEL_W = 420
COLORS = {"recognize": (255, 51, 204), "detect": (0, 230, 255)}


def panel(img: Image.Image, doc: dict | None, label: str) -> Image.Image:
    s = PANEL_W / img.width
    p = img.resize((PANEL_W, int(img.height * s))).convert("RGB")
    p = Image.blend(p, Image.new("RGB", p.size, (0, 0, 0)), 0.35)
    d = ImageDraw.Draw(p)
    if doc:
        k = PANEL_W / doc["width"]
        for r in doc["regions"]:
            pts = [(x * k, y * k) for x, y in r["poly"]]
            if len(pts) >= 2:
                d.line(pts + [pts[0]], fill=COLORS.get(r["source"], (255, 255, 0)), width=2)
    bar = Image.new("RGB", (PANEL_W, 26), (20, 20, 20))
    ImageDraw.Draw(bar).text((8, 6), label, fill=(235, 235, 235), font=ImageFont.load_default())
    out = Image.new("RGB", (PANEL_W, p.height + 26))
    out.paste(bar, (0, 0))
    out.paste(p, (0, 26))
    return out


def main() -> None:
    posters = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "posters"
    dirs = [Path(a) for a in sys.argv[2:]] or sorted(p for p in (ROOT / "data" / "regions").iterdir() if p.is_dir())
    out = ROOT / "out" / "compare"
    out.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(posters.glob("*.jpg")):
        img = Image.open(img_path)
        panels = []
        for d in dirs:
            f = d / f"{img_path.stem}.json"
            doc = json.loads(f.read_text()) if f.exists() else None
            n = len(doc["regions"]) if doc else 0
            panels.append(panel(img, doc, f"{d.name}  ({n})"))
        cols = min(3, len(panels))
        rows = (len(panels) + cols - 1) // cols
        ph = max(p.height for p in panels)
        sheet = Image.new("RGB", (cols * PANEL_W + (cols - 1) * 6, rows * ph + (rows - 1) * 6), (255, 255, 255))
        for i, p in enumerate(panels):
            sheet.paste(p, ((i % cols) * (PANEL_W + 6), (i // cols) * (ph + 6)))
        sheet.save(out / f"{img_path.stem}.jpg", quality=85)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
