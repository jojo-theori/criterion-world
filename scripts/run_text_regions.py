"""텍스트 영역 파이프라인을 한 번에 돌린다. 이미 처리한 커버는 건너뛰므로 posters/ 에 커버를 더 넣고 다시 실행하면 된다.

  1. 50% 축소본 만들기            out/scaled/s50/
  2. Vision (원본, 50%)           data/regions/vision, data/regions/vision@s50
  3. PaddleOCR 검출 다각형         data/regions/paddle-det-640
  4. 합치기                       data/regions/combined   <- 최종 결과
  5. 평가                         out/eval/

사용
  .venv/bin/python scripts/run_text_regions.py            # 새 커버만
  .venv/bin/python scripts/run_text_regions.py --force    # 전부 다시
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "bin" / "python")
POSTERS = ROOT / "posters"
REGIONS = ROOT / "data" / "regions"
S50 = ROOT / "out" / "scaled" / "s50"


def run(*cmd: str) -> None:
    print("$", " ".join(Path(c).name if "/" in c else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    force = ["--force"] if "--force" in sys.argv else []

    S50.mkdir(parents=True, exist_ok=True)
    for f in sorted(POSTERS.glob("*.jpg")):
        dest = S50 / f.name
        if force or not dest.exists():
            im = Image.open(f)
            im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(dest, quality=90)

    vision = str(ROOT / "scripts" / "engines" / "vision.swift")
    run("swift", vision, str(POSTERS), str(REGIONS / "vision"), *force)
    run("swift", vision, str(S50), str(REGIONS / "vision@s50"), *force)
    run(PY, "-W", "ignore", str(ROOT / "scripts" / "engines" / "paddleocr_engine.py"),
        str(POSTERS), str(REGIONS / "paddle-det-640"), "--variant", "det-640", *force)
    run(PY, str(ROOT / "scripts" / "engines" / "combine.py"), str(REGIONS / "paddle-det-640"),
        str(REGIONS / "combined"), str(REGIONS / "vision@s50"), str(REGIONS / "vision"))
    run(PY, str(ROOT / "scripts" / "evaluate.py"), str(REGIONS / "vision"), str(REGIONS / "vision@s50"),
        str(REGIONS / "paddle-det-640"), str(REGIONS / "combined"))


if __name__ == "__main__":
    main()
