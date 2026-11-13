"""에이전트가 나눠 쓴 묘사 조각(out*.json)을 data/analysis/captions_sonnet_v3.json 에 합친다. 여덟 줄이 다 있는 것만.

사용: python3 scripts/merge_captions.py <조각 폴더>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "analysis" / "captions_sonnet_v3.json"
FIELDS = ["People", "Composition", "Medium", "Typography", "Color and light", "Story it suggests", "Visual idea", "Feels like"]


def main() -> None:
    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    before, bad = len(d), []
    for f in sorted(Path(sys.argv[1]).glob("out*.json")):
        try:
            part = json.loads(f.read_text())
        except json.JSONDecodeError:
            print("broken", f.name)
            continue
        for k, v in part.items():
            if all(isinstance(v.get(x), str) and v[x].strip() for x in FIELDS):
                d[k] = {x: v[x].strip() for x in FIELDS}
            else:
                bad.append(k)
    OUT.write_text(json.dumps(dict(sorted(d.items())), ensure_ascii=False, indent=1))
    total = len(list((ROOT / "posters").glob("*.jpg")))
    print(f"{before} -> {len(d)} / {total}", "incomplete:", bad)


if __name__ == "__main__":
    main()
