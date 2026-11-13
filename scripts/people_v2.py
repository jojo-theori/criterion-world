"""People 범주 v2: None · No-face · One · Two · 3+ · Ambiguous (+ 태그 repeated, animal, 이유)

토큰을 더 쓰지 않고 기존 판정(people_from_captions.json: 묘사의 People 줄을 Sonnet 이 읽어 정한 label·n·ambiguous)을 옮긴 뒤,
People 줄의 낱말로 새 기준을 보정한다.
  - 사람은 그림이든 사진이든 같게 센다. 얼굴이 하나라도 보이면 얼굴 보이는 사람 수로(One · Two · 3+).
  - 사람이 분명한데 얼굴이 전혀 안 보이면 No-face(뒷모습, 손·다리만, 얼굴 없는 실루엣).
  - 사람인지부터 불분명하면 Ambiguous(조각상, 마네킹, 인형, 가면, 그림자 …), 이유를 reason 에.
  - 같은 사람이 여러 번 나오면 보이는 수대로 세고(세 번 이상이면 3+) repeated 태그. 동물만 있으면 None 에 animal 태그.
사람 라벨(data/labels/people.json)도 같은 범주로 옮겨 human 에 둔다(some → No-face, animal → None).

출력: data/analysis/people_v2.json {파일: {cat, n, repeated, animal, reason, human?}}
사용: python3 scripts/people_v2.py
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
CATS = ["None", "No-face", "One", "Two", "3+", "Ambiguous"]
BASE = {"none": "None", "animal": "None", "some": "No-face", "one": "One", "two": "Two", "many": "3+"}

NOFACE = re.compile(r"from behind|(?:back|backs) (?:is |are )?(?:turned )?to(?:ward)? (?:the )?(?:viewer|camera|us)|seen from (?:the )?(?:back|behind|rear)|"
                    r"rear view|back view|faceless|no (?:visible )?faces?\b|without (?:a |any )?faces?|headless|"
                    r"faces? (?:is |are )?(?:hidden|obscured|not visible|unseen|turned away|out of frame|cropped|cut off|concealed|covered)", re.I)
AMBIG = {"statue": r"statue|sculpture|bust\b|monument", "mannequin": r"mannequin|dummy", "doll": r"\bdolls?\b|puppet|marionette",
         "mask": r"\bmask", "shadow": r"shadow", "skeleton": r"skeleton|skull", "robot": r"robot|android|cyborg",
         "ghost": r"ghost|spectral|phantom"}
REPEAT = re.compile(r"same (?:man|woman|person|face|figure|boy|girl|child|actor|actress|portrait|head)|repeated|repetition|"
                    r"duplicated|multiple (?:copies|times|versions)|identical (?:faces|figures|copies)|cloned", re.I)
ANIMAL = re.compile(r"\b(?:dog|cat|horse|bird|fish|cow|bull|wolf|lion|tiger|bear|monkey|ape|gorilla|snake|insect|animal|creature|"
                    r"crow|owl|deer|pig|sheep|goat|rabbit|fox|elephant|dinosaur|monster)s?\b", re.I)


MANY = re.compile(r"\b(?:three|four|five|six|seven|eight|nine|ten|eleven|twelve|dozens?|eighteen|twenty|several|many|multiple)\b|"
                  r"\d+\s*x\s*\d+|grid|contact sheet|strip of|row of|rows of", re.I)


def judge(text: str, old: dict) -> dict:
    cat = BASE[old["label"]]
    n = int(old.get("n") or 0)
    out = {"cat": cat, "n": n, "repeated": bool(REPEAT.search(text)), "animal": old["label"] == "animal", "reason": ""}
    if cat == "None" and ANIMAL.search(text) and old["label"] != "none":
        out["animal"] = True
    hits = [k for k, p in AMBIG.items() if re.search(p, text, re.I)]
    if out["repeated"] and cat in ("One", "Two", "No-face") and MANY.search(text):
        out["cat"], out["reason"] = "3+", "same person repeated"
    elif old.get("ambiguous") and hits and cat in ("One", "No-face", "None"):
        out["cat"], out["reason"] = "Ambiguous", ",".join(hits)
    elif cat == "One" and NOFACE.search(text):
        out["cat"], out["reason"] = "No-face", "no face visible"
    elif cat == "No-face":
        out["reason"] = "body part" if re.search(r"hand|leg|feet|foot|arm|torso|eye|lips|mouth", text, re.I) else "no face visible"
    return out


def main() -> None:
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    old = json.loads((A / "people_from_captions.json").read_text())
    human = {k: BASE[v["label"]] for k, v in json.loads((ROOT / "data" / "labels" / "people.json").read_text()).items()}
    out = {}
    for n in sorted(old):
        r = judge(cap[n]["People"], old[n])
        if n in human:
            r["human"] = human[n]
        out[n] = r
    (A / "people_v2.json").write_text(json.dumps(out, ensure_ascii=False, indent=0))
    print("all  ", collections.Counter(r["cat"] for r in out.values()))
    print("tags  repeated", sum(r["repeated"] for r in out.values()), "animal", sum(r["animal"] for r in out.values()))
    both = [n for n in out if "human" in out[n]]
    agree = sum(out[n]["cat"] == out[n]["human"] for n in both)
    print(f"agree {agree}/{len(both)} = {agree / len(both):.3f}")
    print("rows = my label, cols = caption")
    print(" " * 10 + "".join(f"{c:>10}" for c in CATS))
    for a in CATS:
        print(f"{a:>10}" + "".join(f"{sum(1 for n in both if out[n]['human'] == a and out[n]['cat'] == b):10d}" for b in CATS))


if __name__ == "__main__":
    main()
