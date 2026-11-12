"""커버를 사람이 아는 범주로 나눈다. 묶음(family) 하나가 뷰어의 축 하나이고, 한 묶음 안에서 커버는 정확히 한 범주에 든다.

묶음과 범주(우선순위 순, 앞의 조건을 먼저 본다)
  People       Crowd · Couple · Talking heads · Big face · Lone figure · Figures · No one
  Type         Type only · Big centered title · Big title · Quiet type · Title on top · Title in the middle · Title at the bottom
  Composition  Grid / collage · Symmetric · Off-center subject · Silhouette · Busy frame · Close crop · Single object · Open space · Diagonal · Wide scene
               (기하 신호로 확실한 것을 먼저, 남은 커버는 SigLIP 구도 문장 z-점수가 가장 높은 범주. Other 는 두지 않는다)
  Medium       Photo · Illustration · Painting · Graphic · Typographic   (SigLIP 2 제로샷, 확신이 낮으면 둘째 신호로)

재료(모두 커버 크기 대비 비율이라 해상도와 무관하다)
  얼굴 data/analysis/faces.json(YuNet) + OWLv2 face 상자(그림 얼굴 보완) · 사람 상자 OWLv2
  글자 data/regions/combined · 레이아웃 칸·획 data/analysis/strokes_layout · 구도선·대칭·현저성 data/analysis/visual
  임베딩 data/analysis/embed

기준값은 480px 커버에서 맞췄다. 원본으로 다시 돌리면 몇 편의 범주가 바뀔 수 있어, 이전 결과가 있으면 바뀐 편수를 출력한다.

출력
  data/analysis/tropes.json   {파일: {"people": 범주, "type": …, "composition": …, "medium": …, "f": {판정에 쓴 값}}}
  out/tropes/<묶음>.jpg        범주별 접촉 인화지(확인용)
  out/tropes/index.html       범주별 전체 커버 확인 페이지(브라우저로 연다, web/thumbs 를 읽는다)

사용
  .venv-ml/bin/python scripts/build_tropes.py            # 판정 + 접촉 인화지
  .venv-ml/bin/python scripts/build_tropes.py --no-sheet # 판정만
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
POSTERS = ROOT / "posters"
A = ROOT / "data" / "analysis"
REGIONS = ROOT / "data" / "regions" / "combined"
OUT = A / "tropes.json"
SHEETS = ROOT / "out" / "tropes"
THUMBS = ROOT / "web" / "thumbs"
EMBED = A / "embed" / "siglip2-base-patch16-384.npz"
SIGLIP = "google/siglip2-base-patch16-384"

FAMILIES = {
    "people": ["Crowd", "Couple", "Talking heads", "Big face", "Lone figure", "Figures", "Some human", "No one"],
    "type": ["Type only", "Big centered title", "Big title", "Quiet type", "Title on top", "Title in the middle", "Title at the bottom"],
    "composition": ["Grid / collage", "Symmetric", "Off-center subject", "Silhouette", "Busy frame", "Close crop",
                    "Single object", "Open space", "Diagonal", "Wide scene"],
    "medium": ["Photo", "Illustration", "Painting", "Graphic", "Typographic"],
}
PERSON_LABELS = {"person", "silhouette of a person", "couple embracing", "child"}
MEDIUM_PROMPTS = {
    "Photo": ["a film poster made from a photograph", "a black and white photograph", "a film still"],
    "Illustration": ["an illustrated poster", "a pen and ink drawing", "a cartoon illustration", "a hand-drawn sketch"],
    "Painting": ["a painting", "an oil painting", "a watercolor painting"],
    "Graphic": ["a minimal graphic design poster with flat shapes", "an abstract geometric poster", "a screen print with flat colors"],
    "Typographic": ["a poster made only of large typography", "a text-only design with big letters"],
}
# 구도 문장(2026-11-12 Other 630편을 보고 정함). 판정은 composition() 에서 기하 신호와 함께 쓴다
COMP_PROMPTS = {
    "Close crop": ["an extreme close-up, tightly cropped", "a face cropped by the edges of the frame", "a detail filling the whole frame"],
    "Wide scene": ["a wide shot of a landscape with small figures", "a long shot of a street or a room", "a view of a city or a field from far away"],
    "Silhouette": ["a black silhouette against a plain background", "a figure seen in profile as a dark shape", "a shadow of a person"],
    "Busy frame": ["a repeating pattern of many similar shapes", "many repeated small figures in rows", "a dense texture of repeated elements"],
    "Collage": ["a collage of several photographs", "a grid of multiple film stills", "several images arranged side by side"],
    "Centered subject": ["a single subject in the center of the frame", "one object centered on a plain background"],
    "Off-center subject": ["a person placed on one side of the frame with empty space beside",
                           "a subject on the left edge of the picture", "a subject on the right edge of the picture"],
}
# 얼굴·사람 검출을 보완하는 신호(검출기가 놓친 극단적 클로즈업, 그림 얼굴)
CUE_PROMPTS = {
    "face_close": ["a close-up of a human face", "an extreme close-up portrait", "a large face filling the frame"],
    "people": ["people", "a person standing", "a group of people", "a drawing of a person"],
    # 얼굴 없이 몸의 일부만(손, 다리, 몸통, 발) — 사용자: 이런 포스터가 No one 에 들어가 있다
    "body_part": ["a close-up of a human hand", "a person's legs", "a human body without a face", "a naked torso", "human feet"],
}


def area(b: list[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: list[float], b: list[float]) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    u = area(a) + area(b) - inter
    return inter / u if u > 0 else 0.0


# ---------- 재료 모으기 ----------

def text_boxes(stem: str) -> list[list[float]]:
    """글자 영역을 비율 상자로. 크라이테리온 세로 띠(왼쪽 가장자리의 가늘고 긴 상자)는 뺀다."""
    p = REGIONS / f"{stem}.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    W, H = d["width"], d["height"]
    out = []
    for r in d["regions"]:
        xs, ys = [q[0] / W for q in r["poly"]], [q[1] / H for q in r["poly"]]
        b = [min(xs), min(ys), max(xs), max(ys)]
        w, h = b[2] - b[0], b[3] - b[1]
        if b[0] < 0.2 and h > 3 * w and w < 0.08:      # 세로 띠 글자
            continue
        if b[1] < 0.06 and h < 0.03:                      # 맨 위 띠 글자(THE CRITERION COLLECTION)
            continue
        out.append(b)
    return out


def image_symmetry(path: Path, tboxes: list[list[float]]) -> float:
    """글자 영역을 가린 윤곽(Sobel) 지도와 그 좌우 거울의 상관(-1–1). 레이아웃 칸보다 그림 자체의 대칭을 본다."""
    import cv2
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    g = cv2.resize(g, (160, 200), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 1.2)
    e = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    m = np.ones_like(e, bool)
    for b in tboxes:
        m[int(b[1] * 200):int(np.ceil(b[3] * 200)), int(b[0] * 160):int(np.ceil(b[2] * 160))] = False
    m[:, :12] = False                                   # 왼쪽 띠 자리
    mm = m & m[:, ::-1]
    if mm.sum() < 2000:
        return 0.0
    a, b = e[mm], e[:, ::-1][mm]
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-9))


def features(name: str, faces: dict, objects: dict) -> dict:
    stem = name[:-4]
    vis = json.loads((A / "visual" / f"{stem}.json").read_text())
    sl = json.loads((A / "strokes_layout" / f"{stem}.json").read_text())
    f: dict = {}

    # 얼굴: YuNet + 겹치지 않는 OWLv2 face(그림 얼굴)
    fb = [d["box"] for d in faces.get(name, [])]
    for o in objects.get(name, []):
        # 커버 절반보다 큰 얼굴(꽉 찬 클로즈업)은 점수가 충분할 때만(크라이테리온 로고 같은 오검출을 피한다)
        if o["label"] == "face" and o["score"] >= 0.2 and all(iou(o["box"], b) < 0.3 for b in fb) and (area(o["box"]) < 0.5 or o["score"] >= 0.3):
            fb.append(o["box"])
    fa = sorted((area(b) for b in fb), reverse=True)
    f["faces"] = len(fb)
    f["face_max"] = fa[0] if fa else 0.0
    f["face_sum"] = sum(fa)
    big = [b for b in fb if area(b) >= 0.006]
    f["big_faces"] = len(big)
    # 두 얼굴이 맞닿거나 겹치면(얼굴 너비의 1.3배 이내) 연인·두 사람 장면으로 본다
    if len(big) == 2:
        (a0, a1), wv = big, max(big[0][2] - big[0][0], big[1][2] - big[1][0])
        d = np.hypot((a0[0] + a0[2] - a1[0] - a1[2]) / 2, (a0[1] + a0[3] - a1[1] - a1[3]) / 2)
        f["faces_touch"] = bool(d < 1.3 * wv)
    else:
        f["faces_touch"] = False
    f["embrace"] = max((o["score"] for o in objects.get(name, []) if o["label"] == "couple embracing"), default=0.0)
    ys = [(b[1] + b[3]) / 2 for b in big]
    f["face_y_spread"] = (max(ys) - min(ys)) if len(ys) > 1 else 0.0
    pb = [o["box"] for o in objects.get(name, []) if o["label"] in PERSON_LABELS and o["score"] >= 0.25]
    f["person_max"] = max((area(b) for b in pb), default=0.0)
    f["persons"] = len(pb)
    hp = _PARSE.get(name, {})
    f["p_person"] = hp.get("person", 0.0)
    f["p_face"] = hp.get("face", 0.0) + hp.get("hair", 0.0)
    f["p_limbs"] = hp.get("arms", 0.0) + hp.get("legs", 0.0) + hp.get("shoes", 0.0)
    f["p_torso"] = hp.get("torso", 0.0) + hp.get("lower", 0.0)
    # 손(얼굴과 따로 잡힌 손): 점수 × 넓이의 최댓값
    f["hand"] = max((o["score"] * (0.3 + area(o["box"]) ** 0.5) for o in objects.get(name, [])
                     if o["label"] == "hand" and o["score"] >= 0.25), default=0.0)
    # 주인공: 가장 큰 얼굴 또는 사람 상자의 가로 중심과 넓이(치우친 구도 판정)
    sb = max(fb + pb, key=area, default=None)
    f["subj_cx"] = (sb[0] + sb[2]) / 2 if sb else 0.5
    f["subj_area"] = area(sb) if sb else 0.0

    # 글자
    tb = text_boxes(stem)
    f["text_area"] = sum(area(b) for b in tb)
    if tb:
        t = max(tb, key=lambda b: (b[3] - b[1]))          # 가장 큰 글자 줄 = 제목으로 본다
        f["title_h"] = t[3] - t[1]
        f["title_cx"] = (t[0] + t[2]) / 2
        f["title_cy"] = (t[1] + t[3]) / 2
        f["title_w"] = t[2] - t[0]
    else:
        f["title_h"] = f["title_cx"] = f["title_cy"] = f["title_w"] = 0.0

    # 레이아웃 칸(40×50): 좌우 거울 일치도, 면 비율
    L = sl["layout"]
    g = np.array(list(L["grid"])).reshape(50, 40)
    ink = (g == "f") | (g == "t")
    mirror = ink[:, ::-1]
    f["mirror_iou"] = float((ink & mirror).sum() / max(1, (ink | mirror).sum()))
    f["figure_share"] = L["shares"]["figure"]
    f["text_share"] = L["shares"]["text"]
    f["empty_share"] = L["shares"]["empty"]
    f["weight_cx"] = L["weight_centroid"][0]
    f["title_band"] = L["title_band"]

    # 구도선·대칭·현저성(글자 가린 그림 부분)
    art = vis["art"]
    f["diag"] = art["lines"]["diagonal_share"]
    # 길게 뻗은 가로·세로 선(칸 나눔): 폭·높이의 40% 넘는 선분 수
    segs = vis["full"]["lines"]["segments"]
    f["long_h"] = sum(1 for x0, y0, x1, y1 in segs if abs(y1 - y0) < 0.01 and abs(x1 - x0) > 0.4)
    f["long_v"] = sum(1 for x0, y0, x1, y1 in segs if abs(x1 - x0) < 0.01 and abs(y1 - y0) > 0.4 and min(x0, x1) > 0.1)
    f["img_sym"] = image_symmetry(POSTERS / name, tb)
    f["lr_sym"] = vis["full"]["lr_symmetry"]
    f["sal_cx"], f["sal_cy"] = art["saliency"]["centroid"]
    f["sal_spread"] = art["saliency"]["spread"]
    f["ink"] = sl["strokes"]["ink_share"]
    f["gray"] = vis["full"]["color"]["color_mode"] == "grayscale"
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in f.items()}


def medium_scores(names: list[str]) -> dict[str, dict[str, float]]:
    """SigLIP 2 제로샷: 범주마다 문장 여럿의 평균 글 임베딩과 커버 임베딩의 코사인을, 커버 전체에 대해 표준화(z)한다.
    코사인을 그대로 비교하면 늘 'Photo' 가 이긴다(문장마다 기본 점수가 달라서)."""
    import torch
    from transformers import AutoModel, AutoProcessor
    d = np.load(EMBED, allow_pickle=True)
    vec = dict(zip([str(x) for x in d["files"]], d["vectors"]))
    model = AutoModel.from_pretrained(SIGLIP).eval()
    proc = AutoProcessor.from_pretrained(SIGLIP)
    cls = {}
    with torch.no_grad():
        for k, ps in {**MEDIUM_PROMPTS, **CUE_PROMPTS, **COMP_PROMPTS}.items():
            t = proc(text=[f"this is {p}." for p in ps], padding="max_length", max_length=64, return_tensors="pt")
            o = model.get_text_features(**t)
            e = o if isinstance(o, torch.Tensor) else o.pooler_output
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)
            cls[k] = torch.nn.functional.normalize(e, dim=0).numpy()
    ks = list(cls)
    have = [n for n in names if n in vec]
    V = np.stack([vec[n] / (np.linalg.norm(vec[n]) + 1e-9) for n in have])
    S = V @ np.stack([cls[k] for k in ks]).T
    Z = (S - S.mean(0)) / (S.std(0) + 1e-9)
    return {n: dict(zip(ks, map(float, z))) for n, z in zip(have, Z)}


# ---------- 판정(묶음마다 우선순위 규칙, 커버당 한 범주) ----------

def people(f: dict, s: dict[str, float] | None = None) -> str:
    s = s or {}
    if f["faces"] >= 5:
        return "Crowd"
    if f["big_faces"] == 2 and (f["faces_touch"] or f["embrace"] >= 0.3):
        return "Couple"
    if 2 <= f["big_faces"] <= 4:
        return "Talking heads"
    if f["face_max"] >= 0.035 or (f["faces"] == 0 and s.get("face_close", 0) > 0.7):   # 검출기가 놓친 꽉 찬 얼굴은 SigLIP 단서로
        return "Big face"
    if f["faces"] + f["persons"] >= 1 and f["face_max"] < 0.006 and f["person_max"] < 0.12 and f["empty_share"] > 0.3:
        return "Lone figure"
    has_parse = bool(_PARSE)
    if f["faces"] + f["persons"] >= 1 or s.get("people", 0) > 0.7 or s.get("face_close", 0) > 0.5:
        if has_parse and f["faces"] == 0 and f["p_person"] < 0.01:    # 분할로 보면 사람이 거의 없다(오검출)
            return "No one"
        return "Figures"
    if has_parse and f["p_person"] >= 0.03 and f["p_face"] < 0.01:  # 얼굴 없이 몸 부위만(손, 다리, 몸통, 뒷모습)
        return "Some human"
    if (f["hand"] >= 0.15 or s.get("body_part", 0) > 0.9) and (not has_parse or f["p_person"] >= 0.01):
        return "Some human"
    return "No one"


def typ(f: dict) -> str:
    if f["text_share"] > 0.3 and f["figure_share"] < 0.15:
        return "Type only"
    if f["title_h"] >= 0.2:                 # 제목 덩어리(여러 줄이면 한 덩어리) 높이가 커버의 1/5 이상
        return "Big centered title" if abs(f["title_cx"] - 0.5) < 0.08 else "Big title"
    if f["text_area"] < 0.05:
        return "Quiet type"
    return {"top": "Title on top", "middle": "Title in the middle", "bottom": "Title at the bottom"}.get(f["title_band"], "Title in the middle")


def composition(f: dict, s: dict[str, float] | None = None) -> str:
    z = {k: (s or {}).get(k, 0.0) for k in COMP_PROMPTS}
    if (f["long_h"] + f["long_v"] >= 5 and f["long_h"] >= 1 and f["long_v"] >= 1) or z["Collage"] > 1.9:
        return "Grid / collage"
    if f["img_sym"] >= 0.55:
        return "Symmetric"
    if 0.04 < f["subj_area"] < 0.5 and abs(f["subj_cx"] - 0.5) > 0.17:
        return "Off-center subject"
    if z["Silhouette"] > 1.5:
        return "Silhouette"
    if z["Busy frame"] > 1.5:
        return "Busy frame"
    if f["face_max"] >= 0.12 or z["Close crop"] > 1.5:
        return "Close crop"
    if abs(f["sal_cx"] - 0.5) < 0.08 and f["sal_spread"] < 0.27 and f["empty_share"] > 0.35:
        return "Single object"
    if f["empty_share"] >= 0.5:
        return "Open space"
    if f["diag"] >= 0.6:
        return "Diagonal"
    if z["Wide scene"] > 0.8:
        return "Wide scene"
    best = max(["Close crop", "Wide scene", "Silhouette", "Busy frame", "Centered subject", "Off-center subject"], key=lambda k: z[k])
    return "Single object" if best == "Centered subject" else best


def medium(f: dict, s: dict[str, float] | None) -> str:
    if f["text_share"] > 0.3 and f["figure_share"] < 0.15:
        return "Typographic"
    if not s:
        return "Graphic"
    return max(MEDIUM_PROMPTS, key=lambda k: s.get(k, -9))


# ---------- 접촉 인화지 ----------

def sheet(family: str, assign: dict[str, str], per: int = 16) -> Path:
    tw, th, lw, pad = 84, 105, 220, 6
    cats = FAMILIES[family]
    im = Image.new("RGB", (lw + per * (tw + pad), len(cats) * (th + 2 * pad)), (0, 0, 0))
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
    except OSError:
        font = ImageFont.load_default()
    rng = random.Random(7)
    for r, c in enumerate(cats):
        names = sorted(n for n, v in assign.items() if v == c)
        pick = rng.sample(names, min(per, len(names)))
        y = r * (th + 2 * pad) + pad
        dr.text((10, y + th / 2 - 18), c, fill=(235, 235, 235), font=font)
        dr.text((10, y + th / 2 + 2), f"{len(names)} covers", fill=(140, 140, 140), font=font)
        for k, n in enumerate(pick):
            p = THUMBS / n
            if p.exists():
                im.paste(Image.open(p).convert("RGB").resize((tw, th)), (lw + k * (tw + pad), y))
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / f"{family}.jpg"
    im.save(out, quality=85)
    return out


def review_page(res: dict) -> Path:
    """범주별 전체 커버를 spine 순으로 늘어놓은 확인 페이지. 커버에 마우스를 올리면 번호와 제목."""
    import html
    import re
    import sqlite3
    db = sqlite3.connect(ROOT / "data" / "criterion.sqlite")
    titles = {s: t for s, t in db.execute("select spine, title from films")}
    info = {n: {"spine": int(n[:4]), "title": titles.get(int(n[:4]), n)} for n in res}
    th = SHEETS / "thumbs"          # 확인용 작은 썸네일(posters 에서 만든다, 레포에 올리지 않는 out/ 안)
    th.mkdir(parents=True, exist_ok=True)
    for n in res:
        if not (th / n).exists():
            Image.open(POSTERS / n).convert("RGB").resize((112, 140), Image.LANCZOS).save(th / n, quality=80)
    body = []
    for fam, cats in FAMILIES.items():
        body.append(f'<h2 id="{fam}">{fam.capitalize()}</h2>')
        for c in cats:
            names = sorted((n for n, v in res.items() if v[fam] == c), key=lambda n: info.get(n, {}).get("spine", 0))
            imgs = "".join(f'<img loading="lazy" src="thumbs/{html.escape(n)}" '
                           f'title="#{info.get(n, {}).get("spine", "")} {html.escape(info.get(n, {}).get("title", n))}">' for n in names)
            body.append(f'<h3>{c} <span>{len(names)}</span></h3><div class="g">{imgs}</div>')
    nav = " ".join(f'<a href="#{f}">{f.capitalize()}</a>' for f in FAMILIES)
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Tropes review</title>'
            '<meta name="viewport" content="width=device-width, initial-scale=1"><style>'
            'body{margin:0;padding:24px 32px 80px;background:#000;color:#ebebeb;font-family:"Helvetica Neue",Helvetica,Arial,sans-serif}'
            'nav{position:sticky;top:0;padding:12px 0;background:#000;z-index:2;display:flex;gap:18px;font-size:14px}'
            'nav a{color:#8a8a8a;text-decoration:none}nav a:hover{color:#ebebeb}'
            'h2{font-weight:400;font-size:30px;margin:40px 0 8px}h3{font-weight:400;font-size:15px;margin:22px 0 8px}h3 span{color:#8a8a8a;margin-left:8px}'
            '.g{display:flex;flex-wrap:wrap;gap:4px}.g img{width:56px;height:70px;object-fit:cover}'
            f'</style></head><body><nav>{nav}</nav>{"".join(body)}</body></html>')
    SHEETS.mkdir(parents=True, exist_ok=True)
    out = SHEETS / "index.html"
    out.write_text(page, encoding="utf-8")
    return out


PARSE = A / "human_parse.json"                                # embed_more.py: 사람 부위 분할(화면 비율)
LABELS = ROOT / "data" / "labels" / "people.json"             # label_server.py: 사람이 고른 답
_PARSE: dict = {}


def human_override(lab: str, model: str, f: dict) -> str:
    """사람이 고른 사람 수 → People 범주. 모델 범주가 같은 뜻이면 그 세부(큰 얼굴/작은 인물 등)를 살린다."""
    if lab in ("none", "animal"):
        return "No one"
    if lab == "some":
        return "Some human"
    if lab == "one":
        return model if model in ("Big face", "Lone figure", "Figures") else ("Big face" if f.get("face_max", 0) >= 0.035 else "Figures")
    if lab == "two":
        return model if model in ("Couple", "Talking heads") else "Talking heads"
    if lab == "many":
        return "Crowd"
    return model


def main() -> None:
    _PARSE.update(json.loads(PARSE.read_text()) if PARSE.exists() else {})
    human = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    faces = json.loads((A / "faces.json").read_text())
    objects = json.loads((A / "objects" / "owlv2.json").read_text())
    names = sorted(p.name for p in POSTERS.glob("*.jpg") if (A / "visual" / f"{p.stem}.json").exists()
                   and (A / "strokes_layout" / f"{p.stem}.json").exists())
    feats = {n: features(n, faces, objects) for n in names}
    ms = medium_scores(names)
    res = {}
    for n in names:
        f = feats[n]
        pp = people(f, ms.get(n))
        if n in human:   # 사람이 고른 답이 모델보다 먼저
            pp = human_override(human[n]["label"], pp, f)
        res[n] = {"people": pp, "type": typ(f), "composition": composition(f, ms.get(n)), "medium": medium(f, ms.get(n)),
                  "f": f, "medium_scores": {k: round(v, 4) for k, v in (ms.get(n) or {}).items()}}
        if n in human:
            res[n]["human"] = human[n]["label"]
    if OUT.exists():
        old = json.loads(OUT.read_text())
        moved = {fam: sum(1 for n in res if n in old and old[n].get(fam) != res[n][fam]) for fam in FAMILIES}
        print("changed vs previous:", moved)
    OUT.write_text(json.dumps(res, ensure_ascii=False))
    for fam, cats in FAMILIES.items():
        cnt = {c: sum(1 for r in res.values() if r[fam] == c) for c in cats}
        print(f"{fam:12s} " + " · ".join(f"{c} {k}" for c, k in cnt.items()))
        if "--no-sheet" not in sys.argv:
            print("   ->", sheet(fam, {n: r[fam] for n, r in res.items()}))
    if "--no-sheet" not in sys.argv:
        print("review ->", review_page(res))
    print("->", OUT)


if __name__ == "__main__":
    main()
