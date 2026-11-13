"""임베딩마다 UMAP 지도를 겹치지 않는 격자로 그려 확인한다(사이트에 넣기 전): out/gemma/maps.html

지도: Gemma 이미지 / 묘사 전체 / 묘사 + 이미지 / 묘사의 한 줄만(People, Composition, Medium, Typography, Color). 위 탭으로 바꾼다.
색칠: People(people_v2.json) · Composition · Medium · Type(묘사 그 줄의 낱말로 고른 대략적인 범주). 지도마다 기본 색칠이 있고 바꿀 수 있다.
UMAP 결과는 out/gemma/umap_cache.npz 에 둔다(임베딩 파일이 바뀌면 다시 계산).
배치: 기본은 UMAP 좌표 그대로(겹침 허용, 거리가 정확). 'Grid' 를 켜면 빈칸을 조금 둔 격자 칸에 헝가리안 배정(겹치지 않음).
조작: 마우스 휠·트랙패드 핀치 = 커서 기준 확대(최대 8배, 크게 보이면 web/large 480px 로 바꿔 끼움), 트랙패드 두 손가락·끌기 = 이동, 더블클릭 = 확대, 0 = 전체, +/− 키.
People 색칠에서 'My labels' 를 켜면 사람 라벨이 있는 545편만 사람 라벨 색.
범례를 누르면 그 범주만 밝게. 커버에 올리면 제목·범주·그 색칠의 묘사 줄.
사용: .venv-ml/bin/python scripts/preview_maps.py
"""

from __future__ import annotations

import html
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
EMB = A / "embed"
OUT = ROOT / "out" / "gemma"
MAPS = {  # 이름: (임베딩, 기본 색칠)
    "Gemma image": ("gemma2-plain.npz", "People"), "Caption (text)": ("gemma2-cap-all.npz", "People"),
    "Caption + image": ("gemma2-cap-img.npz", "People"), "People line": ("gemma2-cap-people.npz", "People"),
    "Composition line": ("gemma2-cap-composition.npz", "Composition"), "Medium line": ("gemma2-cap-medium.npz", "Medium"),
    "Typography line": ("gemma2-cap-typography.npz", "Type"), "Color line": ("gemma2-cap-color.npz", "Medium")}
PAL = ["#ffffff", "#f5d000", "#ff3b30", "#2f8cff", "#34c759", "#bf5af2", "#ff9500", "#6e6e6e"]
SCHEMES = {  # 색칠 기준: 범주 → 색. People 은 people_v2.json, 나머지는 묘사 그 줄의 낱말로 고른 범주(대략적인 확인용)
    "People": {"None": "#6e6e6e", "No-face": "#2f8cff", "One": "#ffffff", "Two": "#f5d000", "3+": "#ff3b30", "Ambiguous": "#bf5af2"},
    "Composition": dict(zip(["Grid", "Frame in frame", "Circle", "Symmetric", "Diagonal", "Open space", "Centered", "Other"], PAL)),
    "Medium": dict(zip(["Photo", "Painting", "Drawing", "Print", "Collage", "Graphic", "Other"], PAL[:6] + ["#6e6e6e"])),
    "Type": dict(zip(["Top", "Middle", "Bottom", "Vertical", "Diagonal", "Hand-drawn", "Other"], PAL[:6] + ["#6e6e6e"])),
}
COMP = [("Grid", r"grid|panels|contact sheet|tiled|rows of|strip of|mosaic|checkerboard"),
        ("Frame in frame", r"frame within|framed by|inner frame|window|doorway|keyhole|inset|border (?:frames|encloses)|picture within"),
        ("Circle", r"circle|circular|round |disc|oval|sphere|ring "), ("Symmetric", r"symmetr|mirror"),
        ("Diagonal", r"diagonal|tilted|slanted|angled"), ("Open space", r"empty space|negative space|vast|large (?:blank|empty)|mostly empty|sparse"),
        ("Centered", r"cent(?:er|re|ral|ered)")]
from medium_v2 import MED  # noqa: E402  Medium 범주 규칙은 한 곳에
TYP = [("Vertical", r"vertical|stacked letter|reads downward|rotated 90"), ("Diagonal", r"diagonal|tilted|slanted|angled"),
       ("Hand-drawn", r"hand-?drawn|hand-?lettered|handwritten|brush lettering|scrawled|painted letter"),
       ("Top", r"\b(?:top|upper)\b"), ("Bottom", r"\b(?:bottom|lower|foot)\b"), ("Middle", r"middle|cent(?:er|re)|across the")]


def first_by_priority(text: str, rules) -> str:
    for name, pat in rules:
        if re.search(pat, text, re.I):
            return name
    return "Other"


def earliest(text: str, rules) -> str:
    best, pos = "Other", 10 ** 9
    for name, pat in rules:
        m = re.search(pat, text, re.I)
        if m and m.start() < pos:
            best, pos = name, m.start()
    return best


TW, TH, GAP = 40, 50, 4          # 칸 크기(월드 단위)
SLACK = 1.12                     # 칸 수 = 편수 × SLACK (빈칸을 조금 둬서 무리 사이가 갈라져 보이게)


def thumb(n: str) -> str:
    t = OUT / "thumbs_l" / n
    if not t.exists():
        t.parent.mkdir(parents=True, exist_ok=True)
        Image.open(ROOT / "posters" / n).convert("RGB").resize((240, 298), Image.LANCZOS).save(t, quality=85)
    return f"thumbs_l/{n}"


def grid(xy: np.ndarray) -> tuple[np.ndarray, int, int]:
    n = len(xy)
    cols = math.ceil(math.sqrt(n * SLACK * TH / TW))
    rows = math.ceil(n * SLACK / cols)
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    cells = np.stack([gx.ravel() / (cols - 1), gy.ravel() / (rows - 1)], 1)
    p = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    cost = ((p[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    out = np.zeros((n, 2), int)
    out[r] = np.stack([gx.ravel()[c], gy.ravel()[c]], 1)
    return out, cols, rows


def main() -> None:
    import umap
    v2 = json.loads((A / "people_v2.json").read_text())
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    files = sorted(cap)
    cache_f = OUT / "umap_cache.npz"
    cache = dict(np.load(cache_f, allow_pickle=True)) if cache_f.exists() else {}
    maps = {}
    for name, (f, scheme) in MAPS.items():
        p = EMB / f
        if not p.exists():
            continue
        key = f"{f}|{p.stat().st_mtime_ns}"
        if key in cache:
            xy = cache[key]
        else:
            d = np.load(p, allow_pickle=True)
            idx = {str(n): i for i, n in enumerate(d["files"])}
            V = d["vectors"].astype(np.float32)[[idx[n] for n in files]]
            xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(V)
            cache[key] = xy
        g, cols, rows = grid(xy)
        q = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))     # UMAP 좌표 그대로(겹침 허용, 'Exact')
        maps[name] = {"pos": g.tolist(), "xy": np.round(q, 4).tolist(), "cols": cols, "rows": rows, "scheme": scheme}
        print(name, flush=True)
    np.savez(cache_f, **cache)
    items = []
    for n in files:
        c = cap[n]
        items.append({"src": thumb(n), "t": n[5:-4].replace("-", " "), "h": v2[n].get("human"), "r": v2[n]["reason"],
                      "k": {"People": v2[n]["cat"], "Composition": first_by_priority(c["Composition"], COMP),
                            "Medium": earliest(c["Medium"], MED), "Type": first_by_priority(c["Typography"], TYP)},
                      "l": {"People": c["People"], "Composition": c["Composition"], "Medium": c["Medium"], "Type": c["Typography"]}})
    for sc in SCHEMES:
        print(sc, dict(sorted(((k, sum(it["k"][sc] == k for it in items)) for k in SCHEMES[sc]), key=lambda x: -x[1])))
    data = json.dumps({"large": "../../web/large/", "items": items, "maps": maps, "schemes": SCHEMES,
                       "tw": TW, "th": TH, "gap": GAP}, ensure_ascii=False)
    (OUT / "maps.html").write_text(PAGE.replace("/*DATA*/", data), encoding="utf-8")
    print("->", OUT / "maps.html")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Maps</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
:root{--bg:#000;--fg:#ebebeb;--mute:#8a8a8a}
html,body{margin:0;height:100%;overflow:hidden;background:var(--bg);color:var(--fg);font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;touch-action:none}
header{position:fixed;top:0;left:0;right:0;z-index:5;display:flex;flex-wrap:wrap;align-items:center;gap:6px 18px;padding:12px 20px;font-size:13px;background:linear-gradient(#000 60%,#0000)}
header button{all:unset;cursor:pointer;color:var(--mute)}header button:hover,header button.on{color:var(--fg)}
.sp{flex:1}#schemes{color:var(--mute)}#schemes button.on{color:var(--fg);text-decoration:underline;text-underline-offset:4px}.key{display:inline-flex;flex-wrap:wrap;gap:4px 14px}.key button{display:inline-flex;align-items:center;gap:6px}.key i{width:9px;height:9px;border-radius:50%;display:inline-block}
.key button.off{opacity:.35}
#vp{position:fixed;inset:0;cursor:grab}#vp.drag{cursor:grabbing}
#stage{position:absolute;left:0;top:0;transform-origin:0 0}
#stage img{position:absolute;object-fit:cover;outline-width:var(--ow,2px);outline-style:solid;outline-offset:0;transition:left .6s cubic-bezier(.2,.8,.2,1),top .6s cubic-bezier(.2,.8,.2,1),opacity .2s}
#stage img.dim{opacity:.12}#stage img:hover{z-index:2}
#tip{position:fixed;z-index:6;max-width:320px;pointer-events:none;font-size:12px;line-height:1.45;color:#d6d6d6;background:#000c;padding:8px 10px;border-radius:6px;display:none}
#tip b{display:block;color:#fff;font-weight:500;font-size:13px}#tip em{font-style:normal;color:var(--mute)}
</style></head><body>
<header><span id="tabs"></span><span class="sp"></span><span id="schemes"></span><span class="key" id="key"></span><button id="mine">My labels</button><button id="mode">Grid</button></header>
<div id="vp"><div id="stage"></div></div><div id="tip"></div>
<script>
const D=/*DATA*/;
const vp=document.getElementById('vp'),stage=document.getElementById('stage'),tip=document.getElementById('tip');
const names=Object.keys(D.maps);let cur=names[0],mine=false,focus=null,sch=D.maps[cur].scheme;
const imgs=D.items.map((it,i)=>{const im=new Image();im.src=it.src;im.decoding='async';im.style.width=D.tw+'px';im.style.height=D.th+'px';im.dataset.i=i;stage.appendChild(im);return im});
function catOf(it){return mine&&sch==='People'?it.h:it.k[sch]}
function paint(){imgs.forEach((im,i)=>{const it=D.items[i],c=catOf(it);im.style.outlineColor=c?D.schemes[sch][c]:'transparent';
  im.classList.toggle('dim',!!focus&&c!==focus)})}
// exact = UMAP 좌표 그대로(겹침 허용, 거리가 정확), 아니면 겹치지 않는 격자 칸
let exact=true;
function place(){const m=D.maps[cur],W=(m.cols-1)*(D.tw+D.gap),H=(m.rows-1)*(D.th+D.gap);
  imgs.forEach((im,i)=>{let x,y;if(exact){[x,y]=m.xy[i];x*=W;y*=H}else{x=m.pos[i][0]*(D.tw+D.gap);y=m.pos[i][1]*(D.th+D.gap)}
    im.style.left=x+'px';im.style.top=y+'px'})}
document.getElementById('mode').onclick=e=>{exact=!exact;e.target.classList.toggle('on',!exact);place()};
// 탭·범례
const tabs=document.getElementById('tabs');
names.forEach(n=>{const b=document.createElement('button');b.textContent=n;b.onclick=()=>{cur=n;[...tabs.children].forEach(x=>x.classList.toggle('on',x===b));setScheme(D.maps[n].scheme);place()};if(n===cur)b.classList.add('on');tabs.appendChild(b);tabs.append(' ')});
const key=document.getElementById('key'),schB=document.getElementById('schemes'),mineB=document.getElementById('mine');
Object.keys(D.schemes).forEach(n=>{const b=document.createElement('button');b.textContent=n;b.dataset.s=n;b.onclick=()=>setScheme(n);schB.appendChild(b);schB.append(' ')});
function setScheme(n){sch=n;focus=null;[...schB.children].forEach(x=>x.classList.toggle('on',x.dataset.s===n));mineB.style.display=n==='People'?'':'none';key.innerHTML='';
  Object.entries(D.schemes[n]).forEach(([k,c])=>{const b=document.createElement('button');b.innerHTML=`<i style="background:${c}"></i>${k}`;
    b.onclick=()=>{focus=focus===k?null:k;[...key.children].forEach(x=>x.classList.toggle('off',!!focus&&x!==b));paint()};key.appendChild(b)});paint()}
mineB.onclick=e=>{mine=!mine;e.target.classList.toggle('on',mine);paint()};
// 화면 변환: 목표값으로 부드럽게 따라감
let s=1,tx=0,ty=0,S=1,TX=0,TY=0,vx=0,vy=0;const MAXS=8;
function worldSize(){const m=D.maps[cur];return [m.cols*(D.tw+D.gap),m.rows*(D.th+D.gap)]}
function fit(instant){const [w,h]=worldSize();S=Math.min(innerWidth/(w+80),(innerHeight-60)/(h+80));TX=(innerWidth-w*S)/2;TY=60+(innerHeight-60-h*S)/2;if(instant){s=S;tx=TX;ty=TY}}
function zoomAt(f,cx,cy){const ns=Math.max(fitS()*.8,Math.min(MAXS,S*f));TX=cx-(cx-TX)*ns/S;TY=cy-(cy-TY)*ns/S;S=ns}
function fitS(){const [w,h]=worldSize();return Math.min(innerWidth/(w+80),(innerHeight-60)/(h+80))}
function tick(){if(!drag){TX+=vx;TY+=vy;vx*=.92;vy*=.92;if(Math.abs(vx)<.05)vx=0;if(Math.abs(vy)<.05)vy=0}
  const k=.22;s+=(S-s)*k;tx+=(TX-tx)*k;ty+=(TY-ty)*k;stage.style.transform=`translate(${tx}px,${ty}px) scale(${s})`;stage.style.setProperty('--ow',(2/s)+'px');
  if(Math.abs(S-s)<.01&&Math.abs(TX-tx)<1&&Math.abs(TY-ty)<1&&!drag)hires();requestAnimationFrame(tick)}
let hiT=0;function hires(){const now=performance.now();if(now-hiT<250)return;hiT=now;if(s*D.th<90)return;
  imgs.forEach(im=>{if(im.dataset.hi)return;const r=im.getBoundingClientRect();if(r.right<0||r.left>innerWidth||r.bottom<0||r.top>innerHeight)return;
    im.dataset.hi=1;const hi=new Image();hi.onload=()=>{im.src=hi.src};hi.src=D.large+im.src.split('/').pop()})}
// 마우스 휠인지 트랙패드인지: 휠은 줄 단위(deltaMode 1)이거나, 가로 없이 큰 정수 칸(보통 100·120)으로 온다
function isMouseWheel(e){return e.deltaMode!==0||(e.deltaX===0&&Number.isInteger(e.deltaY)&&Math.abs(e.deltaY)>=50)}
vp.addEventListener('wheel',e=>{e.preventDefault();
  if(e.ctrlKey||e.metaKey){zoomAt(Math.exp(-Math.max(-40,Math.min(40,e.deltaY))*.008),e.clientX,e.clientY)}   // 트랙패드 핀치
  else if(isMouseWheel(e)){const d=e.deltaMode===1?e.deltaY*33:e.deltaY;zoomAt(Math.exp(-Math.sign(d)*Math.min(Math.abs(d),240)*.0035),e.clientX,e.clientY)}   // 마우스 휠 = 커서 기준 확대
  else{TX-=e.deltaX;TY-=e.deltaY}},{passive:false});   // 트랙패드 두 손가락 = 이동
let drag=null,last=null;
vp.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};last={x:e.clientX,y:e.clientY,t:performance.now()};vx=vy=0;vp.classList.add('drag');vp.setPointerCapture(e.pointerId)});
vp.addEventListener('pointermove',e=>{if(drag){const dx=e.clientX-last.x,dy=e.clientY-last.y;TX+=dx;TY+=dy;tx+=dx;ty+=dy;vx=dx;vy=dy;last={x:e.clientX,y:e.clientY,t:performance.now()};tip.style.display='none';return}
  const im=e.target.closest('img');if(!im){tip.style.display='none';return}const it=D.items[im.dataset.i];
  const c=it.k[sch];tip.innerHTML=`<b>${it.t}</b><em>${c}${sch==='People'&&it.r?' · '+it.r:''}${sch==='People'&&it.h&&it.h!==c?' · my label '+it.h:''}</em><br>${it.l[sch]}`;
  tip.style.display='block';const r=tip.getBoundingClientRect();tip.style.left=Math.min(e.clientX+14,innerWidth-r.width-8)+'px';tip.style.top=Math.min(e.clientY+14,innerHeight-r.height-8)+'px'});
vp.addEventListener('pointerup',e=>{if(drag&&performance.now()-last.t>60){vx=vy=0}drag=null;vp.classList.remove('drag')});
vp.addEventListener('dblclick',e=>zoomAt(2.2,e.clientX,e.clientY));
// Safari 핀치
let g0=1;addEventListener('gesturestart',e=>{e.preventDefault();g0=1});addEventListener('gesturechange',e=>{e.preventDefault();zoomAt(e.scale/g0,e.clientX,e.clientY);g0=e.scale});
addEventListener('keydown',e=>{if(e.key==='0'||e.key==='Escape')fit();if(e.key==='='||e.key==='+')zoomAt(1.6,innerWidth/2,innerHeight/2);if(e.key==='-')zoomAt(1/1.6,innerWidth/2,innerHeight/2)});
addEventListener('resize',()=>fit());
place();setScheme(sch);fit(true);tick();
</script></body></html>"""


if __name__ == "__main__":
    main()
