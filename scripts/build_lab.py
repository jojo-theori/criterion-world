"""CCW Lab: 축·배치·범주를 사이트에 넣기 전에 실험하는 로컬 전용 지도(out/lab/index.html, 공개하지 않음)

무엇으로 배치하나(Arrange by): 임베딩마다 UMAP 2D. 묘사 글(전체, 한 줄씩)과 그림(Gemma, SigLIP, Gemma 글자 지운 그림, 그림+묘사).
무엇으로 색칠하나(Color by): People(people_v2) · Medium(medium_v2) · Composition·Type(사이트 판정, tropes.json).
배치: Exact = UMAP 좌표 그대로(겹침 허용) / Grid = 빈칸을 조금 둔 격자에 헝가리안 배정(겹치지 않음).
그림: canvas 에 사이트 아틀라스(web/atlas/cover_s.webp, 40×50)로 그리고, 크게 보이면 web/thumbs(112×140) → web/large(480) 로 바꾼다.
조작: 마우스 휠·핀치 = 커서 기준 확대, 끌기·두 손가락 = 이동, 0 = 전체, 커버 클릭 = 그 임베딩에서 가장 가까운 12장 강조.
편집(Edit): 커버를 끌어 옮기고, 커버 위에서 숫자 키로 범주를 고친다(0 = 모델 판정으로). Export 로 JSON 도 받을 수 있다.
UMAP 결과는 out/lab/umap_cache.npz 에 둔다(임베딩 파일이 바뀌면 다시 계산).
사용: .venv-ml/bin/python scripts/build_lab.py, 그다음 python3 scripts/lab_server.py → http://localhost:8777 (이 Mac 에서만 열린다)
편집·라벨은 서버가 data/labels/lab.json 에 저장한다(Label 화면과 같은 파일). build_web.py 가 그 라벨을 판정보다 앞세운다.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "data" / "analysis"
EMB = A / "embed"
OUT = ROOT / "out" / "lab"
TW, TH, GAP = 40, 50, 4
SLACK = 1.12
K_NB = 12

# (이름, 묶음, 임베딩 파일, 기본 색칠, 설명)
MAPS = [
    ("Description", "text", "gemma2-cap-all.npz", "People", "All eight lines of the description"),
    ("People", "text", "gemma2-cap-people.npz", "People", "People line of the description"),
    ("Composition", "text", "gemma2-cap-composition.npz", "Composition", "Composition line of the description"),
    ("Medium", "text", "gemma2-cap-medium.npz", "Medium", "Medium line of the description"),
    ("Type", "text", "gemma2-cap-typography.npz", "Type", "Typography line of the description"),
    ("Color", "text", "gemma2-cap-color.npz", "Medium", "Color and light line of the description"),
    ("Picture", "image", "gemma2-plain.npz", "Medium", "The cover image (EmbeddingGemma 2)"),
    ("Picture, no text", "image", "gemma2-art.npz", "Composition", "The cover with its lettering erased (EmbeddingGemma 2)"),
    ("SigLIP", "image", "siglip2-base-patch16-384.npz", "Medium", "The cover image (SigLIP 2)"),
    ("Picture + description", "image", "gemma2-cap-img.npz", "People", "Description text and image together (EmbeddingGemma 2)"),
]
PAL = ["#ffffff", "#f5d000", "#ff3b30", "#2f8cff", "#34c759", "#bf5af2", "#ff9500", "#64d2ff", "#ff2d92", "#a2845e"]
SCHEMES = {
    "People": {"None": "#6e6e6e", "No-face": "#2f8cff", "One": "#ffffff", "Two": "#f5d000", "3+": "#ff3b30", "Ambiguous": "#bf5af2"},
    "Medium": dict(zip(["Photo", "Painting", "Drawing", "Print", "Collage", "Graphic", "Other"], PAL[:6] + ["#6e6e6e"])),
    "Composition": dict(zip(["Grid / collage", "Symmetric", "Off-center subject", "Silhouette", "Busy frame", "Close crop",
                             "Single object", "Open space", "Diagonal", "Wide scene"], PAL)),
    "Type": dict(zip(["Type only", "Big centered title", "Big title", "Quiet type", "Title on top", "Title in the middle",
                      "Title at the bottom"], PAL[:7])),
}
LINE = {"People": "People", "Medium": "Medium", "Composition": "Composition", "Type": "Typography"}


def grid(xy: np.ndarray) -> tuple[np.ndarray, int, int]:
    n = len(xy)
    cols = math.ceil(math.sqrt(n * SLACK * TH / TW))
    rows = math.ceil(n * SLACK / cols)
    gx, gy = np.meshgrid(np.arange(cols), np.arange(rows))
    cells = np.stack([gx.ravel() / (cols - 1), gy.ravel() / (rows - 1)], 1)
    cost = ((xy[:, None, :] - cells[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    out = np.zeros((n, 2), int)
    out[r] = np.stack([gx.ravel()[c], gy.ravel()[c]], 1)
    return out, cols, rows


def main() -> None:
    import umap
    data = (ROOT / "web" / "data.js").read_text().split("\n")
    covers = json.loads(data[1].split("=", 1)[1].strip().rstrip(";"))
    atlas = {c["file"]: k for k, c in enumerate(covers)}
    meta = {c["file"]: c for c in covers}
    cap = json.loads((A / "captions_sonnet_v3.json").read_text())
    pv2 = json.loads((A / "people_v2.json").read_text())
    mv2 = json.loads((A / "medium_v2.json").read_text())
    tropes = json.loads((A / "tropes.json").read_text())
    files = [c["file"] for c in covers if c["file"] in cap]
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT / "umap_cache.npz"
    cache = dict(np.load(cache_f, allow_pickle=True)) if cache_f.exists() else {}
    maps = []
    for name, group, f, scheme, desc in MAPS:
        p = EMB / f
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        idx = {str(n): i for i, n in enumerate(d["files"])}
        if not all(n in idx for n in files):
            continue
        V = d["vectors"].astype(np.float32)[[idx[n] for n in files]]
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-9
        key = f"{f}|{p.stat().st_mtime_ns}|{len(files)}"
        if key in cache:
            xy = cache[key]
        else:
            xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=0).fit_transform(V)
            cache[key] = xy
        q = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
        g, cols, rows = grid(q)
        S = V @ V.T
        np.fill_diagonal(S, -9)
        nb = np.argsort(-S, axis=1)[:, :K_NB]
        maps.append({"name": name, "group": group, "scheme": scheme, "desc": desc, "cols": cols, "rows": rows,
                     "xy": np.round(q, 4).tolist(), "pos": g.tolist(), "nb": nb.tolist()})
        print(name, flush=True)
    np.savez(cache_f, **cache)
    items = []
    for n in files:
        c, m, t = cap[n], meta[n], tropes.get(n, {})
        items.append({"f": n, "t": m.get("title") or n[5:-4], "y": m.get("year"), "d": m.get("directors") or "", "a": atlas[n],
                      "h": pv2[n].get("human"),
                      "k": {"People": pv2[n]["cat"], "Medium": mv2.get(n, "Other"), "Composition": t.get("composition"), "Type": t.get("type")},
                      "l": {s: c[LINE[s]] for s in SCHEMES}})
    D = {"items": items, "maps": maps, "schemes": SCHEMES, "tw": TW, "th": TH, "gap": GAP,
         "atlas": {"src": "/atlas/cover_s.webp", "cols": 32, "tw": 40, "th": 50}, "thumbs": "/thumbs/", "large": "/large/"}
    (OUT / "index.html").write_text(PAGE.replace("/*DATA*/", json.dumps(D, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print("->", OUT / "index.html")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>CCW Lab</title>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
:root{--t1:clamp(10px,9.6px + .095vw,11.5px);--t2:clamp(11px,10.6px + .095vw,12.5px);--t3:clamp(12px,11.6px + .095vw,13.5px);
  --t6:clamp(24px,20.3px + .95vw,36px);--fg:#ebebeb;--dim:#8a8a8a;--out:cubic-bezier(.22,1,.36,1)}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#000;color:var(--fg);font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;overflow:hidden;touch-action:none}
button{font:inherit}
header{position:fixed;top:0;left:0;right:0;height:56px;background:linear-gradient(#000 70%,#0000);display:flex;align-items:center;gap:14px;padding:0 32px;z-index:4}
.group{position:relative;display:flex;gap:2px;background:#111;border-radius:999px;padding:4px}
.group button{position:relative;z-index:1;border:0;background:transparent;color:var(--dim);font-size:var(--t3);height:28px;padding:0 14px;
  border-radius:999px;cursor:pointer;white-space:nowrap;transition:color .3s var(--out)}
.group button:not(.on):hover{color:var(--fg)}.group button.on{color:#000}
.group .sep{width:1px;margin:6px 4px;background:#333}
.pill{position:absolute;top:4px;bottom:4px;left:0;width:0;border-radius:999px;background:var(--fg);transition:transform .45s var(--out),width .45s var(--out);pointer-events:none}
.sp{flex:1}
.tog{border:0;background:#111;color:var(--dim);height:36px;padding:0 16px;border-radius:999px;cursor:pointer;font-size:var(--t3);transition:background .3s var(--out),color .3s var(--out)}
.tog:hover{color:var(--fg)}a.tog{display:inline-flex;align-items:center;text-decoration:none}.tog.on{background:var(--fg);color:#000}
.count{font-size:var(--t2);color:var(--dim)}
canvas{position:fixed;inset:0;width:100%;height:100%;cursor:grab;display:block}
canvas.drag{cursor:grabbing}
.foot{position:fixed;left:32px;bottom:28px;z-index:3;pointer-events:none;max-width:calc(100vw - 64px)}
.foot .k{font-size:var(--t2);color:var(--dim)}
.foot h1{font-weight:400;font-size:var(--t6);margin:4px 0 2px;letter-spacing:-.01em}
.foot .desc{font-size:var(--t2);color:var(--dim);margin-bottom:14px}
.foot .row{display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px;pointer-events:auto}
.foot .group button{font-size:var(--t2);height:24px;padding:0 12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:12px;pointer-events:auto}
.legend button{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:6px;font-size:var(--t2);color:var(--fg);transition:opacity .3s}
.legend button.off{opacity:.3}.legend i{width:8px;height:8px;border-radius:50%;display:inline-block}
.legend em{font-style:normal;color:var(--dim)}
#tip{position:fixed;z-index:5;max-width:340px;pointer-events:none;font-size:var(--t2);line-height:1.45;color:#cfcfcf;background:#000d;padding:10px 12px;border-radius:10px;opacity:0;transition:opacity .15s}
#tip b{display:block;color:#fff;font-weight:500;font-size:var(--t3)}#tip em{font-style:normal;color:var(--dim);display:block;margin-bottom:4px}
.zoom{position:fixed;right:32px;bottom:28px;z-index:3;display:flex;gap:2px;background:#111;border-radius:999px;padding:4px}
.zoom button{border:0;background:transparent;color:var(--dim);width:28px;height:28px;border-radius:50%;cursor:pointer;font-size:16px}
.zoom button:hover{color:var(--fg)}
</style></head><body>
<header><div class="group" id="maps"></div><span class="sp"></span><span class="count" id="count"></span>
<a class="tog" href="/label">Label</a><button class="tog" id="grid">Grid</button><button class="tog" id="edit">Edit</button><button class="tog" id="export">Export</button><button class="tog" id="reset">Reset edits</button></header>
<canvas id="cv"></canvas>
<div class="foot"><div class="k">Arranged by</div><h1 id="title"></h1><div class="desc" id="desc"></div>
  <div class="row"><span class="k">Colored by</span><div class="group" id="schemes"></div></div><div class="legend" id="legend"></div></div>
<div class="zoom"><button id="zin" aria-label="Zoom in">+</button><button id="zout" aria-label="Zoom out">−</button></div>
<div id="tip"></div>
<script>
const D=/*DATA*/;
const N=D.items.length,cv=document.getElementById('cv'),ctx=cv.getContext('2d'),tip=document.getElementById('tip');
const CW=D.tw+D.gap,CH=D.th+D.gap;
let mapI=Math.max(0,D.maps.findIndex(m=>m.name==='People')),scheme=D.maps[mapI].scheme,exact=true,edit=false,focusCat=null,sel=null,hover=-1;
// 편집 내용(서버가 data/labels/lab.json 에 저장): labels[scheme][file]=범주, moves[map|layout][file]=[x,y]
let E={labels:{},moves:{}},saveT=0;
const save=()=>{clearTimeout(saveT);saveT=setTimeout(()=>fetch('/api/edits',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(E)}),300)};
const catOf=i=>{const it=D.items[i],o=(E.labels[scheme]||{})[it.f];return o||it.k[scheme]};
// ---------- 이미지: 아틀라스(저화질) → thumbs → large ----------
const atlas=new Image();atlas.src=D.atlas.src;atlas.onload=()=>dirty=true;
const mid={},big={};let loading=0;
function want(store,base,i){const f=D.items[i].f;if(store[f]||loading>12)return;const im=new Image();store[f]=im;loading++;
  im.onload=im.onerror=()=>{loading--;im.ok=im.complete&&im.naturalWidth>0;dirty=true};im.src=base+f}
// ---------- 자리: 지금 자리(cx,cy)가 목표(gx,gy)로 부드럽게 ----------
const cx=new Float32Array(N),cy=new Float32Array(N),gx=new Float32Array(N),gy=new Float32Array(N);
const moveKey=()=>D.maps[mapI].name+(exact?'|exact':'|grid');
function targets(){const m=D.maps[mapI],W=(m.cols-1)*CW,H=(m.rows-1)*CH,mv=E.moves[moveKey()]||{};
  for(let i=0;i<N;i++){const o=mv[D.items[i].f];if(o){gx[i]=o[0];gy[i]=o[1];continue}
    if(exact){gx[i]=m.xy[i][0]*W;gy[i]=m.xy[i][1]*H}else{gx[i]=m.pos[i][0]*CW;gy[i]=m.pos[i][1]*CH}}}
// ---------- 화면 변환 ----------
let s=1,tx=0,ty=0,S=1,TX=0,TY=0,vx=0,vy=0,dirty=true,dpr=1;
function worldSize(){const m=D.maps[mapI];return[m.cols*CW,m.rows*CH]}
const fitS=()=>{const[w,h]=worldSize();return Math.min(innerWidth/(w+120),(innerHeight-260)/(h+40))};
function fit(now){const[w,h]=worldSize();S=fitS();TX=(innerWidth-w*S)/2;TY=70+(innerHeight-260-h*S)/2;if(now){s=S;tx=TX;ty=TY}}
function zoomAt(f,x,y){const ns=Math.max(fitS()*.7,Math.min(14,S*f));TX=x-(x-TX)*ns/S;TY=y-(y-TY)*ns/S;S=ns}
function resize(){dpr=devicePixelRatio||1;cv.width=innerWidth*dpr;cv.height=innerHeight*dpr;dirty=true}
// ---------- 그리기 ----------
function draw(){ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,innerWidth,innerHeight);
  const w=D.tw*s,h=D.th*s,col=D.schemes[scheme],nbs=sel!==null?new Set(D.maps[mapI].nb[sel]):null;
  const useMid=h>=56,useBig=h>=150,settled=Math.abs(S-s)<.01*S;
  const order=[];for(let i=0;i<N;i++)if(i!==hover)order.push(i);if(hover>=0)order.push(hover);
  for(const i of order){const x=cx[i]*s+tx,y=cy[i]*s+ty;if(x>innerWidth||y>innerHeight||x+w<0||y+h<0)continue;
    const c=catOf(i),f=D.items[i].f;
    let a=1;if(focusCat&&c!==focusCat)a=.1;if(nbs&&i!==sel&&!nbs.has(i))a=.1;ctx.globalAlpha=a;
    let src=null;if(useBig){if(settled)want(big,D.large,i);if(big[f]&&big[f].ok)src=big[f]}
    if(!src&&useMid){if(settled)want(mid,D.thumbs,i);if(mid[f]&&mid[f].ok)src=mid[f]}
    if(src)ctx.drawImage(src,x,y,w,h);
    else if(atlas.complete){const k=D.items[i].a,ac=D.atlas.cols;ctx.drawImage(atlas,(k%ac)*D.atlas.tw,Math.floor(k/ac)*D.atlas.th,D.atlas.tw,D.atlas.th,x,y,w,h)}
    if(col[c]){ctx.strokeStyle=col[c];ctx.lineWidth=Math.max(1,Math.min(2.5,h/22));ctx.strokeRect(x+.5,y+.5,w-1,h-1)}
    if(i===sel||(nbs&&nbs.has(i))){ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.strokeRect(x-2,y-2,w+4,h+4)}
    const edited=(E.labels[scheme]||{})[f]||(E.moves[moveKey()]||{})[f];
    if(edited&&a>.5){ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(x+w-3,y+3,Math.max(2,Math.min(4,h/14)),0,7);ctx.fill()}}
  ctx.globalAlpha=1}
function tick(){let moving=false;
  if(!drag){TX+=vx;TY+=vy;vx*=.92;vy*=.92;if(Math.abs(vx)<.05)vx=0;if(Math.abs(vy)<.05)vy=0;if(vx||vy)moving=true}
  const k=.22;if(Math.abs(S-s)>1e-4||Math.abs(TX-tx)>.05||Math.abs(TY-ty)>.05){s+=(S-s)*k;tx+=(TX-tx)*k;ty+=(TY-ty)*k;moving=true}
  for(let i=0;i<N;i++){const dx=gx[i]-cx[i],dy=gy[i]-cy[i];if(dx*dx+dy*dy>.01){cx[i]+=dx*.14;cy[i]+=dy*.14;moving=true}}
  if(moving||dirty){draw();dirty=false}requestAnimationFrame(tick)}
// ---------- 맞히기: 위에 그려진 것부터 ----------
function hit(px,py){const w=D.tw*s,h=D.th*s;if(hover>=0){const x=cx[hover]*s+tx,y=cy[hover]*s+ty;if(px>=x&&px<=x+w&&py>=y&&py<=y+h)return hover}
  for(let i=N-1;i>=0;i--){const x=cx[i]*s+tx,y=cy[i]*s+ty;if(px>=x&&px<=x+w&&py>=y&&py<=y+h)return i}return -1}
function showTip(i,e){if(i<0){tip.style.opacity=0;return}const it=D.items[i],c=catOf(i),o=(E.labels[scheme]||{})[it.f];
  tip.innerHTML=`<b>${it.t}</b><em>${it.y||''}${it.d?' · '+it.d:''}<br>${c||'—'}${o?' (edited, model: '+it.k[scheme]+')':''}${scheme==='People'&&it.h&&it.h!==c?' · my label '+it.h:''}</em>${it.l[scheme]}`;
  tip.style.opacity=1;const r=tip.getBoundingClientRect();tip.style.left=Math.min(e.clientX+16,innerWidth-r.width-12)+'px';tip.style.top=Math.min(e.clientY+16,innerHeight-r.height-12)+'px'}
// ---------- 입력 ----------
const isMouseWheel=e=>e.deltaMode!==0||(e.deltaX===0&&Number.isInteger(e.deltaY)&&Math.abs(e.deltaY)>=50);
cv.addEventListener('wheel',e=>{e.preventDefault();
  if(e.ctrlKey||e.metaKey)zoomAt(Math.exp(-Math.max(-40,Math.min(40,e.deltaY))*.01),e.clientX,e.clientY);
  else if(isMouseWheel(e)){const d=e.deltaMode===1?e.deltaY*33:e.deltaY;zoomAt(Math.exp(-Math.sign(d)*Math.min(Math.abs(d),240)*.0035),e.clientX,e.clientY)}
  else{TX-=e.deltaX;TY-=e.deltaY}},{passive:false});
let drag=null;
cv.addEventListener('pointerdown',e=>{const i=hit(e.clientX,e.clientY);vx=vy=0;
  drag={x:e.clientX,y:e.clientY,lx:e.clientX,ly:e.clientY,t:performance.now(),moved:false,item:edit&&i>=0?i:-1,down:i};
  cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{
  if(drag){const dx=e.clientX-drag.lx,dy=e.clientY-drag.ly;if(Math.hypot(e.clientX-drag.x,e.clientY-drag.y)>4)drag.moved=true;
    if(drag.moved){cv.classList.add('drag');tip.style.opacity=0;
      if(drag.item>=0){const i=drag.item;gx[i]+=dx/s;gy[i]+=dy/s;cx[i]=gx[i];cy[i]=gy[i];hover=i;dirty=true}
      else{TX+=dx;TY+=dy;tx+=dx;ty+=dy;vx=dx;vy=dy;dirty=true}}
    drag.lx=e.clientX;drag.ly=e.clientY;drag.t=performance.now();return}
  const i=hit(e.clientX,e.clientY);if(i!==hover){hover=i;dirty=true}showTip(i,e)});
cv.addEventListener('pointerup',e=>{cv.classList.remove('drag');
  if(drag){if(drag.item>=0&&drag.moved){const i=drag.item,k=moveKey();(E.moves[k]=E.moves[k]||{})[D.items[i].f]=[gx[i],gy[i]];save();vx=vy=0;updateCount()}
    else if(!drag.moved){sel=drag.down>=0&&drag.down!==sel?drag.down:null;dirty=true}
    else if(performance.now()-drag.t>60)vx=vy=0;
    if(drag.item>=0)vx=vy=0}
  drag=null});
cv.addEventListener('pointerleave',()=>{hover=-1;tip.style.opacity=0;dirty=true});
let g0=1;addEventListener('gesturestart',e=>{e.preventDefault();g0=1});addEventListener('gesturechange',e=>{e.preventDefault();zoomAt(e.scale/g0,e.clientX,e.clientY);g0=e.scale});
addEventListener('keydown',e=>{if(e.metaKey||e.ctrlKey)return;
  if(e.key==='0'&&!edit){fit();return}if(e.key==='Escape'){sel=null;focusCat=null;legend();fit();return}
  if(e.key==='='||e.key==='+')zoomAt(1.6,innerWidth/2,innerHeight/2);if(e.key==='-')zoomAt(1/1.6,innerWidth/2,innerHeight/2);
  // 편집: 커버 위에서 숫자 키 = 그 번호 범주로 고침, 0 = 모델 판정으로 되돌림
  if(edit&&hover>=0&&/^[0-9]$/.test(e.key)){const cats=Object.keys(D.schemes[scheme]),f=D.items[hover].f,L=(E.labels[scheme]=E.labels[scheme]||{});
    const n=+e.key;if(n===0)delete L[f];else if(cats[n-1])L[f]=cats[n-1]===D.items[hover].k[scheme]?undefined:cats[n-1];if(L[f]===undefined)delete L[f];
    save();legend();updateCount();dirty=true;showTip(hover,{clientX:parseFloat(tip.style.left)-16,clientY:parseFloat(tip.style.top)-16})}});
addEventListener('resize',()=>{resize();fit()});
document.getElementById('zin').onclick=()=>zoomAt(1.8,innerWidth/2,innerHeight/2);
document.getElementById('zout').onclick=()=>zoomAt(1/1.8,innerWidth/2,innerHeight/2);
// ---------- 머리줄·바닥 ----------
function segment(el,labels,onIdx,cur,seps=[]){el.innerHTML='<span class="pill"></span>';labels.forEach((l,k)=>{if(seps.includes(k)){const s=document.createElement('span');s.className='sep';el.appendChild(s)}
  const b=document.createElement('button');b.textContent=l;b.onclick=()=>{onIdx(k);mark(el,k)};el.appendChild(b)});requestAnimationFrame(()=>mark(el,cur))}
function mark(el,k){const bs=[...el.querySelectorAll('button')];bs.forEach((b,j)=>b.classList.toggle('on',j===k));const b=bs[k],p=el.querySelector('.pill');
  if(b){p.style.width=b.offsetWidth+'px';p.style.transform=`translateX(${b.offsetLeft}px)`}}
const gstart=D.maps.findIndex(m=>m.group==='image');
segment(document.getElementById('maps'),D.maps.map(m=>m.name),k=>{mapI=k;sel=null;setScheme(D.maps[k].scheme);targets();updateFoot();updateCount();fit()},mapI,gstart>0?[gstart]:[]);
const SCH=Object.keys(D.schemes);
function setScheme(n){scheme=n;focusCat=null;mark(document.getElementById('schemes'),SCH.indexOf(n));legend();dirty=true}
segment(document.getElementById('schemes'),SCH,k=>setScheme(SCH[k]),SCH.indexOf(scheme));
function legend(){const el=document.getElementById('legend'),cnt={};for(let i=0;i<N;i++){const c=catOf(i);cnt[c]=(cnt[c]||0)+1}el.innerHTML='';
  Object.entries(D.schemes[scheme]).forEach(([c,col],k)=>{const b=document.createElement('button');b.innerHTML=`<i style="background:${col}"></i>${edit?(k+1)+' ':''}${c} <em>${cnt[c]||0}</em>`;
    b.classList.toggle('off',!!focusCat&&focusCat!==c);b.onclick=()=>{focusCat=focusCat===c?null:c;legend();dirty=true};el.appendChild(b)})}
function updateFoot(){const m=D.maps[mapI];document.getElementById('title').textContent=m.name;document.getElementById('desc').textContent=m.desc+(exact?'':' · grid')}
function updateCount(){const nl=Object.values(E.labels).reduce((a,o)=>a+Object.keys(o).length,0),nm=Object.values(E.moves).reduce((a,o)=>a+Object.keys(o).length,0);
  document.getElementById('count').textContent=`${N} covers`+(nl||nm?` · ${nl} relabeled · ${nm} moved`:'')}
document.getElementById('grid').onclick=e=>{exact=!exact;e.target.classList.toggle('on',!exact);targets();updateFoot()};
document.getElementById('edit').onclick=e=>{edit=!edit;e.target.classList.toggle('on',edit);legend()};
document.getElementById('export').onclick=()=>{const b=new Blob([JSON.stringify({exported:new Date().toISOString(),...E},null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='ccw-lab-edits.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.getElementById('reset').onclick=()=>{delete E.moves[moveKey()];if(E.labels[scheme])delete E.labels[scheme];save();targets();legend();updateCount();dirty=true};
function start(){resize();targets();for(let i=0;i<N;i++){cx[i]=gx[i];cy[i]=gy[i]}updateFoot();updateCount();setScheme(scheme);fit(true);tick()}
fetch('/api/edits').then(r=>r.json()).then(d=>{E={labels:d.labels||{},moves:d.moves||{}};start()}).catch(start);
</script></body></html>"""


if __name__ == "__main__":
    main()
