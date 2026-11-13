# The Criterion Collection World (CCW)

크라이테리온 컬렉션 커버를 AI로 분석하는 개인 웹 인터랙티브 프로젝트의 작업 폴더.

## 폴더

```
posters/                 커버 원본 (<spine 네 자리>_<제목 slug>.jpg)
data/
  criterion.sqlite       영화 메타데이터 DB (films 테이블)
  films.csv              같은 내용의 CSV
  cover_checklist.csv    커버 수집 체크리스트 (저장할 파일 이름, 확보 여부)
  raw/                   Wikidata, Wikipedia 원본 캐시, cover_sources.csv (커버 출처 기록)
  regions/<엔진>/        커버별 텍스트 영역 JSON
  analysis/visual/       커버별 색·대비·구도선 JSON
scripts/
  build_film_db.py       메타데이터 DB 생성
  cover_checklist.py     수집 체크리스트 갱신
  fetch_covers.py        커버 받기 (Internet Archive 보관본 → Criterion S3)
  run_text_regions.py    텍스트 영역 파이프라인 전체 실행
  engines/               vision.swift, paddleocr_engine.py, combine.py
  evaluate.py            엔진 비교 평가
  render_compare.py      엔진별 결과를 나란히 그린 확인용 이미지
  analyze_visual.py      색, 대비, 구도선 분석
  embed_siglip.py        SigLIP 2 임베딩, 기법 제로샷 (.venv-ml)
  render_embed.py        임베딩 확인용 이미지 (.venv-ml)
  build_web.py           웹 뷰어 데이터, 축별 지도, 썸네일·흔적 아틀라스 (.venv-ml)
  browser_check.mjs      실제 시간 헤드리스 Chrome 으로 뷰어 조작·스크린샷 (Node 22)
  detect_objects.py      OWLv2 물체 검출 (.venv-ml)
  analyze_strokes_layout.py  획(선화 통계), 레이아웃(글자·그림·빈 공간 칸) (.venv-ml)
  fetch_box_covers.py    박스 세트 커버 시험 수집 → posters_box/
  caption_prompt.txt     커버 묘사(Sonnet 에이전트) 지시문, merge_captions.py 로 합침
  embed_captions.py      묘사 글 임베딩(EmbeddingGemma 2, 전체·줄별·글+이미지) (.venv-ml)
  people_v2.py, medium_v2.py  묘사에서 People·Medium 범주
  build_lab.py, lab_server.py Lab: 실험 지도 + 라벨링, http://localhost:8777 (이 Mac 에서만)
docs/references.md       참고 자료 요약·우선순위
tools/label/             Lab 라벨링 화면
web/                     뷰어 (index.html, data.js, trace/, atlas/, thumbs/, large/, strokes/)
out/
  compare/               확인용 이미지
  eval/                  평가 결과
  visual/                시각 분석 확인용 이미지 (--render)
  embed/                 임베딩 확인용 이미지
EPFLECALLab_PosterWorld_JPG/  레퍼런스 사진 (Poster World 설치, 저작권 EPFL+ECAL Lab, 공개 금지)
```

## 준비

```
python3 -m venv .venv
.venv/bin/pip install paddlepaddle paddleocr pandas lxml
```

opencv-contrib-python 은 PaddleOCR 과 함께 설치된다(시각 분석에서 saliency, LSD 를 쓴다).

macOS Vision 은 시스템 내장이라 설치할 것이 없다.

## 영화 메타데이터 DB

```
.venv/bin/python scripts/build_film_db.py            # 캐시 사용
.venv/bin/python scripts/build_film_db.py --refresh  # 다시 받아오기
```

- 단독 발매작 1,099편, spine 1–1344. 박스 세트에만 든 영화와 LaserDisc 전용 타이틀은 넣지 않는다.
- 기준 목록은 Wikipedia "List of films in the Criterion Collection"(CC BY-SA), Wikidata(CC0)를 spine 번호(P12279)로 붙여 IMDb, TMDB, Criterion ID 를 채운다.
- 두 출처의 연도가 다르면 `year`(Wikipedia)와 `year_wikidata` 를 둘 다 남긴다. 현재 8건.
- 실행할 때 출처 간 불일치(연도, spine 번호 수정)를 출력한다.

## 커버 추가

같은 영화당 최신판 커버 하나만 모은다.

```
.venv/bin/python scripts/fetch_covers.py                 # 없는 커버 전부
.venv/bin/python scripts/fetch_covers.py --spines 11-20  # 범위만
```

- Internet Archive CDX 에서 criterion.com/films/<criterion_id> 의 가장 최근 정상 보관본을 찾고, 그 HTML 의 og:image 주소로 Criterion S3 에서 `_original`(1288×1600) 을 받는다. criterion.com 에는 직접 요청하지 않는다.
- 기존 posters/0001 과 같은 파일이 받아지는 것을 확인했다. 최근 보관본 중에는 Cloudflare 확인 화면이 저장된 것(403)이 있어 200 만 쓴다.
- 보관 시점의 판본이라 그 뒤 새 판본이 나왔으면 반영되지 않는다. 출처와 보관 시점은 data/raw/cover_sources.csv.
- 한 편에 약 30초(대부분 archive.org 응답 대기). 기본 3개 동시, 분당 약 12회 요청.

받은 뒤:

1. (수동으로 넣을 때) `data/cover_checklist.csv` 의 `filename` 이름으로 `posters/` 에 저장
2. `.venv/bin/python scripts/run_text_regions.py` (새 커버만 처리)
3. `.venv/bin/python scripts/analyze_visual.py` (새 커버만 처리)
4. `.venv/bin/python scripts/cover_checklist.py` (확보 여부 갱신)

## 텍스트 영역 파이프라인

```
.venv/bin/python scripts/run_text_regions.py          # 새 커버만 처리
.venv/bin/python scripts/run_text_regions.py --force  # 전부 다시
```

posters/ 에 커버를 넣고 다시 실행하면 새 커버만 처리한다. 최종 결과는 `data/regions/combined/`.

| 단계 | 엔진 | 역할 |
|---|---|---|
| 1 | macOS Vision, 원본과 50% 축소본 | 글자 읽기 |
| 2 | PaddleOCR 검출 (PP-OCRv6, 긴 변 640px, 다각형) | 영역 모양 |
| 3 | combine | 다각형 안에 단어를 읽는 순서대로 붙인다 |

결과 JSON 의 영역 하나:

```json
{"source": "recognize", "text": "GRAND", "confidence": 0.93, "poly": [[x, y], ...]}
```

- `poly` 는 원본 픽셀 좌표, 원점 왼쪽 위
- `source: "detect"` 는 글자를 못 붙인 영역이다. 로고, 무늬, 읽지 못한 레터링이 여기에 들어간다.

## 평가 방법

정답 박스를 손으로 그리지 않는다. 파일명의 spine 번호로 DB 에서 제목, 감독을 찾아 기대 단어로 쓰므로 커버가 늘어도 정답이 자동으로 따라온다.

- read: 기대 단어를 읽어낸 비율 (철자 유사도 0.75 이상 인정)
- located: 기대 단어 위치를 덮은 비율. 위치는 어느 엔진이든 그 단어를 읽어낸 박스로 정한다.
- extra: 기대 단어와 겹치지 않는 영역 수 (오탐 또는 DB 밖 글자)

10장 기준 결과:

| 엔진 | 제목 읽기 | 감독 읽기 | 브랜드 읽기 | 제목 위치 | 커버당 시간 |
|---|---|---|---|---|---|
| Vision 원본 | 81.8% | 85.7% | 90.0% | 100% | 0.1초 |
| Vision 50% | 86.4% | 85.7% | 100% | 95% | 0.05초 |
| PaddleOCR 인식 640 | 77.3% | 85.7% | 70.0% | 95% | 1.5초 |
| PaddleOCR 검출 640 | - | - | - | 100% | 0.6초 |
| combined | 90.9% | 85.7% | 100% | 100% | 0.8초 |

1,092장 기준 (2026-11-11, 순서대로 제목 읽기, 감독 읽기, 브랜드 읽기 | 제목·감독·브랜드 위치):

| 엔진 | 읽기 | 위치 |
|---|---|---|
| Vision 원본 | 83.5% / 88.3% / 97.2% | 98.6% / 99.2% / 98.6% |
| Vision 50% | 84.8% / 87.5% / 92.8% | 98.8% / 99.2% / 93.5% |
| PaddleOCR 검출 640 | - | 98.2% / 98.3% / 98.1% |
| combined | 85.9% / 88.3% / 98.7% | 99.8% / 99.9% / 99.9% |

- 휘어진 글자(Grand Illusion)는 PaddleOCR 검출 다각형만 모양을 제대로 따라간다.
- Vision 은 50% 축소본이 원본보다 오탐이 적고 더 잘 읽는다. 25%(322×400)에서는 확실히 떨어진다.
- PaddleOCR 검출은 25% 축소본에서도 거의 그대로다. 썸네일급 커버가 섞여 들어와도 영역은 잡힌다.
- PaddleOCR 은 원본 크기 그대로 검출하면 큰 글자를 놓친다(Grand Illusion 에서 2개만 검출). 긴 변 640px 로 줄여야 한다.
- 1,100편 기준 약 15분 (M4 Pro, CPU).

## 색, 대비, 구도선 분석

```
.venv/bin/python scripts/analyze_visual.py            # 새 커버만 처리
.venv/bin/python scripts/analyze_visual.py --force    # 전부 다시
.venv/bin/python scripts/analyze_visual.py --render   # 확인용 이미지도 out/visual/ 에
```

텍스트 영역 결과(`data/regions/combined/`)가 있어야 한다. 긴 변 800px 로 줄여 계산하고 커버당 0.2–0.5초.

커버마다 두 버전을 저장한다.

- `full`: 커버 전체. 텍스트 영역별 글자 대 바탕 대비(WCAG 비)를 함께 잰다.
- `art`: 텍스트 다각형 전체와 크라이테리온 틀을 가린 그림 부분. `visible_share` 가 남은 면적 비율.

크라이테리온 틀은 브랜드 글자가 읽힌 위치로 템플릿을 정하고 템플릿의 고정 영역을 통째로 가린다. 로고는 검출이 불안정해서(10장 중 2장) 검출 결과를 쓰지 않는다.

| frame.kind | 판정 | 가리는 영역 (정규화 x0, y0, x1, y1) |
|---|---|---|
| modern-strip | 브랜드 글자가 왼쪽 가장자리 세로 | 띠 (0, 0.535, 0.075, 0.93), 로고 (0.035, 0.025, 0.195, 0.165) |
| dvd-topband | 브랜드 글자가 위쪽 가로 | (0, 0, 1, 0.075) |
| unknown | 그 밖 | 브랜드 글자 다각형을 1% 넓혀서 |

| 축 | 사람이 읽는 지표 | 비교용 벡터 |
|---|---|---|
| 색 | Lab k-means 팔레트 6색, Hasler–Süsstrunk colorfulness, 채도, color_mode (grayscale / single-hue / color) | Lab 결합 히스토그램 144 |
| 대비 | RMS contrast, 명도 백분위, 어두운·밝은 비율, 텍스트 영역 WCAG 비 | 명도 히스토그램 16 |
| 구도선 | LSD 선분, 수평·수직·대각 비율, 선 밀도. 긴 선분 300개 좌표 | 방향 히스토그램 12 |
| 무게중심 | Spectral Residual saliency 무게중심, 퍼짐, 3분할선·교차점·중심 거리 | 8×10 격자 |
| 기타 | 에지 밀도, 좌우 대칭도 | |

- 벡터는 축마다 합 1 로 정규화해 따로 둔다. Poster World 처럼 축을 골라 짝지을 수 있게.
- 선분과 팔레트는 웹에서 원본 없이 흔적만 그리는 용도로도 쓴다. 좌표는 0–1, 원점 왼쪽 위.
- `art` 의 saliency 는 가린 곳을 inpaint 로 메운 뒤 계산한다. 원본에서 계산하고 가리면 글자의 번짐이 마스크 둘레에 남는다.

10장 결과 (art 기준):

| 커버 | 틀 | 남은 면적 | color_mode | colorfulness | RMS | 수평/수직/대각 |
|---|---|---|---|---|---|---|
| Grand Illusion | dvd | 0.76 | single-hue | 92.9 | 0.234 | 0.34/0.30/0.36 |
| Seven Samurai | modern | 0.64 | grayscale | 1.0 | 0.034 | 0.44/0.31/0.25 |
| The Lady Vanishes | modern | 0.44 | color | 117.0 | 0.223 | 0.37/0.04/0.59 |
| Amarcord | modern | 0.87 | color | 62.0 | 0.276 | 0.39/0.30/0.30 |
| The 400 Blows | modern | 0.76 | grayscale | 0.0 | 0.261 | 0.04/0.13/0.83 |
| Beauty and the Beast | modern | 0.89 | grayscale | 3.1 | 0.292 | 0.30/0.41/0.28 |
| A Night to Remember | modern | 0.82 | grayscale | 0.2 | 0.210 | 0.41/0.09/0.50 |
| The Killer | dvd | 0.81 | single-hue | 71.2 | 0.313 | 0.18/0.30/0.52 |
| Hard Boiled | dvd | 0.69 | color | 48.6 | 0.300 | 0.32/0.09/0.60 |
| Walkabout | modern | 0.92 | color | 22.9 | 0.207 | 0.65/0.26/0.09 |

## 임베딩 (SigLIP 2)

PaddleOCR 환경과 섞지 않으려고 별도 venv 를 쓴다.

```
uv venv --python 3.12 .venv-ml
VIRTUAL_ENV=.venv-ml uv pip install torch torchvision transformers sentencepiece protobuf umap-learn scikit-learn matplotlib pillow controlnet_aux
.venv-ml/bin/python scripts/embed_siglip.py     # 새 커버만, data/analysis/embed/
.venv-ml/bin/python scripts/render_embed.py     # 확인용 이미지, out/embed/
```

- 기본 모델 google/siglip2-base-patch16-384. 이미지 임베딩(L2 정규화)과 기법 제로샷 15개 라벨 sigmoid 점수.
- render_embed.py 는 Poster World 표시 문법을 따른다: 격자에 맞춘 UMAP 지도(global_map), 기준 커버와 가장 가까운 두 장(triptych_*), 라벨별 상위 커버(technique_top).

## 물체 검출 (OWLv2)

```
.venv-ml/bin/python scripts/detect_objects.py            # 새 커버만, data/analysis/objects/owlv2.json
.venv-ml/bin/python scripts/detect_objects.py --force    # 어휘(VOCAB)를 바꿨을 때
```

- google/owlv2-base-patch16-ensemble, 영화 커버용 어휘 73개(사람·몸, 무기, 탈것, 동물, 자연, 건물, 물건, 그래픽 요소). 점수 0.15 이상, 라벨별 NMS, 커버당 30개까지. M1 Max 에서 장당 약 0.3초.
- 크라이테리온 C 로고가 moon/circle 로, 인쇄된 글자가 letter 로 잡힌다. letter 는 어휘에서 뺐고(envelope 로 대체), build_web.py 가 틀·텍스트 영역에 절반 넘게 덮인 박스를 버린다.

## 획과 레이아웃

```
.venv-ml/bin/python scripts/analyze_strokes_layout.py           # 새 커버만
.venv-ml/bin/python scripts/analyze_strokes_layout.py --force   # 전부 다시
```

텍스트 영역(data/regions/combined)과 시각 분석의 틀 위치가 있어야 한다. 장당 약 0.15초(M1 Max, MPS).

- 획: controlnet_aux 의 LineartDetector(Informative Drawings)로 선화를 뽑고, 텍스트 다각형과 틀을 가린 그림 부분만 잰다. 잉크 양, 진하기, 굵기(스켈레톤 위 거리 변환 ×2, 잉크 면적 ÷ 선 길이), 방향(구조 텐서, 12구간), 결(coherence: 1 에 가까울수록 한 방향으로 정돈된 선).
  - 선화 모델은 원래 가는 선을 그려서 굵기 분포가 한 구간에 몰린다. 차이는 굵은 꼬리와 평균 굵기에서 난다.
- 레이아웃: 커버를 40×50 칸으로 나눠 글자 / 그림 / 빈 공간 / 틀. 그림 = 바탕색(가장자리 중앙값, Lab 거리 18 초과)과 다르거나 선화 밀도가 있는 곳. 지표는 구역별 면적, 무게중심, 제목 위치(top/middle/bottom), 좌우 균형.
- 선화는 web/strokes/ 에 PNG(로컬용)와 WebP(배포용)로 둔다.

## 얼굴과 범주 (tropes, 작업 중)

```
.venv-ml/bin/python scripts/detect_faces.py      # 새 커버만, data/analysis/faces.json
.venv-ml/bin/python scripts/build_tropes.py      # data/analysis/tropes.json, out/tropes/ (접촉 인화지, index.html)
open out/tropes/index.html                        # 범주별 전체 커버 확인
```

- 얼굴: OpenCV YuNet(opencv_zoo, data/models 에 자동으로 받음). 긴 변 960px 로 키워 찾고, 좌표는 커버 대비 비율.
- 범주: 묶음(People · Type · Composition · Medium)마다 우선순위 규칙으로 커버당 한 범주. 재료는 얼굴, OWLv2 상자, 글자 영역, 레이아웃 칸, 구도선·대칭·현저성, 획, SigLIP 2 제로샷(커버 전체에 대해 z-점수). 기준값은 480px 에서 맞춘 첫 시안이라 원본으로 다시 맞춘다.
- .venv-ml 에는 opencv-python-headless 와 scipy 를 더 넣는다. mediapipe 최신판은 controlnet_aux 와 충돌하므로 넣지 않는다.

## 웹 뷰어

```
.venv-ml/bin/python scripts/embed_siglip.py   # 전체 인상 축용 임베딩 (새 커버만)
.venv-ml/bin/python scripts/build_web.py      # data.js, trace/, atlas/, thumbs/, large/ 갱신 (약 1분)
open web/index.html                           # 서버 없이 file:// 로 열린다
```

레퍼런스는 Poster World(EPFLECALLab_PosterWorld_JPG/). 웹은 미리 만든 데이터를 읽기만 한다.

축 하나가 두 화면을 함께 정한다. 머리줄의 축과 원본/흔적 토글은 두 화면이 같이 쓴다.

| 축 | 격자(전체 보기) | 세 폭 보기의 양옆 | 흔적 |
|---|---|---|---|
| Spine | 1–1344 순서, 빈 칸 = 박스 세트·미확보 | 앞뒤 번호 | 번호 |
| 개봉 연도 | 해마다 한 열 | 같은 해 앞뒤 | 연도 |
| 전체 인상 | SigLIP 2 UMAP 지도 | 코사인 최근접 | 12×15 모자이크 |
| 색 | 가로 색상(330°에서 자름), 세로 명도, 오른쪽 무채색 | Lab 히스토그램 Hellinger | 팔레트 띠 |
| 구도선 | 선 방향 지도 | 방향 히스토그램 Hellinger | 흐린 선분 |
| 텍스트 배치 | 글자 점유 지도 | 점유 격자 코사인 | 글자 영역 |
| 대비 | 명도 분포 지도 | 명도 히스토그램 Hellinger | 흑백 흐림 |
| 획 | 획 통계 지도 | 방향·굵기·잉크·결 코사인 | 선화 |
| 레이아웃 | 글자·그림 점유 지도 | 코사인 | 글자/그림/빈 공간/틀 블록 |
| 물체 | 물체 라벨 지도(IDF) | 코사인 | 검출 박스 |

- 지도는 build_web.py 가 축마다 UMAP → 격자 칸 일대일 배정(헝가리안)으로 미리 만든다.
- 세 폭 보기: 가운데를 다시 누르면 다음 축, 양옆을 누르면 그 커버가 가운데로. 위·아래 줄(각 3칸, 3×3 배치)은 다른 클러스터로 빠지는 출구다.
  - 위 줄: 지금 축에서는 멀지만 다른 축에서 가장 가까운 커버 셋(칸마다 다른 축, 열 때마다 축 순서를 돌린다)
  - 아래 줄: 지금 축 지도에서 먼 35% 중 무작위 셋(메타 축은 전체 인상 지도에서)
  - 띠는 커버를 잘라 보여 주므로, 가운데로 날아갈 때 출발점은 띠가 아니라 잘린 커버 전체의 사각형이다(비율이 찌그러지지 않게).
  - 간격은 가로세로 모두 --gap 하나.
- 검색(머리줄 오른쪽, `/`): 제목·감독·나라·연도·spine(`#2`, `2`)·물체 라벨(영어, 예: horse). 여러 단어는 모두 맞아야 하고 악센트는 무시한다. 맞는 커버만 밝게 남고(셰이더 강조, 300ms) 개수는 'N / 1096편'. 목록 최대 8개, 항목에 올리면 격자 미리보기, Enter·클릭이면 세 폭으로. ×는 늘 보인다.
- 세 폭 안에서 가운데로 날아가는 커버는 날아가는 동안 z-index 를 올린다. 안 올리면 DOM 순서상 뒤에 오는 오른쪽 패널·아래 띠의 검은 배경 밑으로 지나간다.
- 키: 1–9, 0 축 · Space 원본/흔적 · ←→ 양옆 · ↑↓ 출구 · Backspace 뒤로 · Esc 닫기
- 주소 해시: `#<축>[,t][,tri=<spine>]` (예: `#strokes,t,tri=4`). 예전 형식도 읽는다.

성능과 전환

- 격자는 WebGL2 캔버스 한 장, 인스턴싱으로 1,096장을 그리기 호출 한 번에 그린다. 원본은 atlas/cover, 흔적은 atlas/<축>(80×100 타일, 32열)에서 읽는다.
- 축별 화면 좌표는 처음과 창 크기 변경 때만 계산해 캐시한다. 전환 순간에 출발·도착 좌표와 타일별 시차를 GPU 버퍼에 한 번 올리고(약 0.4ms), 이후 프레임은 시간 값 하나만 넘긴다. 보간·시차·원본/흔적 크로스페이드는 정점 셰이더가 한다. 움직이는 것이 없으면 그리지 않는다.
- 아틀라스는 텍스처로 한 번 올리고 밉맵을 만든다(가장자리 1.5px 안쪽에서 읽어 번짐 방지). 흔적 텍스처는 최근 3개만 GPU 에 둔다. 첫 화면 뒤 쉬는 시간에 모든 흔적 아틀라스를 미리 받아 디코드한다.
- WebGL 은 file:// 이미지를 텍스처로 못 쓴다(교차 출처). 그래서 build_web.py 가 atlas/<축>.js(data URL)도 만든다. jpg 는 CSS(확대 미리보기)용.
- 이력: DOM 타일(width/height 전환이 매 프레임 레이아웃) → 캔버스 2D(매 프레임 2,192번 고품질 축소) → WebGL2.
- 화면 사이 이동은 위층의 이미지 한 장(#flyer)이 격자 타일 ↔ 가운데 패널 사이를 난다. 나는 동안 격자의 그 타일과 패널은 숨긴다. 흔적 모드에서는 흔적이 난다.
- 확대 미리보기(#lens) 한 장이 타일 사이를 미끄러진다. 타일을 키우지 않는다(겹침 꼬리 버그).
- 글자 전환은 빠지는 애니메이션을 끝나는 즉시 cancel 한다(fill 을 남기면 글자가 다시 사라진다).
- 시스템 '동작 줄이기'가 켜져 있으면 애니메이션을 끈다.

확인: `node scripts/browser_check.mjs <file://…/web/index.html#축> <출력 폴더> <steps.json>` 이 실제 시간으로 헤드리스 Chrome 을 띄워 클릭·키·입력·스크린샷을 재현한다(steps: wait, shot, moveTile, clickTile, clickSel, key, type, burst, eval). 전환은 burst(연속 촬영)로 프레임을 보고 판단한다. `--virtual-time-budget` 스크린샷은 캔버스 rAF·CSS 전환과 맞지 않아 쓰지 않는다.

- thumbs/, large/, strokes/, atlas/ 는 커버에서 나온 이미지다. 2026-11-12 결정으로 일단 ccw.umjo.me 에 공개한다(원본 posters/ 는 올리지 않는다). trace/ 와 data.js 는 기계가 추출한 흔적과 메타데이터다.

## 알려진 한계

- 그래픽을 글자로 잡는다: C 로고, Seven Samurai 의 동그라미 무늬. `source: "detect"` 로 구분은 된다.
- 손글씨, 장식 레터링은 영역은 잡아도 철자가 틀린다(Seren Samuran, Beautza Beast). 영역이 목적이라 문제는 작다.
- Vision 이 라틴 글자를 키릴 문자로 읽는 경우가 있다(Hard Boiled 의 HARD → HАРП).
- 세로 띠의 연도는 일부만 읽힌다.
- 시각 분석의 `art` 는 텍스트 다각형을 모두 가리므로 글자로 잘못 잡힌 그림도 빠진다. Seven Samurai 는 동그라미 6개 중 4개가 가려진다.
- 텍스트 대비의 glyph/ground 구분은 면적이 작은 쪽을 글자로 본다. 굵은 글자가 상자를 거의 채우면 뒤바뀐다(The 400 Blows 의 blows). WCAG 비 자체는 대칭이라 영향 없다.
- 커버 바깥 테두리(Grand Illusion 의 액자선, Walkabout 의 회색 테두리)는 그대로 선분으로 잡힌다.
