# 크라이테리온월드 — 작업 인계

**사용자에게는 항상 한국어로 답한다.**

폴더 구조, 실행 방법, 평가 결과는 README.md 에 있다. 이 문서는 README 에 없는 맥락과 결정, 다음 할 일을 담는다.

## 프로젝트

크라이테리온 컬렉션 커버를 AI로 분석해 탐색하는 개인 웹 인터랙티브. 수익화하지 않는 팬 프로젝트.
영감은 취리히 디자인 박물관의 EPFL+ECAL Lab "Poster World"(색, 구도선, 물체, 타이포그래피, 텍스트 위치, 대비 축으로 포스터 세 장을 짝지어 보여주는 설치 작품).

- 레퍼런스 사진: EPFLECALLab_PosterWorld_JPG/ (설치 전경 10장, © EPFL+ECAL Lab / Daniela&Tonatiuh). 공개 레포·웹에 올리지 않는다. 읽어낸 문법:
  - 세로 패널 세 개에 짝지은 포스터, 하단에 "The 3 posters are associated based on <축>" 라벨. 축: Color, Typography layout, Object recognition, Main composition line.
  - 다이얼로 원본 ↔ 특징 시각화 전환. 색은 주요 색 비율을 가로 띠로 쌓고, 구도선은 푸리에 분석으로 주된 방향만 남긴 흐린 흑백. 시각화마다 질문 한 줄("Which color catches your eye?").
  - Global analysis: 검은 바탕에 전체 포스터를 작은 타일로 격자에 맞춰 세 패널에 걸쳐 배치.
  - 조작은 터치패드(포스터 고르기) + 언어 버튼 + About.
- 방향: 해체주의적 접근 + 기술. 물리엔진으로 부수기, 바운딩 박스 글리치 같은 흔한 문법보다 의미 층위의 균열을 선호하는 쪽으로 논의됨.
- 후보 컨셉
  - 오독 아카이브: VLM 에게 커버만 보여주고 영화를 추측하게 한 뒤 실제 정보와의 간극을 보여준다. 커버에 제목이 인쇄되어 있으므로 텍스트 영역을 지운 커버를 줘야 성립한다(텍스트 영역 작업을 먼저 한 이유).
  - 문법 교환: SAM 으로 분리한 레이어를 임베딩상 가장 먼 커버와 맞바꾸고 "크라이테리온다움" 점수를 매긴다. 저작권 리스크가 가장 큼.
  - 버전 사이 경로: 같은 영화의 판본 커버 사이 임베딩 경로. 최신판만 모으기로 해서 현재는 보류.
- 크라이테리온 고유 축 후보: spine 번호(정전의 순번), 개봉 포스터와의 거리(TMDB), 영화 프레임 사용 여부.

## 결정 사항

- 커버는 같은 영화당 최신판 하나만 모은다.
- 박스 세트도 포함한다(2026-11-12 사용자: "박스팩 다 포함해", 이전 결정 '박스 세트 수록작은 뺀다'를 뒤집음). DB 에 kind(film/box)와 box_set 열. 박스 수록 영화(자기 spine 있는 것) + 박스 자체(spine). spine 없이 박스에만 든 영화(Wikipedia 두 번째 박스 표, 약 128행)는 아직 안 넣었다.
- 텍스트 영역에는 크라이테리온 세로 띠, 상단 띠 글자도 포함한다(디자인이 시기마다 달라서).
- 무료 오픈 모델만 쓴다. 분석은 로컬에서 일괄 처리해 JSON 으로 저장하고, 웹은 그 데이터를 읽기만 한다.
- 저작권: 고해상도 원본(posters/)과 레퍼런스 사진은 공개 레포·서버에 올리지 않는다. 뷰어용 커버(web/large 480px, thumbs, 아틀라스)는 2026-11-12 사용자 결정으로 일단 공개(cc.umjo.me), 장기적으로 다시 고민한다. 메인 화면은 기계가 추출한 흔적(팔레트, 영역, 선)으로 구성하는 방향이 리스크와 컨셉 양쪽에 맞다고 논의됨.

## 현재 상태 (2026-11-11)

- 영화 DB: data/criterion.sqlite, 1,344행 = spine 1–1344 전부(단독 1,099, 박스 수록 155, 박스 자체 90). 박스 자체의 criterion_id 는 'box:<id-slug>'(data/raw/boxsets_cdx.txt 의 Internet Archive 주소와 이름으로 맞춤, 이름 속 1–9 숫자가 다르면 거절). 이전 DB 는 scratchpad 에 백업했었다.
- 커버 수집 완료(2026-11-11): 1,096/1,099. 빠진 3편은 DB 에 criterion_id 가 없다: 212 Ingmar Bergman Makes a Movie(실은 Bergman 3부작 박스 보너스 디스크 → BOX_SET_ONLY 에 넣을지 사용자 결정 대기), 406 Martha Graham: Dance on Film, 1329 Welcome II the Terrordome. scripts/fetch_covers.py (Internet Archive 보관본 → Criterion S3). 결과는 data/raw/cover_sources.csv, 로그는 out/logs/fetch_covers.log. 실패 건은 이 CSV 의 status 로 확인.
- 텍스트 영역 파이프라인 완료: PaddleOCR 검출 다각형(모양) + macOS Vision 원본·50%(글자) 결합. scripts/run_text_regions.py 가 새 커버만 처리한다.
- 색·대비·구도선 분석 완료(2026-11-11): data/analysis/visual/, 커버 전체(full)와 그림 부분(art) 두 버전, 지표와 축별 비교 벡터.
- SigLIP 2 임베딩 시험(2026-11-11): .venv-ml(Python 3.12, uv, torch·transformers 5)을 따로 만들었다. 기존 .venv(3.9, PaddleOCR)와 섞지 않는다. scripts/embed_siglip.py → data/analysis/embed/, scripts/render_embed.py → out/embed/ (격자 UMAP 지도, 세 폭 짝, 기법 제로샷 상위). base 모델로 667장 25초(M1 Max, MPS).
  - 짝이 시각보다 의미로 묶인다(Seven Samurai → Sanjuro, Three Outlaw Samurai). SigLIP 이 커버의 제목 글자를 읽기 때문. 시각 축으로 쓰려면 art 버전(텍스트 가린 이미지)을 넣어야 한다.
  - 기법 제로샷은 sigmoid 확률이 0.00–0.09로 낮아 절대값으로는 못 쓴다. 라벨 안 순위는 typographic, collage, watercolor 정도가 그럴듯하고 photograph 는 틀린다.
- 웹 뷰어(2026-11-11, 사용자 피드백으로 여러 번 개편): web/index.html. 축 10개(Spine, 개봉 연도, 전체 인상, 색, 구도선, 텍스트 배치, 대비, 획, 레이아웃, 물체)가 격자 배치와 세 폭 짝을 함께 정한다. 원본/흔적 토글 공용. 격자는 WebGL2 인스턴싱(좌표 보간은 셰이더, 아틀라스는 data URL JS), 화면 이동은 #flyer. 세 폭 보기는 3×3: 위 줄 다른 축으로 이어진 커버, 아래 줄 먼 곳의 커버. 세부는 README.
  - 사용자 취향: 담백하게. 설명 문구·수치 캡션·조작 안내는 뺐다. 격자 라벨은 'Arranged by', 세 폭 보기 라벨은 'Grouped by'(2026-11-12 사용자 선택, 이전 'Covers are sorted according to' / 'Neighboring covers are associated based on'). 포스터 밑은 제목, #spine · 연도 · 감독만.
  - 2026-11-12 개편: 세 폭 보기는 4열(이전 · 지금 · 상세 정보 · 다음) + 위·아래 줄. 이동은 축마다 토러스 격자(META.rings, 좌우 = 줄 안 순환, 위아래 = 줄 사이 순환, 모든 커버를 돈다). 최근접 둘을 양옆에 두면 A↔B 고리가 생겨 버렸다.
  - UI 는 영어로 통일(축 버튼 Spine/Year/Look/Color/Lines/Type/Contrast/Strokes/Layout/Objects). 상세 정보의 번호는 '#834'.
  - 단축키는 e.code(물리 키)로 읽어 한/영 상태와 무관. 이동 q w e / a s d / z x c(화면 3×3과 같은 배치, s = 다음 축), 축 1–9·0, space 눈, f·/ 검색, esc, backspace 뒤로, 방향키 보조. 팝오버(0.2초)는 머리줄 버튼에만, 포스터에는 띄우지 않는다.
  - 흔적 캔버스는 눈을 켰을 때만 그린다. 흐림은 필터 대신 작게 그려 키운다(구도선·대비 축 버벅임 원인이었음).
  - 검증은 scripts/browser_check.mjs(실제 시간 CDP). 헤드리스 --virtual-time-budget 스크린샷은 캔버스와 맞지 않는다.
  - 2026-11-12 추가: 순회 줄 = 이름 붙일 수 있는 범주(build_web.py item_categories → META.rowNames, 1열 왼쪽에 세로로 표시). 색 = 색상 이름, 대비 = 명도×대비 삼분위, 구도선 = 선 양×주 방향, Type = 제목 위치×글자 양, 획 = 잉크×굵기, Layout = 주된 면×제목 위치, Objects = 그 커버의 드문 물체, Look = KMeans 묶음(시대·나라·물체로 이름). Spine·Year 는 고르게 나눈 범위. 사용자: "Row 말고 유의미한 정보, 적을 게 없으면 적을 게 있는 방식으로 필터를 적용".
  - 상세 정보: Wikidata(scripts/fetch_wikidata_extra.py → data/raw/wikidata_extra.json: 상영 시간, 화면비, 흑백/컬러, 언어, 장르, 촬영, 음악, 각본, 출연, 원제)와 criterion.com 보관본 Film Info(scripts/fetch_criterion_meta.py → data/raw/criterion_meta.json: 상영 시간, 흑백/컬러, 화면비, 언어, 판매 형식 4K/Blu-ray/DVD). Criterion 쪽이 우선. 보관본 수집은 archive.org 제한 때문에 매우 느리다(분당 몇 장). criterion.com 직접 접근은 봇 차단이라 쓰지 않는다.
  - 머리줄: 눈(맨 왼쪽) · 축 · … · 개수 · 검색 · 닫기. 오른쪽 아래 크레딧(umjo.me, Poster World, Not affiliated). favicon.svg(눈), og.jpg.
  - 격자에서 흔적을 켠 채 축을 바꾸면 제자리에서 흔적이 번져 바뀐 뒤(SWAP 420ms) 움직인다. 세 폭 보기 등장 순서는 q w e a s d z x c, 정보 글자는 맨 끝.
  - 구도선 축은 full 버전이라 글자 획이 선으로 많이 잡힌다. art 버전과 비교할 것.
  - v0.3.0 모바일(폭 720px 이하, CSS 와 JS narrow() 가 같은 기준): 축 버튼 가로 스크롤, 검색은 펼치면 머리줄을 덮음, 세로 화면이면 지도 축·연도 축을 전치. 터치는 훑으면 미리보기(가장 가까운 타일 18px 안), 같은 커버를 한 번 더 누르거나 미리보기를 누르면 연다. 세 폭 보기는 이전·지금·다음 한 줄(양옆은 안쪽 가장자리만) + 위·아래 띠 + 아래 정보, 스와이프로 이동(밀어 올리면 아래 줄).
  - 같은 버전: 세 폭 보기를 열 때만 pushState(브라우저 뒤로 가기 = 닫기, ×·esc 는 history.back()), 격자 방향키 이동 + enter, 세 폭 보기 포커스 이동·복귀, WebGL 2 없으면 안내 문구. 개수 표시는 늘 '밝은 커버 / 전체 편수'.
  - 줄 이름은 가운데 줄 것만(사용자), 1열 왼쪽 여백에 세로로, 글자는 작게(--t1, 키운 것은 사용자가 되돌림). 가운데 커버 윗변부터 아래 띠 끝(폰은 정보 영역 옆)까지 쓸 수 있다. 폰도 양옆 여백 26px(NARROW_GUTTER)을 두고 같은 세로쓰기이며, 검은 정보 영역은 양옆 여백까지 덮는다(--gut, 날아오르는 커버가 틈으로 보이지 않게). 줄 이름은 그 위 층(z 7). 폰 크레딧은 왼쪽 한 줄(이름 · 버전) · 오른쪽 네 줄(링크 앞 줄바꿈, 두 묶음 사이 6px). 380px 미만 폰은 큰 축 이름을 크레딧 위로 올린다. 축 드롭다운은 클릭·탭으로 열면 포커스를 옮기지 않고(키보드일 때만), 호버하면 옅은 바탕.
  - 화면 구간은 세 개(사용자): S ≤ 720(폰 배치) · M 721–1024 · L ≥ 1025(BP_S, BP_M). 격자 지도 전치만 폭이 아니라 화면 방향(세로가 1.1배 넘게 길 때) 기준.
  - 글자 크기는 구간마다 끊지 않는 가변 토큰 --t1–--t6(폰 390px 값과 1440px 값을 잇는 clamp 직선). 폰 전용 글자 크기 지정은 두지 않는다.
  - 축 드롭다운: S·M 에서, 그리고 L 이라도 머리줄이 넘치면(fitAxes() 가 잰다) body.axcompact 로 바꿔 지금 축 버튼 하나 + 세로 목록. 목록은 가로 세그먼트와 같은 결(#111 바탕, 28px 알약 버튼, 고른 것은 흰 알약, Spine·Year 와 나머지 사이 1px 선). 바깥 누름·esc 로 닫고, 위아래 화살표로 이동. 세 폭 보기에서 그 커버에 없는 축은 비활성.
  - 머리줄 높이는 모두 36px(축 버튼 28 + 여백 4×2). 눈(18px)·돋보기(16px)·닫기(16px SVG)는 같은 선 굵기(약 1.2px).
  - v0.3.1 상세 정보: #spine · 제목 · 감독 · 연도는 커버 윗변에, 표와 Show More 는 아랫변에 붙인다(.body). 표는 늘 보이는 Country · Runtime · Aspect ratio · Language 네 줄(영화마다 일관되게), 표 아래 Show More 를 누르면 Color · Cinematography · Music · Written by 가 높이를 키우며 펼쳐지고 줄마다 떠오른다(접을 때는 높이를 줄인 뒤 다시 그림. infoOpen, 다른 커버로 옮겨도 유지). 정보 칸은 절대 스크롤하지 않는다(사용자). .info 는 3열 전체(grid-row 1/4)를 차지하고 평소엔 padding 으로 커버 높이만 쓴다(infoPads, 가운데 커버 offsetTop 기준). 넘치면 fillInfo 가 넘치는 만큼만 padding 을 위아래 반씩 줄여 가운데를 중심으로 늘리고(padding transition .42s, 펼치기 애니메이션과 같은 시간), 그래도 넘치면 값 한 줄 → 제목 축소 → 줄 간격 축소 → 아래 줄부터 뺀다. 좁은 화면은 정보 칸 258px 고정(NARROW_INFO, 접힌 내용의 95% 가 들어가는 높이, 열고 닫아도 커버 크기 그대로)이고 정보 영역이 검은 덮개 층(z 6)이다: 펼치면 펼친 줄 높이만큼 정확히 위로 늘어나(Show More 아래는 제자리, 제목부터 표까지 한 덩어리로 같은 속도), 그래도 넘치면 글자 크기를 바꾸지 않고(사용자) 더 위로 늘어나 아래 띠를 덮고(가운데 커버 윗변까지), 아래 줄로 이동할 때 날아오르는 커버(flyWithin z 5)는 그 뒤에서 나온다. 펼친 상태로 펼칠 것이 없는 커버에 가도 Show Less 는 남는다. criterion.com 링크(새 탭)는 #spine 번호에 건다. 번호 뒤 작은 화살표(Lucide arrow-up-right)를 늘 흐리게, 가리키면 점선 밑줄. 제목에 링크를 걸거나 Show More 를 호버 때만 보이게 한 시도는 사용자가 되돌렸다.
- 박스 세트 커버 시험 수집(2026-11-11, 사용자: "일단 받고 생각하자"): scripts/fetch_box_covers.py → posters_box/, 기록 data/raw/box_cover_sources.csv. 박스 세트 쪽 spine 245개 중 Wikidata 에 Criterion ID 가 있는 151편. DB 에는 넣지 않았다. 넣을지는 사용자 결정 대기.
  - 결과: 151편 전부 받음, 박스 세트 49개. 이미지 주소는 151개 모두 다르지만 그림이 같은 쌍이 14개(28편): World Cinema Project 1–4 는 디스크 한 장(2편)마다 커버 하나, Gates of Heaven/Vernon, Florida, The Emigrants/The New Land, I Am Curious Yellow/Blue(부제만 다름). 자기 커버가 있는 것은 123편(예: 284 Kanal, DVD 시기 디자인).
  - 같은 그림 판정: 24×30 흑백 상관 0.99 이상. World Cinema Project 끼리는 같은 템플릿이라 0.9–0.94 로 높게 나오니 기준을 낮추지 말 것.
- 1,096장 전부 텍스트 영역·시각 분석·임베딩·물체 검출·획/레이아웃 완료(2026-11-11). 틀: modern-strip 978, dvd-topband 93, none·unknown 21(1,092장 기준).
- 틀 판정 실패: 틀 안 작은 글자를 못 읽은 경우. 상단 띠(RoboCop, Alphaville, Nanook, Henry V, Most Dangerous Game, Il Posto, Tout va bien), 세로 띠(Spartacus, Grey Gardens, Made in U.S.A., Boyhood, Election, Now, Voyager 등). 글자 대신 띠 위치·모양으로 찾는 보조 판정을 넣을 것.
- 1, 8, 9번 커버는 옛 DVD 디자인. 절판이라 최신판일 가능성이 높지만 확인 안 됨.

## 다음 할 일

1. 색, 대비, 구도선 분석은 끝났다(scripts/analyze_visual.py, 결과와 판단은 README). 사용자가 확인용 이미지(out/visual/)를 보고 판단할 것:
   - art 버전이 텍스트 다각형을 전부 가려 Seven Samurai 동그라미처럼 글자로 잘못 잡힌 그림도 빠진다. `source: "detect"` 영역을 남길지.
   - 틀 템플릿 좌표는 10장에서 잰 값이다. frame.kind 가 none/unknown 인 커버와 다른 시기 디자인을 확인한다(위 틀 판정 실패 참고).
2. SigLIP 2: art 버전으로 다시 임베딩해 의미 묶음과 시각 묶음을 비교한다(둘의 차이 자체가 "오독" 컨셉 재료). 기법 제로샷은 so400m 모델이나 라벨 간 softmax 로 다시 본다.
   이후 후보: 기법(SigLIP 제로샷: 목판화, 수채화, 사진 등), 무게중심 축(saliency 벡터 있음), 오독 아카이브 축(의미 짝 vs 시각 짝 차이). 구도 유형 분류 모델 KU-PCP 는 야외 사진 학습이라 일러스트에 안 맞을 수 있음.
   ML 모델은 .venv-ml 에 설치한다(torch 설치는 2026-11-11 사용자 승인, controlnet_aux 추가). 큰 패키지를 새로 넣을 때는 사용자에게 확인한다.
3. 커버가 들어오면: (.venv) run_text_regions.py → analyze_visual.py → cover_checklist.py, (.venv-ml) embed_siglip.py → detect_objects.py → analyze_strokes_layout.py → build_web.py. 모두 새 커버만 처리한다(build_web 은 전부 다시, 약 1분).
4. 사용자 결정 대기: 212 를 BOX_SET_ONLY 에 넣을지, Spine 축 격자의 박스 세트 빈 칸 처리(흐린 테두리 + 박스 이름 제안), 박스 세트 커버(posters_box/)를 뷰어에 넣을지.

## 배포 (2026-11-12 준비)

- 레포 https://github.com/thumbjiho/criterion-world (public). 배포는 scripts/deploy.sh: git 에 올라간 web/ 만 모아 Cloudflare Workers 정적 자산(Worker 이름 criterion-world, 도메인 cc.umjo.me)으로 올린다. wrangler 4.160 은 pages project create 를 Workers 로 넘겨서 Pages 대신 이 방식. GitHub 와 자동 연결은 없다(커밋·푸시 후 deploy.sh). .gitignore 가 원본·레퍼런스·중간 결과·file:// 용 산출물(atlas *.js/*.jpg, strokes *.png)을 뺀다. 레포 약 123MB, 5,300 파일.
- http 에서는 아틀라스를 WebP 로 읽고 원본 아틀라스는 반 크기(cover_s.webp 0.75MB)를 먼저 올린다. 흔적 아틀라스 미리 받기는 큰 원본 아틀라스가 온 뒤. 캐시는 web/_headers.
- 로컬에서 file:// 로 열면 예전처럼 data URL JS 를 쓴다.
- 첫 로딩 화면(#boot): http 에서는 data.js 와 cover_s.webp 를 fetch 로 받으며 실제 진행률(풀린 바이트 기준, 크기는 build_web.py 가 index.html 의 /*SIZES*/ 에 적음)을 보이고, 다 받으면 본 스크립트(<script type="text/plain" id="main">)를 실행한다. file:// 에서는 ⌘⇧R 때만 더미 진행.
- 줄 이름은 모두 다르게(사용자 요청): 큰 범주는 두 번째 지표(SPLIT: 색=밝기, 대비=채도, 구도선=좌우 대칭, Type=글자 가로 위치, 획=결, Layout=그림 세로 위치)로 나누고 9단계 낱말을 붙인다. Look·Objects 는 그 줄에 많은 장르·물체를 붙인다. 마지막 수단으로 II, III.
- 버전: 오른쪽 아래 'v0.1.0 · data 2026-11-12'. 화면·동작을 바꿔 배포할 때 web/index.html 의 VERSION 을 올린다. 데이터 날짜는 build_web.py 가 META.built 에 적는다.

## 주의할 점

- 프로젝트를 ~/Desktop/Claude Code/사이드/ 에서 지금 위치로 옮긴 뒤 .venv 의 바이너리 237개에 com.apple.quarantine 이 붙어 import 가 막히고, macOS 가 막은 .so 를 지웠다. 같은 버전으로 전부 재설치하고 .venv/bin 스크립트의 옛 경로를 고쳐 복구했다(2026-11-11). 다시 옮기면 venv 를 새로 만드는 편이 낫다.
- 옛 경로는 "사이드"는 NFC, "크라이테리온월드"는 NFD 로 섞여 있어 grep/sed 로 안 잡혔다. 한글 경로 문자열은 정규화 형식을 의심한다.
- 커버 수집 시도 기록: 이전 세션에서 criterion.com, cinematerial.com 은 curl 이 Cloudflare 403 을 받았고, 그 세션의 Chrome 확장은 정책으로 막혔다. 세션·환경마다 다를 수 있으니 금지 규칙이 아니라 참고 기록이다. 2026-11-11 세션에서도 Chrome 확장은 연결되지 않았다.
- Wikipedia API, Wikidata SPARQL 은 curl 로 접근된다. User-Agent 는 개인정보 없는 일반 문자열을 쓴다.
- 시스템 Python 이 3.9 라 .venv 도 3.9. 타입 힌트에 `X | None` 을 쓰면 `from __future__ import annotations` 가 필요하다.
- 스크립트 이름을 라이브러리와 같게 짓지 않는다(paddle.py 로 했다가 `import paddle` 이 꼬였음).
- PaddleOCR 은 원본 크기로 검출하면 큰 글자를 놓친다. 긴 변 640px(det-640)이 기준.
- PaddleOCR 모델은 ~/.paddlex/official_models 에 캐시된다. 실행 시 PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True.
- Vision 은 25% 축소본(322×400)에서 성능이 크게 떨어진다. 썸네일급 커버가 들어오면 확인할 것.
- Wikipedia 목록에 오류가 있다(Inside Llewyn Davis spine 795 → 실제 794). build_film_db.py 가 Wikidata 와 대조해 출력하니 갱신할 때 메모를 확인한다.
- 백그라운드에서 다른 스크립트가 끝나길 기다릴 때 `pgrep -f "<스크립트 이름>"` 은 기다리는 셸 자신의 명령줄에도 걸려 영원히 기다린다. PID 로 기다릴 것.
- 문서를 정규식으로 고칠 때 re.S 와 `.*` 를 같이 쓰면 파일 끝까지 지운다(2026-11-11 CLAUDE.md 뒷부분을 지웠다가 복원).
