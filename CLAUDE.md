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
  예외(2026-11-13 사용자): 커버 묘사(8줄 캡션)는 Claude Sonnet(구독 에이전트)으로 만든다 → data/analysis/captions_sonnet_v3.json. 로컬 Qwen3-VL-8B 는 매체·글자 묘사가 자주 틀려서 뺐다. 에이전트에 준 지시문은 scripts/caption_prompt.txt.
- 저작권: 고해상도 원본(posters/)과 레퍼런스 사진은 공개 레포·서버에 올리지 않는다. 뷰어용 커버(web/large 480px, thumbs, 아틀라스)는 2026-11-12 사용자 결정으로 일단 공개(ccw.umjo.me), 장기적으로 다시 고민한다. 메인 화면은 기계가 추출한 흔적(팔레트, 영역, 선)으로 구성하는 방향이 리스크와 컨셉 양쪽에 맞다고 논의됨.

## 현재 상태 (2026-11-11)

- 영화 DB: data/criterion.sqlite, 1,344행 = spine 1–1344 전부(단독 1,099, 박스 수록 155, 박스 자체 90). 박스 자체의 criterion_id 는 'box:<id-slug>'(data/raw/boxsets_cdx.txt 의 Internet Archive 주소와 이름으로 맞춤, 이름 속 1–9 숫자가 다르면 거절). 이전 DB 는 scratchpad 에 백업했었다.
- 커버 수집 완료(2026-11-11): 1,096/1,099. 빠진 3편은 DB 에 criterion_id 가 없다: 212 Ingmar Bergman Makes a Movie(실은 Bergman 3부작 박스 보너스 디스크 → BOX_SET_ONLY 에 넣을지 사용자 결정 대기), 406 Martha Graham: Dance on Film, 1329 Welcome II the Terrordome. scripts/fetch_covers.py (Internet Archive 보관본 → Criterion S3). 결과는 data/raw/cover_sources.csv, 로그는 out/logs/fetch_covers.log. 실패 건은 이 CSV 의 status 로 확인.
- 빈 칸 채움(2026-11-12): spine 1–1344 전부 커버가 있다(박스 세트 자체 포함, 사용자: "세트 자체도 넣어도 될 것 같은데 빈 곳 없이"). 빈 칸 23편 중 10편은 Wayback CDX 로 영화·박스 페이지를 제목으로 찾아 보관본의 원본 이미지(og:image 또는 페이지 안 criterion-production S3 주소)를 받았고(옛 디자인 보관본에는 og:image 가 없어 시점을 거슬러 올라가야 할 때가 많다), 13편은 사용자가 직접 넣었다(data/raw/cover_sources.csv 의 status manual). 212 는 박스 보너스 디스크지만 자기 커버로 넣었다. criterion_id 21편 보충(79, 1341 은 빈 값). 원본 23장은 공개 레포에 올리지 않으므로 다른 컴퓨터에는 따로 전달한다(ccw-missing-covers-23.zip 을 posters/ 에 푼다).
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
  - 줄 이름은 가운데 줄 것만(사용자), 1열 왼쪽 여백에 세로로, 글자는 작게(--t1, 키운 것은 사용자가 되돌림). 가운데 커버 윗변부터 아래 띠 끝(폰은 정보 영역 옆)까지 쓸 수 있다. 폰도 양옆 여백 26px(NARROW_GUTTER)을 두고 같은 세로쓰기이며, 검은 정보 영역은 양옆 여백까지 덮는다(--gut, 날아오르는 커버가 틈으로 보이지 않게). 줄 이름은 그 위 층(z 7). 폰 크레딧은 가운데 정렬: 위는 Inspired by … 와 Not affiliated … 를 살짝 띄운 한 줄(폭이 모자라면 두 줄, 사용자가 두 줄도 좋다고 함), 맨 아래 이름 · 버전. 축 드롭다운은 클릭·탭으로 열면 포커스를 옮기지 않고(키보드일 때만), 호버하면 옅은 바탕.
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

## 진행 중: 범주 축(tropes 브랜치, 2026-11-12)

사용자 방향: 축마다 모델 하나가 순서를 정하는 대신, 여러 모델과 단순 처리기의 신호로 사람이 아는 범주(대칭, 가운데 큰 제목, Talking heads …)를 판정한다. "묶음(family) 하나 = 축 하나, 한 축 안에서 커버는 정확히 한 범주(줄)" — 격자 자리 하나, 세 폭 보기 위아래 이동, 줄 이름이 모두 이 전제 위에 있어서 여러 범주 소속은 쓰지 않는다. 한 커버가 여러 범주에 해당하는 효과는 축을 바꿀 때 드러난다.

- 축 구성(사용자 동의): Spine · Year | Color | People · Type · Composition · Medium (10개 → 7개). Contrast 는 Color 에(흑백·듀오톤·포인트 컬러 같은 범주로), Layout 은 Type·Composition 에, Lines 는 Composition 에, Strokes 는 Medium 에, Objects 는 People·Composition 에 흡수. Look 은 축에서 빼고 신호로만(사용자: 아깝지 않다). 빠지는 축의 흔적 그림은 새 축으로 옮긴다(People = 얼굴·인물 상자, Type = 글자 다각형, Composition = 구도선·대칭축·무게중심, Medium = 선화, Color = 팔레트 띠).
- 묶음 안 판정: 우선순위 규칙으로 하나만. 애매하면 점수 높은 쪽, 어디에도 확실하지 않으면 그 묶음의 Other. 둘째 후보는 데이터에만.
- 근거 데이터(이전 축들): 축별 최근접 이웃 6개 겹침은 대부분 1–4%(우연 0.5%), Type ↔ Layout 17.6%(중복), Color ↔ Contrast 7.8%.
- 만든 것: scripts/detect_faces.py(OpenCV YuNet, 모델은 data/models 에 자동으로 받음. mediapipe 는 최신판이 controlnet_aux 와 충돌해 쓰지 않음), scripts/build_tropes.py(판정 → data/analysis/tropes.json, out/tropes/<묶음>.jpg 접촉 인화지, out/tropes/index.html 범주별 전체 확인 페이지).
- 첫 시안은 원본이 없는 다른 Mac 에서 web/large(480px)를 posters/ 로 복사해 전체 파이프라인을 다시 돌려 만들었다. 사용자 평: "방향은 나쁘지 않은데 완성도는 아직", 고화질로 다시 하기로 함. 480px 에서 보인 한계: 크라이테리온 띠 글씨를 못 읽어 frame 판정이 비고(띠가 글자로 섞임, build_tropes 가 가장자리 세로 띠·맨 위 띠 상자를 따로 거름), 여러 줄 제목이 한 덩어리로 잡혀 title_h 가 덩어리 높이가 됨, 작은 얼굴을 놓침.
- 첫 시안 품질(내 판단): People 이 가장 좋음(Crowd·Big face 잘 맞음, Talking heads 에 두 사람 키스 장면 섞임 → Couple 로 나눌 것, Lone figure 들쭉날쭉). Type 괜찮음(Quiet type·Type only 좋음). Composition 가장 약함(대칭·대각선 납득 안 됨, Left/Right-weighted 는 사람이 쓰는 범주가 아님). Medium 은 Photo 정확, Painting 에 사진이 섞임(제로샷 한계, 코사인을 그대로 비교하면 늘 Photo 가 이겨 z-점수로 바꿈).
- 다음 할 일(원래 컴퓨터에서)
  1. 원본 posters/ 로 run_text_regions → analyze_visual → embed_siglip → detect_objects → analyze_strokes_layout → detect_faces → build_tropes. build_tropes 가 이전 결과와 바뀐 편수를 출력한다.
  2. Composition 재설계: 레이아웃 칸 대신 그림 자체를 좌우로 뒤집어 비교하는 대칭, 범주는 가운데 하나의 물체 · 프레임 속 프레임 · 격자/콜라주 · 넓은 여백 · 대각선처럼 사람이 쓰는 말로.
  3. Medium 은 범주당 대표 커버 10–20장을 사용자에게 골라 받아 SigLIP 임베딩 위 소수 예시 분류(프로토타입 또는 로지스틱 회귀).
  4. People: Couple 분리, Lone figure 기준. Color 에 Contrast 흡수.
  5. 범주가 사용자 눈에 맞으면 build_web.py 를 새 7개 축으로 바꾸고(격자 배치 = 범주별 묶음, 줄 = 범주, 줄 이름 = 범주 이름) 흔적을 옮긴다.
- 2026-11-12 저녁: 원본 커버(1,344편, 빈 칸 채움 포함)로 전체 파이프라인을 다시 돌렸다. 바뀐 판정 규칙: People 에 Couple(큰 얼굴 2개가 맞닿거나 OWLv2 couple embracing ≥ 0.3), 검출기가 놓친 얼굴은 SigLIP 단서(face_close, people)로 보완. Composition 은 Grid / collage(길게 뻗은 가로·세로 선) · Symmetric(글자 가린 윤곽 지도와 거울의 상관 ≥ 0.55) · Single object · Open space · Diagonal · Full-bleed image · Other 로 다시 짰다. Medium 제로샷은 커버 전체에 대한 z-점수. 결과: People Crowd 103 · Couple 97 · Talking heads 182 · Big face 318 · Lone figure 88 · Figures 335 · No one 221 / Type Type only 22 · Big centered 204 · Big 119 · Quiet 171 · Top 281 · Middle 289 · Bottom 258 / Composition Grid 50 · Symmetric 81 · Single object 112 · Open space 138 · Diagonal 177 · Full-bleed 158 · Other 628(가장 큰 남은 과제) / Medium Photo 434 · Illustration 222 · Painting 274 · Graphic 203 · Typographic 211. 확인 페이지는 posters 에서 만든 썸네일과 DB 제목을 쓴다(out/tropes/thumbs).
- 사이트 적용(v0.4.0, 2026-11-12 밤, 사용자: "이전보다 나으니 적용, 더 개선해 나가면 될듯" + 화면비·길이·나라 뷰): 축 = Spine · Year · Country · Runtime · Aspect | Color | People · Type · Composition · Medium(키 1–9·0). Look·Lines·Contrast·Strokes·Layout·Objects 와 이전 Type(text) 축은 뺐다. build_web.py category_axes: 범주 축의 격자 = 순회 줄을 그대로 쌓은 것(rings_grid, 왼쪽 여백에 범주 이름), 줄 안 순서 = Look 지도(SigLIP UMAP) 가로. 큰 범주는 두 번째 말(People·Type·Composition → Medium, Medium → People)로 나누고 그래도 겹치면 연도 범위. 나라 = 지역 순, 한 줄 이상인 나라는 따로(나라, 연도 범위), 나머지는 지역 이름 줄, 8편 미만 나머지는 같은 지역 큰 나라 줄에 붙임(United Kingdom & Ireland). 상영 시간 = 짧은 순 범위 줄, 화면비 = 1.19 · 1.33 · 1.37 · 1.66 · 1.78 · 1.85 · 2:1 이상 → 연도. 흔적: People = 얼굴(채움)·사람(테두리) 상자(trace.people), Type = 글자 다각형, Composition = 구도선, Medium = 선화(TRACE_AS), Country·Runtime·Aspect = 글자. 나라 축의 세계 지도식 격자는 아직(지금은 지역 줄을 쌓은 격자).
- 자유 배치(v0.5.0, 사용자: "재미없게 직사각형으로 배치하지 말고 나라는 지도 모양, 화면비는 진짜 화면비 모양, 다른 애들도 자유롭게"): build_web.py free_layouts 가 축마다 목표 모양(화면 단위)을 그리고 pack()(헝가리안, 빈칸 허용)으로 칸에 배정한다. META.gridLabels[축] = 덩어리 이름표(칸 좌표). Country = 경위도(대양 줄이고 유럽 넓힌 proj) 위 나라별 해바라기 원반(미국·일본·이탈리아 등은 SHAPE 로 늘이고 돌림, 오래된 영화가 가운데). Aspect = 비율마다 실제 비율 모양 블록을 나란히(아래 맞춤). Runtime = 3분 막대(폭 2칸) 분포. Color = 색상환 원판(무채색 가운데, 유채색 각도 = 색상 순위, 반지름 = 어두울수록 바깥). People·Type·Composition·Medium = 범주 섬(섬 자리 = 범주 평균 임베딩 PCA, 겹침 밀어내기, 섬 안 = 범주 안 PCA 를 원반에 일대일). 자유 배치 축은 폰에서도 돌리지 않는다. 세 폭 보기 순회(META.rings)는 그대로 범주 줄.
- v0.6.0(사용자 피드백): Color 는 원래 배치(색상 × 명도 격자)로 되돌림("원래가 낫"). Country 뒤에 Natural Earth 1:110m 육지 윤곽(data/raw/ne_110m_land.geojson, public domain)을 같은 투영으로 희미하게(META.gridBg, 뷰어 #gridbg SVG). Runtime·Aspect 는 사용자 마음에 듦. People·Type·Composition·Medium 은 원형 섬이 아쉽다고 해서 범주 뜻을 그린 픽토그램 모양으로 채운다(build_web.py _pict / pictogram_cells: Crowd = 점 여러 개, Couple = 겹친 두 원, Big face = 얼굴, Type = 포스터 틀 안 제목 띠 위치, Grid = 3×3, Diagonal = 사선 띠, Photo = 카메라, Typographic = A 등). 그림 안은 Look UMAP 을 헝가리안으로 맞춤, 그림은 줄지어 놓고 아래에 이름. 닫기 버튼은 숨을 때 폭을 접어 검색이 오른쪽 끝에 맞는다.
- v0.7.0(사용자 피드백): 픽토그램은 '너무 별로'라 뺐다. People·Type·Composition·Medium 격자 = 그 축의 모델 신호 벡터(People: 얼굴 수·크기·위치, 사람 상자, 포옹, SigLIP face_close/people / Type: 글자 넓이·제목 크기·위치 + 글자 분포 PCA / Composition: 긴 선, 대칭, 현저성, 여백, 구도 문장 z + 구도선 분포 PCA / Medium: 매체 문장 z, 잉크, 흑백 + 획 분포 PCA)를 표준화해 UMAP → pack(빈칸 허용 연속 지도). 사용자: '모델 결과값이 벡터성을 가지고 비슷한 위치끼리 상관성을 찾을 수 있게, 맘대로 배치하지 말고'. 범주 이름은 그 범주가 모인 자리(centroid_labels). Country = 실제 국경(Natural Earth 1:50m admin-0) 모양을 영화 수에 맞게 키우고 줄여 그 안을 채운 비연속 카토그램(country_cartogram), 큰 나라부터 실제 위치 가까이 한 칸 띄워 놓음, 배경 = 놓인 나라 윤곽(web/map/country.js, 축을 열 때 받음). 사용자: '일본도 일본 나라 모양으로'. Color 는 원래 격자. 줌: 1 → 2.2 → 4.5배(ZOOMS), 더 들어가면 그 자리 커버 상세. 트랙패드 핀치(ctrl+휠, Safari gesture), 터치 두 손가락, +/− 버튼(#zoom)·키(=, -), 확대 중 끌기·두 손가락 스크롤로 이동, esc 로 전체. 셰이더 uView, hit/near/rect·이름표·배경이 같은 변환을 쓴다. 축을 바꾸면 전체 보기로. browser_check.mjs 에 drag·wheel·code 단계.
- v0.7.3 줌 다듬기(사용자): 단계 1 → 2 → 4 → 8배로 딱 떨어지게(핀치 한 번 = 한 단계, 200ms 조용할 때까지 다음 단계 막음, 터치는 두 손가락 거리 1.35배마다). 8배에서 더 들어가면 상세. 4배 이상은 보이는 커버만 #hires 이미지로 덮음(4배 thumbs, 8배 large; 흔적·검색·축 이동 중엔 덮지 않음). 브라우저 자체 줌 금지(ctrl+휠, ⌘/ctrl = - 0 은 사이트 줌, Safari gesture, viewport user-scalable=no, touch-action pan-x pan-y). 상세를 닫으면 줌을 유지한 채 마지막 커버를 가운데로(centerOn). 상세 안에서 축을 바꿔도 줌 유지. 끌기·스크롤 이동이 끝나면 관성(속도×260ms) 뒤 화면 가운데에 가장 가까운 커버가 정가운데 오게 맞춰 선다(G.snap). 호버 미리보기는 타일 자리에서 커지고 이동 .34s.
- v0.8.x(사용자): People 은 UMAP 대신 뜻이 분명한 좌표 — 가로 = 사람 수(max(얼굴, 사람 상자), log, 10+), 세로 = 사람이 화면을 차지하는 비율(max(person_max, 2.5×face_max, 2×face_sum), √). 눈금(0 1 2 3 5 10+, 5% 20% 50% 100%)과 축 이름(META.gridExtra.people.axes, 아래 가운데·왼쪽 가운데). Couple 판정 커버의 10–90% 범위에 배경 영역 네모(gridExtra.regions), 평소엔 거의 안 보이고 호버하면 밝아지며 이름. 줌 이동 물리: 끄는 동안 1:1, 놓으면 속도×TAU(325ms) 만큼 간 자리에서 가장 가까운 커버를 목적지로 임계 감쇠 스프링(OMEGA 9.5/s, 놓을 때 속도 이어받음, G.release). 트랙패드 스크롤은 속도를 재다 80ms 끊기면 release. 잡으면 진행 중인 움직임을 멈추고 이어받는다.
- v0.9.x(사용자): 용어 '클러스터 라벨' = 격자 위 무리 이름(나라 이름, Couple 영역, 범주 이름 모두). People·Type 은 느슨한 축(soft_axes: 축 좌표에서 출발한 UMAP 을 축 방향으로 돌려 세우고 35% 당김) — 사용자: '가로세로 축이 정확할 필요는 없다, 사람들이 몰랐던 비슷한 포스터끼리 모여 공통점·패턴·인사이트를 찾게'. Type 재료에 텍스트 기울기(text_tilt: 틀·왼쪽 띠 뺀 글자 영역 minAreaRect, 0–90°)를 대각선(15–75°, 두 배 무게)과 세로쓰기(75°+)로 나눠 넣음. 영역 네모(dense_region: 가장 빽빽한 곳 중심 60%): Couple, Diagonal type, Vertical type. Country 배경 = 모든 나라(흐리게, 작은 섬·남극 생략) + 영화 있는 나라 키운 윤곽(조금 밝게). Aspect = 액자 바깥 비율이 실제 화면비와 같게 칸 간격을 실수로 조율, 빈자리는 고르게 흩음, 배경에 액자 선(map/aspect.js).
- v1.0(사용자): Country = 늘리지 않은 메르카토르 지도(v1.0.2, 처음엔 Equal Earth)(country_map, 폭 1000 단위 연속 좌표 META.grids.country={free,w,h}, 커버마다 타일 크기 c.gw.country). 사용자: '지도를 늘려 포스터에 맞추지 말고 포스터를 지도에 맞춰'. 커버를 실제 국경 안 격자점에 채우고 나라 넓이에 비해 영화가 많으면 타일을 작게(최대 4.5 단위, 너무 작은 나라는 최소 1.1 로 그 자리에 모음). 배경 = 모든 나라(흐리게) + 영화 있는 나라(조금 밝게). 줌 1·2·4·8·16배. 고화질 덮개는 화면 타일 크기 34px 이상에서(150 기기 px 넘으면 large). People 클러스터 라벨 = 자연스럽게 갈린 무리에 No people · Big face · Couple · Crowd(Couple 을 억지로 떼지 않음). 영역 = 그 무리 핵심 칸만(사이에 낀 다른 커버 제외), 라벨은 은은히 늘 보이고 호버하면 밝게. 얼굴이 꽉 찬 커버가 No one 으로 잡히던 것: OWLv2 큰 얼굴(점수 0.3+) 허용, SigLIP face_close 0.7 로 Big face. 사용자 제안: 라벨링을 모델에만 맡기지 말고 사용자가 직접(다음 할 일: 영역 그려 이름 붙이는 편집 모드).
- v1.1(사용자): People 에서 Couple 을 떼던 신호(embrace, faces_touch)를 뺌. People·Type·Composition·Medium 은 칸에 맞추지 않는 소수점 좌표(relax: 목표 자리에서 겹치는 타일만 밀어냄). 클러스터 라벨은 모든 축에서 평소 숨김, 영역이나 그 무리 커버에 호버하면 영역이 밝아지며 이름이 떠오름(애니메이션), 활성 라벨은 화면 밖이어도 가장자리에 붙어 보임. Type·Composition·Medium 의 범주 이름도 클러스터 영역으로. Country 는 나라별 국경 경로가 영역(isPointInFill, 그 나라 커버 호버로도). Year 는 10년 단위 영역(1960s). 좌표 라벨(눈금)은 확대해서 화면 밖으로 나가면 가장자리에 붙음. Aspect 라벨은 늘 보이고 가운데 정렬(.axis 페이드인 애니메이션이 transform 을 덮어써 translate 속성으로). Spine 한 줄은 10 의 배수(5 의 배수가 타일을 12% 넘게 키우면 5), 왼쪽 번호 자리 34px 늘 남김.
- v1.2(사용자): 포스터끼리 겹치면 안 됨 → deoverlap(겹침이 0 이 될 때까지 밀어내고 남으면 가장 가까운 빈자리로, 틈 0.06; Country 는 나라 사이도). 땅에 비해 영화가 적은 나라는 k-평균 중심(국경 안 점)으로 고르게, 1 편이면 한가운데. 클러스터 영역 = 무리 밀도 등고선의 부드러운 곡선 테두리(smooth_outline: 가우시안 + 등고선 + 차이킨), 포스터 위층 #gridtop 에 평소 옅은 점선, 호버하면 밝은 선과 옅은 채움, 판정은 곡선 안(isPointInFill). Year 는 10 년 직사각형, 반응 범위는 그 열 전체(위 빈 곳과 아래 연도 숫자 포함).
- v1.2.5: 클러스터에 호버하면 그 범주 전체(all)는 그대로, 나머지는 밝기 .4 로(검색 강조는 .16, uFloor). 켜는 커버는 그 범주 중 중심이 테두리 안에 있는 것. 사용자 질문 '휴먼 라벨링으로 분류를 더 좋게' → 제안: 고쳐 주기 + 대표 예시 고르기를 로컬 전용 편집 모드로(data/labels.json), 나중에 헷갈리는 것만 묻기·세 장 비교.
- v1.3(사용자): 클러스터끼리 겹치지 않게 partition_regions(범주마다 밀도, 각 지점은 밀도 최고 범주가 차지), 테두리는 수직·수평선만(3칸 블록 경계 rect_outline). 영역 안 포스터 중 그 범주가 45% 미만이면 표시 안 함, People 에서 Couple 라벨은 뺌(사용자: 억지스럽다). Type 좌표: 가로 = 제목 좌우, 세로 = 제목 위아래(위쪽 제목이 위, 격자를 포스터 한 장처럼), 제목 크기 3배 무게로 큰 것끼리·작은 것끼리.
- v1.3.4(사용자): 손·몸·하반신만 나오는 포스터가 No one 으로 가던 것 → People 범주 'Some human'(얼굴·사람 상자 없이 OWLv2 hand 점수×넓이 ≥ .15 또는 SigLIP body_part z > .9). People 벡터에 hand, body_part 추가, 클러스터 라벨 Some human. People 라벨: No people · Big face(Couple 포함) · Crowd · Some human.
- v1.3.5(사용자): People 구조 = 왼쪽 위 One big face, 오른쪽 위 Couples, 많은 얼굴은 Crowd. 사람 수 3배 무게·12명에서 자르지 않음, 작은 얼굴 다수 신호, 커플 신호(faces_touch, embrace) 다시 넣음, soft_axes pull .5. 라벨 One big face · Couples · Crowd · No people · Some human.
- v1.3.6(사용자): Type 클러스터 라벨을 위치로 — Left title, Right title(title_cx < .38 / > .62), 가운데면 Title on top · in the middle · at the bottom, 그리고 Big centered title · Quiet type · Type only · Diagonal/Vertical type. Big title 이 동떨어지던 것: 크기 무게 3→2배, 좌우 위치 2배, pull .5.
- v1.3.7: Country 흔적(눈)은 두 글자 국가 코드(c.cc, Natural Earth ISO_A2, 옛 나라 SU·CS·YU). Year 호버 반응은 연도 숫자까지(그 아래 제외).
- v1.3.8: 축을 바꿀 때 제자리에서 사라지거나 생기던 포스터 = 그 축에 자리가 없던 커버(Runtime 98, Aspect 100). 모르는 커버도 자리를: Runtime 오른쪽 끝 '?' 막대, Aspect 맨 끝 Unknown 블록(액자 없음). 이제 모든 축에서 1,344편이 모두 날아서 이동. 대만 코드 TW.
- v1.4(사용자: '데이터가 부족하다', 2·3·4 진행): scripts/embed_more.py → 글자를 지운 그림(out/art, 공개 안 함), SigLIP 그림 임베딩, DINOv2-base(원본·그림), 사람 부위 분할 SegFormer b2 clothes(data/analysis/human_parse.json). People 판정: 분할로 사람 1% 미만이면 No one, 얼굴 없이 몸 3% 이상이면 Some human, 사람이 고른 답(data/labels/people.json) 우선(human_override). 지도: People 에 분할 비율, Composition 에 DINOv2 그림, Medium 에 DINOv2·SigLIP 그림. 라벨링: scripts/label_server.py(localhost:8777, tools/label/people.html, 2026-11-13 Lab 서버로 합침) 틴더식 1–6(none, animal, some, one, two, many), 틀렸을 법한 것부터 100장씩. 다음: 편집 모드(고치기, 대표 ★, 같은 무리 Y/N, 영역 이름, 물어봐), 사람 답으로 메트릭 러닝·지도 학습 UMAP, 클러스터 대조 설명·VLM 이름 후보(Qwen2-VL 2B 4.5GB 는 받기 전에 사용자 확인).
- v1.4.2(2026-11-12 마무리): 사람 라벨 545편(data/labels/people.json, 키 0/1/2/3 사람 수, 9 일부, A 동물). 모델 일치 52%, 모델은 사람 수를 적게 센다(두 사람 이상을 한 사람으로, 뒷모습을 사람 없음으로). 라벨이 있으면 People 판정을 덮어쓰고 지도 가로(사람 수)도 그 수를 씀. 축을 바꾸면 클러스터 강조(어둡게) 해제. 다음: 남은 800편 라벨링, 라벨로 People 분류기 학습(SigLIP·DINOv2·분할 특징 위 로지스틱 회귀), Animal 범주 검토, 편집 모드, 클러스터 대조 설명.
- v1.4.3(사용자): 터치 입력이면(pointerType, body.touchmode, 진짜 마우스가 움직이면 해제) 손가락 호버 미리보기·클러스터 설명 없음. 확대 중 화면 정가운데 점(#reticle)이 닿은 커버(테두리)·클러스터(영역·라벨)에 호버 효과, 움직임이 160ms 멈추면 그 커버 이름을 아래에(window.onViewChange → reticleTick). 전체 보기에서 탭 = 그 자리 한 단계 확대, 확대 중 탭 = 바로 상세. browser_check.mjs 에 tap·tdrag.
- Composition 재설계(2026-11-12 밤, 이 Mac 에서 원본 1,344편): Other 630편을 보고 범주를 사람이 쓰는 구도 말로 늘렸다. Grid / collage 119 · Symmetric 77 · Off-center subject 281(가장 큰 얼굴·사람 상자 중심이 가운데에서 0.17 넘게) · Silhouette 114 · Busy frame 111 · Close crop 214 · Single object 104 · Open space 87 · Diagonal 94 · Wide scene 143. Other 는 없앴다(기하 규칙이 안 걸리면 SigLIP 구도 문장 z-점수 최고 범주). 육안: Close crop·Wide scene 좋음, Silhouette 대체로, Busy frame 은 '반복 패턴'보다 '빽빽한 화면', Off-center 에 여러 인물 장면이 섞임. Full-bleed image 는 뺐다(구도 말이 아님). 사용자 확인 대기(out/tropes/index.html).
- 나라 축(사용자 제안, 메타 묶음: Spine · Year · Country): 격자는 세계 지도처럼 나라 자리에 커버를 모아 지역이 덩어리로 보이게, 세 폭 보기 순회는 지역 줄(East Asia, South Asia, Western Europe, Nordic, Eastern Europe, Africa, Middle East, Latin America, North America, Oceania …) 안에서 나라 → 연도 순. 격자 xy 로 움직이지 않는다(사용자: "지역 그룹별로 순회해야지 지도를 xy좌표로 움직이는 방식으로 작동하면 안되거든"). 지금 구조(격자 = c.g, 순회 = META.rings 범주 줄)가 이미 그렇다. 미국 533편은 시대별 줄로 나눌 것. 공동 제작은 첫 번째 나라.
- 이름: The Criterion Collection World(CCW)(2026-11-12 사용자, 이전 Criterion World). 화면 title·로딩·크레딧·og 를 바꿨고 GitHub 레포도 ccw 로 바꿨다(옛 주소는 GitHub 가 넘겨준다). 주소는 ccw.umjo.me(옛 cc.umjo.me 는 Worker 에 그대로 붙어 있고 index.html 맨 앞 스크립트가 새 주소로 넘긴다). Cloudflare Worker 이름 criterion-world 는 그대로.

## 묘사·축별 임베딩(ccw-embeddings 브랜치, 2026-11-13)

사용자 방향: 커버를 글로 자세히 묘사하고, 그 글을 임베딩해 범주 판정과 지도에 쓴다. 판단은 숫자보다 경향으로(사람 라벨에도 실수가 있다).

- 묘사(data/analysis/captions_sonnet_v3.json, 1,344편, 레포에 올림): 원본 커버를 Sonnet 에이전트가 보고 여덟 줄을 쓴다(People, Composition, Medium, Typography, Color and light, Story it suggests, Visual idea, Feels like). 커버의 글자는 단서로 읽어도 되지만 영화 지식은 쓰지 않는다. 지시문 scripts/caption_prompt.txt, 30장씩 나눈 목록을 에이전트 여러 개에 주고 scripts/merge_captions.py 로 합친다. 장당 약 5,800 토큰, 전체 약 700만. 에이전트가 파일 이름 철자를 '고쳐' 쓰는 일이 있다(0250, 0791, 0832) → 합친 뒤 posters/ 와 키를 대조할 것. Visual idea·Feels like 는 만들되 아직 임베딩에 쓰지 않는다.
- 임베딩: EmbeddingGemma 2(google/embeddinggemma-2, 768d). scripts/embed_gemma.py(이미지·방향 문장), scripts/embed_captions.py(묘사 글: 전체 cap-all, 한 줄씩 cap-people/composition/medium/typography/color, 글+이미지 cap-img). 글은 "title: none | text: …", 범주 문장은 "task: classification | query: …".
- 평가(사람 People 라벨 545편, scripts/eval_captions.py --all): probe(선형 분류 균형 정확도)·문장 검색 p@30 — Gemma 이미지 .66·.47, 묘사 전체 .74·.67, People 줄만 .78·.48(예시 검색은 .55 로 최고), 묘사+이미지 .71·.63. 이미지를 더해도 People 에는 도움이 안 된다. 방향 문장(이미지 앞에 붙이는 글, scripts/steer_sweep.py) 6개는 모두 이미지만보다 낮아 그만뒀다.
- 지도(지금은 Lab: scripts/build_lab.py, 겹침 허용·격자 + 확대·이동): 축마다 그 줄만 임베딩한 지도가 그 축으로 가장 잘 모인다. 이웃 8칸 같은 범주 비율 — People 줄 .65(무작위 .24), Medium 줄 .73(무작위 .35, 사진 · 콜라주 · 회화 · 그래픽·판화·드로잉이 덩어리로), Composition 줄 .26(무작위 .17), Typography 줄 .25(.17). Composition·Type 의 색칠 범주는 낱말 규칙이라 엉성하다 → Sonnet 이 글만 읽고 판정하게 한 뒤 다시 볼 것. Color 축은 기존 색상 × 명도 격자가 낫다.
- People v2(scripts/people_v2.py → data/analysis/people_v2.json, 레포에 올림): None · No-face · One · Two · 3+ · Ambiguous. 사용자와 정한 기준: 그림이든 사진이든 사람이면 센다, 얼굴이 하나라도 보이면 얼굴 보이는 사람 수, 사람이 분명한데 얼굴이 전혀 없으면 No-face(뒷모습, 손·다리만), 사람인지부터 불분명하면 Ambiguous(조각상, 마네킹, 인형, 가면, 그림자, 이유는 reason), 같은 인물이 여러 번 나오면 보이는 수대로(세 번 이상 3+) + repeated 태그, 동물만 있으면 None + animal 태그. 먼저 Sonnet 이 People 줄 글만 읽고 none/animal/some/one/two/many 로 판정(data/analysis/people_from_captions.json, 로컬)하고, 토큰 없이 낱말 규칙으로 v2 로 옮겼다. 결과 One 423 · 3+ 396 · Two 267 · None 134 · No-face 70 · Ambiguous 54.
  - 사람 라벨과의 일치: 옛 범주 83%(애매함 없는 것 91%, 기존 검출기 판정은 52%). 사용자: 모델이 사람 라벨보다 맞게 본 사례가 꽤 많다. 그래서 묘사 판정을 기본으로 하고 사람 라벨은 사용자가 확인한 것만 덮어쓰기로. v2 기준으로는 76.9%(차이 대부분은 기준 변화: 예전에 뒷모습을 One 으로 라벨). 엇갈린 커버는 scripts/review_people.py → out/gemma/people_review.html.
- 사이트 적용(v1.4.4): build_web.py 가 people_v2.json·medium_v2.json(scripts/medium_v2.py, Medium 줄의 가장 먼저 나온 매체 낱말: Photo 716 · Painting 215 · Graphic 185 · Drawing 122 · Collage 49 · Print 36 · Other 21)이 있으면 tropes 의 people·medium 을 덮어쓰고, 두 축 격자를 묘사 줄 임베딩(gemma2-cap-people/medium.npz, embed_captions.py) UMAP 으로 그린다. 임베딩이 없으면 이전 방식. Type·Composition 은 묘사 줄 지도가 나아지지 않아(.25–.26) 그대로.
  - 주의: 이 Mac 에서 build_web 을 통째로 돌리면 이 Mac 의 분석 데이터(글자 영역·레이아웃·물체·틀)가 사이트를 만든 컴퓨터와 달라 Type·Composition·Color 배치와 흔적 625개까지 바뀐다(틀 41편 사라짐 포함). 그래서 v1.4.4 는 새 빌드에서 people·medium 항목(META grids·rings·rowNames·gridExtra, COVERS g·nb)만 기존 data.js 에 옮겨 넣었고 흔적·아틀라스는 그대로 뒀다. 두 컴퓨터의 분석 데이터를 맞추기 전까지는 이렇게 부분만 옮긴다.
  - 마우스 휠 = 한 칸에 한 단계 확대(커서 기준, 260ms 잠금). deltaMode 가 줄 단위이거나 가로 없이 큰 정수 칸이면 휠, 아니면 트랙패드(두 손가락 = 이동, 핀치 = 단계 확대).
- Lab(2026-11-13 사용자: 지도를 베타처럼 써서 축·배치·범주를 실험하고 마음에 드는 형태를 사이트에 도입): scripts/build_lab.py → out/lab/index.html, scripts/lab_server.py → http://localhost:8777(127.0.0.1 에만 열려 이 Mac 에서만 접속). 화면은 사이트와 같은 머리줄·알약 버튼. 위 = 무엇으로 배치하나(묘사 전체·줄별 Description/People/Composition/Medium/Type/Color, 그림 Picture/Picture no text/SigLIP/Picture + description), 아래 = 무엇으로 색칠하나(People·Medium = v2, Composition·Type = 사이트 판정). Exact(UMAP 그대로, 겹침 허용)/Grid. canvas + 사이트 아틀라스(cover_s 40×50) → 크게 보이면 thumbs → large. 커버 클릭 = 그 임베딩에서 가까운 12장. Edit = 커버 끌어 옮기기, 커버 위 숫자 키로 범주 고치기(0 = 되돌림). /label = 한 장씩 보는 라벨링(옛 label_server 를 합침, People v2 키 1–6, 옛 라벨과 엇갈린 것·Ambiguous·No-face 부터). 둘 다 data/labels/lab.json 에 저장하고 build_web.py 가 이 라벨을 판정보다 앞세운다. 옛 people.json 은 평가용으로 둔다. admin 하위 도메인은 Cloudflare Access 설정과 저장소(KV 등)가 필요해 보류.
- 참고 자료: docs/references.md(우선순위·링크·자료별 요약). 원문 PDF 는 references/(git 제외). Composition·Type 개선 방향: 내용을 지운 구조 이미지(구도선·대칭·중요도 지도) 임베딩, 제목 상자 크롭 + 기하 값, 축별 세 장 비교 평가 세트를 먼저.

## 다음 할 일 (2026-11-13 기준, 위에서부터)

지금 상태: PR #5(ccw-embeddings) 열림, 배포 안 함. 사이트 v1.4.4 = People·Medium 이 묘사 기반, 마우스 휠 확대. 실험은 Lab(localhost:8777)에서 하고 마음에 드는 것만 사이트로 옮긴다.

1. 머지 후 배포: PR #5 머지 → scripts/deploy.sh(wrangler 로그인 필요). 배포 전에 People·Medium 을 사이트에서 한 번 더 본다.
2. Lab 으로 People 라벨 다듬기(사용자): /label 에서 옛 라벨과 엇갈린 126편부터 People v2 로 확인. 답은 data/labels/lab.json → build_web 이 앞세운다. 몇십 편 쌓이면 eval_captions 처럼 묘사 판정과의 일치를 다시 재고, People 줄 지도에서 Edit 로 고친 커버가 어디 몰리는지 본다.
3. 두 컴퓨터의 분석 데이터 맞추기: 이 Mac 의 data/regions·data/analysis(글자 영역·레이아웃·물체·틀)가 사이트를 만든 컴퓨터와 달라 build_web 을 통째로 돌리면 Type·Composition·Color·흔적이 바뀐다. 어느 쪽이 맞는지(틀 41편이 사라지는 쪽이 이 Mac) 정하고 한쪽으로 맞춘 뒤에 전체 빌드. 그전까지는 바꾼 축만 data.js 에 옮겨 넣는다(v1.4.4 방식).
4. Composition·Type 개선(docs/references.md 우선순위). 먼저 평가 세트, 그다음 입력 바꾸기:
   a. 축별 세 장 비교 평가 세트: 기준 커버 · 그 축으로 비슷한 커버 · People·Medium 은 같지만 그 축은 다른 커버. 축마다 30–50개, Lab 에서 고르는 화면을 만든다(사용자 확인).
   b. Composition: 글자 지운 그림(out/art)에서 구도선·대칭축·중요도 지도를 그린 '구조 이미지'를 만들어 DINOv2(또는 DINOv3, 받기 전 확인)로 임베딩 → Lab 에 지도로 추가해 비교.
   c. Type: 제목 상자 크롭 임베딩 + 위치·크기·각도·면적 비율. 상자 배치끼리 헝가리안 gIoU 로 '글자 배치가 같은 커버' 이웃. Lab 에 지도로 추가.
   d. 그래도 범주가 엉성하면 Sonnet 이 Composition·Typography 줄 글만 읽고 범주 판정(축당 30만 토큰 안팎, 범주 목록은 사용자 확인).
5. Lab 다듬기 후보: 지도 두 개 섞기(가중치 슬라이더), 범주별 '가장 전형적인 / 가장 이상한' 커버, 흔적 켜기, 고른 커버들 사이 경로. Poster World 조사에서 미묘한 축은 흔적 없이는 공통점을 거의 못 알아봤다 → Composition·Type 은 세 폭 보기에서 흔적을 기본으로 켜는 것도 검토.
6. 컨셉 쪽 후보(docs/references.md): 오독 층(검출기가 틀린 것·모델끼리 엇갈린 판정), 흔적의 평균(연대·나라·매체별 글자 영역·구도선·얼굴 위치 평균).

이전 후보(아직 유효)
- 커버가 들어오면: (.venv) run_text_regions.py → analyze_visual.py → cover_checklist.py, (.venv-ml) embed_siglip.py → detect_objects.py → analyze_strokes_layout.py → build_web.py. 묘사도 새 커버만 Sonnet 으로(scripts/caption_prompt.txt, merge_captions.py) → embed_captions.py → people_v2.py · medium_v2.py.
- 틀 판정 확인(frame.kind none/unknown), art 버전이 글자로 잘못 잡힌 그림까지 지우는 문제(`source: "detect"` 영역).
- 사용자 결정 대기: 박스 세트 커버(posters_box/)를 뷰어에 넣을지.
- ML 모델은 .venv-ml 에 설치한다. 큰 패키지·새 모델(DINOv3, SAM 3, UMSI 등)은 받기 전에 사용자에게 확인한다.

## 작업 내역 (2026-11-13, ccw-embeddings 브랜치)

- Sonnet 커버 묘사 1,344편(장당 약 5,800 토큰). Qwen3-VL 시험 후 Sonnet 으로 결정.
- EmbeddingGemma 2 임베딩(이미지, 방향 문장, 묘사 전체·줄별·글+이미지)과 평가. 방향 문장은 효과 없어 중단.
- People v2 범주 정의(사용자와 기준 합의)와 판정, 엇갈림 확인 페이지.
- Medium 범주를 묘사에서.
- 사이트 v1.4.4: People·Medium 묘사 기반 지도, 마우스 휠 확대.
- 참고 자료 정리(docs/references.md, 논문은 references/ 로컬).
- Lab(실험 지도 + 라벨링, 로컬 전용) 만들고 옛 라벨링 서버를 합침.

## 배포 (2026-11-12 준비)

- 레포 https://github.com/thumbjiho/ccw (public). 배포는 scripts/deploy.sh: git 에 올라간 web/ 만 모아 Cloudflare Workers 정적 자산(Worker 이름 criterion-world, 도메인 cc.umjo.me)으로 올린다. wrangler 4.160 은 pages project create 를 Workers 로 넘겨서 Pages 대신 이 방식. GitHub 와 자동 연결은 없다(커밋·푸시 후 deploy.sh). .gitignore 가 원본·레퍼런스·중간 결과·file:// 용 산출물(atlas *.js/*.jpg, strokes *.png)을 뺀다. 레포 약 123MB, 5,300 파일.
- http 에서는 아틀라스를 WebP 로 읽고 원본 아틀라스는 반 크기(cover_s.webp 0.75MB)를 먼저 올린다. 흔적 아틀라스 미리 받기는 큰 원본 아틀라스가 온 뒤. 캐시는 web/_headers.
- 로컬에서 file:// 로 열면 예전처럼 data URL JS 를 쓴다.
- 느린 회선 스켈레톤(v0.3.3): 큰 커버(large/, thumbs/) img 에 class lq 와 밑그림(대표 색 + 이미 받은 반 크기 원본 아틀라스 타일)을 깔고, 받는 동안 숨 쉬듯 깜빡인다. load/error 는 document 캡처로 받아 .ok 를 붙인다(window 에는 img load 가 오지 않는다).
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
- archive.org 는 동시 요청(3개)에 Connection refused 로 한동안 막는다. 페이지 하나가 403·404 인 것은 막힘이 아니라 그 페이지 문제다. fetch_criterion_meta.py --fill 은 화면비 없는 단독 영화만 한 장씩 4초 간격으로 받고, 짧은 주소 → 미래 날짜(2099, 최신 보관본으로 넘겨줌) → 제목이 붙은 주소(CDX prefix) 순으로 찾는다. 2026-11-12 이렇게 81편 → 2편(Che 는 페이지에 화면비 칸이 비어 있음, A Master Builder 는 보관본 403).
- 줄 이름(META.rowNames)은 build_web.py 가 마지막에 scripts/row_names.py 의 shorten_all 로 줄인다(축 이름이 이미 말하는 낱말·mostly 같은 군더더기 삭제, 겹치면 원래대로). 이 컴퓨터에서는 같은 함수로 data.js 의 META 만 갈아 끼웠다.
- 이 컴퓨터(분석 데이터·.venv-ml 없음)에서는 build_web.py 를 못 돌려, criterion_meta.json 을 고친 뒤 build_web 과 같은 규칙(rt, ar, clr, language, fmt)으로 data.js 의 COVERS 만 갈아 끼우고 index.html 의 SIZES 를 맞췄다. 원래 컴퓨터에서 build_web 을 돌리면 같은 결과가 나온다.
