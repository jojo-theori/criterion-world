# 참고 자료

사용자가 모은 참고 자료(2026-11-13)를 읽고, CCW 의 범주 판정·지도·조작을 개선하는 데 쓸 점을 정리했다. 논문 PDF 원문은 저작권 때문에 레포에 올리지 않고 로컬 `references/`(git 제외)에 둔다. 아래 부록 A–C 는 에이전트가 원문을 읽고 자기 말로 요약한 노트다(영어).

## 우선순위(세 노트를 합친 제안)

약한 축(Composition, Type)부터.

1. **Composition 은 내용을 지운 구조 이미지로 임베딩한다.** 글자를 지운 그림에서 구도선·대칭축·중요도(saliency) 지도를 그린 이미지를 DINOv3/DINOv2 로 임베딩한다. 묘사 글이나 원본 이미지는 내용(무엇이 그려졌나)에 끌려간다. Poster World 가 Canny·FFT·CRAFT 출력에 ResNet 을 쓴 것, VFCNet 이 saliency·GVF 렌더링으로 내용과 무관한 구도 유사도를 낸 것과 같은 방향이다.
2. **Type 은 제목 상자 잘라낸 이미지 + 기하 값으로.** 제목 영역 크롭을 임베딩하고, 위치·크기·각도·면적 비율·상자 수와 합친다. 상자끼리 헝가리안 매칭 gIoU(RADIO1D 의 구성 점수 아이디어)로 "글자 배치가 같은 커버"를 학습 없이 바로 찾을 수 있고, 매칭된 상자를 흔적으로 그릴 수 있다.
3. **축별 평가 세트를 먼저 만든다.** GeneCIS·PICD 방식의 세 장 비교(기준, 같은 축으로 비슷한 것, People·Medium 은 같지만 그 축은 다른 것)를 축마다 수십 개. 모델·입력을 바꿀 때마다 이걸로 잰다. VLM 구도 분류가 약하다는 보고(PhotoFramer)가 있어 Sonnet 구도 판정도 이걸로 검증한다.
4. **축별 투영 헤드.** 고정된 이미지 임베딩 위에 작은 선형 헤드를 축마다 학습한다(정답 = 그 축의 Sonnet 묘사 줄 임베딩, 또는 사람 라벨). FocalLens 의 조건부 임베딩을 가볍게 흉내 내는 것. 매개변수형 UMAP 이면 빌드마다 지도가 덜 흔들린다.
5. **중요도 지도를 디자인용으로(UMSI).** 포스터로 학습한 중요도 모델로 일반 saliency 를 대체하면 무게중심·균형·3분할 거리·제목 지배도 같은 설명 가능한 값이 나온다.
6. **흔적을 설명으로.** Poster World 이용자 조사에서 미묘한 축은 흔적 그림 없이 공통점을 거의 알아채지 못했다. Composition·Type 은 세 폭 보기에서 흔적을 기본으로 켜고, 이웃도 지도와 같은 공간에서 고른다.
7. **오독을 층으로(신승백 김용훈).** 검출기가 틀린 것(글자 속 얼굴, 그림 속 놓친 글자)과 모델끼리 엇갈린 판정을 따로 보여 주는 층. "의미 층위의 균열" 방향과 가장 잘 맞는다.
8. **흔적의 평균(Salavon).** 픽셀이 아니라 흔적(글자 영역, 구도선, 얼굴 위치)을 연대·나라·매체별로 평균한 그림. 크라이테리온의 시기별 문법을 드러내고 저작권 부담도 적다.
9. **도구.** Embedding Atlas 로 축별 이웃을 빠르게 점검, PixPlot 의 핫스폿·배치 전환을 UI 패턴으로. DINOv3, SAM 3(텍스트 프롬프트 마스크: frame, circle, silhouette, title text)은 받기 전에 사용자 확인.

## 링크

전시·컬렉션 인터페이스
- Poster World (EPFL+ECAL Lab): https://epfl-ecal-lab.ch/research-projects/posterworld · https://epfl-ecal-lab.ch/poster-world/ · https://www.planetdigital.uzh.ch/en/project/poster-world
- Poster World 논문: https://backend.epfl-ecal-lab.ch/wp-content/uploads/2022/12/EPFLECALLab_2022-12-01_Andrade-et-al.pdf
- Barnes Collection Online: https://collection.barnesfoundation.org · https://medium.com/barnes-foundation/computer-vision-so-good-3f4162a35d3f · https://www.barnesfoundation.org/press/press-releases/launch-of-new-framework-for-searching-collection-online
- Collection Space Navigator: https://arxiv.org/pdf/2305.06809
- VIKUS Viewer: https://vikusviewer.fh-potsdam.de · https://github.com/cpietsch/vikus-viewer/ · https://uclab.fh-potsdam.de/projects/vikus-viewer/
- Principal Components (Nasjonalmuseet): https://nasjonalmuseet.no/en/about-the-national-museum/collection-management---behind-the-scenes/digital-collection-management/project-principal-components · https://auduno.github.io/2018/10/27/visualizing-an-art-collection/
- Google Art Palette: https://www.blog.google/topics/arts-culture/machine-learning-meets-culture/ · https://digitaltrends.com/web/google-arts-and-culture-experiments

문화 분석·작가
- Time 표지 4,535장: https://lab.culturalanalytics.info/2016/04/timeline-4535-time-magazine-covers-1923.html · 프로젝트 목록 https://lab.culturalanalytics.info/p/projects.html
- Cinemetrics: https://www.creativeapplications.net/project/cinemetrics-processing/ · https://flowingdata.com/2012/01/12/cinemetrics-creates-a-visual-fingerprint-for-movies
- Jason Salavon: https://salavon.com/work/EveryPlayboyCenterfoldDecades/ · https://en.wikipedia.org/wiki/Jason_Salavon
- 신승백 김용훈: https://www.creativeapplications.net/people/shinseungback-kimyonghun/ · https://petapixel.com/2025/02/05/how-ai-sees-us-analytic-portrait-series-combines-ai-vision-tech-with-real-portraits

기술
- GeneCIS: https://arxiv.org/pdf/2306.07969.pdf · FocalLens: https://arxiv.org/html/2504.08368v1
- PixPlot: https://dhlab.yale.edu/projects/pixplot · UMSI: https://research.adobe.com/publication/predicting-visual-importance-across-graphic-design-types
- RADIO1D: https://arxiv.org/pdf/2607.03624 · DINOv3: https://ai.meta.com/blog/dinov3-self-supervised-vision-model/ · SAM 3: https://docs.ultralytics.com/models/sam-3
- OCR 비교: https://roboflow.com/blog/best-open-source-ocr-models · https://pasqualepillitteri.it/en/news/25568/best-local-ocr-pc-server-2026
- PhotoFramer: https://arxiv.org/pdf/2512.00993 · PICD: https://cvpr.thecvf.com/virtual/2025/poster/34575 · VFCNet: https://arxiv.org/html/2604.16500
- Embedding Atlas: https://apple.github.io/embedding-atlas/tool.html · EmbeddingGemma 2: https://huggingface.co/google/embeddinggemma-2

로컬 `references/` 파일: posterworld_andrade2022.pdf, collection_space_navigator_2305.06809.pdf, genecis_2306.07969.pdf, focallens_2504.08368.pdf, radio1d_2607.03624.pdf, photoframer_2512.00993.pdf, vfcnet_2604.16500.pdf


---

## 부록 A: collection-exploration projects for CCW

Load status: the three Poster World pages and both PDFs were read in full. The Barnes Medium blog returned 403, and a mirror was blocked. The Barnes collection site only exposed icon labels. The Nasjonalmuseet page failed the domain check; the auduno blog loaded. Google's blog post and the Digital Trends article both loaded, but neither has technical detail.

---

### 1. Poster World (EPFL+ECAL Lab, project pages)

**What:** A 2022 installation at Museum für Gestaltung Zürich. It covers about 2,000 Swiss posters (1907–2021) and groups them in trios by one chosen visual feature.

**How:** The three web pages repeat the same summary. Each poster gets features for dominant colours, main composition lines, objects, typography, text placement and contrast. One chosen feature is re-rendered as a new graphic. The collection view re-sorts by that feature, and trios are drawn from that similarity. The pages say the pairing was "trained" on curatorial associations from the museum's own poster books, not generic datasets. No models are named on the pages; the paper gives them (next section).

**Borrow:**
- The framing that the AI amplifies curation rather than replacing it fits CCW's human-label override (people.json). Make that override visible as part of the work, not hidden plumbing.
- Keep the content set fixed and curated. Poster World swaps in new curated subsets over time. CCW could do the same with themed "rooms" (for example "Kurosawa", "1960s Japan" or "Janus era") as one filter on the same maps.

### 2. Poster World paper (Henchoz, Andrade et al., PAD #23, 2022)

**What:** A design-research paper on how the installation was built and how three interaction scenarios tested with 61 users.

**How:**
- **Feature extraction:**
  - Hand-tuned classical image processing, one method per feature:
    - Dominant colours: k-means.
    - Shape strokes: Canny edges.
    - Maximum contrast: minimum-cross-entropy threshold.
    - Composition lines: FFT (the main direction(s) are kept).
  - Objects: YOLO boxes.
  - Typography layout: the CRAFT character-region heatmap.
  - Global analysis: a convolutional autoencoder trained on the whole digitised archive, rendered as a low-res dominant-colour image.
- **Pairing:** The key trick is to run an ImageNet ResNet-18 on the feature visualisation image (the edge map, threshold map, FFT lines or CRAFT heatmap), not on the poster. t-SNE then reduces that to 2D, and similarity for the trio is distance in t-SNE 2D, so the trio and the map agree by construction. A separate "narrative" model clusters objects in the curated books to build sequences, validated with graphic designers.
- **Feature list:** Global analysis, dominant colours, maximum contrast, shape strokes, composition lines, main composition line, typography layout and object recognition. Designer, client and keyword metadata add context.

**Interaction and findings:**
- **Three posters is the sweet spot.** Pilot tests (n=14) found that three gave variety while the shared trait stayed clear. A random association scored only slightly lower than colour or theme pairings, so loose associations still interest people.
- **Scenarios tested:**
  - Scenario 1: pick two features as the X/Y axes.
  - Scenario 2: one feature, with a toggle between the original and the feature rendering.
  - Scenario 3: X = chosen feature, Y = year, and the user picks the poster.
- **Results:**
  - Scenario 2, the feature rendering, was judged the most inventive and drew the most praise.
  - Two-axis choice confused people. Users wanted to pick a poster themselves.
  - Before using the installation, 89% recognised colour pairings, but only 15% recognised typography layout and 20% composition line. After use these rose only slightly (to 92%, 18% and 26%).
  - Novices and experts scored the same.
- **Final installation:**
  - Standby mode cycles through linked posters.
  - Touch opens the collection map for one feature, and changing the feature animates the map.
  - Picking a poster shows it 1:1 in the centre with its two most similar posters on either side.
  - The feature rendering can then be revealed on all three.

**Borrow:**
- **Embed the trace, not the cover, for the weak axes.** For Composition, render each cover's composition trace (CCW already has LSD lines, symmetry map, saliency and line drawing) as a small grey image. Embed it with a vision model (DINOv2 or SigLIP) and run UMAP on that. Do the same for Type with the text-polygon mask, as a binary or blurred heatmap like CRAFT. This strips subject and meaning, which is exactly why caption embeddings fail on these two axes.
- **Derive neighbours from the drawn map.** Compute three-panel neighbours from the same 2D layout the user sees (or at least the same embedding). Visible adjacency then explains the pairing.
- **Expect people to miss composition and type pairings.** Because they do, the trace overlay (CCW's eye toggle) is the explanation, not decoration. Consider turning the trace on automatically for the three panels on the Type and Composition axes.
- **Add an idle mode.** An attract/standby mode could auto-walk the torus ring (current, next, next) with an axis change every few steps. It is cheap to build and matches the installation.

### 3. Barnes Collection Online (Barnes Foundation)

**What:** The Barnes Foundation's online collection, launched in 2017. You can browse it by the formal elements light, line, colour and space, as well as by artist, medium or movement.

**How:**
- From the press release: Elasticsearch handles metadata search. "Visually related" results come from ArtPI, Artrendex's art-similarity API. Computer vision also generated object keywords.
- The site itself only exposed control labels: colour, light, line and space tools. Line types are diagonal, horizontal, vertical, curvy, broken and unbroken; there is also shuffle and zoom.
- So formal attributes are discrete, human-readable facets combined with an opaque similarity neighbour list.
- The Medium post on the CV pipeline (which tagging services were used, how they were checked against human tags) failed to load (403).

**Borrow:**
- **Make line quality a facet.** Diagonal / horizontal / vertical / curvy / broken / unbroken is a ready-made vocabulary for CCW Composition. CCW already has the LSD direction histogram and stroke stats, so these can be rules (dominant direction, curvature, segment continuity) instead of caption keywords.
- **Use formal-element facets as filters that dim non-matching covers.** This would sit on top of any axis map, much like CCW's search/cluster dimming. That lets people combine "Medium map + only diagonal-line covers".
- **Add a shuffle button.** It would serve casual browsing in the three-panel view (random jump on the current axis).

### 4. Collection Space Navigator (Ohm, Canet Solà, Karjus, Schich, 2023, arXiv 2305.06809)

**What:** An open-source (MIT) browser tool for exploring image collections through switchable 2D projections plus multidimensional range filters.

**How:**
- The centre is a zoomable thumbnail scatter (UMAP, t-SNE, PCA, or a plain x/y such as PC1 over time).
- Switching projection animates the thumbnails. The authors note the in-between motion itself shows how the projections differ.
- **Left panel:** hover preview, metadata, and colour-coded thumbnail borders for a chosen category.
- **Right panel:**
  - Range sliders, each topped by a live histogram. Histograms update for filtered and unfiltered items, and the hovered item's bin is highlighted.
  - Clicking a histogram enters "bin mode", where scrubbing the bars shows one bin at a time.
  - Boolean text queries.
  - CSV/PNG export.
- Embeddings shown include ResNet and interpretable "compression ensemble" features.

**Borrow:**
- **Histogram scrubbers for CCW's numeric traces.** Good candidates are colourfulness, RMS contrast, text area, title size, face count and runtime. Hovering a cover would mark its bin in every histogram, giving a compact "where this cover sits" profile for the detail panel.
- **Bin mode** (scrub a histogram and see only that slice lit on the current map) is a strong way to test whether an axis map really orders by a property. It would help diagnose the weak Composition and Type maps.
- **Category-coloured borders on the map.** For example, colour Medium category borders while viewing the People map. This shows how two axes cross without leaving the map.
- **Interpretable features over opaque embeddings for formal axes.** The compression-ensemble idea (compress filtered versions of the image; use the compression ratios as features) is a cheap, explainable texture and complexity feature that CCW could add for Composition ("busy" vs "open").

### 5. VIKUS Viewer (Christopher Pietsch, UCLab FH Potsdam)

**What:** An open-source web framework for browsing cultural collections on a zoomable canvas along "time, themes, texture". Examples include Past Visions, Van Gogh and the Women's March signs.

**How:**
- **Stack:** A static site built with PIXI.js and d3. Config comes from `data.csv` (id, keywords, year), `timeline.csv` and `config.json`.
- **Layouts:**
  - Group layout: items stacked in columns per year, so the overview is a histogram made of the images.
  - CSV layout: x/y from a t-SNE file.
  - Layouts animate into each other.
- **Precomputed with vikus-viewer-script:**
  - Three texture tiers (256 sprite sheets, 1024 detail, optional 4096).
  - A t-SNE layout from ImageNet logits (a CLIP notebook is an alternative), with a spacing pass to stop overlap.
- **Keyword filter:** A tag cloud with counts and AND filtering. There are also hierarchical colon-separated tags and a crossfilter mode with faceted counts.
- **Zoom:** Zooming in adds timeline text and switches to high-res images.

**Borrow:**
- **Keyword tag cloud with counts.** Build it from CCW's caption categories (People v2, Medium, and later Composition/Type) and the "repeated" / "animal" tags. Selecting tags would dim the map, and the counts show the shape of the collection at a glance.
- **Hierarchical tags.** For example `Medium:Print:Woodcut` or `People:Two:Couple`. This resolves CCW's one-category-per-axis constraint for rows and grid, while still allowing finer sub-tags for filtering.
- **Timeline annotations on the Year axis.** Short notes on Criterion-relevant events (Janus Films, the move from LaserDisc to DVD to Blu-ray, the cover-design eras) appear only when zoomed in.

### 6. Principal Components (Nasjonalmuseet, Norway; Audun Øygard, 2018)

**What:** The National Museum of Norway's ML maps of its collection (the "VY" site). There were separate maps for paintings, drawings and faces, plus GAN experiments. The museum page failed to load (domain check), so these notes come from the author's blog.

**How:**
- They trained their own classifiers for style and for motif on the collection, using the last-layer features as embeddings. Accuracy was modest and deliberately secondary.
- Maps were made with t-SNE, then parametric t-SNE, so new works can be added without re-running and the map stays stable.
- Faces used a photo face-similarity embedding. Fine-tuning it on painted portraits did not help, and the face map was the least coherent.
- Regions were named by inspection (seascapes, fjords, bearded men…).
- Each period got two summaries: the works nearest the decade's mean embedding, and its outliers. Outliers often pointed ahead to later trends.

**Borrow:**
- **Train small supervised heads per axis on your own labels** (style/motif classifiers here; CCW's 545 People labels and future Medium/Composition picks). Then map the penultimate layer. Supervised features shape the map toward the axis, which is the direct fix for weak Composition and Type.
- **Use a parametric projection** (parametric UMAP, or UMAP `transform`) so new covers and re-runs land in stable places. The layout then does not reshuffle at each data build.
- **Show "most typical" and "most unusual" per group.** Nearest to and farthest from the centroid per decade, category or cluster label. This makes a natural cluster-hover readout and a "Criterion-ness" angle.
- **Be wary of photo-trained detectors on illustrated covers.** This echoes CCW's own YuNet/People findings, and supports the decision to use caption-based judgement there.

### 7. Google Arts & Culture: Art Palette and related experiments

**What:** Google Arts & Culture Lab experiments (2018). In Art Palette you pick a palette or take a photo and get artworks with matching colours. Related experiments are t-SNE Map, X Degrees of Separation and auto-Tags.

**How:**
- Both articles are non-technical. The Google blog says only that Art Palette uses a combination of computer vision algorithms; it extracts a few dominant colours and matches palettes across partner collections.
- t-SNE Map: a 3D zoomable landscape of artworks placed by visual similarity.
- X Degrees of Separation: finds a chain of visually similar works linking any two chosen artworks.
- Tags: machine-assigned labels ranging from concrete to abstract ("azure", "skin", "map").
- Neither page names models or similarity metrics.

**Borrow:**
- **Palette query.** Let the user build or edit a 5-swatch palette (or pick one from a cover) and order or dim the Color map by palette distance. Earth-mover or matched Lab distance would work on CCW's existing k-means palettes and needs no new model.
- **A "path between two covers" mode.** Pick two covers and walk the shortest chain of neighbours through the embedding of the chosen axis, shown as a strip in the three-panel view. It also revives the old "path between versions" concept with a different input.
- **Abstract tags next to formal ones.** Google mixed concrete and abstract tags. CCW's unused caption lines (Visual idea, Feels like) could become a soft "mood" filter rather than an axis.

---

### Top takeaways for CCW

1. **Map the weak axes from trace images.** For Composition and Type, embed a rendered trace image (composition lines, symmetry or line drawing; text-region heatmap) with DINOv2/SigLIP and UMAP that, as Poster World did with ResNet on Canny/FFT/CRAFT outputs. Don't map them from captions. Removing subject matter is the point.
2. **Add supervised per-axis heads.** Use CCW's human labels to train a small classifier or metric head per axis (as Nasjonalmuseet did with style and motif). Map its penultimate features with parametric UMAP, so the layout is axis-shaped and stable across rebuilds.
3. **Name the Composition and Line categories with Barnes's formal vocabulary.** Diagonal, horizontal, vertical, curvy, broken and unbroken line, plus light and space. Compute them by rule from the LSD and stroke statistics, so the categories are explainable.
4. **Keep three panels and make the trace the explanation.** On subtle axes, people in the Poster World study rarely recognised the shared trait (15–26%) without the rendering. Turn the trace on by default for Type and Composition, and derive neighbours from the same space the map shows.
5. **Add filter layers that cut across axes.** CSN-style histogram scrubbers and bin mode, plus VIKUS-style tag counts. Both dim the current map, so people can cross two axes and check that a map really orders by a property.
6. **Add playful entry points.** A palette query on the Color axis, a "path between two covers" chain, an idle auto-walk mode, and "most typical / most unusual" per cluster.

---

## 부록 B: Cultural analytics, averaging, machine-vision art

Fetch status: CAN pages (Cinemetrics, SSB people page) and FlowingData returned 403. Petapixel returned 403. Cinemetrics' own site has an expired TLS certificate. Cinemetrics is summarised from background knowledge, marked as such. For SSB I used the artists' own site (ssbkyh.com).

---

### 1. Time magazine covers 1923–2009 (Cultural Analytics Lab)

**What:** Manovich and Douglass's visual study of 4,535 Time covers, from the first issue to 2009.

**Method:** They cropped the covers to remove margins and the red border. They made two views:
- A full grid in publication order, read left to right and top to bottom, for judging medium, colour versus B/W, hue, contrast and content by eye.
- A scatter of covers with date on x and one automatic measure on y. The measure is mean brightness for B/W covers and mean saturation for colour ones.

The page names no tools; the lab's ImagePlot is implied. Findings: medium shifts (photo → painting → photo → software collage), B/W and colour coexisting for years, hue "periods", and contrast and saturation rising until the late 1990s and then falling. Almost all change was gradual.

**Borrow for CCW:**
- Use a Spine-ordered grid as the "raw reading" view. With 1,344 covers and Spine standing in for publication time, the medium and colour periods should show up before any axis is applied.
- For Year and Spine, plot the covers themselves at (time, one metric). Make the y-metric switchable (brightness, saturation, colourfulness, text area %).
- Crop out the Criterion frame and band before measuring, as they cropped the red border. This matches the existing "picture-only" mask.
- Measure B/W and colour covers on different scales: brightness for B/W, saturation for colour.

### 2. Cultural Analytics Lab projects (selected)

**What:** The lab's index of image-collection studies, 2008–2018. These five are the most relevant to CCW.

### a. Exploring *Sovetskoe Foto* (2018)
- **Method:** 401 covers of a Soviet photo magazine, 1926–1991, scraped from Archive.org with R and cut out with Automator. They are shown as date-sorted montages and as split montages (pre- and post-WWII, single decades). The analysis is qualitative and done by eye.
- **Borrow:** Split views, i.e. two montages side by side for two eras or two design regimes. For CCW: covers from before and after a house-design change (the vertical band versus the top band), or laserdisc/DVD-era versus Blu-ray-era designs.

### b. Google Logo Space (2012)
- **Method:** 587 Google doodles mapped on two custom "difference from the original" axes. x is how much the logo was modified; y is where the change happened (upper versus lower part). Image processing extracted the features. Result: a Gaussian cloud, with themed clusters (flags, athletes).
- **Borrow:**
  - Measure distance from a canonical form. For CCW, that form could be the median Criterion cover, or the film's theatrical poster from TMDB.
  - Encode where the change happens as an axis. For CCW: where the text sits, top versus bottom, which is already in the text-region data.
  - Say outright that "difference" can be defined many ways and show one at a time.

### c. Mondrian vs Rothko (2011)
- **Method:** Two painters' works in one brightness × saturation scatter, made with ImagePlot. Each artist's footprint (dense areas, gaps, outliers) shows which region of the space they explored.
- **Borrow:**
  - Draw a footprint hull or density contour per subgroup on the Colour axis: per designer, decade or country.
  - Show the empty regions of the space as a finding in their own right.

### d. MoMA Photography Collection (2014)
- **Method:** 18,941 photos, 1837–2012. The lab extracted tonality, contrast, palette, texture, line orientation, and face presence and position, along with metadata (artist, nationality, process). The layouts include a radial view, where radius is year and angle is mean brightness, and a vertical year-sorted strip.
- **Borrow:**
  - Try a radial layout for Spine or Year: radius is time and angle is hue or brightness. This works well as a zoomable WebGL "rose".
  - MoMA measured the same trace set as CCW (palette, line orientation, faces). That supports CCW's axis list and shows those measures work on a heterogeneous collection.

### e. Science vs Popular Science (2010)
- **Method:** Chronological montages of sampled issues and pages, including 9,801 *Science* pages. Read by eye, they show illustrations and photos disappearing until graphs dominate.
- **Borrow:** A montage can carry an argument about the medium, not just about style. CCW's Medium axis could show when photography, painting and illustration dominated across the spine sequence.

(Also relevant but not fetched: ImagePlot, Style Space, What colour is Slumdog Millionaire, Veja.vis covers.)

### 3. Cinemetrics (Frederic Brodbeck, 2011)

**Status:** Both CAN and FlowingData returned 403, and the author's site has an expired certificate. The summary below is from background knowledge.

**What:** A graduation project (KABK, The Hague) that turns each film into a generative "fingerprint" graphic.

**Method:** It parses a film into shots and measures shot length (editing rhythm), dominant colours per shot, and the amount of motion. Each shot becomes a segment of a circular glyph:
- Segment size is shot length.
- Fill is the shot's palette.
- Edge wobble or amplitude is motion.

Films can be compared side by side, and their remakes too. Built with Processing, plus Python/OpenCV for analysis.

**Borrow:**
- Give each cover a fingerprint glyph drawn only from its traces (palette ring, line-orientation rose, text-area %, face count). This is a copyright-safe stand-in for the image at low zoom or with the "eye" set to traces-only.
- Use one glyph grammar for every cover, so all 1,344 can be compared in small multiples.
- Cinemetrics compared remakes. Use the same idea to put a cover glyph next to its theatrical poster glyph (the "distance from release poster" axis).

### 4. Jason Salavon — Every Playboy Centerfold, The Decades (2002)

**What:** Four prints, each the pixel-mean of every Playboy centerfold from one decade, 1960s to 1990s.

**Method:** Every foldout from 1960 to 1999 is aligned and mean-averaged per decade into a blurred composite figure. The composites are printed as large C-prints and made with custom C/C++ and ImageMagick. Related works:
- *100 Special Moments*: averages of 100 commemorative photos each.
- *The Top Grossing Film of All Time, 1x1*: every frame of *Titanic* reduced to one average colour and tiled.
- *Emblem*: films as concentric colour rings.

Salavon frames this as tracking a genre's evolution as a mass.

**Borrow:**
- Make a mean cover per group: per decade, country, designer or medium. What stays sharp in the average is the house grammar (title placement, band, logo). What blurs is the individual design.
- Average the trace layers rather than the pixels: mean text-region heatmaps, mean line maps, mean face-box positions. This sidesteps copyright and shows conventions such as "faces sit in the upper third".
- Use the *Titanic 1x1* idea for the low-zoom view: one cell per cover at its mean or dominant colour, in Spine order, so the whole canon reads as a colour strip.

### 5. Shinseungback Kimyonghun (SSB)

**What:** A Seoul artist duo who make work from computer-vision misperception, especially face detection.

**Method (works fetched from ssbkyh.com):**
- **Cloud Face:** A camera and a face detector scan the sky. Clouds the algorithm reads as faces are printed as a grid, alongside a video of the detection process.
- **Nonfacial Portrait:** Painters make portraits under the rule that three face-detection algorithms watching the canvas must not find a face. They show the result with the painting process.
- **Portrait:** Faces detected in every 24th frame of a film are averaged into a single face per film, a Salavon-like composite made through a detector.
- **Analytic Portrait (2024):** 14 vision models run on one sitter, and their outputs are combined into the final portrait. The models are SIFT, OpenCV, Dlib, DSFD, DensePose, keypoints, expression (Ad-Corre), gaze (L2CS-Net), FairFace attributes, hair matting, a beauty-score dataset and Google Cloud Vision. A GIF shows the layers building up.

**Borrow:**
- Make machine misreadings a first-class layer. In the People axis, keep the false-positive faces (faces "found" in typography, patterns or objects) and give them their own view, e.g. covers a detector says contain faces that a human says do not. Do the same for faces a detector misses on illustrated or abstract covers. This is the meaning-level crack CCW wants, and it fits the "오독 아카이브" concept directly.
- Stack the "eye" toggle like Analytic Portrait: reveal traces one model at a time (OCR boxes → palette → LSD lines → face boxes → Claude description), so the cover is rebuilt from machine views. Label each layer with its model name.
- Use SSB's Portrait as a source for a face-average per group: the mean face crop across all covers in a decade or by one designer.
- Show model disagreement. Where OCR, Vision and the VLM disagree (text found or not, face or not), mark the cover. The disagreement is a trace in itself.

---

### Top takeaways for CCW

1. **Elevate misreadings to an axis or view (SSB).** Show detector false positives and false negatives (faces in type, text missed in illustration) and cross-model disagreement as their own layer. This is the strongest fit for the "meaning-level crack" stance.
2. **Build the "eye" toggle as a layered reconstruction (Analytic Portrait).** Reveal traces one model at a time, each labelled with its tool, so the cover reads as the sum of the machine views.
3. **Average the traces, not the pixels (Salavon).** Make mean text-region, line and face-position maps per decade, country, designer or medium. This exposes the house grammar while staying copyright-light.
4. **Image-as-point scatters with switchable metrics (Time, Mondrian/Rothko).** Plot time (or one metric) against one metric, using B/W-aware measures. Add per-group footprint hulls and treat the empty regions as findings.
5. **Distance-from-canonical axes (Google Logo Space).** Measure distance from the median Criterion cover or from the release poster, plus where the deviation occurs. Say that "difference" is one chosen definition.
6. **A per-cover trace glyph (Cinemetrics) for low zoom.** Use one fixed grammar (palette ring, line rose, text %, face count) so all 1,344 can be compared as small multiples. Add a radial Spine "rose" (MoMA) as an alternative global layout.

---

## 부록 C: conditional similarity, composition, typography, tools

Scope: what each source is, what it found, whether it runs on the M1 Max, and how CCW could use it. Weak axes today are Composition and Typography.

---

### GeneCIS (Vaze, Carion, Misra; Meta/Oxford, CVPR 2023)
**What:** A zero-shot benchmark for conditional image similarity: given a reference image and a text condition, pick the most similar image in a small gallery.

**Key idea and results**
- Four tasks: focus on / change an attribute, focus on / change an object. Galleries have 10 to 15 images with distractors that resemble either the reference or the condition, so a model has to use both.
- Plain CLIP baselines do poorly. Average R@1: image only 11.5, text only 8.1, image + text averaged 12.6. Their Combiner, trained on 1.6M triplets mined automatically from CC3M captions (subject, predicate, object parsing), reaches 16.8.
- Scores correlate only weakly with ImageNet accuracy, so bigger backbones alone do not solve conditional similarity. For "colour"-type conditions, plain image embeddings already work reasonably well.

**Local / licence:** The benchmark is small and the model is a CLIP RN50x4 plus a small Combiner MLP, so it runs easily on MPS. Code licence is not stated in the paper text I read (project page: sgvaze.github.io/genecis).

**Use in CCW**
- Copy the evaluation format, not the model. Build a small CCW-GeneCIS: for each reference cover, a condition (for example "Composition: frame in frame" or "Typography: hand-drawn title") and a gallery of 10 to 15 covers with one positive plus distractors that share the subject or medium but not the condition. Sonnet's description lines can supply the labels. Report R@1 and R@3 per axis, so every method below gets one comparable score.
- The image + text averaging baseline costs nothing with EmbeddingGemma 2 because image and text share one space. Try `normalize(img_emb + λ·text_emb("composition: <tag>"))` as a quick per-axis query vector. Expect small gains at most, as in the paper.
- Their recipe for mining training triplets from captions maps directly onto CCW's eight-line descriptions: pairs of covers whose Composition lines share a parsed relation (for example "figure inside frame") become positives for a lightweight Combiner or projection head.

---

### FocalLens (Hsieh et al.; UW + Apple, preprint 2025)
**What:** A vision encoder that takes a natural-language instruction as an extra input and returns an image embedding tuned to that instruction.

**Key idea and results**
- Trained contrastively on visual instruction-tuning triplets (image, instruction, answer): the conditioned image embedding must match the text embedding of the answer.
- Two variants. FocalLens-CLIP adds the instruction's text embedding as an extra token inside a CLIP ViT-L/14-336. FocalLens-MLLM reads a summary token from a LLaVA-style model.
- On a toy colour/shape set, plain CLIP gets 57 mAP when only colour matters and about 90 when only shape matters. Both FocalLens variants are much more balanced.
- Improves CelebA-attribute retrieval by about 8 points and GeneCIS by about 9 points (FocalLens-CLIP average R@3 43.5 vs CLIP 34.5). Also gains on SugarCrepe and MMVP-VLM, and keeps generic classification performance.

**Local / licence:** A ViT-L-sized model would run on MPS. The paper text I read does not mention released weights or a licence. Assume it is not available until a repo is found.

**Use in CCW**
- This is the closest published match to what CCW wants: one image, one embedding per aspect. If weights turn up, embed every cover once per axis ("How is the frame composed?", "How is the title set?") and keep separate neighbour lists.
- Without weights, the idea can be approximated cheaply. Train a small projection head per axis on frozen SigLIP 2 / DINO / EmbeddingGemma image features, contrastively aligned to the EmbeddingGemma embedding of the matching Sonnet line (the "answer"). That is FocalLens with a frozen backbone, and 1,344 pairs per axis is enough for a linear or 2-layer head.
- Its finding that CLIP-like embeddings are biased toward shape and object identity explains why People and Medium separate well in CCW while Composition does not.

---

### PixPlot (Yale DHLab)
**What:** An open-source tool that lays tens of thousands of images out in a 2D WebGL map, with similar images placed together.

**Key idea and results:** Inception (TensorFlow) image features, then UMAP (tunable `n_neighbors`, `min_dist`, metric), then HDBSCAN "hotspots" with a KMeans fallback, viewed in a custom WebGL viewer. Demos have around 24k to 31k images, and the docs say it scales past 100k with a smaller cell size. The metadata CSV supports category, tags, year, description and a label column for supervised UMAP. Note: the dhlab.yale.edu project URL now redirects to the new library site. These details come from the GitHub readme (YaleDHLab/pix-plot).

**Local / licence:** MIT. Runs locally (Python 3.x, conda). The TensorFlow feature extractor is dated, so treat it as a UI and layout reference rather than an analysis tool.

**Use in CCW**
- UI pattern: hotspot clusters with a representative thumbnail and label, plus switching between layouts (UMAP, by year, by metadata). CCW can generalise this to switching between per-axis maps that share the same point identities and animate between them.
- Its `label` column for supervised UMAP is a cheap way to pull Composition clusters apart for display. Supervised UMAP on Sonnet composition tags gives a "composition map" even if the raw embedding is weak, but label it honestly as tag-guided.
- Do not reuse its Inception features. CCW's DINOv2 / SigLIP 2 embeddings are stronger.

---

### UMSI: Predicting Visual Importance Across Graphic Design Types (Fosco et al., UIST 2020, Adobe + MIT)
**What:** A single model that predicts per-pixel visual importance for graphic designs and saliency for natural photos.

**Key idea and results:** Trained jointly on several design types plus natural images, with a built-in classifier that recognises the input type so the user does not have to label it. Comes with the Imp1k dataset: 1,000 crowd-annotated designs including webpages, **movie posters**, mobile UIs, infographics and ads. The authors report that it matches dedicated natural-image saliency models and beats earlier importance models. The Adobe page lists no numbers. Demo uses: re-weighting element importance and retargeting a design to new aspect ratios.

**Local / licence:** Code (`github.com/diviz-mit/predimportance-public`), Keras `.h5` weights and Imp1k are linked from predimportance.mit.edu. No licence is stated. It needs a Keras/TensorFlow environment (tensorflow-macos) or a conversion. The model is small, so 1,344 covers are trivial to process.

**Use in CCW**
- Replace or supplement the current OpenCV saliency with a design-aware importance map. Movie posters are in its training domain, which matters because generic saliency tends to fire on texture, while design importance fires on titles and faces.
- Composition vector: importance centroid, spread, entropy, distance to thirds and centre, left/right and top/bottom balance, and a coarse 4×4 or 8×8 pooled importance grid as a layout fingerprint.
- Typography: the share of total importance that falls inside the OCR title boxes (title-dominant vs image-dominant covers). This is an interpretable axis that current embeddings cannot express.
- It also feeds VFCNet-style inputs (below) if DeepGaze is inconvenient.

---

### RADIO1D (Heinrich, Ranzinger et al.; NVIDIA, arXiv 2607.03624, July 2026)
**What:** A vision encoder that compresses an image into a variable-length 1D token sequence. It is distilled from DINOv3, SigLIP2 and SAM3, and the paper includes an analysis of how SigLIP2 and DINOv3 features differ.

**Key idea and results**
- Analysis: DINOv3 patch features are spatially coherent (strong neighbour correlations, highest off-diagonal CKA). SigLIP2 patch features are weakly spatial, and a few fixed-position tokens act as hidden global summarisers. VLM fine-tuning pushes encoders toward the SigLIP2 pattern (CKAoff drops from 0.28 to 0.035).
- RADIO1D uses nested dropout so the first tokens summarise globally. One token still gives 40 mIoU on ADE20K, so rough scene layout lives in very few tokens.
- It introduces a composition-aware retrieval metric. For a query and a retrieved image, matched object boxes score same category × (gIoU+1)/2, the pairs are assigned with the Hungarian algorithm, and the score is normalised by the best possible match in the database. On Comp@1, RADIO1D-H scores 0.516 on COCO vs SigLIP2 0.505 and DINOv3 0.481, with larger gains on nuImages.
- In VLMs it offers an accuracy/latency dial through the token count.

**Local / licence:** The authors say checkpoints are released under a permissive licence. The H model (ViT-H class, roughly 0.6B or more parameters) should run on MPS for 1,344 images, likely via `trust_remote_code`. Check the exact Hugging Face repo and licence before use.

**Use in CCW**
- **Use the metric as a similarity function, not only for evaluation.** CCW already has boxes: OWLv2 objects, YuNet faces, SegFormer person parts, and OCR text boxes. "Hungarian-matched gIoU over same-class boxes" gives a direct, explainable **Composition similarity** when computed over object and face boxes, and a **Typography layout similarity** when computed over text boxes (class = title / credit / Criterion band). No new model is needed.
- Use it as the score for a composition benchmark. Comp@K over CCW's own boxes can rank SigLIP 2, DINOv2/v3, RADIO1D and EmbeddingGemma on how well their neighbours preserve layout.
- Try the RADIO1D 1-to-8-token embeddings as a "layout summary" vector for the Composition axis, since early tokens hold coarse scene structure.
- The analysis explains why CCW's pooled SigLIP 2 vector is weak on composition. For layout, prefer DINO patch-token grids pooled to a coarse spatial grid, for example 4×4×C then PCA.

---

### DINOv3 (Meta, Aug 2025)
**What:** Meta's third-generation self-supervised vision foundation model, a 7B ViT teacher trained on 1.7B images without labels, plus distilled smaller models.

**Key idea and results:** It is 7× larger and trained on 12× more data than DINOv2. Its strongest point is high-quality dense features: a frozen backbone with simple heads reaches state of the art on detection, segmentation and depth. It matches or beats SigLIP 2 and Perception Encoder on many global benchmarks across 60+ benchmarks. Released sizes include distilled ViT-S/B/L (and larger) and ConvNeXt T/S/B/L. The blog does not describe Gram anchoring; the paper does.

**Local / licence:** Weights on Hugging Face, gated behind Meta's DINOv3 licence, which permits commercial use. ViT-B/L run comfortably on M1 Max MPS via transformers (supported in recent versions). The 7B is impractical locally.

**Use in CCW**
- Swap DINOv2-base for DINOv3 ViT-B/16 or ViT-L/16 as the "visual structure" embedding. VFCNet (below) shows raw DINOv3 CLS already matches specialised composition models on PICD.
- Dense patch features give a layout fingerprint: pool patch tokens to a 3×3 or 4×4 grid on the de-texted cover (CCW already masks text regions), then use that as the Composition vector.
- Patch-feature clustering or PCA on each cover can produce figure/ground segmentation and region graphs for visual traces on the main screen, which fits the "machine-extracted traces" display direction.

---

### SAM 3 (Meta; Ultralytics docs)
**What:** Promptable Concept Segmentation: segments every instance of a concept given a text noun phrase or an image exemplar. It also keeps SAM 2-style point and box prompts and video tracking.

**Key idea and results:** 473.6M parameters (3.45 GB). LVIS zero-shot mask AP is 47.0 (previous best 38.5). SA-Co/Gold CGF1 is 65.0 vs OWLv2 at 34.3. COCO zero-shot box AP is 53.5. On an RTX PRO 6000 via Ultralytics it takes about 2.9 s per image; Meta quotes 30 ms on an H200. A SAM 3.1 checkpoint exists for images only.

**Local / licence:** `pip install -U ultralytics`. Weights (`sam3.pt`) are gated on Hugging Face and must be downloaded manually. The docs state no licence (Meta's SAM licence applies) and say nothing about MPS. Expect it to run on MPS or CPU at a few seconds per image, which is fine for 1,344 covers.

**Use in CCW**
- Replace OWLv2 boxes with SAM 3 masks from text prompts that target composition primitives: "frame", "window", "circle", "silhouette", "horizon", "doorway", "hand", "face". Masks give area, centroid, overlap and nesting, and nesting directly detects frame-in-frame.
- Use exemplar prompting for recurring Criterion motifs (for example a circular vignette): give one example box and retrieve similar shapes across covers.
- Typography: prompt "title text" or "hand-lettered text" to get pixel masks for glyph shapes. Mask outline statistics (stroke irregularity, baseline curvature) help separate hand-drawn from typeset titles, which OCR boxes cannot.
- Also enables the "grammar swap" concept (layer separation) with better masks than SAM 2.

---

### Open-source OCR comparisons (Roboflow; pasqualepillitteri.it 2026)
**What:** Two 2026 roundups of local OCR models, one general and one focused on local hardware.

**Key idea and results**
- Document-parsing leaders on OmniDocBench v1.6: PaddleOCR-VL-1.6 (0.9B, Apache-2.0) 96.3, MinerU2.5-Pro (1.2B, Apache-2.0 plus usage conditions) 95.8, GLM-OCR (0.9B, MIT) 95.2. Then dots.ocr, DeepSeek-OCR 2 and olmOCR.
- For scene text and bounding boxes, both pieces point to PP-OCR (now v6, 1.5M to 34.5M parameters, Apache-2.0) and Florence-2 (0.23B/0.77B, MIT). Other entries: SmolVLM2, Qwen 3.5 for reasoning over text, Chandra OCR 2 for handwriting (restricted licence), EasyOCR.
- Mac: PaddleOCR-VL-1.6 and GLM-OCR run through MLX-VLM, Granite-Docling has an MLX build, and Marker runs on MPS. vLLM and SGLang are not available on Mac. The benchmarks cover only English and Chinese documents.

**Local / licence:** As listed above. Everything relevant to CCW runs on the M1 Max.

**Use in CCW**
- The current PaddleOCR detection plus macOS Vision recognition pipeline is already the right kind of tool (scene-text detection). Document-parsing leaderboards say little about stylised cover titles, so do not switch on their basis.
- Cheap upgrade worth testing: PP-OCRv6 detection at det-640, compared against the current boxes on the hand-lettered covers where detection is hardest.
- Florence-2 `<OCR_WITH_REGION>` (MIT) is a second opinion for text boxes and also gives region captions. Agreement between the two pipelines can flag uncertain text regions.
- For Typography, OCR's job is to provide clean boxes and role labels (title / director / credits / Criterion band). Style should come from embeddings of the text crops (see takeaways), not from OCR models.

---

### PhotoFramer (You et al.; CAS / CUHK / Adobe, arXiv 2512.00993, v2 2026)
**What:** A multimodal model that, given a poorly composed photo, explains in text how to reframe it (shift, zoom in or change viewpoint) and generates an example of the improved frame.

**Key idea and results**
- 207K pairs of poor and good images built from cropping datasets, plus synthetic view-change pairs made by a learned "composition degradation" model. Qwen2.5-VL-32B wrote the guidance text.
- As a component, they trained a composition-assessment model (Qwen2.5-VL-7B with GRPO) on CADB, GAIC, KU-PCP and AVA. It reaches SRCC about 0.76 on CADB and 0.80 on GAIC.
- Key number for CCW: on CADB composition **classification**, off-the-shelf Qwen2.5-VL-32B scores about 10% accuracy, while their fine-tuned 7B scores about 58%. KU-PCP and CADB both use photographic classes (rule of thirds, centre, symmetric, diagonal, curved, triangle, pattern, horizontal, vertical).
- The paper uses U2Net saliency plus DINOv2 masked features to check subject consistency.

**Local / licence:** A 7B Qwen2.5-VL derivative would run on M1 Max via MLX (4 to 8-bit). The text I read does not state whether weights are released or under what licence. Check zhiyuanyou.github.io/photoframer.

**Use in CCW**
- Evidence that general VLMs describe composition poorly, so CCW's Sonnet "Composition" lines are likely noisy labels. Embedding them more cleverly will not fully fix that axis. Treat them as weak supervision and validate with geometric features.
- If the 7B assessment model is released, run it on covers to get a composition score plus a KU-PCP/CADB class as an extra label set. Expect a domain gap: it is trained on photos, and many covers are illustrations.
- The U2Net + masked-DINOv2 check is a reusable recipe: compare covers by subject appearance only (masked features), separately from background and layout.

---

### PICD: Photographic Image Composition Dataset (Zhao et al., CVPR 2025 Highlight)
**What:** A large benchmark for composition embedding: "Can machines understand composition?"

**Key idea and results**
- The paper version has 36,857 images in 24 composition categories across 355 scenes. The repo version has 49,123 images and is still growing.
- The taxonomy crosses element type (points / lines / shapes, after Kandinsky) with arrangement (rule of thirds, centred, diagonal, vertical, horizontal, triangle, C/O/S-curve, radial, dense, scatter, and others).
- Metric: Composition Discrimination Accuracy (CDA), a triplet test. CDA-1 asks whether the same-composition image is closer than a different-composition one. CDA-2 adds semantic interference: the negative shares the anchor's content.
- The paper finds that models trained for composition classification on small sets are near chance under CDA, and that cropping or aesthetic models carry more composition signal.
- Note: the CVPR virtual page returned HTTP 500. These details come from the GitHub repo (CV-xueba/PICD_ImageComposition) and search abstracts.

**Local / licence:** Data under CC BY-NC-SA 4.0 plus Terms of Use (non-commercial, fine for CCW). 44,577 images are downloadable; the rest come via ID/URL mapping. Labels are in `labels_PICD.csv`. Negatives, quality scores and text descriptions are marked "coming soon".

**Use in CCW**
- Adopt the **CDA-2 protocol** for CCW: triplets where the negative shares People or Medium with the anchor but differs in Composition tag. This is exactly the test for whether a "composition embedding" ignores subject matter. Pair it with the GeneCIS-style set above.
- Train a linear or MLP probe on DINOv3 features using PICD's 24 classes, apply it to covers, and use the 24-d probability vector as a Composition embedding. Its vocabulary (points / lines / shapes × arrangement) suits graphic covers better than KU-PCP's photographic rules.
- Expect a domain gap from photos to illustrated covers. Validate on about 50 hand-checked covers before trusting it.

---

### VFCNet (Dadras et al.; TU Wien / UAS St. Pölten, arXiv 2604.16500, Apr 2026)
**What:** A composition classifier and embedding that sees only saliency and edge-flow fields, not the RGB image, so it is robust to content.

**Key idea and results**
- Input is a 3-channel tensor: DeepGaze IIE saliency plus the averaged x/y components of two gradient vector flow (GVF) fields, one from image gradients and one pulled toward salient regions (μ 0.15, 10 iterations).
- Backbone: DINOv3 ViT-B/16, fine-tuned, with pooled features from low, mid and high blocks, plus divergence / curl / magnitude statistics of the GVF at three scales. Trained on KU-PCP (4,244 images, 9 classes), evaluated zero-shot on PICD.
- Results: CDA-1 0.683 and CDA-2 0.629, against CACNet 0.513 / 0.462, DINOv3 fine-tuned on RGB 0.621 / 0.505, and plain frozen DINOv3 CLS 0.489 / 0.392. Ablations show that edges plus saliency alone carry most composition information.

**Local / licence:** Code at github.com/ADadras/VFCNet (licence not stated in the paper). ViT-B plus a 30-epoch fine-tune on 4k images is feasible on M1 Max MPS. GVF is a cheap NumPy iteration, and DeepGaze IIE is PyTorch.

**Use in CCW**
- Most directly applicable composition method. Compute the saliency + GVF 3-channel image for every cover, ideally on the de-texted version so titles do not dominate, and embed it with the released VFCNet (512-d) or simply with frozen DINOv3. This gives a **Composition embedding that is stripped of semantics by construction**, so People and Medium should stop leaking into composition neighbours.
- Use UMSI importance (design-aware) in place of DeepGaze, since covers are designs. Compare both on the CCW CDA-2 set.
- The GVF divergence / curl / magnitude statistics are interpretable features ("flow converges on the centre", "rotational") for the composition vector and for a "flow-field" visual trace on the main screen. Copyright-safe, since no pixels are shown.
- Running the same input on text-only masks (the text region raster) gives a typography-layout flow embedding.

---

### Embedding Atlas (Apple)
**What:** An open-source web tool for exploring large datasets through embedding maps with linked metadata views.

**Key idea and results:** It computes embeddings (or takes precomputed vectors or x/y) and projects them with UMAP. Points are density-scaled. Selecting a point shows its nearest neighbours, either from a precomputed `neighbors` column or computed. It supports SQL filtering and cross-table joins over metadata, optional graph metrics (PageRank, degree), image columns, user-supplied labels, and an MCP server for agents. Launch with `pip install embedding-atlas` / `uvx embedding-atlas data.parquet`, as a widget or Streamlit app, or export it as a static web app.

**Local / licence:** MIT, runs locally (default localhost:5055).

**Use in CCW**
- Fastest internal QA tool for per-axis embeddings. Write one Parquet with cover id, metadata (year, spine, director, Sonnet tags) and one x/y plus `neighbors` column per axis. Browse each axis's neighbours with filters such as "Medium = illustration AND Composition tag = frame in frame".
- Precomputed `neighbors` columns let CCW load its own conditional neighbour lists (for example Hungarian-gIoU typography neighbours) and inspect them visually before building the real UI.
- Keep it local. The static export would embed cover thumbnails, which conflicts with the no-public-originals rule unless only traces are exported.

---

### EmbeddingGemma 2 (Google, model card)
**What:** A roughly 740M-parameter multimodal embedding model (270M text core, 170M vision encoder, 300M audio encoder) that maps text, images, video and audio, alone or interleaved, into one 768-d space.

**Key idea and results:** Matryoshka truncation to 512, 256 or 128 (256 is near full quality; 128 hurts multimodal quality; re-normalise after truncation). Context is 8,192 tokens shared across modalities, and an image costs 280 tokens by default, configurable from 70 to 1,120. Text uses task prefixes: asymmetric tasks use `task: search result | query: …` and `title: … | text: …`, while symmetric tasks (classification, clustering, similarity) use the same prefix on both sides. Media takes no prefix and is placed via `<|image|>` placeholders. Benchmarks: MTEB multilingual v2 61.4, MIEB-lite 65.7, MMEB v2 image 57.3, VisDoc 67.8. Use bf16 or fp32, not fp16.

**Local / licence:** Apache 2.0. sentence-transformers (`SentenceTransformer("google/embeddinggemma-2")`, `prompt_name`, `truncate_dim`). Disable the audio encoder with `config_kwargs={"audio_config": None}`. Already running in CCW.

**Use in CCW**
- **Interleaved input is a free FocalLens-style test.** Embed `<|image|>` plus an axis instruction ("task: clustering | query: the layout and framing of this poster") and check on the CCW-GeneCIS / CDA-2 sets whether neighbours shift toward the instructed aspect. The card does not promise that instructions steer image features, so measure before adopting.
- Raise the vision token budget (560 or 1,120) for Typography. At the default 280, small and fine lettering on 1,344 covers is probably below resolution. Also embed the **text-region crops** alone (title crop as image) as a typography-style vector.
- Use symmetric prefixes for description-line embeddings (classification/clustering), not the query/document pair, because cover-to-cover similarity is symmetric. Truncate to 256-d for the UMAP and kNN layers to cut noise.

---

### Top takeaways for CCW
Ranked by expected gain on the weak Composition and Typography axes first.

1. **Composition from structure-only inputs (VFCNet idea).** Embed a saliency or importance + GVF rendering of each de-texted cover with DINOv3, or with released VFCNet weights. This strips subject matter by construction and is the best-documented gain on content-robust composition (PICD CDA-2 0.63 vs 0.39 for raw DINOv3).
2. **Box-matching similarity as a direct conditional metric (RADIO1D's Comp score).** Run Hungarian-matched gIoU over existing boxes: text boxes with roles give Typography layout neighbours, and object/face/SAM 3 masks give Composition neighbours. No training, fully explainable, and the per-pair match can be drawn on screen as traces.
3. **Build a CCW evaluation set before choosing models.** Use GeneCIS-style conditional triplets plus PICD-style CDA-2, where the negative shares People or Medium but not the target aspect. Score every candidate per axis, since Sonnet composition labels are probably noisy (PhotoFramer: an off-the-shelf 32B VLM gets about 10% on composition classification).
4. **Typography from text-region crops, not whole covers.** Embed the title-box crops (SigLIP 2 / DINOv3 / EmbeddingGemma at a higher vision-token budget). Combine with geometry (position, size, angle, area share, box count) and the UMSI importance share inside title boxes.
5. **Design-aware importance maps (UMSI, trained on posters).** Replace generic saliency. They give interpretable composition stats (centroid, balance, thirds distance, coarse grid) and a "title dominance" typography feature.
6. **Cheap learned conditioning (FocalLens without its weights).** Train per-axis projection heads on frozen image embeddings, aligned contrastively to the EmbeddingGemma embedding of that axis's Sonnet line. Also test EmbeddingGemma 2 interleaved image + instruction input, and the GeneCIS image + text averaging baseline, as zero-cost options.
7. **Upgrade the backbone and masks.** Use DINOv3 ViT-B/L, with coarse-grid pooled patch tokens for layout, instead of DINOv2-base. Use SAM 3 text-prompted masks ("frame", "circle", "silhouette", "title text") for frame-in-frame nesting and hand-drawn-glyph shape statistics. Optionally train a PICD 24-class probe as an extra composition vector.
8. **Tooling.** Use Embedding Atlas locally (Parquet with per-axis x/y and `neighbors` columns) to QA conditional neighbours quickly. Take PixPlot's hotspots and layout switching as UI patterns. No change to OCR is needed beyond an optional PP-OCRv6 or Florence-2 cross-check on hand-lettered titles.
