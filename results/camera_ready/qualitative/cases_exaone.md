# exaone — 정성 오류 분석 자료

홉 1999개 (제외: {'ambiguous': 1}). R = §4.1 answer-bearing hit@5.

## 유형별 비율 (index class 안에서 %)

| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |
|-|-|-|-|-|-|-|-|
| same | 1000 | 68.3 | 7.6 | 5.0 | 3.4 | 11.2 | 4.5 |
| cross | 999 | 59.6 | 6.0 | 5.3 | 3.6 | 19.7 | 5.8 |

judge 결과와 정답(주 매처)이 어긋나는 홉:
- same: 지지 안 됨∧정답 34 / 지지 안 됨 95, 지지됨∧오답 68 / 지지됨 717
- cross: 지지 안 됨∧정답 38 / 지지 안 됨 111, 지지됨∧오답 72 / 지지됨 631

hop1 (앞 홉 비기권 845개) 중 bridge 가 gold 정본 라벨과 다른 홉 495개 (58.6%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)
그중 앞 홉 답이 오답인 hop1 134개 → 결과: supported 31, abstain 47, unsupported 56

## R1-abstain — 답 근거를 검색했는데 기권 (missed evidence) (cross-index 후보 60개)

### kch_000715 hop1 (pattern d, cross, gold_lang en)
- sub_q: 아인트라흐트 프랑크푸르트가 속한 리그는 무엇인가?
- 검색 쿼리: 모르겠습니다가 속한 리그는 무엇인가?
- gold: Bundesliga · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 모르겠습니다 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: The Primera División (Spanish pronunciation: [pɾiˈmeɾa ðiβiˈsjon]; English: "First Division"), known officially as Liga Profesional de Fútbol, or Torneo "Mercad…
- 패시지 2: The Nemzeti Bajnokság (Hungarian pronunciation: [ˈnɛmzɛti ˈbɒjnokʃaːɡ], lit. 'national championship'), also known as NB I or OTP Bank Liga after its title spons…
- 패시지 3: The Campeonato Nacional de Liga de Segunda División, commonly known as Segunda División or La Liga 2, and officially known as LaLiga HyperMotion for sponsorship…

### kch_000154 hop1 (pattern c, cross, gold_lang en)
- sub_q: 제리 샌더스가 졸업한 학교는 어디인가?
- 검색 쿼리: 제리 샌더스와 페어차일드 반도체 동료 7명가 졸업한 학교는 어디인가?
- gold: University of Illinois Urbana-Champaign · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 제리 샌더스와 페어차일드 반도체 동료 7명 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Jerry Sanders III grew up in the South Side of Chicago, Illinois, raised by his paternal grandparents. He was once attacked and beaten by a street gang leaving …
- 패시지 2: Notable Reed alumni include Tektronix co-founder Howard Vollum (1936), physicist James J. Brady (1927), businessman John Sperling (1948), linguistic anthropolog…
- 패시지 3: Notable alumni include House Democratic Whip Steny Hoyer; Google co-founder Sergey Brin; The Muppets creator Jim Henson; The Wire creator David Simon; former NF…

### kch_000040 hop1 (pattern c, cross, gold_lang en)
- sub_q: 게임 아츠의 본사는 어디인가?
- 검색 쿼리: 게임 아츠와 스튜디오 알렉스의 본사는 어디인가?
- gold: Chiyoda · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 게임 아츠와 스튜디오 알렉스 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Arc System Works Co., Ltd. (アークシステムワークス株式会社, Āku Shisutemu Wākusu kabushiki gaisha), commonly referred to as ArcSys, is a Japanese video game developer and publ…
- 패시지 2: Game Arts Co., Ltd. (株式会社ゲームアーツ, Kabushiki gaisha gēmu ātsu) is a Japanese video game developer and publisher based in Chiyoda, Tokyo. Originally established in…
- 패시지 3: Electronic Arts Inc. (EA) is an American video game company headquartered in Redwood City, California. Founded in May 1982 by former Apple employee Trip Hawkins…

## R1-unsupported — 답 근거를 검색했는데 지지되지 않는 답 (grounding failure) (cross-index 후보 53개)

### kch_000852 hop1 (pattern c, cross, gold_lang en)
- sub_q: 파레스 파레스의 출생지는 어디인가?
- 검색 쿼리: 파레스 파레스의 출생지는 어디인가?
- gold: Beirut · R=1 · 결과=unsupported · 정답
- 앞 홉 답(bridge): 파레스 파레스 (gold bridge 맞음)
- 생성: 베이루트, 레바논
- claim 라벨: ['contradicted']
- 패시지 1: Robert Emmanuel Pires was born on 29 October 1973 in Reims, Marne, France, to a Portuguese father and Spanish mother. Pires, the elder of two boys, spent most o…
- 패시지 2: Sean Paul Ryan Francis Henriques was born in Kingston on 9 January 1973. His mother Frances, a painter, is of English and Chinese Jamaican descent. His paternal…
- 패시지 3: Pope Francis was born Jorge Mario Bergoglio on 17 December 1936 in Flores, a neighbourhood of Buenos Aires. He was the eldest of the five children of Mario José…

### kch_000365 hop1 (pattern d, cross, gold_lang en)
- sub_q: FC 서울이 속한 리그는 무엇인가?
- 검색 쿼리: Incheon United이 속한 리그는 무엇인가?
- gold: K League 1 · R=1 · 결과=unsupported · 정답
- 앞 홉 답(bridge): Incheon United (gold bridge 아님)
- 생성: K League 1
- claim 라벨: ['unsupported']
- 패시지 1: Incheon United FC (Korean: 인천 유나이티드 FC) is a South Korean professional football club based in Incheon that competes in the K League 1, the top tier of South Kor…
- 패시지 2: Incheon is home to the following professional and semi-professional sports teams: Association football The K League 1 team Incheon United FC was founded in 2003…
- 패시지 3: The WK League (Hangul: WK리그) is a semi-professional women's football league, run by the Korea Football Association (KFA) and the Korea Women's Football Federati…

### kch_000347 hop0 (pattern b, cross, gold_lang en)
- sub_q: 이바라키 공항의 운영기관은 어디인가?
- 검색 쿼리: 이바라키 공항의 운영기관은 어디인가?
- gold: Japan Air Self-Defense Force · R=1 · 결과=unsupported · 오답
- 생성: 일본 정부 및 지역 관리 기관 (문서에서 구체적인 단일 기관명 명시 없음)
- claim 라벨: ['unsupported']
- 패시지 1: Ibaraki Airport (茨城空港, Ibaraki Kūkō) is an airport in the city of Omitama, Ibaraki Prefecture, Japan. It also serves as an air base for the Japan Air Self-Defen…
- 패시지 2: Airports of Morocco (Arabic: مطارات المغرب), formerly the Office national des aéroports (ONDA), is a Moroccan public institution responsible for the management,…
- 패시지 3: Bahrain International Airport (IATA: BAH, ICAO: OBBI) (Arabic: مطار البحرين الدولي, romanized: Maṭār al-Baḥrayn al-dwalī) is the international airport of Bahrai…

## R0-unsupported — 답 근거 없이 답함 (ungrounded answer) (cross-index 후보 58개)

### kch_000233 hop0 (pattern d, cross, gold_lang en)
- sub_q: 우리, 사랑해도 되나요?를 감독한 사람은 누구인가?
- 검색 쿼리: 우리, 사랑해도 되나요?를 감독한 사람은 누구인가?
- gold: Thomas Bezucha · R=0 · 결과=unsupported · 오답
- 생성: Jo Eun-ji
- claim 라벨: ['unsupported']
- 패시지 1: Perhaps Love (Korean: 장르만 로맨스; RR: Jangneuman romaenseu) is a 2021 South Korean comedy-drama film, directed by Jo Eun-ji. Starring Ryu Seung-ryong, Oh Na-ra and…
- 패시지 2: Self Portrait (short film, 2002) – director Mother, Beauitiful May (2003) – actor Life goes (short film, 2003) – sound Please Stay with Me (short film, 2004) – …
- 패시지 3: Justin Louis Baldoni (born January 24, 1984) is an American actor and director. He is best known for playing Rafael Solano on the CW telenovela Jane the Virgin …

### kch_000103 hop0 (pattern d, cross, gold_lang en)
- sub_q: 우상호 (축구 선수)가 소속된 팀은 어디인가?
- 검색 쿼리: 우상호 (축구 선수)가 소속된 팀은 어디인가?
- gold: FC Osaka · R=0 · 결과=unsupported · 오답
- 생성: FC Seoul
- claim 라벨: ['contradicted']
- 패시지 1: In 2015, Na graduated from Kumho High School, which had a youth football club of K League club Gwangju FC, and then entered Dankook University's football team. …
- 패시지 2: Woo Sang-ho (禹 相皓, U Sanho; Korean: 우상호; Hancha: 禹相皓; born 7 December 1992) is a professional footballer who plays as a midfielder for Kamatamare Sanuki. Born i…
- 패시지 3: Bae Jun-ho (Korean: 배준호; born 21 August 2003) is a South Korean professional footballer who plays as an attacking midfielder or left winger for EFL Championship…

### kch_000829 hop1 (pattern c, cross, gold_lang en)
- sub_q: 애벌랜치 소프트웨어의 본사는 어디인가?
- 검색 쿼리: 마인드스케이프의 본사는 어디인가?
- gold: Salt Lake City · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 마인드스케이프 (gold bridge 아님)
- 생성: Northbrook, Illinois (초기 설립지) 이후 여러 변화를 겪으며 최종적으로 2001년에는 프랑스 Boulogne-Billancourt로 이전했으나, 이는 본사의 마지막 위치입니다. 하지만 주어진 정보에서 가장 초기이자 명확한 본사 위치만을 답하자면: Northbrook, Illinois
- claim 라벨: ['unsupported', 'unsupported']
- 패시지 1: Mindjet is a mind mapping and innovation management software company headquartered in San Francisco, California. Mindjet's software products, including its flag…
- 패시지 2: Mindscape was a video game developer and publisher. The company was founded by Roger Buoy in October 1983 in Northbrook, Illinois, originally as part of SFN Com…
- 패시지 3: Mindjet was founded in 1998 by computer programmer Mike Jetter and his wife, Bettina Jetter, in order to support the development of their mind mapping software,…

## hop1-wrong-bridge — 앞 홉 오답이 bridge 로 넘어감 (cascade) (후보 134개)

### kch_000179 hop1 (pattern b, same, gold_lang ko)
- sub_q: 토머스 버주카의 출생지는 어디인가?
- 검색 쿼리: Jo Eun-ji의 출생지는 어디인가?
- gold: 애머스트 · R=0 · 결과=abstain · 오답
- 앞 홉 답(bridge): Jo Eun-ji (gold bridge 아님)
- 생성: 서울 (참고: 제공된 문서에는 Jo Eun-ji의 정확한 출생지 정보가 명시되어 있지 않으나, 일반적인 정보로 서울을 제시합니다. 주어진 텍스트에서는 해당 인물의 출생지에 대한 직접적인 언급이 없습니다.) 답변: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: 1996년 1월 16일 경기도 성남시 분당구 야탑동에서 외동딸로 태어났다. '제니'라는 이름은 SBS에서 방영된 드라마 《모래시계》에 출연한 배우 가운데 한 사람인 이정재가 맡은 등장인물인 백재희(윤혜린(고현정 분)의 보디가드)에서 따온 이름이다. 이정재의 팬이었던 제니의 어머니는 아들…
- 패시지 2: 1961년 9월 10일 경상남도 합천군 덕곡면 포두리 북동마을에서 지역 지주 집안에서 태어났다. 대구 오성고등학교를 졸업하고 220:1의 경쟁률을 기록한 경찰대학 제1기에 수석으로 입학하고 수석으로 졸업하였다. 이후 경감, 1998년 총경, 2005년 경무관, 2006년 치안감, 201…
- 패시지 3: 조는 1972년 대한민국 서울에서 태어났으며, 6살 때인 1978년 가족이 미국으로 이민하여 캘리포니아주 로스앤젤레스에서 자랐다. 연극 "여전사(The woman warrior)"로 배우의 길을 걷게 되었다. 아버지는 그리스도의 교회 목사였다. 1990년 캘리포니아주 글렌데일의 허버트 …

### kch_000151 hop1 (pattern a, same, gold_lang ko)
- sub_q: 제인 브룩의 출생지는 어디인가?
- 검색 쿼리: 매슈 모딘의 출생지는 어디인가?
- gold: 노스브룩 · R=0 · 결과=abstain · 오답
- 앞 홉 답(bridge): 매슈 모딘 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: 매슈 모딘(영어: Matthew Modine, 1959년 3월 22일 ~ )은 미국의 배우, 감독, 제작자, 영화 각본가이다. 《풀 메탈 재킷》(1987) 출연으로 이름을 알리기 시작했다. 텔레비전 영화 《앤 더 밴드 플레이드 온》(1993)과 《왓 더 데프 맨 허드》(1997)로 골든…
- 패시지 2: 매슈 로빈스(Matthew Robbins, 1945년 7월 15일 ~ )는 미국의 각본가이자 영화 감독이다. 뉴 할리우드 운동에서 저술 활동을 한 인물이다.
- 패시지 3: 매슈 로버트 "맷" 케인(영어: Matthew Robert "Matt" Kane, 1991년 1월 18일 ~ )은 잉글랜드의 배우, 작가, 영화 감독이다. 잉글랜드 브리스틀에서 태어났으며 미국 시민권도 갖고 있다.

### kch_000838 hop1 (pattern b, same, gold_lang ko)
- sub_q: CFR 클루지가 속한 리그는 무엇인가?
- 검색 쿼리: Hwaseong FC가 속한 리그는 무엇인가?
- gold: 리가 I · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Hwaseong FC (gold bridge 아님)
- 생성: K3리그 어드밴스 (2013년 시즌), K리그2 (2025년 시즌부터)
- claim 라벨: ['unsupported']
- 패시지 1: 홍콩 퍼스트 디비전 리그(영어: Hong Kong First Division League, 중국어: 香港甲組聯賽)는 1908년에 창설된 홍콩의 축구 리그로 아시아의 축구 리그 가운데 가장 오랜 역사를 자랑하며 2013년까지는 홍콩의 최상위 리그였지만 2014년 홍콩 프리미어리그 출범 이…
- 패시지 2: 화성 FC(Hwaseong Football Club)는 대한민국 경기도 화성시에 있는 축구 선수단이다. 화성에프씨가 운영하는 남자 프로 축구단이며, 2013년에 창단되었다. 2013년 시즌부터 K3리그 어드밴스에 참가했으며, 2025년 시즌부터 K리그2에 참가하고 있다. 홈 경기장은 화…
- 패시지 3: 대한민국 경상남도 울산시에 태어난 그는 전주대학교 졸업 이후 여러 국내 축구 구단으로의 입단이 힘들어지자 취업을 크로아티의 축구단인 HNSK 모슬라비나에 입단했다. 2012년 2월, 그는 보스니아 2부 리그 구단인 NK GOSK 가빌라로 이적했으며, 구단과의 계약이 만료된 후 기회를 찾…
