# sonnet — 정성 오류 분석 자료

홉 1997개 (제외: {'ambiguous': 3}). R = §4.1 answer-bearing hit@5.

## 유형별 비율 (index class 안에서 %)

| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |
|-|-|-|-|-|-|-|-|
| same | 999 | 73.3 | 4.7 | 2.6 | 3.7 | 12.6 | 3.1 |
| cross | 998 | 63.4 | 5.9 | 2.0 | 4.2 | 21.5 | 2.9 |

judge 결과와 정답(주 매처)이 어긋나는 홉:
- same: 지지 안 됨∧정답 14 / 지지 안 됨 57, 지지됨∧오답 86 / 지지됨 769
- cross: 지지 안 됨∧정답 16 / 지지 안 됨 49, 지지됨∧오답 98 / 지지됨 675

hop1 (앞 홉 비기권 864개) 중 bridge 가 gold 정본 라벨과 다른 홉 454개 (52.5%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)
그중 앞 홉 답이 오답인 hop1 147개 → 결과: supported 25, abstain 64, unsupported 58

## R1-abstain — 답 근거를 검색했는데 기권 (missed evidence) (cross-index 후보 59개)

### kch_000743 hop0 (pattern b, cross, gold_lang en)
- sub_q: 털리도 미술관을 설계한 건축가는 누구인가?
- 검색 쿼리: 털리도 미술관을 설계한 건축가는 누구인가?
- gold: Frank Gehry · R=1 · 결과=abstain · 오답
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Frank Owen Gehry ( GAIR-ee; né Goldberg; February 28, 1929 – December 5, 2025) was a Canadian and American architect and designer known for his postmodern desig…
- 패시지 2: Shōzō Uchii (内井 昭蔵, Uchii Shōzō) was a Japanese architect and academic authority on the works of Frank Lloyd Wright. Known for the design of landmark structures…
- 패시지 3: The park was designed by Bernard Tschumi, a French architect of Swiss origin, who built it from 1984 to 1987 in partnership with Colin Fournier, on the site of …

### kch_000139 hop0 (pattern d, cross, gold_lang en)
- sub_q: 이바라키 공항의 운영기관은 어디인가?
- 검색 쿼리: 이바라키 공항의 운영기관은 어디인가?
- gold: Japan Air Self-Defense Force · R=1 · 결과=abstain · 오답
- 생성: 답변: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Ibaraki Airport (茨城空港, Ibaraki Kūkō) is an airport in the city of Omitama, Ibaraki Prefecture, Japan. It also serves as an air base for the Japan Air Self-Defen…
- 패시지 2: Airports of Morocco (Arabic: مطارات المغرب), formerly the Office national des aéroports (ONDA), is a Moroccan public institution responsible for the management,…
- 패시지 3: Bahrain International Airport (IATA: BAH, ICAO: OBBI) (Arabic: مطار البحرين الدولي, romanized: Maṭār al-Baḥrayn al-dwalī) is the international airport of Bahrai…

### kch_000009 hop1 (pattern c, cross, gold_lang en)
- sub_q: 항공자위대가 속한 국가는 어디인가?
- 검색 쿼리: 모르겠습니다.가 속한 국가는 어디인가?
- gold: Japan · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 모르겠습니다. (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Italy, Japan, Liechtenstein, Luxembourg, Malta, Monaco, Netherlands, New Zealand, Norway, Portugal, San Marino, South Korea, Spain, Sweden, Switzerland, Turkey,…
- 패시지 2: The United States of America (USA), also known as the United States (U.S.) or America, is a country primarily located in North America. It is a federal republic…
- 패시지 3: This list includes all internationally recognised sovereign countries falling even partially under any common geographical or political definitions of Europe. W…

## R1-unsupported — 답 근거를 검색했는데 지지되지 않는 답 (grounding failure) (cross-index 후보 20개)

### kch_000379 hop0 (pattern d, cross, gold_lang en)
- sub_q: 네덜란드 왕궁의 본사는 어디에 있는가?
- 검색 쿼리: 네덜란드 왕궁의 본사는 어디에 있는가?
- gold: Amsterdam · R=1 · 결과=unsupported · 정답
- 생성: 답변: 암스테르담
- claim 라벨: ['contradicted']
- 패시지 1: Huis ten Bosch (Dutch: Paleis Huis ten Bosch, pronounced [paːˈlɛis ˌɦœys tɛm ˈbɔs]; English: "House in the Woods") is a royal palace in The Hague, Netherlands. …
- 패시지 2: The Palace of Laeken or Castle of Laeken (French: Château de Laeken; Dutch: Kasteel van Laken; German: Schloss zu Laeken) is the official residence of the King …
- 패시지 3: The Hague (Dutch: Den Haag, short for 's-Gravenhage) is the capital city of the South Holland province of the Netherlands. With a population of over half a mill…

### kch_000365 hop1 (pattern d, cross, gold_lang en)
- sub_q: FC 서울이 속한 리그는 무엇인가?
- 검색 쿼리: 인천 유나이티드이 속한 리그는 무엇인가?
- gold: K League 1 · R=1 · 결과=unsupported · 정답
- 앞 홉 답(bridge): 인천 유나이티드 (gold bridge 아님)
- 생성: 답변: K League 1
- claim 라벨: ['unsupported']
- 패시지 1: Incheon United FC (Korean: 인천 유나이티드 FC) is a South Korean professional football club based in Incheon that competes in the K League 1, the top tier of South Kor…
- 패시지 2: The National League is a professional association football league in England. The National League is the first division of the National Leagues and step 1 of th…
- 패시지 3: Source: League Second Division / Championship (level 2) Champions: 1934–35 Play-off winners: 2021 Third Division South / Third Division / League One (level 3) C…

### kch_000813 hop0 (pattern d, cross, gold_lang en)
- sub_q: 월드의 설립자는 누구인가?
- 검색 쿼리: 월드의 설립자는 누구인가?
- gold: Sam Altman · R=1 · 결과=unsupported · 정답
- 생성: 답변: Edward Daly (World Airways 기준) 또는 Sam Altman, Max Novendstern, Alex Blania (World/Worldcoin 기준)
- claim 라벨: ['unsupported']
- 패시지 1: World, also known as World Network, is a humanity verification project and financial network. It was developed and launched as the cryptocurrency project Worldc…
- 패시지 2: Jimmy Donal Wales (born August 8, 1966) is an American and British internet entrepreneur and former financial trader. He is best known for co-founding Wikipedia…
- 패시지 3: Mokichi Okada (岡田茂吉 Okada Mokichi; 23 December 1882 – 10 February 1955) was the founder of the World Church of Messiah, that later became the Church of World Ме…

## R0-unsupported — 답 근거 없이 답함 (ungrounded answer) (cross-index 후보 29개)

### kch_000211 hop1 (pattern d, cross, gold_lang en)
- sub_q: 연세대학교가 속한 행정구역은 어디인가?
- 검색 쿼리: 연세대학교, 캘리포니아대학교 샌디에이고가 속한 행정구역은 어디인가?
- gold: Seoul · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 연세대학교, 캘리포니아대학교 샌디에이고 (gold bridge 아님)
- 생성: 답변: 샌디에이고 카운티 (San Diego County)
- claim 라벨: ['unsupported']
- 패시지 1: The University of California, San Diego (UC San Diego or UCSD) is a public land-grant research university in the La Jolla neighborhood of San Diego, California,…
- 패시지 2: San Diego ( SAN dee-AY-goh; Spanish: [san ˈdjeɣo]) is a city on the Pacific coast of Southern California, adjacent to the Mexico–United States border. It is the…
- 패시지 3: San Bernardino ( SAN BUR-nə-DEE-noh) is a city in and the county seat of San Bernardino County, California, United States. Located in the Inland Empire region o…

### kch_000829 hop1 (pattern c, cross, gold_lang en)
- sub_q: 애벌랜치 소프트웨어의 본사는 어디인가?
- 검색 쿼리: 마인드스케이프의 본사는 어디인가?
- gold: Salt Lake City · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 마인드스케이프 (gold bridge 아님)
- 생성: 답변: Boulogne-Billancourt, France
- claim 라벨: ['unsupported']
- 패시지 1: Mindjet is a mind mapping and innovation management software company headquartered in San Francisco, California. Mindjet's software products, including its flag…
- 패시지 2: Mindscape was a video game developer and publisher. The company was founded by Roger Buoy in October 1983 in Northbrook, Illinois, originally as part of SFN Com…
- 패시지 3: Mindjet was founded in 1998 by computer programmer Mike Jetter and his wife, Bettina Jetter, in order to support the development of their mind mapping software,…

### kch_000209 hop1 (pattern d, cross, gold_lang en)
- sub_q: CFR 클루지가 속한 리그는 무엇인가?
- 검색 쿼리: Hwaseong FC가 속한 리그는 무엇인가?
- gold: Liga I · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Hwaseong FC (gold bridge 아님)
- 생성: 답변: K League 2
- claim 라벨: ['unsupported']
- 패시지 1: In September 2012, Hwaseong City formed a committee to promote the establishment of Hwaseong FC. In January 2013, the team received approval from the Korea Foot…
- 패시지 2: Hwaseong FC (Korean: 화성 FC) is a South Korean professional football club based in Hwaseong, Gyeonggi Province. They play in K League 2, the second tier of South…
- 패시지 3: Hwaseong is home to the V-League women's volleyball team Hwaseong IBK Altos and the K League 2 football team Hwaseong FC, both of which are based at Hwaseong Sp…

## hop1-wrong-bridge — 앞 홉 오답이 bridge 로 넘어감 (cascade) (후보 147개)

### kch_000936 hop1 (pattern d, cross, gold_lang en)
- sub_q: 고레에다 히로카즈가 졸업한 학교는 어디인가?
- 검색 쿼리: Frank Capra가 졸업한 학교는 어디인가?
- gold: Waseda University · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Frank Capra (gold bridge 아님)
- 생성: 답변: California Institute of Technology
- claim 라벨: ['unsupported']
- 패시지 1: Soon after graduating from college, Capra was commissioned in the United States Army as a second lieutenant, having completed campus ROTC. In the Army, he taugh…
- 패시지 2: Capra was born Francesco Rosario Capra on May 18, 1897, in Bisacquino, a village near Palermo, Sicily, Italy. He was the youngest of seven children of Salvatore…
- 패시지 3: Frank Russell Capra (born Francesco Rosario Capra; May 18, 1897 – September 3, 1991) was an Italian-born American film director, producer, and screenwriter who …

### kch_000159 hop1 (pattern b, same, gold_lang ko)
- sub_q: 로스앤젤레스 FC가 속한 리그는 무엇인가?
- 검색 쿼리: 대전 하나 시티즌가 속한 리그는 무엇인가?
- gold: 메이저 리그 사커 · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 대전 하나 시티즌 (gold bridge 아님)
- 생성: 답변: K리그2
- claim 라벨: ['unsupported']
- 패시지 1: 대전 하나 시티즌 FC B(Daejeon Hana Citizen Football Club B)는 대한민국 대전광역시에 있었던 축구 선수단이다. 하나금융축구단이 운영했던 남자 U-23 축구단이며, 2008년에 창단되었고 2024년에 해단되었다. 2022년 시즌부터 2024년 시즌까지 K4…
- 패시지 2: K리그1의 클럽 중 하나인 FC 서울의 유스팀인 오산고등학교 축구부 출신으로 2017년 FC 서울의 우선지명을 받고 동국대학교에 진학하여 학교 축구부에 입단했다. 그러나 3학년 때인 2019년에 전방십자인대부상을 입으며 선수 생활이 어려워지면서 서울의 우선지명이 취소되었다. 이후 재활 …
- 패시지 3: 시티즌 AA(영어: The Citizen Athletic Association, Citizen AA) 또는 시티즌 체육회, 공민 체육회(중국어: 公民體育會)는 홍콩의 축구 클럽으로, 현재 홍콩 1부 리그에서 활동하고 있다.

### kch_000716 hop1 (pattern d, cross, gold_lang en)
- sub_q: 유르고르덴 IF가 속한 리그는 무엇인가?
- 검색 쿼리: FC Seoul가 속한 리그는 무엇인가?
- gold: Allsvenskan · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): FC Seoul (gold bridge 아님)
- 생성: 답변: K League 1
- claim 라벨: ['unsupported']
- 패시지 1: FC Seoul (Korean: FC 서울) is a South Korean professional football club based in Seoul that competes in the K League 1, the top flight of South Korean football. T…
- 패시지 2: FC Seoul K League 1: 2016 Korean League Cup: 2006 Korean FA Cup runner-up: 2016 Jeonnam Dragons Korean FA Cup: 2007 Korean League Cup runner-up: 2008 Ulsan Hyun…
- 패시지 3: FC Seoul K League 1: 2000, 2010, 2012 Korean FA Cup: 2002, 2009, 2010 Korean League Cup: 2010 Korean Super Cup: 2001 AFC Champions League runner-up: 2001–02, 20…
