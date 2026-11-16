# gemini — 정성 오류 분석 자료

홉 2000개 (제외: 없음). R = §4.1 answer-bearing hit@5.

## 유형별 비율 (index class 안에서 %)

| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |
|-|-|-|-|-|-|-|-|
| same | 1000 | 75.5 | 3.1 | 2.7 | 3.0 | 12.5 | 3.2 |
| cross | 1000 | 62.9 | 5.0 | 2.3 | 3.9 | 21.1 | 4.8 |

judge 결과와 정답(주 매처)이 어긋나는 홉:
- same: 지지 안 됨∧정답 16 / 지지 안 됨 59, 지지됨∧오답 89 / 지지됨 785
- cross: 지지 안 됨∧정답 14 / 지지 안 됨 71, 지지됨∧오답 106 / 지지됨 668

hop1 (앞 홉 비기권 870개) 중 bridge 가 gold 정본 라벨과 다른 홉 524개 (60.2%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)
그중 앞 홉 답이 오답인 hop1 160개 → 결과: supported 30, abstain 61, unsupported 69

## R1-abstain — 답 근거를 검색했는데 기권 (missed evidence) (cross-index 후보 50개)

### kch_000878 hop1 (pattern c, cross, gold_lang en)
- sub_q: 노벨 문학상을 수여하는 기관은 어디인가?
- 검색 쿼리: 한국소설문학상, 오늘의 젊은 예술가상, 이상문학상 대상, 국제 부커상, 노벨 문학상, 말라파르테 상을 수여하는 기관은 어디인가?
- gold: Swedish Academy · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 한국소설문학상, 오늘의 젊은 예술가상, 이상문학상 대상, 국제 부커상, 노벨 문학상, 말라파르테 상 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: The Nobel Prize in Literature, here meaning for Literature (Swedish: Nobelpriset i litteratur), is a Swedish literature prize that is awarded annually to an aut…
- 패시지 2: The Booker Prize, formerly the Booker Prize for Fiction (1969–2001) and the Man Booker Prize (2002–2019), is a literary award conferred each year for the best s…
- 패시지 3: The Premio Malaparte (lit. 'Malaparte Prize') is an Italian literary award given annually to an international writer. It was created in 1983 by Graziella Lonard…

### kch_000139 hop0 (pattern d, cross, gold_lang en)
- sub_q: 이바라키 공항의 운영기관은 어디인가?
- 검색 쿼리: 이바라키 공항의 운영기관은 어디인가?
- gold: Japan Air Self-Defense Force · R=1 · 결과=abstain · 오답
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Ibaraki Airport (茨城空港, Ibaraki Kūkō) is an airport in the city of Omitama, Ibaraki Prefecture, Japan. It also serves as an air base for the Japan Air Self-Defen…
- 패시지 2: Airports of Morocco (Arabic: مطارات المغرب), formerly the Office national des aéroports (ONDA), is a Moroccan public institution responsible for the management,…
- 패시지 3: Bahrain International Airport (IATA: BAH, ICAO: OBBI) (Arabic: مطار البحرين الدولي, romanized: Maṭār al-Baḥrayn al-dwalī) is the international airport of Bahrai…

### kch_000009 hop1 (pattern c, cross, gold_lang en)
- sub_q: 항공자위대가 속한 국가는 어디인가?
- 검색 쿼리: 모르겠습니다가 속한 국가는 어디인가?
- gold: Japan · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 모르겠습니다 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Italy, Japan, Liechtenstein, Luxembourg, Malta, Monaco, Netherlands, New Zealand, Norway, Portugal, San Marino, South Korea, Spain, Sweden, Switzerland, Turkey,…
- 패시지 2: The United States of America (USA), also known as the United States (U.S.) or America, is a country primarily located in North America. It is a federal republic…
- 패시지 3: The Kingdom of the Netherlands is a founding member state of the European Union. Although originally both Suriname and the Netherlands Antilles were explicitly …

## R1-unsupported — 답 근거를 검색했는데 지지되지 않는 답 (grounding failure) (cross-index 후보 23개)

### kch_000445 hop1 (pattern d, cross, gold_lang en)
- sub_q: 토트넘 홋스퍼 FC가 속한 리그는 무엇인가?
- 검색 쿼리: FC 서울 (Anyang LG Cheetahs), PSV 에인트호번, 토트넘 홋스퍼, 강원 FC가 속한 리그는 무엇인가?
- gold: Premier League · R=1 · 결과=unsupported · 오답
- 앞 홉 답(bridge): FC 서울 (Anyang LG Cheetahs), PSV 에인트호번, 토트넘 홋스퍼, 강원 FC (gold bridge 아님)
- 생성: 답변: K League 1
- claim 라벨: ['unsupported']
- 패시지 1: FC Seoul (Korean: FC 서울) is a South Korean professional football club based in Seoul that competes in the K League 1, the top flight of South Korean football. T…
- 패시지 2: Anyang is home to the HL Anyang ice hockey team which play in the Asia League Ice Hockey. They won their first Asian hockey league title in 2010 and were the fi…
- 패시지 3: FC Anyang (Korean: FC 안양) is a South Korean professional football club based in Anyang that competes in the K League 1, the top tier of South Korean football. T…

### kch_000379 hop0 (pattern d, cross, gold_lang en)
- sub_q: 네덜란드 왕궁의 본사는 어디에 있는가?
- 검색 쿼리: 네덜란드 왕궁의 본사는 어디에 있는가?
- gold: Amsterdam · R=1 · 결과=unsupported · 정답
- 생성: 답변: 암스테르담
- claim 라벨: ['contradicted']
- 패시지 1: Huis ten Bosch (Dutch: Paleis Huis ten Bosch, pronounced [paːˈlɛis ˌɦœys tɛm ˈbɔs]; English: "House in the Woods") is a royal palace in The Hague, Netherlands. …
- 패시지 2: The Palace of Laeken or Castle of Laeken (French: Château de Laeken; Dutch: Kasteel van Laken; German: Schloss zu Laeken) is the official residence of the King …
- 패시지 3: The Hague (Dutch: Den Haag, short for 's-Gravenhage) is the capital city of the South Holland province of the Netherlands. With a population of over half a mill…

### kch_000248 hop1 (pattern d, cross, gold_lang en)
- sub_q: 예일 대학교가 속한 행정구역은 어디인가?
- 검색 쿼리: 예일 대학교가 속한 행정구역은 어디인가?
- gold: New Haven · R=1 · 결과=unsupported · 정답
- 앞 홉 답(bridge): 예일 대학교 (gold bridge 아님)
- 생성: 답변: 뉴헤이븐 (New Haven)
- claim 라벨: ['unsupported']
- 패시지 1: IE University, known in Spanish as IE Universidad or Universidad Instituto de Empresa (lit. University Institute of Business), is a private university with camp…
- 패시지 2: Zaria (formerly Zazzau) is a metropolitan city in Nigeria, located within four local government areas in Kaduna State. It serves as the capital of the Zazzau Em…
- 패시지 3: The Yale School of Management (also known as Yale SOM) is the graduate business school of Yale University, a private Ivy League research university in New Haven…

## R0-unsupported — 답 근거 없이 답함 (ungrounded answer) (cross-index 후보 48개)

### kch_000993 hop1 (pattern c, cross, gold_lang en)
- sub_q: 아인트라흐트 프랑크푸르트가 속한 리그는 무엇인가?
- 검색 쿼리: 화성 FC가 속한 리그는 무엇인가?
- gold: Bundesliga · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 화성 FC (gold bridge 아님)
- 생성: 답변: K League 2
- claim 라벨: ['unsupported']
- 패시지 1: Hwaseong FC (Korean: 화성 FC) is a South Korean professional football club based in Hwaseong, Gyeonggi Province. They play in K League 2, the second tier of South…
- 패시지 2: Only six clubs have been champions since the introduction of the Super League: Galatasaray 26 times, Fenerbahçe 19 times, Beşiktaş 16 times (see note below), Tr…
- 패시지 3: Ulsan HD FC (Korean: 울산 HD FC), formerly Ulsan Hyundai FC and commonly known as Ulsan (울산), is a South Korean professional football club based in Ulsan that com…

### kch_000115 hop1 (pattern c, cross, gold_lang en)
- sub_q: 파나소닉의 설립자는 누구인가?
- 검색 쿼리: FDK의 설립자는 누구인가?
- gold: Kōnosuke Matsushita · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): FDK (gold bridge 아님)
- 생성: 답변: 알렉세이 나발니
- claim 라벨: ['unsupported']
- 패시지 1: The Militant League for German Culture (German: Kampfbund für deutsche Kultur, KfdK), was a nationalistic antisemitic political society during the Weimar Republ…
- 패시지 2: Jack Friedman (July 9, 1939 – May 3, 2010) was an American entrepreneur, businessman, veteran of the toy and video game industries, and a philanthropist. Regard…
- 패시지 3: The Anti-Corruption Foundation (ACF; Russian: Фонд борьбы с коррупцией (ФБК), romanized: Fond borby s korruptsiyey, FBK; lit. 'Foundation for combating corrupti…

### kch_000905 hop1 (pattern c, cross, gold_lang en)
- sub_q: 기드온 사르의 국적은 어디인가?
- 검색 쿼리: 박원순의 국적은 어디인가?
- gold: Israel · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 박원순 (gold bridge 아님)
- 생성: 대한민국 (또는 South Korea)
- claim 라벨: ['unsupported']
- 패시지 1: Park Won-soon (Korean: 박원순; February 11, 1955 or March 26, 1956 – July 9, 2020) was a South Korean politician, lawyer and activist who served as the mayor of Se…
- 패시지 2: Park Won-soon was born on February 11, 1955 or March 26, 1956, in Changnyeong, South Korea. He was enrolled at Kyunggi High School in 1971 and graduated in 1974…
- 패시지 3: Park Geun-hye (; Korean: 박근혜 [pak‿k͈ɯn.hje] ; born 2 February 1952) is a South Korean politician who served as the 11th president of South Korea from 2013 until…

## hop1-wrong-bridge — 앞 홉 오답이 bridge 로 넘어감 (cascade) (후보 160개)

### kch_000867 hop1 (pattern a, same, gold_lang ko)
- sub_q: 아인트라흐트 프랑크푸르트가 속한 리그는 무엇인가?
- 검색 쿼리: 부산 아이콘스가 속한 리그는 무엇인가?
- gold: 푸스발-분데스리가 · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 부산 아이콘스 (gold bridge 아님)
- 생성: 답변: K리그
- claim 라벨: ['unsupported']
- 패시지 1: 부산 아이파크 FC(Busan IPark Football Club)는 대한민국 부산광역시에 있는 축구단이다. 아이파크스포츠가 운영하는 남자 프로 축구단이며, 1979년에 창단되었다. 2026시즌 장호익이 주장이고 안현범, 김찬이 부주장이다. 모체는 1979년 11월 22일에 창단된 새한…
- 패시지 2: 롯데 자이언츠(영어: Lotte Giants)는 대한민국 부산광역시를 연고지로 하는 KBO 소속 프로 야구 팀이다. 주식회사로서 롯데그룹의 계열사이다.
- 패시지 3: 삼성 라이온즈(영어: Samsung Lions)는 대한민국의 대구광역시를 연고지로 하는 KBO 소속 프로 야구 팀이다. 법적으로는 주식회사이며, 제일기획의 자회사이다. 홈구장은 대구 삼성 라이온즈 파크이고, 제2홈구장은 포항야구장이다. 자회사로는 서울특별시 서초구에 위치한 삼성 레포츠 …

### kch_000131 hop1 (pattern a, same, gold_lang ko)
- sub_q: SC 브라가가 속한 리그는 무엇인가?
- 검색 쿼리: 국가대표가 속한 리그는 무엇인가?
- gold: 프리메이라리가 · R=0 · 결과=abstain · 오답
- 앞 홉 답(bridge): 국가대표 (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: 내셔널리그(National League)는 대한민국의 축구 리그 시스템에서 남자 축구 3부 리그에 해당했던 축구 대회이다. 한국실업축구연맹이 주관했던 남자 세미프로 축구 리그이며, 2003년 시즌부터 2019년 시즌까지 매년 개최되었다. 2020년 시즌에 K3리그, K4리그로 개편되면서…
- 패시지 2: 1966년 3월 20일 경상남도 진주시에서 출생했다. 진주 봉래초등학교, 진주중학교, 영등포공업고등학교를 졸업했다. 연세대학교에 입학하여 1학년을 마친 뒤 1985년 포철 아톰즈에 입단해 프로 축구 선수에 데뷔했고 서울 용산구 이태원에도 카페를 차리며 유흥가 사장님으로 변신했다. 포항에…
- 패시지 3: 초등학교 5학년 때 배구부 감독의 권유로 배구를 시작했으며, 2006년 아시안 게임에서 대한민국 국가대표 배구 선수이다. 2007-2008 시즌 드래프트 1순위로 LIG에 입단하였으며 2007년 12월 20일 현대캐피탈전에서 처음으로 출전하였다. 2012년에는 이경수가 부상당하자 그를 …

### kch_000936 hop1 (pattern d, cross, gold_lang en)
- sub_q: 고레에다 히로카즈가 졸업한 학교는 어디인가?
- 검색 쿼리: 프랑크 카프라가 졸업한 학교는 어디인가?
- gold: Waseda University · R=0 · 결과=abstain · 오답
- 앞 홉 답(bridge): 프랑크 카프라 (gold bridge 아님)
- 생성: 답변: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: matriculated at Hofstra University in 1955 as a theater arts major. There, he was awarded a scholarship in playwriting. This furthered his interest in directing…
- 패시지 2: The École nationale d'administration (French pronunciation: [ekɔl nɑsjɔnal dadministʁɑsjɔ̃]; ENA; English: National School of Administration) was a French grand…
- 패시지 3: The Canton of Geneva's public school system has écoles primaires (ages 4–12) and cycles d'orientation (ages 12–15). Students can leave school at 15, but seconda…
