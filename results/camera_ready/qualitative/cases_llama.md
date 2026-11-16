# llama — 정성 오류 분석 자료

홉 2000개 (제외: 없음). R = §4.1 answer-bearing hit@5.

## 유형별 비율 (index class 안에서 %)

| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |
|-|-|-|-|-|-|-|-|
| same | 1000 | 73.6 | 0.3 | 5.6 | 4.5 | 2.7 | 13.3 |
| cross | 1000 | 61.9 | 1.4 | 6.8 | 4.7 | 3.9 | 21.3 |

judge 결과와 정답(주 매처)이 어긋나는 홉:
- same: 지지 안 됨∧정답 23 / 지지 안 됨 189, 지지됨∧오답 135 / 지지됨 781
- cross: 지지 안 됨∧정답 15 / 지지 안 됨 281, 지지됨∧오답 147 / 지지됨 666

hop1 (앞 홉 비기권 968개) 중 bridge 가 gold 정본 라벨과 다른 홉 545개 (56.3%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)
그중 앞 홉 답이 오답인 hop1 313개 → 결과: supported 68, abstain 24, unsupported 221

## R1-abstain — 답 근거를 검색했는데 기권 (missed evidence) (cross-index 후보 14개)

### kch_000708 hop0 (pattern b, cross, gold_lang en)
- sub_q: 포드 에스코트의 제조사는 어디인가?
- 검색 쿼리: 포드 에스코트의 제조사는 어디인가?
- gold: Ford Motor Company · R=1 · 결과=abstain · 오답
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: The Ford Motor Company (commonly known as Ford, sometimes abbreviated as FoMoCo) is an American multinational automobile manufacturer headquartered in Dearborn,…
- 패시지 2: Peugeot (UK: , US: ; French: [pøʒo] ) is a French automobile brand owned by Stellantis. Peugeot was founded as a family business in 1810 and Automobiles Peugeot…
- 패시지 3: Polestar Automotive Holding UK PLC, or simply Polestar, is a Swedish automotive manufacturer that produces electric cars. Principally owned by Li Shufu's PSD In…

### kch_000050 hop0 (pattern b, cross, gold_lang en)
- sub_q: 서브노티카의 개발사는 어디인가?
- 검색 쿼리: 서브노티카의 개발사는 어디인가?
- gold: Unknown Worlds Entertainment · R=1 · 결과=abstain · 오답
- 생성: Unknown Worlds Entertainment
- claim 라벨: (claim 없음)
- 패시지 1: Subnautica was announced by Unknown Worlds Entertainment on December 17, 2013, with Charlie Cleveland as the director and lead gameplay programmer, and Hugh Jer…
- 패시지 2: Gnosia's development first began in 2015 by four-person indie studio Petit Depotto, being their second video game released after Unholy Heights. Unholy Heights …
- 패시지 3: Subnautica is a 2018 action-adventure survival game developed and published by Unknown Worlds Entertainment. The player controls Ryley Robinson, a survivor of a…

### kch_000002 hop1 (pattern d, cross, gold_lang en)
- sub_q: 베이징시가 속한 국가는 어디인가?
- 검색 쿼리: 모릅니다.가 속한 국가는 어디인가?
- gold: People's Republic of China · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 모릅니다. (gold bridge 아님)
- 생성: 답변: 모르겠습니다.
- claim 라벨: (claim 없음)
- 패시지 1: Italy, Japan, Liechtenstein, Luxembourg, Malta, Monaco, Netherlands, New Zealand, Norway, Portugal, San Marino, South Korea, Spain, Sweden, Switzerland, Turkey,…
- 패시지 2: Malta, officially the Republic of Malta, is an island country in Southern Europe located in the Mediterranean Sea, between Sicily and North Africa. It consists …
- 패시지 3: This list includes all internationally recognised sovereign countries falling even partially under any common geographical or political definitions of Europe. W…

## R1-unsupported — 답 근거를 검색했는데 지지되지 않는 답 (grounding failure) (cross-index 후보 68개)

### kch_000498 hop1 (pattern d, cross, gold_lang en)
- sub_q: 로버트 슈워츠먼의 출생지는 어디인가?
- 검색 쿼리: Anne Hathaway, Julie Andrews의 출생지는 어디인가?
- gold: Los Angeles · R=1 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Anne Hathaway, Julie Andrews (gold bridge 아님)
- 생성: Sacramento, California
- claim 라벨: ['unsupported']
- 패시지 1: Hathaway was born Henri Léopold de Fiennes, in Sacramento, California, the son of actors Rhody and Jean Hathaway. The family's surname was originally "de Fienne…
- 패시지 2: Mary Doyle, politician James Ashton, politician Lou Bennett, musician Frank Block, politician Roy Cameron, pathologist Travis Fimmel, actor and model Henry Hopw…
- 패시지 3: Christie was born on 14 April 1940 at Singlijan Tea Estate, Chabua, Assam, British India, to Rosemary (née Ramsden), a Welsh-born painter, and Frank St John Chr…

### kch_000464 hop0 (pattern b, cross, gold_lang en)
- sub_q: 청두 셰페이롄의 본사는 어디에 있는가?
- 검색 쿼리: 청두 셰페이롄의 본사는 어디에 있는가?
- gold: Chengdu · R=1 · 결과=unsupported · 오답
- 생성: Mianyang Branch
- claim 라벨: ['unsupported']
- 패시지 1: Beijing Chengfeng Football Club (Chinese: 北京橙丰; pinyin: Běijīng Chéngfēng) was a professional Chinese football club that last participated in the Chinese League…
- 패시지 2: Chengdu Tiancheng F.C. (simplified Chinese: 成都天诚; traditional Chinese: 成都天誠; pinyin: Chéngdū Tiānchéng) was a Chinese professional football club based in Chengd…
- 패시지 3: Hebei Airlines (河北航空公司 Héběi Hángkōng gōngsī) is a Chinese airline which has its corporate headquarters in the Shijiazhuang World Trade Plaza Hotel (S: 石家庄世贸广场酒…

### kch_000438 hop0 (pattern b, cross, gold_lang en)
- sub_q: 로터스 에메야의 제조사는 어디인가?
- 검색 쿼리: 로터스 에메야의 제조사는 어디인가?
- gold: Lotus Cars · R=1 · 결과=unsupported · 오답
- 생성: 중국
- claim 라벨: ['unsupported']
- 패시지 1: The Lotus Emeya is a battery electric executive car manufactured by the Chinese-owned British sports car manufacturer Lotus Cars, with production beginning in 2…
- 패시지 2: EAMS – Embraer Aircraft Maintenance Services Inc. (Nashville, TN, U.S.) – maintenance services site. OGMA – Indústria Aeronáutica de Portugal (Alverca do Ribate…
- 패시지 3: On 29 August 2023, Lotus confirmed that the production version of the Type 133 will be called the Emeya. It was revealed officially on 7 September 2023 in New Y…

## R0-unsupported — 답 근거 없이 답함 (ungrounded answer) (cross-index 후보 213개)

### kch_000192 hop1 (pattern c, cross, gold_lang en)
- sub_q: 마다브 네팔이 졸업한 학교는 어디인가?
- 검색 쿼리: 마다브 쿠마르 네팔이 졸업한 학교는 어디인가?
- gold: Tribhuvan University · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 마다브 쿠마르 네팔 (gold bridge 아님)
- 생성: 모하메드 (문서에서 모하메드의 출생지와 관련된 내용을 찾았습니다.)
- claim 라벨: ['unsupported']
- 패시지 1: Mohamed was born in 1952 in Souali, situated in the northern Obock region of present-day Djibouti. He studied at the University of Limoges in France, where he e…
- 패시지 2: LaGuardia High School of Music & Art and Performing Arts. His acceptance into the school was a turning point in his appreciation for acting. His sophomore-year …
- 패시지 3: Since Indian independence in 1947, Patiala has emerged as a major education centre in the state of Punjab. The city houses the Thapar Institute of Engineering a…

### kch_000904 hop1 (pattern d, cross, gold_lang en)
- sub_q: 다카하타 이사오가 졸업한 학교는 어디인가?
- 검색 쿼리: Joseph Eggleston Johnston II가 졸업한 학교는 어디인가?
- gold: University of Tokyo · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Joseph Eggleston Johnston II (gold bridge 아님)
- 생성: Charterhouse School
- claim 라벨: ['unsupported']
- 패시지 1: Joseph Eggleston Johnston II (born May 13, 1950) is an American film director, producer, writer, and visual effects artist. He is best known for directing visua…
- 패시지 2: He was born in East Dereham in Norfolk, the son of the Francis Wollaston (1737–1815), a noted amateur astronomer, and his wife Althea Hyde. He was one of 17 chi…
- 패시지 3: George Edwin Bergstrom was born in Neenah, Wisconsin, of Norwegian immigrant ancestry. His father, George O. Bergstrom, was a prominent businessman and local po…

### kch_000125 hop0 (pattern b, cross, gold_lang en)
- sub_q: 전국자위대에 출연한 배우는 누구인가?
- 검색 쿼리: 전국자위대에 출연한 배우는 누구인가?
- gold: Sonny Chiba · R=0 · 결과=unsupported · 오답
- 생성: 답변: Meg Ryan
- claim 라벨: ['unsupported']
- 패시지 1: Santanu Bose Enzo Cilenti Alexander Gifford Paul Forman Trevor Cooper Misha Crosby Roma Downey Cory English Lisa Goldman Jack Greenlees Aiysha Hart Diana Hayden…
- 패시지 2: Andre Akpan, soccer player Larry D. Alexander, artist and writer Rodney Anderson, Republican member of the Texas House of Representatives from Grand Prairie (20…
- 패시지 3: Bob Akin (1936–2002), business executive, journalist, television commentator and champion sports car racing driver Fay Baker (1917–1987), actress and author Kat…

## hop1-wrong-bridge — 앞 홉 오답이 bridge 로 넘어감 (cascade) (후보 313개)

### kch_000913 hop1 (pattern d, cross, gold_lang en)
- sub_q: 더 콜렉티브의 본사는 어디인가?
- 검색 쿼리: MachineGames의 본사는 어디인가?
- gold: Newport Beach · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): MachineGames (gold bridge 아님)
- 생성: Uppsala, Sweden
- claim 라벨: ['unsupported']
- 패시지 1: MachineGames' founding team—consisting of Kjell Emanuelsson, Jerk Gustafsson, Magnus Högdahl, Jim Kjellin, Fredrik Ljungdahl, Jens Matthies, and Michael Wynne—w…
- 패시지 2: MachineGames Sweden AB is a Swedish video game developer based in Uppsala. The studio was founded in 2009 by seven former employees of Starbreeze Studios, inclu…
- 패시지 3: was later changed to MachineGames Sweden AB. After the acquisition, MachineGames resumed hiring. According to Gustafsson, around 70% of the company's employees …

### kch_000156 hop1 (pattern c, cross, gold_lang en)
- sub_q: FC 서울이 속한 리그는 무엇인가?
- 검색 쿼리: 부산 아이파크이 속한 리그는 무엇인가?
- gold: K League 1 · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 부산 아이파크 (gold bridge 아님)
- 생성: K League 2
- claim 라벨: ['unsupported']
- 패시지 1: Busan IPark FC (Korean: 부산 아이파크 FC) is a South Korean professional football club based in Busan that competes in K League 2, the second tier of the South Korean…
- 패시지 2: The League of Ireland is a national association football league consisting of professional clubs in the Republic of Ireland and Derry City F.C. in Northern Irel…
- 패시지 3: Bayern Munich II 3. Liga: 2019–20 Regionalliga Bayern: 2018–19 Bayern Munich Bundesliga: 2018–19 SC Freiburg DFB-Pokal runner-up: 2021–22 South Korea U14 Asian …

### kch_000967 hop1 (pattern d, cross, gold_lang en)
- sub_q: 리처드 해리스의 국적은 어디인가?
- 검색 쿼리: Daniel Radcliffe, Rupert Grint, Emma Watson의 국적은 어디인가?
- gold: Ireland · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Daniel Radcliffe, Rupert Grint, Emma Watson (gold bridge 아님)
- 생성: 영국
- claim 라벨: ['unsupported']
- 패시지 1: Daniel Jacob Radcliffe was born on 23 July 1989 at Queen Charlotte's and Chelsea Hospital in the Hammersmith district of London, the only child of casting agent…
- 패시지 2: Daniel Radcliffe as Harry Potter: An 11-year-old orphan living with his unwelcoming aunt, uncle, and cousin, who learns of his own fame as a wizard known to hav…
- 패시지 3: Daniel Jacob Radcliffe (born 23 July 1989) is an English actor best known for portraying the title character in all eight films of the Harry Potter film series …
