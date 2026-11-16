# aya — 정성 오류 분석 자료

홉 2000개 (제외: 없음). R = §4.1 answer-bearing hit@5.

## 유형별 비율 (index class 안에서 %)

| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |
|-|-|-|-|-|-|-|-|
| same | 1000 | 72.6 | 1.4 | 7.3 | 4.4 | 1.8 | 12.5 |
| cross | 1000 | 63.1 | 1.4 | 7.7 | 6.6 | 4.0 | 17.2 |

judge 결과와 정답(주 매처)이 어긋나는 홉:
- same: 지지 안 됨∧정답 38 / 지지 안 됨 198, 지지됨∧오답 93 / 지지됨 770
- cross: 지지 안 됨∧정답 44 / 지지 안 됨 249, 지지됨∧오답 122 / 지지됨 697

hop1 (앞 홉 비기권 968개) 중 bridge 가 gold 정본 라벨과 다른 홉 588개 (60.7%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)
그중 앞 홉 답이 오답인 hop1 229개 → 결과: supported 58, abstain 17, unsupported 154

## R1-abstain — 답 근거를 검색했는데 기권 (missed evidence) (cross-index 후보 14개)

### kch_000683 hop0 (pattern d, cross, gold_lang en)
- sub_q: 차두리가 소속된 팀은 어디인가?
- 검색 쿼리: 차두리가 소속된 팀은 어디인가?
- gold: Eintracht Frankfurt · R=1 · 결과=abstain · 오답
- 생성: 답변: 없음 (문서에서 차두리의 팀에 대한 정보가 제공되지 않음)
- claim 라벨: (claim 없음)
- 패시지 1: Cha Du-ri (Korean: 차두리; Korean pronunciation: [tɕʰa.du.ɾi] or [tɕʰa] [tu.ɾi]; born 25 July 1980) is a South Korean professional manager and former footballer. H…
- 패시지 2: Choi's first senior club in Australia was Parramatta FC, where he was the side's Player of the Year in 2013. On 27 July 2016, Choi scored a 70-metre goal from i…
- 패시지 3: In April 2014, it was announced that the former India cricket captain Sachin Tendulkar, along with PVP Ventures, won the bidding for the Kochi franchise in Indi…

### kch_000064 hop1 (pattern c, cross, gold_lang en)
- sub_q: 너팅 어소시에이츠의 본사는 어디인가?
- 검색 쿼리: 시지기 엔지니어링 (Nolann Bushnell과 Ted Dabney가 설립)의 본사는 어디인가?
- gold: Mountain View · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 시지기 엔지니어링 (Nolann Bushnell과 Ted Dabney가 설립) (gold bridge 아님)
- 생성: 답변: 모르겠습니다 (문서에 시지기 엔지니어링의 본사에 대한 정보가 제공되지 않았습니다.)
- claim 라벨: (claim 없음)
- 패시지 1: While studying at the University of Utah, electrical engineering student Nolan Bushnell had a part-time job at an amusement arcade, where he became familiar wit…
- 패시지 2: cash amount of US$350 each. They approached Ampex and a prior manager of Dabney's with their prototype, but found no interest. The major arcade game manufacture…
- 패시지 3: Founded and headquartered in Newcastle upon Tyne, United Kingdom, the company has grown organically, through acquisitions and, more recently, through subscripti…

### kch_000050 hop0 (pattern b, cross, gold_lang en)
- sub_q: 서브노티카의 개발사는 어디인가?
- 검색 쿼리: 서브노티카의 개발사는 어디인가?
- gold: Unknown Worlds Entertainment · R=1 · 결과=abstain · 오답
- 생성: 답변: Unknown Worlds Entertainment
- claim 라벨: (claim 없음)
- 패시지 1: Subnautica was announced by Unknown Worlds Entertainment on December 17, 2013, with Charlie Cleveland as the director and lead gameplay programmer, and Hugh Jer…
- 패시지 2: Gnosia's development first began in 2015 by four-person indie studio Petit Depotto, being their second video game released after Unholy Heights. Unholy Heights …
- 패시지 3: Subnautica is a 2018 action-adventure survival game developed and published by Unknown Worlds Entertainment. The player controls Ryley Robinson, a survivor of a…

## R1-unsupported — 답 근거를 검색했는데 지지되지 않는 답 (grounding failure) (cross-index 후보 77개)

### kch_000447 hop0 (pattern b, cross, gold_lang en)
- sub_q: 트랜센던스에 출연한 배우는 누구인가?
- 검색 쿼리: 트랜센던스에 출연한 배우는 누구인가?
- gold: Johnny Depp · R=1 · 결과=unsupported · 정답
- 생성: 답변: Johnny Depp, Morgan Freeman, Rebecca Hall, Paul Bettany, Kate Mara, Cillian Murphy, Cole Hauser, Laura Jane Haddock
- claim 라벨: ['contradicted']
- 패시지 1: Transcendence is a 2014 American science fiction thriller film directed by Wally Pfister (in his directorial debut) and written by Jack Paglen. The film stars J…
- 패시지 2: Joseph Timothy Thomerson (born April 8, 1946) is an American actor and comedian. He is best known for his role as Jack Deth in the Trancers film series, his wor…
- 패시지 3: Joshua David Duhamel ( də-MEL; born November 14, 1972) is an American actor. After working as a model, he made his acting debut as Leo du Pres on the ABC daytim…

### kch_000379 hop0 (pattern d, cross, gold_lang en)
- sub_q: 네덜란드 왕궁의 본사는 어디에 있는가?
- 검색 쿼리: 네덜란드 왕궁의 본사는 어디에 있는가?
- gold: Amsterdam · R=1 · 결과=unsupported · 정답
- 생성: 답변: 암스테르담 (Koninklijk Paleis van Amsterdam 또는 Paleis op de Dam)
- claim 라벨: ['contradicted']
- 패시지 1: Huis ten Bosch (Dutch: Paleis Huis ten Bosch, pronounced [paːˈlɛis ˌɦœys tɛm ˈbɔs]; English: "House in the Woods") is a royal palace in The Hague, Netherlands. …
- 패시지 2: The Palace of Laeken or Castle of Laeken (French: Château de Laeken; Dutch: Kasteel van Laken; German: Schloss zu Laeken) is the official residence of the King …
- 패시지 3: The Hague (Dutch: Den Haag, short for 's-Gravenhage) is the capital city of the South Holland province of the Netherlands. With a population of over half a mill…

### kch_000360 hop0 (pattern b, cross, gold_lang en)
- sub_q: 메이저 리그를 감독한 사람은 누구인가?
- 검색 쿼리: 메이저 리그를 감독한 사람은 누구인가?
- gold: David S. Ward · R=1 · 결과=unsupported · 정답
- 생성: 답변: David S. Ward
- claim 라벨: ['contradicted']
- 패시지 1: Major League is a 1989 American sports comedy film produced by Chris Chesser and Irby Smith, written and directed by David S. Ward, that stars Tom Berenger, Cha…
- 패시지 2: Major League Baseball (MLB) is a professional baseball league in North America composed of 30 teams, divided equally between the National League (NL) and the Am…
- 패시지 3: Major League Soccer is the most recent of a series of men's premier professional national soccer leagues established in the United States and Canada. The predec…

## R0-unsupported — 답 근거 없이 답함 (ungrounded answer) (cross-index 후보 172개)

### kch_000254 hop0 (pattern b, cross, gold_lang en)
- sub_q: 서브스턴스를 감독한 사람은 누구인가?
- 검색 쿼리: 서브스턴스를 감독한 사람은 누구인가?
- gold: Coralie Fargeat · R=0 · 결과=unsupported · 오답
- 생성: 답변: Alex Somers
- claim 라벨: ['unsupported']
- 패시지 1: The series is directed by Kim Jung-sik, who previously worked on several hit Korean sitcoms like High Kick! (2006–07), High Kick: Revenge of the Short Legged (2…
- 패시지 2: Severance is an American science fiction psychological thriller television series created by Dan Erickson, and executive produced and primarily directed by Ben …
- 패시지 3: Collateral is a 2004 American neo-noir action thriller film directed and produced by Michael Mann, written by Stuart Beattie, and starring Tom Cruise and Jamie …

### kch_000185 hop1 (pattern d, cross, gold_lang en)
- sub_q: 휴 잭맨의 출생지는 어디인가?
- 검색 쿼리: Ryan Reynolds, Hugh Jackman의 출생지는 어디인가?
- gold: Sydney · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Ryan Reynolds, Hugh Jackman (gold bridge 아님)
- 생성: 답변: Vancouver, British Columbia
- claim 라벨: ['unsupported']
- 패시지 1: Ryan Rodney Reynolds was born in Vancouver, British Columbia, on October 23, 1976. His father, James Chester Reynolds, served as a Royal Canadian Mounted Police…
- 패시지 2: Ryan Rodney Reynolds (born October 23, 1976) is a Canadian and American actor, entrepreneur, and film producer. Known for starring in comedic and superhero film…
- 패시지 3: 1838–1909) lived in Lewisham Delroy Lindo (actor) born in Lewisham Marie Lloyd (entertainer) lived in Lewisham and New Cross Ruben Loftus-Cheek (footballer) Ale…

### kch_000821 hop1 (pattern c, cross, gold_lang en)
- sub_q: 월터 힐이 졸업한 학교는 어디인가?
- 검색 쿼리: 월터 힐이 졸업한 학교는 어디인가?
- gold: Michigan State University · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 월터 힐 (gold bridge 맞음)
- 생성: 답변: Yale University
- claim 라벨: ['contradicted']
- 패시지 1: Phillips Exeter Academy (also known as Exeter or PEA) is an independent, co-educational, college-preparatory school in Exeter, New Hampshire. Established in 178…
- 패시지 2: The Wharton School ( WHOR-tən) is the business school of the University of Pennsylvania, a private research university in Philadelphia. Established in 1881 thro…
- 패시지 3: Wilder began writing plays while at the Thacher School in Ojai, California, where he did not fit in and was teased by classmates as overly intellectual. Accordi…

## hop1-wrong-bridge — 앞 홉 오답이 bridge 로 넘어감 (cascade) (후보 229개)

### kch_000102 hop1 (pattern c, cross, gold_lang en)
- sub_q: 기초과학연구원이 속한 행정구역은 어디인가?
- 검색 쿼리: 포항공과대학교 물리학과이 속한 행정구역은 어디인가?
- gold: Daejeon · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 포항공과대학교 물리학과 (gold bridge 아님)
- 생성: 답변: Pohang, South Korea
- claim 라벨: ['unsupported']
- 패시지 1: Pohang University of Science and Technology (POSTECH, Korean: 포항공과대학교) is a private research university in Pohang, South Korea.
- 패시지 2: The China National Space Administration has its headquarters in the district. The Beijing Aerospace Flight Control Center which serves as the central command ce…
- 패시지 3: The University of Portland currently host two detachments of the Reserve Officers' Training Corps: the Air Force Reserve Officer's Training Corps and the Army R…

### kch_000672 hop1 (pattern d, cross, gold_lang en)
- sub_q: 올리버 오티스 하워드가 졸업한 학교는 어디인가?
- 검색 쿼리: 존 하트포드 (John Harvard)가 졸업한 학교는 어디인가?
- gold: Bowdoin College · R=0 · 결과=supported · 오답
- 앞 홉 답(bridge): 존 하트포드 (John Harvard) (gold bridge 아님)
- 생성: 답변: 하버드 대학 (Harvard College)
- claim 라벨: ['supported']
- 패시지 1: Harvard College was founded in 1636 by vote of the Great and General Court of the Massachusetts Bay Colony. Two years later, the college became home to North Am…
- 패시지 2: Harvard University is a private Ivy League research university in Cambridge, Massachusetts, United States. Founded in 1636, and named Harvard College in 1639 in…
- 패시지 3: The John F. Kennedy School of Government, commonly called the Harvard Kennedy School (HKS), is the graduate school of public policy of Harvard University, a pri…

### kch_000504 hop1 (pattern c, cross, gold_lang en)
- sub_q: 청두시가 속한 국가는 어디인가?
- 검색 쿼리: 홍콩가 속한 국가는 어디인가?
- gold: People's Republic of China · R=1 · 결과=supported · 정답
- 앞 홉 답(bridge): 홍콩 (gold bridge 아님)
- 생성: 중국
- claim 라벨: ['supported']
- 패시지 1: Hong Kong is a special administrative region of China. Situated on China's southern coast just south of Shenzhen, it consists of Hong Kong Island, Kowloon, and …
- 패시지 2: Along with Singapore, South Korea, and Taiwan, Hong Kong is one of the Four Asian Tigers. One of the world's most significant financial centres and commercial p…
- 패시지 3: Hong Kong was a Crown colony of the United Kingdom from 1841 to 1981, and a dependent territory from 1981 to 1997. However, from 1941 to 1945, it was under a pe…
