# qwen — 정성 오류 분석 자료

홉 2000개 (제외: 없음). R = §4.1 answer-bearing hit@5.

## 유형별 비율 (index class 안에서 %)

| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |
|-|-|-|-|-|-|-|-|
| same | 1000 | 70.9 | 0.9 | 7.2 | 4.3 | 6.2 | 10.5 |
| cross | 1000 | 59.7 | 2.4 | 6.8 | 4.5 | 11.0 | 15.6 |

judge 결과와 정답(주 매처)이 어긋나는 홉:
- same: 지지 안 됨∧정답 19 / 지지 안 됨 177, 지지됨∧오답 158 / 지지됨 752
- cross: 지지 안 됨∧정답 24 / 지지 안 됨 224, 지지됨∧오답 224 / 지지됨 642

hop1 (앞 홉 비기권 926개) 중 bridge 가 gold 정본 라벨과 다른 홉 537개 (58.0%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)
그중 앞 홉 답이 오답인 hop1 312개 → 결과: supported 75, abstain 48, unsupported 189

## R1-abstain — 답 근거를 검색했는데 기권 (missed evidence) (cross-index 후보 24개)

### kch_000895 hop1 (pattern d, cross, gold_lang en)
- sub_q: 덴절 워싱턴의 출생지는 어디인가?
- 검색 쿼리: Bob Hoskins, Denzel Washington, Chloe Webb의 출생지는 어디인가?
- gold: Mount Vernon · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): Bob Hoskins, Denzel Washington, Chloe Webb (gold bridge 아님)
- 생성: 답변: Chloe Webb는 뉴욕시 맨해튼의 그린우치 마을, Denzel Washington은 뉴욕시 인근 마운트버논, Bob Hoskins은 문서에 언급되지 않음.
- claim 라벨: (claim 없음)
- 패시지 1: Webb was born June 25, 1956, in the Greenwich Village neighborhood of Manhattan in New York City. She spent her youth in various locations on the East Coast, de…
- 패시지 2: Suffrage, temperance, and peace activist Hannah Johnston Bailey was born in Cornwall in 1839. Olympic speed skater Bonnie Blair was born in Cornwall on March 18…
- 패시지 3: Washington Redskins. Malcolm Koonce, NFL Defensive End for the Las Vegas Raiders Sean Murphy, MLB catcher for the Atlanta Braves George Pataki, former New York …

### kch_000066 hop1 (pattern d, cross, gold_lang en)
- sub_q: 뉴캐슬 유나이티드 제츠 FC가 속한 리그는 무엇인가?
- 검색 쿼리: 제주 유나이티드 FC가 속한 리그는 무엇인가?
- gold: A-League Men · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): 제주 유나이티드 FC (gold bridge 아님)
- 생성: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: JEF United Chiba (ジェフユナイテッド千葉, Jefu Yunaiteddo Chiba), full name JEF United Ichihara Chiba (ジェフユナイテッド市原・千葉, Jefu Yunaiteddo Ichihara Chiba) and also known as JE…
- 패시지 2: The J1 League (Japanese: J1リーグ, Hepburn: Jē-wan Rīgu), a.k.a. the J.League or the Meiji Yasuda J1 League (Japanese: 明治安田J1リーグ, Hepburn: Meiji Yasuda Jē-wan Rīgu…
- 패시지 3: The Hrvatska nogometna liga (pronounced [xř̩ʋaːtskaː nôɡomeːtnaː lǐːɡa]; lit. 'Croatian Football League'), abbreviated as HNL and also known for sponsorship rea…

### kch_000003 hop1 (pattern d, cross, gold_lang en)
- sub_q: 조니 뎁의 출생지는 어디인가?
- 검색 쿼리: Johnny Depp, Morgan Freeman, Rebecca Hall, Paul Bettany, Kate Mara, Cillian Murphy, Cole Hauser의 출생지는 어디인가?
- gold: Owensboro · R=1 · 결과=abstain · 오답
- 앞 홉 답(bridge): Johnny Depp, Morgan Freeman, Rebecca Hall, Paul Bettany, Kate Mara, Cillian Murphy, Cole Hauser (gold bridge 아님)
- 생성: 답변: 모르겠습니다
- claim 라벨: (claim 없음)
- 패시지 1: Depp was born on June 9, 1963, in Owensboro, Kentucky, the youngest of four children of waitress Betty Sue Depp (née Wells; later Palmer) and civil engineer Joh…
- 패시지 2: Lily-Rose Melody Depp was born in the American Hospital of Paris located in Neuilly-sur-Seine, a wealthy suburb of Paris on May 27, 1999. She is the daughter of…
- 패시지 3: Marcheline Bertrand, actress, mother of Angelina Jolie, born in Blue Island Gary Bettenhausen, 1980 & 1983 USAC Silver Crown Series Champion, was born in Blue I…

## R1-unsupported — 답 근거를 검색했는데 지지되지 않는 답 (grounding failure) (cross-index 후보 68개)

### kch_000564 hop0 (pattern b, cross, gold_lang en)
- sub_q: 고로의 산보를 감독한 사람은 누구인가?
- 검색 쿼리: 고로의 산보를 감독한 사람은 누구인가?
- gold: Hayao Miyazaki · R=1 · 결과=unsupported · 오답
- 생성: 답변: 고로 타니구치
- claim 라벨: ['unsupported']
- 패시지 1: Yasuichiro Yamamoto (山本 泰一郎, Yamamoto Yasuichiro) is a Japanese anime director, and storyboard artist. He is best known for directing the majority of Detective …
- 패시지 2: Isao Takahata (高畑 勲, Takahata Isao; October 29, 1935 – April 5, 2018) was a Japanese director, screenwriter and producer. A co-founder of Studio Ghibli, he earn…
- 패시지 3: Kōsei Saitō (斎藤 光正, Saitō Kōsei; 17 July 1932 – 25 November 2012) was a Japanese film director. He is known for directing the film G.I. Samurai .

### kch_000516 hop0 (pattern d, cross, gold_lang en)
- sub_q: 메이저 리그 2를 감독한 사람은 누구인가?
- 검색 쿼리: 메이저 리그 2를 감독한 사람은 누구인가?
- gold: David S. Ward · R=1 · 결과=unsupported · 정답
- 생성: 답변: David S. Ward
- claim 라벨: ['unsupported']
- 패시지 1: Major League II is a 1994 American sports comedy film and sequel to the 1989 film Major League and it is the second installment in the Major League film series.…
- 패시지 2: Major League is a 1989 American sports comedy film produced by Chris Chesser and Irby Smith, written and directed by David S. Ward, that stars Tom Berenger, Cha…
- 패시지 3: appeared in Super Bowl II Rick Carlisle 1979, former NBA player, former coach of the 2011 NBA champion Dallas Mavericks, current coach of the Indiana Pacers Mo …

### kch_000500 hop1 (pattern c, cross, gold_lang en)
- sub_q: 예일 대학교가 속한 행정구역은 어디인가?
- 검색 쿼리: 예일 대학교가 속한 행정구역은 어디인가?
- gold: New Haven · R=1 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 예일 대학교 (gold bridge 맞음)
- 생성: 연방주의 지역
- claim 라벨: ['unsupported']
- 패시지 1: IE University, known in Spanish as IE Universidad or Universidad Instituto de Empresa (lit. University Institute of Business), is a private university with camp…
- 패시지 2: Zaria (formerly Zazzau) is a metropolitan city in Nigeria, located within four local government areas in Kaduna State. It serves as the capital of the Zazzau Em…
- 패시지 3: The Yale School of Management (also known as Yale SOM) is the graduate business school of Yale University, a private Ivy League research university in New Haven…

## R0-unsupported — 답 근거 없이 답함 (ungrounded answer) (cross-index 후보 156개)

### kch_000339 hop1 (pattern c, cross, gold_lang en)
- sub_q: TSMC의 설립자는 누구인가?
- 검색 쿼리: 애플의 설립자는 누구인가?
- gold: Morris Chang · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 애플 (gold bridge 아님)
- 생성: 답변: 스티브 잡스, 스티브 워즈니악, 론 웨이너
- claim 라벨: ['unsupported']
- 패시지 1: Apple Inc. is an American multinational technology company headquartered in Cupertino, California, in Silicon Valley, and known for consumer electronics, softwa…
- 패시지 2: Steven Paul Jobs (February 24, 1955 – October 5, 2011) was an American businessman, inventor, and investor. A pioneer of the personal computer revolution of the…
- 패시지 3: Stephen Gary Wozniak (; born August 11, 1950), also known by his nickname Woz, is an American technology entrepreneur, electrical engineer, computer programmer,…

### kch_000242 hop1 (pattern c, cross, gold_lang en)
- sub_q: 미스트워커의 설립자는 누구인가?
- 검색 쿼리: 미스트워커와 아تون의 설립자는 누구인가?
- gold: Hironobu Sakaguchi · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 미스트워커와 아تون (gold bridge 아님)
- 생성: 답변: Brian Acton, Thomas Preston-Werner
- claim 라벨: ['unsupported']
- 패시지 1: Brian Acton (born 1972) is an American business executive and computer programmer serving as the executive chairperson of Signal Technology Foundation, which he…
- 패시지 2: Thomas Preston-Werner (born May 27, 1979) is an American software developer and billionaire entrepreneur. He is an active contributor within the free and open-s…
- 패시지 3: David Crane (born 1954) is an American video game designer and programmer. He is the co-founder of Activision, which he formed with former Atari co-workers in 1…

### kch_000914 hop1 (pattern c, cross, gold_lang en)
- sub_q: 돈 블루스가 졸업한 학교는 어디인가?
- 검색 쿼리: 돈 블루스가 졸업한 학교는 어디인가?
- gold: Brigham Young University · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 돈 블루스 (gold bridge 맞음)
- 생성: 답변: Brown University
- claim 라벨: ['unsupported']
- 패시지 1: worked at McDonald's as a short-order line cook during the breakfast shift. Bezos attended the Student Science Training Program at the University of Florida. He…
- 패시지 2: attended but did not graduate from Dartmouth; he is, however, the only person to have received two honorary degrees from Dartmouth. Other authors and media pers…
- 패시지 3: Douglas was born in Brooklyn, New York, the second son of an assistant cashier at the National Park Bank. He attended Trinity Chapel School and was of Scottish …

## hop1-wrong-bridge — 앞 홉 오답이 bridge 로 넘어감 (cascade) (후보 312개)

### kch_000163 hop1 (pattern a, same, gold_lang ko)
- sub_q: Mojang Studios의 본사는 어디인가?
- 검색 쿼리: 마인크래프트 레전드의 개발사는 블리자드 엔터테인먼트입니다.의 본사는 어디인가?
- gold: 스톡홀름 · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): 마인크래프트 레전드의 개발사는 블리자드 엔터테인먼트입니다. (gold bridge 아님)
- 생성: 캘리포니아 어바인
- claim 라벨: ['unsupported']
- 패시지 1: 블리자드 엔터테인먼트(영어: Blizzard Entertainment, Inc.)는 미국의 비디오 게임 개발사이다. 1991년 2월 8일 최초 설립 시에는 실리콘 & 시냅스 (Silicon & Synapse)라는 이름으로, 마이클 모하임과 앨런 애드햄, 프랭크 피어스 3인의 UCLA 출…
- 패시지 2: 블리자드 엔터테인먼트는 마이클 모하임, 앨런 애드햄, 프랭크 피어스가 UCLA를 학사 졸업하고 1년 후인 1991년 2월 8일 실리콘 & 시냅스라는 이름으로 설립되었다. 설립 초기에는 다른 게임 제작 스튜디오를 위한 게임 포트 제작에 주력했다. 당시 개발된 포트가 사용된 게임은 J.R.…
- 패시지 3: 블리자드 코리아는 블리자드 계열의 회사로서 주로 블리자드 사의 제품의 대한민국에서의 유통을 담당하는 회사이다. 2005년 대한민국에서 월드 오브 워크래프트의 정식 서비스를 앞두고 블리자드 엔터테인먼트가 직접 관리를 하기 위해 2004년 11월 2004년 11월에 블리자드가 최초로 설립한…

### kch_000967 hop1 (pattern d, cross, gold_lang en)
- sub_q: 리처드 해리스의 국적은 어디인가?
- 검색 쿼리: Daniel Radcliffe, Rupert Grint, Emma Watson의 국적은 어디인가?
- gold: Ireland · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): Daniel Radcliffe, Rupert Grint, Emma Watson (gold bridge 아님)
- 생성: 답변: 영국
- claim 라벨: ['unsupported']
- 패시지 1: Daniel Jacob Radcliffe was born on 23 July 1989 at Queen Charlotte's and Chelsea Hospital in the Hammersmith district of London, the only child of casting agent…
- 패시지 2: Daniel Radcliffe as Harry Potter: An 11-year-old orphan living with his unwelcoming aunt, uncle, and cousin, who learns of his own fame as a wizard known to hav…
- 패시지 3: Daniel Jacob Radcliffe (born 23 July 1989) is an English actor best known for portraying the title character in all eight films of the Harry Potter film series …

### kch_000716 hop1 (pattern d, cross, gold_lang en)
- sub_q: 유르고르덴 IF가 속한 리그는 무엇인가?
- 검색 쿼리: FC 서울가 속한 리그는 무엇인가?
- gold: Allsvenskan · R=0 · 결과=unsupported · 오답
- 앞 홉 답(bridge): FC 서울 (gold bridge 아님)
- 생성: K League 1
- claim 라벨: ['unsupported']
- 패시지 1: FC Seoul (Korean: FC 서울) is a South Korean professional football club based in Seoul that competes in the K League 1, the top flight of South Korean football. T…
- 패시지 2: Source: League Second Division / First Division / Championship (level 2) Champions: 1948–49, 2000–01, 2021–22 Runners-up: 1958–59 Play-off winners: 2018, 2020 T…
- 패시지 3: Fotbal Club FCSB (Romanian pronunciation: [fet͡ʃeseˌbe]), sometimes still referred to as the former name FC Steaua București, is a Romanian professional footbal…
