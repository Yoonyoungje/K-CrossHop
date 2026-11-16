# K-CrossHop 어노테이션 가이드라인 v2.0

> 작성일: 2026년 6월 (2026-07-08 정본 6모델 pooling 샘플로 갱신, sub_q 표시 오류 수정 반영)  
> 목적: Claim 충실성(faithfulness) 인간 판정 — IAA(Inter-Annotator Agreement) 측정용  
> 대상: Annotator 3인 (독립 판정)  
> 도구: 어노테이션 웹앱 (`iaa_annotation_app.html` — 브라우저에서 파일 열기만 하면 실행됨)  
> v1.0 대비 변경: 샘플 규모 조정, 라벨 분포를 정본 기준으로 업데이트, Fleiss' κ 추가  
> 갱신: 샘플을 8B 4종(aya/exaone/llama/qwen) + frontier 2종(sonnet/gemini) pooling으로 재추출 (`audit/sample_iaa.py`). 판정 정의·절차는 변경 없음, `model` 필드 추가.  
> v2.1 갱신: Google Sheets 대신 웹앱으로 전환. hop1 검색 질의가 이전 hop 오류로 원래 질문과 달라지는 경우(§2 참고) 정본 질문을 보여주도록 수정.

---

## 1. 과제 개요

이 과제는 AI가 생성한 텍스트의 각 **claim**이 검색된 문서(passage)에 의해 **근거가 있는지** 판정하는 작업입니다.

### 배경 (간단히)

저희가 만드는 시스템은 질문에 답하기 위해 문서를 검색하고 → 그 문서를 읽고 → 답변을 생성합니다. 이 과정에서 AI가 **검색된 문서에 없는 내용을 만들어내는 현상**을 "환각(hallucination)"이라고 합니다.

저희는 AI 답변의 각 claim이 환각인지 아닌지를 판정할 것입니다.

---

## 2. 용어 설명

이 작업이 처음이신 분들을 위한 배경 설명입니다. 이미 익숙하시면 바로 3장으로 넘어가셔도 됩니다.

### 이 연구가 하는 일 (한 문장 요약)

AI에게 두 단계로 나뉜 질문을 주고, AI가 문서를 검색해서 답을 찾아가는 과정을 관찰합니다. 예를 들어 "봉준호 감독의 영화를 제작한 국가는 어디인가?"를 풀려면 ① 먼저 "봉준호 감독의 영화는?"을 검색해 "기생충"이라는 답을 찾고, ② 그 답("기생충")을 가지고 다시 "기생충을 제작한 국가는?"을 검색해 최종 답을 찾아야 합니다. 이렇게 **질문을 풀기 위해 여러 단계로 나눠 검색하는 것**을 **hop(홉)**이라 부릅니다.

### Hop (홉)
검색 한 단계를 가리키는 말입니다. 위 예시에서 ①번이 hop 0, ②번이 hop 1입니다. 각 hop마다 AI는 (1) 검색 질문(`sub_q`)을 만들고 → (2) 문서를 검색해 몇 개의 문서 조각(`passage`)을 가져오고 → (3) 그 문서를 읽고 답변(`generated_text`)을 생성합니다.

여러분의 판정은 **hop 단위**로 이루어집니다 — hop 0에서 나온 claim은 hop 0의 passage만 보고, hop 1에서 나온 claim은 hop 1의 passage만 보고 판정합니다. 다른 hop의 내용을 끌어와 판정하지 마세요.

### Passage (패시지, 문서 청크)
AI가 검색으로 가져온 **문서 조각 하나**입니다. 위키피디아 같은 긴 문서를 통째로 주지 않고, 관련성 높은 몇 문단만 잘라서(chunk) 줍니다. 한 hop마다 최대 5개가 제공되며, 이것이 여러분이 판정할 때 볼 수 있는 **유일한 증거**입니다.

### Claim (클레임, 판정 대상)
AI가 생성한 답변(`generated_text`) 안에서 뽑아낸 **핵심 주장 하나**입니다. 보통 짧은 개체명이나 구절입니다.

**예시:**
> `박원순`  
> `전라남도 순천시`  
> `Javier Bardem`

하나의 답변에 여러 claim이 있을 수 있고, 각 claim을 따로따로 판정합니다.

> ⚠️ **중요:** 여러분이 알고 계신 배경지식은 판정에 사용하지 마세요. Passage에 없으면 **unsupported**입니다.

### sub_q (검색 질문) / generated_text (생성된 답변)
`sub_q`는 그 hop이 원래 답해야 하는 질문입니다. 화면에는 이 질문을 기준으로 판정하시면 됩니다. `generated_text`는 AI가 그 hop에서 실제로 출력한 답변 전문이며, 여기서 claim이 추출됩니다.

> ⚠️ **hop 1의 검색어가 실제로는 다를 수 있습니다.** hop 1은 hop 0의 답을 이용해 검색어를 새로 만드는데, hop 0가 틀린 답(환각)을 냈다면 그 틀린 내용이 검색어에 그대로 들어갑니다. 예를 들어 원래 질문이 "무하마드 빈 살만의 국적은?"인데 hop 0가 엉뚱하게 "캘빈 클라인"이라고 답했다면, 실제 검색은 "캘빈 클라인의 국적은?"으로 이루어져 캘빈 클라인 관련 passage가 나옵니다. 이런 경우 화면에 "⚠️ 실제 검색에는 다른 질의가 쓰였습니다"라는 안내가 함께 뜹니다 — **그래도 판정 기준은 항상 화면에 보이는 sub_q(정본 질문)와 claim입니다.** 이런 hop이 전체 hop 1의 상당수(표본 기준 약 70~90%)를 차지하니 당황하지 마세요 — 이 자체가 저희가 연구하는 현상(한 단계의 오류가 다음 단계로 전파되는 것) 중 하나입니다.

### 언어 패턴 (a / b / c / d)
질문마다 hop 0과 hop 1이 각각 한국어(KO) 문서를 찾아야 하는지 영어(EN) 문서를 찾아야 하는지가 미리 정해져 있습니다. 이 조합이 4가지 패턴입니다:

| 패턴 | hop0 | hop1 |
|-|-|-|
| a | 한국어 문서 | 한국어 문서 |
| b | 영어 문서 | 한국어 문서 |
| c | 한국어 문서 | 영어 문서 |
| d | 영어 문서 | 영어 문서 |

여러분이 직접 이 패턴을 신경 쓸 필요는 없습니다 — 화면에 왜 a/b/c/d가 표시되는지 참고용으로만 알아두세요.

### model / qid
`model`은 그 답변을 생성한 AI 이름, `qid`는 질문을 구분하는 고유 번호입니다. 판정에는 영향을 주지 않는 식별용 정보입니다.

---

## 3. 판정 단위 — 화면에 표시되는 항목

각 케이스 화면에 아래 항목들이 순서대로 나옵니다.

| 항목 | 설명 |
|---|---|
| model | 답변을 생성한 모델 (aya/exaone/llama/qwen/sonnet/gemini) — 판정과 무관, 식별용 |
| qid | 질문 ID — 판정과 무관, 식별용 |
| pattern / hop / index | 언어 패턴(a/b/c/d), 홉 번호(0/1), 이 홉의 검색 언어(KO/EN) — 참고용 |
| sub_q | 이 홉이 답해야 하는 **정본 질문**. 이 질문 기준으로 판정하세요 |
| (실제 검색 질의 안내) | sub_q와 실제 검색어가 다를 때만 노란 박스로 표시 (§2 참고) |
| generated_text | AI가 생성한 전체 답변 |
| claim | **판정 대상** claim (파란 박스로 강조 표시) |
| passages | AI가 검색한 문서 최대 5개. claim과 일치하는 부분이 있으면 자동으로 강조(하이라이트) 표시됩니다 |
| 라벨 버튼 | 클릭해서 판정 입력 (단축키 `s`/`u`/`c`) |
| 이유 입력란 | 한 줄 이유 작성 |

---

## 4. 라벨 정의

### ✅ supported (지지됨)
**"Passage가 이 claim을 명확하게 뒷받침한다"**

- Passage에 claim의 핵심 내용이 직접 서술되어 있을 때
- 표현이 다르거나 언어가 달라도 의미가 동일하면 supported

**예시:**
```
claim  : 박원순
passage: "박원순 전 서울시장은..."
판정   : ✅ supported
```

```
claim  : Javier Bardem
passage (EN): "Javier Bardem is a Spanish actor..."
판정   : ✅ supported  (언어가 달라도 동일 인물 확인)
```

---

### ❓ unsupported (근거 없음)
**"Passage에 이 claim을 확인하거나 부정할 정보가 없다"**

- Claim의 내용이 passage 어디에도 나오지 않을 때
- Passage가 관련 주제를 다루지만 해당 세부 정보는 없을 때

**예시:**
```
claim  : 전라남도 순천시
passage: "곽재구는 시인이며 순천대학교에서 근무했다." (위치 정보 없음)
판정   : ❓ unsupported
```

---

### ❌ contradicted (모순됨)
**"Passage가 이 claim을 직접 부정하거나 다른 내용을 명시한다"**

- Passage에 claim과 충돌하는 사실이 명시될 때
- 수치, 이름, 날짜, 장소 등이 passage와 다를 때

**예시:**
```
claim  : Kat Coiro
passage: "Marley & Me was directed by David Frankel"
판정   : ❌ contradicted  (다른 감독이 명시됨)
```

> ⚠️ **unsupported vs contradicted 구분:**  
> - Passage가 해당 정보를 **아예 다루지 않으면** → unsupported  
> - Passage가 해당 주제를 **명시적으로 서술하는데 claim과 다르면** → contradicted  
> 애매하시면 **unsupported**로 판정해 주세요.

---

## 5. 판정 절차

**Step 1.** `sub_q`(검색 질문)와 `hop` 번호를 확인 — 지금 어떤 정보를 찾는 단계인지 파악  
**Step 2.** `claim`을 읽어 주세요 — 무슨 사실을 주장하는지 파악  
**Step 3.** `passages`를 차례로 읽어 주세요 — claim 관련 내용이 있는지 확인  
**Step 4.** 아래 트리를 따라 라벨 결정:

```
Passage에 claim 관련 내용이 있나요?
    │
    ├─ 없음 → unsupported
    │
    └─ 있음 → Claim과 일치하나요?
                  │
                  ├─ 일치함 → supported
                  │
                  └─ 불일치 → 불일치가 명확한가요?
                                  │
                                  ├─ 명확함 → contradicted
                                  │
                                  └─ 애매함 → unsupported
```

**Step 5.** 라벨 버튼(`supported` / `unsupported` / `contradicted`) 클릭 — 단축키 `s` / `u` / `c`도 사용 가능  
**Step 6.** 이유 입력란에 한 줄 이유를 작성 (예: "Passage 2에서 대전 위치 명시") — 라벨과 이유를 모두 입력해야 다음 케이스로 넘어갈 수 있습니다

---

## 6. 자주 헷갈리는 케이스

### Case 1 — Claim이 짧은 개체명인 경우
```
claim   : 박원순
passages: 박원순 전 서울시장이 설립한 기관에 관한 내용
판정    : ✅ supported  (해당 인물이 언급되고 문맥이 일치하면)
```

### Case 2 — 언어가 다른 경우 (KO claim / EN passage)
```
claim  (KO): 바르셀로나
passage (EN): "Javier Bardem was born in Las Palmas, Gran Canaria, Spain..."
판정        : ❌ contradicted  (passage는 라스팔마스 출생을 명시)
```
언어가 달라도 의미로 비교하세요.

### Case 3 — Passage가 더 많은 정보를 담고 있는 경우
```
claim  : 고려대학교
passage: "홍창수는 고려대학교, 서울대학교에서 근무했다."
판정   : ✅ supported  (claim이 passage 정보의 일부면 충분)
```

### Case 4 — AI가 추론·요약한 문장
```
claim  : 대전광역시  (추론 결론)
passage: "KAIST는 대전에 위치한다."
판정   : ✅ supported  (passage로부터 직접 도출 가능)
```
추론 과정에서 passage에 없는 전제가 추가됐으면 → unsupported

### Case 5 — 수식어/불확실 표현 포함
```
claim  : 하지만 정확히 언급된 것은 아니고
```
이런 AI의 불확실 표현은 판정 대상 claim이 아닙니다. 핵심 개체/사실만 판정하세요.

---

### 실제 데이터 예시 (calibration sample에서 발췌)

아래는 실제 본 과제의 calibration 세트에서 그대로 가져온 예시입니다. 형식과 난이도를 미리 감 잡는 용도입니다.

**예시 1 — supported (같은 언어, KO→KO)**
```
sub_q  : 페라리 360의 제조사는 어디인가?
claim  : 페라리
passage: "페라리 360(영어: Ferrari 360)은 페라리에서 1999년부터 2004년까지 생산했던 스포츠카이다."
판정   : ✅ supported  (passage가 claim을 직접 서술)
```

**예시 2 — supported (교차 언어, KO 질문→EN passage)**
```
sub_q  : 베니스 유령 살인사건을 감독한 사람은 누구인가?
claim  : Kenneth Branagh
passage: "A Haunting in Venice is a 2023 British mystery ... produced and directed by Kenneth Branagh ..."
판정   : ✅ supported  (언어가 달라도 동일 사실 확인 가능)
```

**예시 3 — unsupported (이전 hop 오류로 검색이 엉뚱한 대상으로 빗나간 경우)**
```
sub_q            : 무하마드 빈 살만 알 사우드의 국적은 어디인가?   ← 판정 기준은 이 질문
⚠️ 실제 검색 질의 : 캘빈 클라인의 국적은 어디인가?                ← hop 0가 "캘빈 클라인"이라고
                                                                    잘못 답해서 이렇게 바뀜
claim            : 미국
passage          : "Calvin Richard Klein (born November 19, 1942) is an American fashion designer ..."
판정             : ❓ unsupported  (passage가 Calvin Klein 얘기만 하고, sub_q가 묻는 인물(무하마드 빈
                    살만)에 대한 정보는 전혀 없음)
```
> 화면에 "⚠️ 실제 검색에는 다른 질의가 쓰였습니다"가 뜨면 이런 경우입니다. sub_q(정본 질문)만 보고 판정하세요 — **passage에 sub_q·claim 관련 정보가 없으면 무조건 unsupported**입니다. 왜 이 passage가 검색됐는지는 안내 문구로 설명돼 있으니 신경 쓰지 않으셔도 됩니다.

**예시 4 — contradicted (장소 불일치)**
```
sub_q  : 야마다 요지의 출생지는 어디인가?
claim  : 교토 부
passage: "Yoji Yamada was born on 13 September 1931 in Toyonaka, Osaka ..."
판정   : ❌ contradicted  (passage는 오사카 도요나카 출생을 명시, 교토가 아님)
```

---

## 7. 판정 금지 사항

| 하면 안 되는 것 | 이유 |
|---|---|
| 배경지식으로 판정하기 | Passage 기반 판정이 원칙 |
| 이전 홉 결과를 이번 홉 판정에 반영하기 | 홉 단위 독립 판정이 원칙 |
| AI 답변의 전반적 품질로 판정하기 | Claim 단위 판정이 원칙 |
| 애매할 때 contradicted 쓰기 | 애매하면 unsupported |
| 판정 이유 없이 라벨만 기록하기 | IAA 분석에 이유가 필요 |

---

## 8. 어노테이션 웹앱 사용법

1. `iaa_annotation_app.html` 파일을 더블클릭(또는 브라우저로 열기) — 설치·로그인 불필요, 인터넷 연결도 필요 없습니다
2. 이니셜이나 이름을 입력하고 시작 — 같은 이름으로 다시 열면 진행상황이 이어집니다 (브라우저에 자동 저장됨)
3. **Calibration 20개**부터 시작됩니다. 각 항목마다 라벨 버튼 클릭 + 이유 입력 → 제출하면 바로 아래에 "🔎 참고 판정"이 뜹니다 (연구자가 미리 준 참고용 답, 정답으로 검증된 건 아니니 토론용으로만 사용)
4. Calibration 20개가 끝나면 자동으로 **본 판정(101건, blind)**으로 넘어갑니다 — 여기서는 참고 판정이 뜨지 않습니다
5. 좌측 목록에서 언제든 이전 항목으로 돌아가 다시 볼 수 있습니다
6. 전체 완료 후 화면에 뜨는 **"최종 결과 다운로드"** 버튼을 눌러 JSON 파일을 받아 연구자에게 전달해 주세요 (이메일/슬랙 등). 중간에 백업이 필요하면 상단의 "진행상황 백업 다운로드"도 언제든 사용 가능합니다

> passage 안에서 claim과 일치하는 부분은 자동으로 노란색 강조 표시됩니다 — 참고용이며, 최종 판단은 직접 읽고 내려주세요.

---

## 9. Calibration Round 절차

본 판정 전에 아래 절차를 함께 진행합니다:

1. **Calibration 20개**를 각자 독립 판정합니다
2. 불일치 케이스를 같이 검토하고 토론합니다
3. 필요 시 이 가이드라인을 수정합니다
4. 수정된 가이드라인으로 **본 샘플 101 claims**를 독립 판정합니다
5. Fleiss' κ를 계산합니다 (3인 기준)

> κ 목표: **≥ 0.75** (상당한 일치)  
> κ < 0.60이면 불일치 케이스 재분석 후 guideline 추가 수정

---

## 10. 참고 — 라벨 분포 예상 (6모델 pooling 기준)

본 샘플은 hallucination을 의도적으로 과표집(40%)했습니다. 실제 본 샘플(101 claims, judge_label 기준):

- **supported**: 76개 (75.2%)
- **unsupported**: 20개 (19.8%)
- **contradicted**: 5개 (5.0%)

모델 분포: llama 13 · sonnet 9 · gemini 7 · aya 7 · exaone 7 · qwen 7 (trajectory 단위, 50개 중)  
패턴 분포: a 13 · b 13 · c 12 · d 12

---

*버전: v2.2 | 2026-07-08 | Google Sheets → 웹앱 전환, hop1 sub_q 표시 오류(정본 질문 대신 오염된 실제 검색어를 보여주던 문제) 수정, calibration 후 추가 수정될 수 있음*
