# answer_bearing_hit Proxy Audit 가이드라인 v1.0

> 작성일: 2026-07-08
> 목적: `answer_bearing_hit`(gold entity가 passage에 등장하는지로 검색 성공을 판정하는 자동 proxy)가
> 실제로 "모델이 답할 수 있었는데 기권했다"를 의미하는지 인간이 검증 — Human IAA와 별개의 audit
> 대상: Annotator 3인 (독립 판정)
> 도구: 어노테이션 웹앱 (`proxy_annotation_app.html` — 브라우저에서 파일 열기만 하면 실행됨)
> 주의: 기존에 sonnet 26건에 대해 임시로 붙여둔 판정(draft)이 있으나,
> 이는 정식 human audit이 아니다(LLM이 작성했을 가능성 있음). **sonnet 케이스도 다시 blind로 판정 대상에 포함되어 있다.**
> 이 draft는 화면에 표시되지 않는다 — 사후 비교용으로만 연구자가 사용합니다.

---

## 1. 과제 개요

`answer_bearing_hit`는 "hop0에서 검색된 passage 중 하나에 gold entity(정답 개체) 문자열이
등장하면 검색 성공"으로 판정하는 자동 규칙입니다. 그런데 실제로는 **entity가 passage에
등장해도, 질문이 요구하는 관계(relation)까지 그 passage에 명시돼 있어야 모델이 답할 수
있습니다.** entity만 등장하고 관계가 없으면 모델의 기권은 정당한 것이고, proxy가 "검색 성공"
이라 표시한 게 틀린 겁니다(false positive).

이 과제는: **cross-index(KO 질문 → EN 검색) + answer_bearing_hit=True + 모델이 기권한 케이스**를
모아, 그 기권이 진짜 proxy의 false positive인지, 아니면 모델이 답할 수 있었는데도 과하게
기권한 것인지 사람이 직접 passage를 읽고 판정하는 작업입니다.

---

## 2. 용어 설명

Human IAA 가이드라인을 먼저 읽으셨다면 아래 상당수가 익숙하실 겁니다. 처음이시라면 천천히 읽어주세요.

### 이 연구가 하는 일 (한 문장 요약)
AI에게 두 단계(hop)로 나뉜 질문을 주고, AI가 문서를 검색해 답을 찾아가는 과정을 관찰합니다. 이 proxy audit은 그중 **첫 번째 검색 단계(hop 0)에서 "검색이 성공했다"고 자동으로 판정된 케이스인데도 AI가 답을 하지 않고 포기(기권)한 경우**만 모아, 그 자동 판정이 맞는지 사람이 검증하는 작업입니다.

### 검색(retrieval)과 index(인덱스)
AI는 질문에 답하기 위해 미리 만들어둔 문서 저장소(index)에서 관련 문서를 찾습니다. 이 저장소는 한국어 문서용(KO index)과 영어 문서용(EN index)이 따로 있습니다. 질문은 한국어로 하는데 정답이 영어 문서에만 있는 경우(cross-index)는 검색이 특히 어려워집니다 — 이 연구가 살펴보는 핵심 현상입니다.

### hop / passage / sub_q / generated_text
- **hop**: 검색 한 단계. AI는 (1) 검색 질문(`sub_q`)을 만들고 → (2) 문서를 검색해 문서 조각(`passage`)을 가져오고 → (3) 답변(`generated_text`)을 생성합니다. 이 과제는 **hop 0**만 다룹니다.
- **passage**: 검색으로 가져온 문서 조각. 위키피디아 같은 긴 문서를 통째로 주지 않고 관련성 높은 부분만 잘라(chunk) 줍니다. 최대 5개가 제공됩니다.
- **sub_q**: 그 hop에서 실제로 검색에 쓰인 질문 문장.
- **generated_text**: AI가 실제로 출력한 답변 전문 (기권 표현 포함 원문).

### gold entity (정답 개체)
그 질문의 진짜 정답에 해당하는 대상(사람/장소/조직 이름 등)입니다. 예: "이바라키 공항의 운영기관은?"이라는 질문의 gold entity는 `Japan Air Self-Defense Force`입니다.

### answer_bearing_hit
"검색된 문서 중 하나에 gold entity 이름이 등장하면 검색 성공"이라고 판정하는 **자동 규칙**입니다. 이름만 등장했는지, 질문이 실제로 묻는 내용(관계)까지 들어있는지는 구분하지 못합니다 — 이 규칙이 맞는지 사람이 확인하는 게 이 과제의 목적입니다.

### 기권 (abstention)
AI가 "모르겠습니다" 같은 표현으로 답을 하지 않은 경우입니다. 이 과제에서는 "검색은 성공했다고 자동 판정됐는데 AI가 기권한" 케이스만 모아뒀습니다.

---

## 3. 판정 단위 — 화면에 표시되는 항목

각 케이스 화면에 아래 항목들이 순서대로 나옵니다. draft 판정은 화면에 아예 표시되지 않습니다(연구자만 사후 비교용으로 보유).

| 항목 | 설명 |
|---|---|
| model / qid / pattern | 어떤 모델·질문·패턴인지 — 판정과 무관, 식별용 |
| gold entity | 정답 개체명 (hop0 기준), 파란 박스로 강조 표시 |
| sub_q | hop0에서 실제로 검색에 쓰인 질문 — proxy audit은 hop0만 다루므로 항상 정본 질문과 동일합니다 |
| generated_text | 모델이 실제로 낸 답변 (기권 표현 포함 원문) |
| passages | 검색된 문서 최대 5개. gold entity가 등장한 passage는 **GOLD 배지**와 굵은 강조로 표시됩니다 |
| 라벨 버튼 | 클릭해서 판정 입력 (단축키 `1`=proxy-FP `2`=과기권 `3`=모호 `4`=파서FP) |
| 이유 입력란 | 한 줄 이유 작성 |

---

## 4. 라벨 정의

### 🔵 proxy-FP (proxy 오탐)
**"gold entity는 passage에 등장하지만, 질문이 묻는 관계는 그 passage에 없다"**

- entity가 다른 맥락(다른 역할, 다른 관계)으로 언급됐을 뿐, 질문에 답하는 데 필요한 사실은 없음
- 이 경우 모델의 기권은 **정당**하고, `answer_bearing_hit=True`가 틀렸다고 봄

### 🟠 과기권 (모델이 과도하게 기권함)
**"질문이 묻는 관계가 passage에 명시돼 있는데도 모델이 기권했다"**

- passage만 보면 명확히 답할 수 있는데 모델이 "모르겠습니다"라고 답한 경우
- proxy는 맞았고(정말 검색 성공), 모델의 기권 판단이 잘못된 것

### 🟡 모호 (경계 사례)
**"관련 문맥은 있지만 relation이 명시적인지 애매하다"**

- proxy-FP와 과기권 중 하나로 확신 있게 고르기 어려운 경우
- 애매하면 proxy-FP보다는 **모호**를 선택하세요 (과기권으로 과대 판정하지 않기 위함)

### ⚪ 파서FP (참고: 흔하지는 않지만 나올 수 있음)
**"모델이 사실은 기권하지 않고 답을 냈는데, 규칙 기반 기권 분류기가 오탐한 경우"**

- 예: 정답 개체명 자체에 "Unknown"이 들어간 경우 (`Unknown Worlds Entertainment` 등),
  기권 분류기가 "unknown"을 기권 표현으로 착각
- `generated_text`를 읽었을 때 실제로는 확정된 답 엔티티가 있으면 **파서FP**로 표시
  (proxy-FP/과기권/모호 판정 대상이 아님 — 애초에 기권이 아니므로)

---

## 5. 판정 절차

**Step 1.** `sub_q`를 읽고 질문이 정확히 무엇을 묻는지 파악 (예: "~의 본사는 어디인가" → 요구 관계 = headquarters)
**Step 2.** `generated_text`를 확인 — 정말 기권인지, 혹시 파서FP인지 먼저 체크
**Step 3.** `gold_passage_idx`가 가리키는 passage(굵게 표시된 부분)를 읽고, **그 관계가 명시돼 있는지** 확인
**Step 4.** 판정 트리:

```
generated_text가 사실 답을 낸 것 아닌가? (기권 표현에 gold_entity 이름이 우연히 포함된 경우 등)
    │
    ├─ 맞음 → 파서FP
    │
    └─ 아님 (진짜 기권) → gold entity가 등장한 passage에 질문이 묻는 관계가 명시돼 있나?
                              │
                              ├─ 명시 안 됨 (entity는 다른 맥락) → proxy-FP
                              │
                              ├─ 명시됨 (모델이 답할 수 있었음) → 과기권
                              │
                              └─ 애매함 → 모호
```

**Step 5.** 라벨 버튼(`proxy-FP` / `과기권` / `모호` / `파서FP`) 클릭 — 단축키 `1`/`2`/`3`/`4`도 사용 가능
**Step 6.** 이유 입력란에 한 줄 이유 작성 (예: "Auckland가 crew base 문맥으로만 등장, headquarters 관계 없음") — 라벨과 이유를 모두 입력해야 다음 케이스로 넘어갈 수 있습니다

---

## 6. 실제 데이터 예시 (calibration 세트에서 발췌)

**예시 1 — proxy-FP**
```
sub_q          : 제트커넥트의 본사는 어디에 있는가?
gold_entity    : Auckland
gold passage   : "...continues to employ pilots and cabin crew based at Auckland and
                  Wellington airports that operate Qantas flights..."
모델 답변      : 답변: 모르겠습니다
판정           : 🔵 proxy-FP
이유           : Auckland가 승무원 근무지(crew base) 문맥으로만 등장하고,
                 Jetconnect의 headquarters(본사) 관계는 이 passage에 없음.
```

**예시 2 — 과기권**
```
sub_q          : 버터 러버에 출연한 배우는 누구인가?
gold_entity    : Olivia Wilde
gold passage   : "...starring Yara Shahidi, Jennifer Garner, Ty Burrell, Olivia Wilde,
                  Rob Corddry, Ashley Greene, Alicia Silverstone, and Hugh Jackman..."
모델 답변      : 모르겠습니다
판정           : 🟠 과기권
이유           : starring list에 Olivia Wilde가 명시적으로 나열되어 있어 답할 수 있었음에도 기권.
```

**예시 3 — 모호**
```
sub_q          : 이바라키 공항의 운영기관은 어디인가?
gold_entity    : Japan Air Self-Defense Force
gold passage   : "Ibaraki Airport ... also serves as an air base for the Japan Air
                  Self-Defense Force (JASDF) under the name Hyakuri Air Base..."
모델 답변      : 답변: 모르겠습니다
판정           : 🟡 모호
이유           : "air base로 쓰인다"는 관계는 있으나 '운영기관(operator)'이라는
                 질문의 정확한 relation과 정확히 일치하는지는 해석에 따라 갈림.
```

> 위 3개는 calibration 세트에 실제로 포함돼 있으니, calibration 라운드에서 이 예시로
> 연습한 뒤 본인 판정과 (제출 후 화면에 뜨는) 기존 판정을 비교해 논의하세요.

---

## 7. 판정 금지 사항

| 하면 안 되는 것 | 이유 |
|---|---|
| draft 판정을 미리 알아내려 하기 (화면엔 안 뜨지만 원본 파일 등에서) | blind 판정 원칙 위반 — 반드시 판정 후에만 비교 |
| 배경지식으로 판정하기 | passage 기반 판정이 원칙 (Human IAA와 동일) |
| 애매할 때 과기권으로 몰기 | 애매하면 모호 |
| gold_passage_idx 외 다른 passage 근거로 과기권 판정 | 질문이 요구하는 관계가 **gold entity가 등장한 passage**에 있는지가 핵심 |
| 이유 없이 라벨만 기록 | Fleiss' κ 분석 후 불일치 케이스 재검토에 이유가 필요 |

---

## 8. 어노테이션 웹앱 사용법

1. `proxy_annotation_app.html` 파일을 더블클릭(또는 브라우저로 열기) — 설치·로그인 불필요
2. 이니셜이나 이름을 입력하고 시작 — 같은 이름으로 다시 열면 진행상황이 이어집니다 (브라우저에 자동 저장됨)
3. **Calibration 16개**부터 시작됩니다. 각 항목마다 라벨 버튼 클릭 + 이유 입력 → 제출하면 바로 아래에 "🔎 참고 판정"이 뜹니다 (draft 판정 — 정답으로 검증된 건 아니니 토론용으로만 사용)
4. Calibration 16개가 끝나면 자동으로 **본 판정(130건, blind — sonnet 8건 포함)**으로 넘어갑니다
5. 좌측 목록에서 언제든 이전 항목으로 돌아가 다시 볼 수 있습니다
6. 전체 완료 후 화면에 뜨는 **"최종 결과 다운로드"** 버튼을 눌러 JSON 파일을 받아 연구자에게 전달해 주세요. 중간 백업이 필요하면 상단의 "진행상황 백업 다운로드"도 언제든 사용 가능합니다

> κ 목표: **≥ 0.70** (proxy 자체가 애매한 관계 판단을 요구하므로 Human IAA보다 다소 낮은 기준 허용)

---

## 9. 참고 — 케이스 분포

- **calibration**: 16건 (proxy-FP 8 · 과기권 4 · 모호 4, 전부 sonnet 기존 판정에서 추출)
- **main**: 130건 (aya 24 · exaone 36 · llama 20 · qwen 20 · sonnet 8 · gemini 22)
- sonnet의 parser_fp 2건(`Unknown Worlds Entertainment` 관련)은 애초에 기권이 아니므로 pool에서 제외됨

---

*버전: v1.1 | 2026-07-08 | Google Sheets → 웹앱 전환, calibration 후 추가 수정될 수 있음*
