# 원 실험 judge 관대함 규모

대상: 6모델 × 3조건의 **yield=1 ∧ 주 매처 오답 ∧ 비기권 ∧ 비-ambiguous** hop 4292건. 판정은 응답이 gold 와 같은 엔티티를 가리키는지를 `gpt-5.4-2026-03-05`로 물었다(원고 judge 와 같은 요청 본문, `temperature 0`·`seed 42`, 입력은 질문·gold KO·EN 표기·응답, **passage 미제공**). 기록 `entity_check.jsonl`. 집계 `judge_leniency_report.py` · numpy `default_rng(42)`, triple 250 클러스터 재표집 B = 10,000.

전체 판정: yes 1047 / no 3210 / unclear 35.

## 1. 모델 × 조건 × same/cross

| 모델 | 조건 | side | 비기권 hop | 대상 | 대상 비율 | same_entity=yes [95% CI] | no | unclear | no 중 passage 내 존재 |
|-|-|-|-|-|-|-|-|-|-|
| Aya Expanse 8B | controlled | same | 968 | 93 | 0.096 | 0.204 [0.111, 0.312] | 74 | 0 | 52/74 (0.703) |
| Aya Expanse 8B | controlled | cross | 946 | 122 | 0.129 | 0.156 [0.077, 0.248] | 103 | 0 | 51/103 (0.495) |
| Aya Expanse 8B | template-matched | same | 966 | 97 | 0.100 | 0.175 [0.088, 0.278] | 80 | 0 | 59/80 (0.738) |
| Aya Expanse 8B | template-matched | cross | 965 | 126 | 0.131 | 0.206 [0.116, 0.305] | 100 | 0 | 55/100 (0.550) |
| Aya Expanse 8B | surface-normalized | same | 966 | 101 | 0.105 | 0.178 [0.090, 0.280] | 83 | 0 | 60/83 (0.723) |
| Aya Expanse 8B | surface-normalized | cross | 970 | 133 | 0.137 | 0.256 [0.161, 0.359] | 99 | 0 | 54/99 (0.545) |
| EXAONE 3.5 8B | controlled | same | 812 | 68 | 0.084 | 0.118 [0.016, 0.246] | 60 | 0 | 44/60 (0.733) |
| EXAONE 3.5 8B | controlled | cross | 742 | 72 | 0.097 | 0.236 [0.096, 0.392] | 55 | 0 | 29/55 (0.527) |
| EXAONE 3.5 8B | template-matched | same | 843 | 72 | 0.085 | 0.125 [0.026, 0.250] | 63 | 0 | 41/63 (0.651) |
| EXAONE 3.5 8B | template-matched | cross | 864 | 87 | 0.101 | 0.218 [0.114, 0.333] | 68 | 0 | 27/68 (0.397) |
| EXAONE 3.5 8B | surface-normalized | same | 842 | 72 | 0.086 | 0.125 [0.026, 0.250] | 63 | 0 | 42/63 (0.667) |
| EXAONE 3.5 8B | surface-normalized | cross | 880 | 88 | 0.100 | 0.227 [0.118, 0.348] | 68 | 0 | 24/68 (0.353) |
| Llama 3.1 8B | controlled | same | 970 | 135 | 0.139 | 0.193 [0.100, 0.294] | 109 | 0 | 104/109 (0.954) |
| Llama 3.1 8B | controlled | cross | 947 | 147 | 0.155 | 0.279 [0.191, 0.372] | 103 | 3 | 71/103 (0.689) |
| Llama 3.1 8B | template-matched | same | 977 | 136 | 0.139 | 0.206 [0.106, 0.312] | 108 | 0 | 105/108 (0.972) |
| Llama 3.1 8B | template-matched | cross | 974 | 158 | 0.162 | 0.399 [0.295, 0.500] | 92 | 3 | 55/92 (0.598) |
| Llama 3.1 8B | surface-normalized | same | 977 | 141 | 0.144 | 0.206 [0.108, 0.311] | 112 | 0 | 107/112 (0.955) |
| Llama 3.1 8B | surface-normalized | cross | 980 | 152 | 0.155 | 0.388 [0.284, 0.493] | 91 | 2 | 60/91 (0.659) |
| Qwen3 8B | controlled | same | 929 | 158 | 0.170 | 0.158 [0.085, 0.238] | 130 | 3 | 105/130 (0.808) |
| Qwen3 8B | controlled | cross | 866 | 224 | 0.259 | 0.375 [0.292, 0.462] | 131 | 9 | 50/131 (0.382) |
| Qwen3 8B | template-matched | same | 941 | 162 | 0.172 | 0.160 [0.086, 0.242] | 133 | 3 | 106/133 (0.797) |
| Qwen3 8B | template-matched | cross | 935 | 236 | 0.252 | 0.411 [0.328, 0.498] | 135 | 4 | 67/135 (0.496) |
| Qwen3 8B | surface-normalized | same | 940 | 166 | 0.177 | 0.163 [0.088, 0.243] | 135 | 4 | 105/135 (0.778) |
| Qwen3 8B | surface-normalized | cross | 939 | 242 | 0.258 | 0.393 [0.311, 0.476] | 143 | 4 | 65/143 (0.455) |
| Claude Sonnet 4.6 | controlled | same | 826 | 86 | 0.104 | 0.140 [0.037, 0.262] | 74 | 0 | 61/74 (0.824) |
| Claude Sonnet 4.6 | controlled | cross | 724 | 98 | 0.135 | 0.255 [0.150, 0.372] | 73 | 0 | 25/73 (0.342) |
| Claude Sonnet 4.6 | template-matched | same | 863 | 86 | 0.100 | 0.140 [0.032, 0.261] | 74 | 0 | 58/74 (0.784) |
| Claude Sonnet 4.6 | template-matched | cross | 842 | 97 | 0.115 | 0.278 [0.163, 0.405] | 70 | 0 | 42/70 (0.600) |
| Claude Sonnet 4.6 | surface-normalized | same | 866 | 90 | 0.104 | 0.133 [0.031, 0.250] | 78 | 0 | 62/78 (0.795) |
| Claude Sonnet 4.6 | surface-normalized | cross | 870 | 94 | 0.108 | 0.287 [0.169, 0.418] | 67 | 0 | 38/67 (0.567) |
| Gemini 3.5 Flash | controlled | same | 844 | 89 | 0.105 | 0.135 [0.039, 0.247] | 77 | 0 | 71/77 (0.922) |
| Gemini 3.5 Flash | controlled | cross | 739 | 106 | 0.143 | 0.302 [0.194, 0.416] | 74 | 0 | 38/74 (0.514) |
| Gemini 3.5 Flash | template-matched | same | 879 | 90 | 0.102 | 0.144 [0.044, 0.258] | 77 | 0 | 70/77 (0.909) |
| Gemini 3.5 Flash | template-matched | cross | 841 | 91 | 0.108 | 0.275 [0.151, 0.410] | 66 | 0 | 36/66 (0.545) |
| Gemini 3.5 Flash | surface-normalized | same | 879 | 92 | 0.105 | 0.141 [0.043, 0.253] | 79 | 0 | 73/79 (0.924) |
| Gemini 3.5 Flash | surface-normalized | cross | 879 | 85 | 0.097 | 0.259 [0.132, 0.400] | 63 | 0 | 36/63 (0.571) |

## 2. 모델별 — 전체 비기권 hop 대비

"judge faithful ∧ 다른 엔티티" = yield=1 ∧ 주 매처 오답 ∧ same_entity=no. 분모는 비기권·비-ambiguous hop 전체(3조건 합).

| 모델 | 비기권 hop | 대상 (yield=1 ∧ 오답) | 대상 비율 | same_entity=no | **judge faithful ∧ 다른 엔티티 비율** | 그중 passage 내 존재 |
|-|-|-|-|-|-|-|
| Aya Expanse 8B | 5781 | 672 | 0.116 | 539 | **0.093** | 331/539 (0.614) |
| EXAONE 3.5 8B | 4983 | 459 | 0.092 | 377 | **0.076** | 207/377 (0.549) |
| Llama 3.1 8B | 5825 | 869 | 0.149 | 615 | **0.106** | 502/615 (0.816) |
| Qwen3 8B | 5550 | 1188 | 0.214 | 807 | **0.145** | 498/807 (0.617) |
| Claude Sonnet 4.6 | 4991 | 551 | 0.110 | 436 | **0.087** | 286/436 (0.656) |
| Gemini 3.5 Flash | 5061 | 553 | 0.109 | 436 | **0.086** | 324/436 (0.743) |

## 3. 표기 차이 유형 (same_entity=yes)

| 모델 | side | yes | 어순 | 음역 차이 | 띄어쓰기·구두점 | 상위·하위 개념 | 기타 |
|-|-|-|-|-|-|-|-|
| Aya Expanse 8B | same | 54 | 0 | 12 | 1 | 15 | 26 |
| Aya Expanse 8B | cross | 79 | 4 | 37 | 6 | 10 | 22 |
| EXAONE 3.5 8B | same | 26 | 0 | 6 | 0 | 8 | 12 |
| EXAONE 3.5 8B | cross | 56 | 2 | 17 | 1 | 12 | 24 |
| Llama 3.1 8B | same | 83 | 0 | 27 | 3 | 13 | 40 |
| Llama 3.1 8B | cross | 163 | 34 | 82 | 4 | 11 | 32 |
| Qwen3 8B | same | 78 | 2 | 43 | 0 | 12 | 21 |
| Qwen3 8B | cross | 276 | 23 | 166 | 4 | 27 | 56 |
| Claude Sonnet 4.6 | same | 36 | 0 | 14 | 1 | 11 | 10 |
| Claude Sonnet 4.6 | cross | 79 | 31 | 30 | 4 | 6 | 8 |
| Gemini 3.5 Flash | same | 38 | 0 | 19 | 0 | 4 | 15 |
| Gemini 3.5 Flash | cross | 79 | 11 | 31 | 8 | 7 | 22 |

## 4. 사례 — same_entity=no (무작위 15건, seed 42)

- **Claude Sonnet 4.6** · template-matched · `kch_000950` hop0 · passage 내 존재 아니오
  - 질문: 에딘버러 왕립학회의 설립자는 누구인가? / gold KO 「콜린 매클로린」
  - 응답 claim: ['문서에는 에딘버러 왕립학회의 특정 설립자가 명시되어 있지 않습니다.', '다만 조지프 블랙, 윌리엄 쿨렌, 존 워커 등 에딘버러 대학교 교수들의 도움으로 1783년에 설립되었다고 기술되어 있습니다.']
  - 판정 근거: 모델 응답은 콜린 매클로린이 아니라 조지프 블랙, 윌리엄 쿨렌, 존 워커 등을 언급하며 특정 설립자를 정답 엔티티로 지시하지 않습니다.
- **Aya Expanse 8B** · surface-normalized · `kch_000634` hop0 · passage 내 존재 예
  - 질문: 송진형이 소속된 팀은 어디인가? / gold KO 「뉴캐슬 유나이티드 제츠 FC」
  - 응답 claim: ['FC 서울']
  - 판정 근거: 정답 엔티티는 '뉴캐슬 유나이티드 제츠 FC(Newcastle Jets FC)'이고 모델 응답은 'FC 서울'로 서로 다른 팀명을 가리킵니다.
- **Aya Expanse 8B** · controlled · `kch_000052` hop0 · passage 내 존재 예
  - 질문: 김태균이 소속된 팀은 어디인가? / gold KO 「지바 롯데 마린스」
  - 응답 claim: ['Jeonbuk Hyundai Motors']
  - 판정 근거: 모델 응답 'Jeonbuk Hyundai Motors'는 정답 엔티티 '지바 롯데 마린스(Chiba Lotte Marines)'와 표기가 전혀 다른 별개의 대상입니다.
- **Gemini 3.5 Flash** · template-matched · `kch_000190` hop0 · passage 내 존재 예
  - 질문: 에네루프의 제조사는 어디인가? / gold KO 「파나소닉」
  - 응답 claim: ['FDK']
  - 판정 근거: 정답 엔티티는 '파나소닉(Panasonic Holdings Corporation)'이고 모델 응답은 'FDK'로, 주어진 표기상 서로 다른 실체입니다.
- **Llama 3.1 8B** · controlled · `kch_000168` hop0 · passage 내 존재 예
  - 질문: 에스티 로더의 설립자는 누구인가? / gold KO 「에스티 로더」
  - 응답 claim: ['조셉 로더']
  - 판정 근거: 정답 엔티티는 '에스티 로더(Estée Lauder)'인데, 모델 응답 '조셉 로더'는 표기가 달라 동일 실체로 보기 어렵습니다.
- **Llama 3.1 8B** · controlled · `kch_000086` hop0 · passage 내 존재 아니오
  - 질문: 문선민이 소속된 팀은 어디인가? / gold KO 「유르고르덴 IF」
  - 응답 claim: ['FC 서울']
  - 판정 근거: 정답 엔티티는 '유르고르덴 IF(Djurgårdens IF Fotboll)'이고, 모델 응답의 'FC 서울'은 표기가 전혀 다른 다른 팀명입니다.
- **EXAONE 3.5 8B** · surface-normalized · `kch_000680` hop1 · passage 내 존재 예
  - 질문: 마이클 커티즈의 출생지는 어디인가? / gold KO 「부다페스트」
  - 응답 claim: ['헝가리']
  - 판정 근거: 정답은 도시인 부다페스트이고, 모델 응답인 헝가리는 국가로서 서로 다른 실체입니다.
- **EXAONE 3.5 8B** · controlled · `kch_000226` hop0 · passage 내 존재 아니오
  - 질문: 오재석이 소속된 팀은 어디인가? / gold KO 「감바 오사카」
  - 응답 claim: ['대전 한아 시민']
  - 판정 근거: 정답 엔티티는 '감바 오사카(Gamba Osaka)'이고, 모델 응답 '대전 한아 시민'은 표기가 전혀 다른 다른 대상을 가리킵니다.
- **Gemini 3.5 Flash** · template-matched · `kch_000572` hop0 · passage 내 존재 예
  - 질문: 안나 성당의 기적에 출연한 배우는 누구인가? / gold KO 「발렌티나 체르비」
  - 응답 claim: ['데릭 루크, 마이클 일리']
  - 판정 근거: 정답 엔티티는 '발렌티나 체르비(Valentina Cervi)'인데, 모델 응답의 '데릭 루크, 마이클 일리'는 서로 다른 다른 인물들입니다.
- **Aya Expanse 8B** · surface-normalized · `kch_000009` hop0 · passage 내 존재 아니오
  - 질문: 이바라키 공항의 운영기관은 어디인가? / gold KO 「항공자위대」
  - 응답 claim: ['이바라키 공항은 이바라키현에서 운영합니다.']
  - 판정 근거: 정답 엔티티는 '항공자위대'인데, 모델 응답은 '이바라키현'을 운영기관으로 지목하여 서로 다른 실체입니다.
- **Claude Sonnet 4.6** · surface-normalized · `kch_000993` hop0 · passage 내 존재 예
  - 질문: 차두리가 소속된 팀은 어디인가? / gold KO 「아인트라흐트 프랑크푸르트」
  - 응답 claim: ['화성 FC']
  - 판정 근거: 모델 응답의 '화성 FC'는 정답 엔티티 '아인트라흐트 프랑크푸르트(Eintracht Frankfurt)'와 표기가 전혀 다른 별개의 팀명입니다.
- **Gemini 3.5 Flash** · template-matched · `kch_000993` hop0 · passage 내 존재 예
  - 질문: 차두리가 소속된 팀은 어디인가? / gold KO 「아인트라흐트 프랑크푸르트」
  - 응답 claim: ['화성 FC']
  - 판정 근거: 모델 응답의 '화성 FC'는 정답 엔티티 '아인트라흐트 프랑크푸르트(Eintracht Frankfurt)'와 표기가 전혀 다른 별개의 팀명입니다.
- **Qwen3 8B** · surface-normalized · `kch_000931` hop0 · passage 내 존재 아니오
  - 질문: 콜롬보 계획의 본사는 어디에 있는가? / gold KO 「콜롬보」
  - 응답 claim: ['스리 라크나']
  - 판정 근거: 정답 엔티티는 '콜롬보(Colombo)'인데, 모델 응답 '스리 라크나'는 표기상 동일하거나 통상적 변형으로 보기 어렵습니다.
- **Aya Expanse 8B** · template-matched · `kch_000467` hop0 · passage 내 존재 예
  - 질문: 장현수가 소속된 팀은 어디인가? / gold KO 「FC 도쿄」
  - 응답 claim: ['Al-Gharafa']
  - 판정 근거: 정답 엔티티는 'FC 도쿄(FC Tokyo)'이고 모델 응답은 'Al-Gharafa'로, 표기가 가리키는 대상이 다릅니다.
- **Claude Sonnet 4.6** · controlled · `kch_000489` hop0 · passage 내 존재 예
  - 질문: 허정무가 소속된 팀은 어디인가? / gold KO 「PSV 에인트호번」
  - 응답 claim: ['대전 하나 시티즌']
  - 판정 근거: 모델 응답의 '대전 하나 시티즌'은 정답 엔티티 'PSV 에인트호번(PSV Eindhoven)'과 표기가 전혀 다른 별개의 팀명입니다.
