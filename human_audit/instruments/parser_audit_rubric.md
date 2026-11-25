# 기권 parser 품질 audit — 라벨링 rubric

목적: 규칙기반 기권 분류기(`utils/answer_parser.py`)를 사람 gold와 대조해
Precision/Recall/F1을 측정한다. 라벨러는 **모델 출력(`generated_text`)만 보고**
그 홉이 실제로 기권인지 판정한다. parser의 결정(어느 시트에서 왔는지)은 보지 않는다(블라인드).

## human_label 3범주

| 라벨 | 정의 | 예 |
|------|------|-----|
| **abstain** | 출력의 핵심이 **거부/불확실**이고, 물은 대상에 대한 **확정 답 엔티티가 없음** | "모르겠습니다", "문서에서 확인할 수 없습니다", "정보가 없습니다" |
| **answer** | 물은 대상에 대해 **답 엔티티를 확정**(정답·오답 무관) | "답변: 도쿄", "Unknown Worlds Entertainment" |
| **unclear** | 위 둘로 판정 곤란(깨진 출력, 다국어 혼입 등) | — |

## 경계 규칙 (반드시 사전 적용)

1. **partial → abstain.** 일부 엔티티를 나열하나 *물은 그 대상*은 "확인 불가"라 하면 **abstain**.
   - 예: "오릭스·소프트뱅크는 퍼시픽리그, **롯데의 리그는 문서에서 확인되지 않습니다**" → 물은 대상(롯데) 미확정 → **abstain**.
   - 근거: yield 정의 = *물은 답을 산출했는가*. 곁가지 엔티티 언급은 답 산출이 아니다.

2. **틀린 답도 answer.** 오답이어도 확정했으면 answer(이건 judge가 unsupported로 처리할 몫이지 기권이 아님).

3. **엔티티명이 기권어를 포함해도 answer.** "**Unknown** Worlds Entertainment", "언노운 월드…"는 답 엔티티지 기권 아님 → **answer**.
   (이게 현재 parser의 주 FP 클래스다.)

4. **설명 후 기권**: 근거를 길게 설명하다 마지막에 "모르겠습니다"로 닫으면 → **abstain**.
   설명 끝에 답을 확정하면 → **answer**.

5. **빈 출력 / 깨진 출력 / 비KO·비EN 지배** → **unclear** (P/R에서 제외).

## 라벨 대상 (시트 우선순위)

전수(1,683 abstain + nonabstain 표본)를 다 볼 필요 없다. 다음만 채우면 P/R이 닫힌다:

- **abstain 시트**: `no_abstain_kw`(전수) → `gold_in_passage` → `long` → `clean`(표본). clean은 FP≈0 기대라 sanity용.
- **nonabstain 시트**: `short`(전수에 가깝게) → `normal`(표본). FN(놓친 기권)은 단문에 숨는다.

채점은 `parser_abstain_audit_prep.py score --filled <csv...>` — (config,stratum)별 표본비율을
모집단으로 재가중해 Precision/Recall/F1 + 층화 부트스트랩 95% CI를 출력한다.

## 권장 절차

1. 2인 라벨러가 독립으로 채운다(블라인드). 최소 overlap 80홉.
2. overlap에서 Cohen's κ 계산, κ<0.6이면 rubric 보강 후 재라벨.
3. 불일치 홉은 합의(adjudication) 후 단일 gold 확정.
4. 부록 보고: confusion matrix + P/R/F1(CI) + named FP class("unknown" 12홉) + κ.
