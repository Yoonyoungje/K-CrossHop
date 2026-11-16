---
tags: [pipeline, flags, dynamic, schema]
status: active
---

# Trajectory Flag Schema

> 관련 문서: [[00_index]]

dynamic 파이프라인에서 `pipeline/05_pilot_runner.py`가 생성하는 trajectory 파일의 flag 정의서.  
annotation 이후 추가되는 필드(hop_faithfulness, hallu 등)는 별도 표기.

**Paper condition names.** 아래 `trajectory_mode` 값은 논문의 조건명과 다음과 같이
대응한다. 배포 파일명 접미사도 함께 적는다 (README §4 참조).

| Paper condition (§3.2) | `run_metadata.trajectory_mode` | 배포 파일 접미사 |
|---|---|---|
| Controlled (Korean-query cascade) | `dynamic_controlled` | `ecological` |
| Template-matched ablation | `dynamic_ablation_raw` | `query_matched` |
| Surface-normalized ablation | `dynamic_ablation_normalized` | `normalized` |

---

## 1. Run-level metadata (`run_metadata` 필드)

각 QA trajectory 최상위에 포함. 파이프라인 간 구분에 필수.

| 필드 | 타입 | 값 예시 | 설명 |
|------|------|---------|------|
| `trajectory_mode` | str | `dynamic_controlled` | 파이프라인 식별자 |
| `query_policy` | str | `ko_dynamic` | hop0/hop1 쿼리 언어 정책 |
| `template_policy` | str | `ko_template` / `lang_matched_template` | template 선택 기준 |
| `bridge_insert_policy` | str | `prev_answer_raw` / `prev_answer_normalized` | hop1 bridge 삽입값 출처 |
| `normalization_policy` | str | `none` / `gold_bridge_alias_match` | 정규화 방식 |
| `index_routing` | str | `oracle_gold_lang` | 검색 index 선택 방식 |

### 파이프라인별 값

| 필드 | `dynamic_subq` (ctrl) | `dynamic_subq_qmatched` (abl_raw) | `dynamic_subq_norm` (abl_norm) |
|------|----------------------|-----------------------------------|-------------------------------|
| `trajectory_mode` | `dynamic_controlled` | `dynamic_ablation_raw` | `dynamic_ablation_normalized` |
| `query_policy` | `ko_dynamic` | `lang_matched_dynamic_raw` | `lang_matched_dynamic_normalized` |
| `template_policy` | `ko_template` | `lang_matched_template` | `lang_matched_template` |
| `bridge_insert_policy` | `prev_answer_raw` | `prev_answer_raw` | `prev_answer_normalized` |
| `normalization_policy` | `none` | `none` | `gold_bridge_alias_match` |

---

## 2. Item-level 필드 (QA 단위)

| 필드 | 타입 | 설명 | 사용 지표 |
|------|------|------|----------|
| `qid` | str | trajectory 고유 식별자 | 모든 지표 |
| `path_cluster_id` | str | `{S}_{B}_{A}` — matched 2×2 묶음 | cluster bootstrap |
| `bridge_cluster_id` | str | `{S}_{B}` — hop0 answer 공유 그룹 | cluster bootstrap |
| `pattern` | str | `a`/`b`/`c`/`d` | LSHD / CLEPR |
| `hop0_gold_lang` | str | `ko` / `en` | same/cross 분류 |
| `hop1_gold_lang` | str | `ko` / `en` | same/cross 분류 |
| `transition_direction` | str | `KO→EN` 등 | CLEPR direction |

> **주의**: 생성 파일에는 bridge를 공유하는 확장 행이 섞일 수 있다. 분석은 항상 `--qa-pairs`로
> 정본 코어 세트(1,000행)에 제한한 뒤 수행한다.

---

## 3. Hop-level 쿼리 구성 flags

각 trajectory entry(hop)마다 기록. dynamic 모드에서만 존재 (`hop_state` 있을 때).

### 3.1 Template 관련

| 필드 | 타입 | 값 예시 | 설명 | 사용 지표 |
|------|------|---------|------|----------|
| `query_template_field` | str | `sub_q` / `sub_q_en` | 어떤 qa_pairs 필드를 template으로 사용 | QMC |
| `query_template_lang` | str | `ko` / `en` | template 자체의 언어 | LSHD / QMC |
| `design_query_index_match` | bool | true/false | template_lang == index_lang 여부 | **LSHD 핵심** |
| `actual_query_lang_detected` | str | `ko` / `en` | 최종 query의 실제 언어 (bridge 삽입 후) | 민감도 분석 |
| `query_contains_gold_answer` | bool | false | query에 정답이 포함됐는지 (leakage 탐지) | 무결성 검사 |

> `design_query_index_match` vs `actual_query_lang_detected`:  
> dyn_ctrl에서 KO template에 EN bridge가 삽입되면 design_match=False(EN-index hop)이나 actual=mixed.  
> 헤드라인 LSHD는 **design_query_index_match** 기준 사용. actual은 appendix 민감도 분석.

### 3.2 Bridge anchor 관련

| 필드 | 타입 | 값 예시 | 설명 |
|------|------|---------|------|
| `bridge_anchor_text` | str\|null | `Beijing` / `베이징시` | template에서 치환 대상이 된 문자열 |
| `bridge_anchor_lang` | str\|null | `en` / `ko` | anchor 언어 |
| `bridge_anchor_source` | str\|null | `sub_q_en_bridge_label` / `sub_q_bridge_label` / `prev_gold_entity` | 어느 필드에서 anchor를 가져왔는지 |
| `anchor_replacement_success` | bool\|null | true | 치환 성공 여부 (hop0은 null) |

`bridge_anchor_source` 값:
- `sub_q_bridge_label` — KO gold label (정상 경로)
- `sub_q_en_bridge_label` — EN gold label (정상 경로)
- `prev_gold_entity` — fallback (bridge_label 패치 누락 시)
- `null` — anchor 없음 (template 그대로 반환)

### 3.3 hop0 output 의존 flags (hop1+에서만)

| 필드 | 타입 | 설명 | 사용 지표 |
|------|------|------|----------|
| `prev_answer_raw` | str | 모델의 hop0 원문 답변 | hop1 query provenance |
| `prev_answer_is_gold_bridge` | bool | hop0 답변이 gold bridge entity와 일치하는지 | CLEPR / propagation |
| `prev_answer_surface_lang` | str | hop0 답변의 언어 (`ko`/`en`) | surface mismatch 분석 |
| `prev_answer_abstained` | bool | hop0 abstain 여부 | abstention cascade |

### 3.4 Bridge insertion / normalization flags

| 필드 | 타입 | dyn_ctrl | dyn_abl_raw | dyn_abl_norm |
|------|------|---------|-------------|-------------|
| `bridge_insert_text` | str | hop0 raw | hop0 raw | normalized or raw |
| `bridge_insert_source` | str | `prev_answer_raw` | `prev_answer_raw` | `normalized_prev_answer` or `prev_answer_raw` |
| `normalization_applied` | bool | false | false | true (alias match 성공 시) |
| `normalization_status` | str | `not_applicable` | `not_applicable` | 아래 참조 |
| `normalization_basis` | str\|null | null | null | `prev_answer_alias_match` or null |

`normalization_status` 값:

| 값 | 조건 | CLEPR 처리 |
|----|------|-----------|
| `not_applicable` | dyn_ctrl / dyn_abl_raw | 해당 없음 |
| `success_gold_bridge_alias` | hop0 답변이 gold bridge alias와 매칭 → canonical label로 정규화 | 정상 포함 |
| `failed_unlinked` | alias match 실패 → raw prev_answer 사용 (dyn_abl_raw와 동일) | 정상 포함 |
| `skipped_abstain` | hop0 abstain → normalize 불가 | abstention cascade로 분리 |
| `oracle_gold_bridge_forced` | gold bridge 강제 복구 (현재 미구현, 미래 대비) | **CLEPR 제외 필수** |

> **핵심 경고**: `oracle_gold_bridge_forced`는 hop0 오류를 hop1에서 지워버리므로 propagation/CLEPR 분석에서 반드시 제외.  
> 현재 구현에서는 발생하지 않음 (alias match 실패 시 raw fallback).

---

## 4. 지표별 필요 flags

### 4.1 Dynamic LSHD

```
필요: trajectory_mode, qid, pattern, hop_idx, gold_lang,
      design_query_index_match, abstained, hop_faithfulness
```

- `same-index` = `design_query_index_match == True`
- `cross-index` = `design_query_index_match == False`
- dyn_ctrl cross-index = b/d의 hop0, c/d의 hop1

### 4.2 Dynamic QMC 분해

```
LSHD_dyn_ctrl   (trajectory_mode=dynamic_controlled)
LSHD_dyn_abl_raw  (trajectory_mode=dynamic_ablation_raw)
LSHD_dyn_abl_norm (trajectory_mode=dynamic_ablation_normalized)

QMC_template_raw  = LSHD_dyn_ctrl - LSHD_dyn_abl_raw
QMC_normalization = LSHD_dyn_abl_raw - LSHD_dyn_abl_norm
QMC_total_norm    = LSHD_dyn_ctrl - LSHD_dyn_abl_norm
```

> `QMC_total_norm`을 static QMC와 같은 이름으로 부르지 말 것. entity normalization 개입이 추가됨.

### 4.3 CLEPR

```
필요: pattern, transition_direction, hop0_hallu, hop1_hallu,
      prev_answer_is_gold_bridge, prev_answer_abstained,
      bridge_insert_source, normalization_status
제외: normalization_status == oracle_gold_bridge_forced
```

### 4.4 Yield / Abstention cascade

```
필요: abstained, hop_idx, design_query_index_match,
      prev_answer_abstained, answer_bearing_hit (annotation 후)
```

---

## 5. Annotation 이후 추가 필드 (pipeline/06_annotate.py)

trajectory에 없고 annotated_*.jsonl에서 추가됨:

| 필드 | 출처 | 설명 |
|------|------|------|
| `hop_faithfulness` | LLM judge | FaithRate 분모/분자 |
| `has_hallucination` | derived | hop_faithfulness < 0.5 |
| `abstained` | derived | 기권 여부 |
| `claims` | LLM judge | claim별 라벨 |

`answer_bearing_hit` 등은 08→09→10 체인 이후 별도 파일에서 병합.
