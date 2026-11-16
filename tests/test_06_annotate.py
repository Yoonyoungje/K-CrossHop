"""
unit tests for pipeline/06_annotate.py:
  - NaN faithfulness serialization (Fix 2)
  - fail-fast on missing gold_lang / retrieval_success (Fix 1/3)
  - retrieval_failed computed from retrieval_success contract field (Fix 3)
  - gold_lang present in serialized hops
"""

import sys
import os
import json
import asyncio
import importlib.util

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from utils.answer_parser import is_judge_target
from utils.metrics import valid_faith, clepr_rates as compute_clepr_rates

# 모듈명이 숫자로 시작하므로 importlib로 로드
_spec = importlib.util.spec_from_file_location("annotate06", os.path.join(ROOT, "pipeline/06_annotate.py"))
annotate06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(annotate06)

HopAnnotation = annotate06.HopAnnotation
TrajectoryAnnotation = annotate06.TrajectoryAnnotation
ClaimLabel = annotate06.ClaimLabel
ann_to_dict = annotate06.ann_to_dict
annotate_trajectory = annotate06.annotate_trajectory
safe_faithfulness = annotate06.safe_faithfulness
compute_clepr = annotate06.compute_clepr
parse_generated_answer = annotate06.parse_generated_answer
ParseResult = annotate06.ParseResult


def is_nan(x) -> bool:
    import math
    return isinstance(x, float) and math.isnan(x)


def _hop(**kw):
    base = dict(
        hop=0, generated_text="", retrieved_langs=["ko"], gold_lang="ko",
        lang_switch=False, abstained=False, claims=[],
        hop_faithfulness=1.0, has_hallucination=False,
    )
    base.update(kw)
    return HopAnnotation(**base)


def _ann(hops):
    return TrajectoryAnnotation(
        qid="q00000", model="m", pattern="c", hops=hops,
        cross_lingual_error_propagation=False, abstained=False,
        retrieval_failed=False, final_answer_correct=False,
    )


# ── Fix 2: NaN faithfulness serialization ────────────────────────────────────

def test_safe_faithfulness_helpers():
    assert is_nan(float("nan")) is True
    assert is_nan(0.5) is False
    assert safe_faithfulness(float("nan")) is None
    assert safe_faithfulness(None) is None
    assert safe_faithfulness(0.5) == 0.5


def test_nan_serializes_to_none():
    """비기권·claim 0개 홉(faithfulness=NaN)은 None으로 직렬화."""
    d = ann_to_dict(_ann([_hop(hop_faithfulness=float("nan"), abstained=False)]))
    assert d["hops"][0]["hop_faithfulness"] is None


def test_dumps_allow_nan_false_succeeds_no_nan_token():
    d = ann_to_dict(_ann([_hop(hop_faithfulness=float("nan"))]))
    s = json.dumps(d, ensure_ascii=False, allow_nan=False)  # NaN이면 ValueError 발생
    assert "NaN" not in s


def test_gold_lang_in_serialized_hop():
    d = ann_to_dict(_ann([_hop(gold_lang="en")]))
    assert d["hops"][0]["gold_lang"] == "en"


# ── Fix 1/3: fail-fast on missing required fields ────────────────────────────

def _traj(steps):
    return {"qid": "q00000", "model": "m", "pattern": "c",
            "trajectory": steps, "final_answer": "모르겠습니다"}


def test_missing_gold_lang_raises():
    traj = _traj([{"hop": 0, "generated_text": "모르겠습니다",
                   "retrieved_passages": [], "retrieved_langs": [],
                   "retrieval_success": False}])  # gold_lang 없음
    with pytest.raises(ValueError, match="gold_lang"):
        asyncio.run(annotate_trajectory(traj, None, {}))


def test_missing_retrieval_success_raises():
    traj = _traj([{"hop": 0, "generated_text": "모르겠습니다",
                   "retrieved_passages": [], "retrieved_langs": [],
                   "gold_lang": "ko"}])  # retrieval_success 없음
    with pytest.raises(ValueError, match="retrieval_success"):
        asyncio.run(annotate_trajectory(traj, None, {}))


# ── Fix 3: retrieval_failed from retrieval_success contract field ────────────

def test_retrieval_failed_from_contract_field():
    """홉 하나라도 retrieval_success=False면 retrieval_failed=True.
    모든 홉을 기권 처리해 LLM judge 호출을 피한다."""
    traj = _traj([
        {"hop": 0, "generated_text": "모르겠습니다", "retrieved_passages": [],
         "retrieved_langs": [], "gold_lang": "ko", "retrieval_success": False},
        {"hop": 1, "generated_text": "모르겠습니다", "retrieved_passages": ["x"],
         "retrieved_langs": ["en"], "gold_lang": "en", "retrieval_success": True},
    ])
    ann = asyncio.run(annotate_trajectory(traj, None, {}))
    assert ann.retrieval_failed is True
    # lang_switch는 gold_lang 전환에서 재계산: [ko, en] → [False, True]
    assert [h.lang_switch for h in ann.hops] == [False, True]


def test_retrieval_not_failed_when_all_success():
    traj = _traj([
        {"hop": 0, "generated_text": "모르겠습니다", "retrieved_passages": ["x"],
         "retrieved_langs": ["ko"], "gold_lang": "ko", "retrieval_success": True},
    ])
    ann = asyncio.run(annotate_trajectory(traj, None, {}))
    assert ann.retrieval_failed is False


# ── CLEPR: 측정 불가(None/NaN) 다음 홉을 "회복 성공"으로 세지 않음 ──────────────

def _dhop(**kw):
    base = dict(abstained=False, hop_faithfulness=1.0,
                has_hallucination=False, lang_switch=False)
    base.update(kw)
    return base


def test_valid_faith():
    assert valid_faith(0.0) is True
    assert valid_faith(1.0) is True
    assert valid_faith(None) is False
    assert valid_faith(float("nan")) is False


def test_clepr_rates_excludes_none_next_hop():
    # k: lang_switch + unfaithful, k+1: 측정 불가(None) → 쌍 전체 제외
    results = [{"hops": [
        _dhop(lang_switch=True, hop_faithfulness=0.0, has_hallucination=True),
        _dhop(hop_faithfulness=None, has_hallucination=False),
    ]}]
    cr = compute_clepr_rates(results)
    assert cr["xling_n"] == 0          # 분모에 안 들어감 (이전엔 회복 성공으로 집계됨)
    assert cr["same_n"] == 0


def test_clepr_rates_counts_measured_next_hop():
    # k+1이 실제 측정값이면 정상 집계
    results = [{"hops": [
        _dhop(lang_switch=True, hop_faithfulness=0.0, has_hallucination=True),
        _dhop(hop_faithfulness=0.2, has_hallucination=True),
    ]}]
    cr = compute_clepr_rates(results)
    assert cr["xling_n"] == 1
    assert cr["xling"] == 1.0


def test_clepr_bool_ignores_none_next_hop():
    # 궤적 단위 bool도 None 다음 홉을 전파로 세지 않음
    hops = [
        _hop(hop=0, lang_switch=True, hop_faithfulness=0.0, has_hallucination=True),
        _hop(hop=1, hop_faithfulness=float("nan"), has_hallucination=False),
    ]
    assert compute_clepr(hops) is False


# ── parse_generated_answer: parser hardening ─────────────────────────────────

def test_short_valid_answer_is_one_claim():
    pr = parse_generated_answer("답변: 경기도 용인시")
    assert pr.abstained is False
    assert pr.ambiguous_answer_marker is False
    assert pr.answer_marker_count == 1
    assert len(pr.claims) == 1
    assert "경기도 용인시" in pr.claims[0]


def test_short_valid_answer_space_colon():
    # 스페이스 포함 마커 '답변 : ...'도 단일 claim으로 추출
    pr = parse_generated_answer("답변 : 경기도 용인시")
    assert pr.abstained is False
    assert pr.answer_marker_count == 1
    assert len(pr.claims) == 1
    assert "경기도 용인시" in pr.claims[0]


def test_single_clear_abstain():
    pr = parse_generated_answer("답변: 모르겠습니다.")
    assert pr.abstained is True
    assert pr.claims == []


def test_single_abstain_space_colon():
    # '답변 : 모르겠습니다.' → abstained
    pr = parse_generated_answer("답변 : 모르겠습니다.")
    assert pr.abstained is True
    assert pr.claims == []
    assert pr.answer_marker_count == 1


def test_single_abstain_fullwidth_colon():
    # 전각 콜론 '답변：모르겠습니다.' → abstained
    pr = parse_generated_answer("답변：모르겠습니다.")
    assert pr.abstained is True
    assert pr.claims == []
    assert pr.answer_marker_count == 1


def test_multi_marker_both_abstain():
    pr = parse_generated_answer("답변: 정보가 없습니다.\n답변: 모르겠습니다.")
    assert pr.abstained is True
    assert pr.claims == []
    assert pr.ambiguous_answer_marker is False


def test_multi_marker_non_abstain_final():
    pr = parse_generated_answer("답변: 설명입니다.\n답변: 용인시")
    assert pr.abstained is False
    assert pr.ambiguous_answer_marker is True
    assert pr.parse_issue == "multi_answer_non_abstain"
    assert pr.claims == []
    assert pr.answer_marker_count >= 2


def test_multi_marker_mixed_colon_variants():
    # 일반 콜론 + 스페이스 콜론 혼합 → ambiguous
    pr = parse_generated_answer("답변: 설명입니다.\n답변 : 용인시")
    assert pr.ambiguous_answer_marker is True
    assert pr.answer_marker_count == 2


def test_fallback_no_answer_marker_in_claims():
    # 답변: 마커가 있는 텍스트 → answer_section 경로, claim에 리터럴 "답변:" 없음
    text = "생각: 먼저 검색해보자\n답변: 경기도입니다.\n이것이 결론입니다."
    pr = parse_generated_answer(text)
    for claim in pr.claims:
        assert "답변:" not in claim


def test_fallback_path_no_marker_leak():
    # 마커가 전혀 없는 텍스트 → fallback 경로, claim에 답변 마커 잔여 없음
    text = "생각: 먼저 검색해보자\n이것이 결론입니다."
    pr = parse_generated_answer(text)
    assert pr.answer_marker_count == 0
    for claim in pr.claims:
        assert "답변:" not in claim
        assert "답변 :" not in claim


def test_nonstandard_abstain_morupnida():
    # '모릅니다' (비표준 기권어) — 마커 없는 짧은 텍스트 → fallback_abstain
    pr = parse_generated_answer("모릅니다.")
    assert pr.abstained is True
    assert pr.claims == []
    assert pr.source == "fallback_abstain"


def test_nonstandard_abstain_morupnida_with_marker():
    # 마커 + 모릅니다 → answer_section_abstain
    pr = parse_generated_answer("답변: 모릅니다.")
    assert pr.abstained is True
    assert pr.claims == []


def test_long_markerless_abstain_no_length_gate():
    # 마커 없는 장황한 기권(>80자) — 게이트 제거 후 marker==1과 동일하게 abstain.
    # 모델이 엔티티를 추측하면서도 "문서엔 명시 안 됨/모르겠습니다"로 비근거를 자인한 케이스.
    text = ('서울특별시 (추정 근거 부족으로 정확한 행정구역 명시 불가) '
            '모르겠습니다만, 제공된 문서에는 기초과학연구원의 구체적인 위치 정보가 '
            '명시되어 있지 않습니다.')
    pr = parse_generated_answer(text)
    assert pr.answer_marker_count == 0
    assert pr.abstained is True
    assert pr.source == "fallback_abstain"
    assert pr.claims == []


@pytest.mark.parametrize("text", [
    "답변: The documents do not mention where the person worked.",
    "답변: The passages do not contain that information.",
    "답변: There is insufficient information to determine the answer.",
    "답변: No information about the location is provided.",
    "답변: It is not possible to determine the year from the documents.",
    "답변: I am unable to find the answer in the provided passages.",
    "답변: This cannot be determined from the given documents.",
    "답변: The document does not specify the director.",
])
def test_frontier_english_abstain(text):
    # frontier 모델의 장황한 영어 거절 표현 → abstained (rev2 무게이트, 8B와 동일 파서)
    pr = parse_generated_answer(text)
    assert pr.abstained is True, f"미탐지: {text!r}"
    assert pr.claims == []


@pytest.mark.parametrize("text", [
    "답변: Seoul National University",        # 정상 영어 답변
    "답변: The film contains many scenes.",   # "contain"이 "do not contain" 거절 아님
    "답변: The movie was directed by Bong Joon-ho.",
    "답변: Unknown Worlds Entertainment",     # 엔티티명의 'Unknown' — 기권 아님 (정답)
    "답변: The Unknown Soldier",              # 고유명사 'Unknown' — 기권 아님
    "답변: Unknown Pleasures",                # 앨범명 — 기권 아님
])
def test_frontier_english_non_abstain(text):
    # 영어 정상 답변은 기권으로 오탐하지 않는다 (거절 구문 한정 매칭)
    pr = parse_generated_answer(text)
    assert pr.abstained is False, f"오탐: {text!r}"


@pytest.mark.parametrize("text", [
    "답변: unknown",                  # bare unknown — 여전히 기권
    "답변: Unknown",
    "답변: unknown.",
    "답변: N/A",
    "답변: The director is unknown",  # 뒤에 대문자 고유명사 없음 → 기권
])
def test_bare_unknown_still_abstain(text):
    # bare 'unknown'/'N/A'은 기권 유지 (엔티티명 안의 substring만 제외)
    pr = parse_generated_answer(text)
    assert pr.abstained is True, f"미탐지: {text!r}"


def test_artifact_cyrillic():
    # 키릴 문자 지배적 출력 → artifact
    pr = parse_generated_answer("Созданный текст не содержит информации о том, где работал 강소천.")
    assert pr.source == "artifact"
    assert pr.claims == []
    assert pr.abstained is False
    assert pr.parse_issue == "artifact_output"


def test_artifact_malayalam():
    # 말라얄람 문자 → artifact
    pr = parse_generated_answer("കെഎച്ച്ഇഎൻ")
    assert pr.source == "artifact"
    assert pr.claims == []


def test_fallback_short_bare_answer():
    # 마커 없이 짧은 명사구만 출력 (v2.3 스타일) → fallback_short, claims에 포함
    pr = parse_generated_answer("서울")
    assert pr.source == "fallback_short"
    assert pr.abstained is False
    assert pr.claims == ["서울"]


def test_fallback_short_not_abstain():
    # fallback_short는 기권이 아님
    pr = parse_generated_answer("고려대학교")
    assert pr.source == "fallback_short"
    assert pr.abstained is False
    assert len(pr.claims) == 1


# ── is_judge_target ──────────────────────────────────────────────────────────

def test_judge_target_non_abstain_with_claims():
    pr = parse_generated_answer("답변: 서울대학교")
    assert is_judge_target(pr, {}) is True


def test_judge_target_abstain_is_not_target():
    pr = parse_generated_answer("답변: 모르겠습니다.")
    assert pr.abstained is True
    assert is_judge_target(pr, {}) is False


def test_judge_target_non_abstain_no_claims():
    # source=none, claims=[] → not a judge target
    pr = ParseResult(claims=[], source="none", abstained=False,
                     answer_marker_count=0, ambiguous_answer_marker=False, parse_issue=None)
    assert is_judge_target(pr, {}) is False


def test_judge_target_empty_content_with_reasoning():
    pr = ParseResult(claims=["서울"], source="fallback_short", abstained=False,
                     answer_marker_count=0, ambiguous_answer_marker=False, parse_issue=None)
    assert is_judge_target(pr, {"empty_content_with_reasoning": True}) is False


def test_judge_target_missing_artifact_fields_default_false():
    # 구형 trajectory 파일에 execution-artifact 필드 없어도 True로 평가
    pr = ParseResult(claims=["고려대학교"], source="fallback_short", abstained=False,
                     answer_marker_count=0, ambiguous_answer_marker=False, parse_issue=None)
    assert is_judge_target(pr, {}) is True


def test_clepr_rates_excludes_ambiguous_hop():
    # ambiguous 홉은 faithfulness=NaN → valid_faith=False → 분모 제외
    results = [{"hops": [
        _dhop(lang_switch=True, hop_faithfulness=0.0, has_hallucination=True),
        _dhop(hop_faithfulness=float("nan"), has_hallucination=False),  # ambiguous hop
    ]}]
    cr = compute_clepr_rates(results)
    assert cr["xling_n"] == 0
