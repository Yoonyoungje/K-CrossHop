"""
unit tests for _build_dynamic_query / _normalize_bridge in pipeline/05_pilot_runner.py

대상 시나리오:
  dynamic_subq (dyn_ctrl):
    - hop0 always uses sub_q (KO), regardless of gold_lang
    - hop1 KO-index: sub_q + bridge_label substitution
    - hop1 EN-index: sub_q + bridge_label substitution (intentional KO→EN mismatch)
    - hop1 fallback: use prev_gold when bridge_label absent

  dynamic_subq_qmatched (dyn_abl_raw):
    - hop0 KO-index: sub_q (KO)
    - hop0 EN-index: sub_q_en (EN)
    - hop1 KO-index: sub_q + bridge_label substitution
    - hop1 EN-index: sub_q_en + sub_q_en_bridge_label substitution
    - hop1 EN-index fallback (no sub_q_en_bridge_label): prev_gold in sub_q_en
    - hop1 EN-index fallback (nothing matches): sub_q_en as-is

  dynamic_subq_norm (dyn_abl_norm):
    - hop0: identical to dyn_abl_raw (lang-matched)
    - hop1 EN-index alias match success: normalizes to EN canonical label
    - hop1 EN-index alias match failure: falls back to raw prev_answer (like dyn_abl_raw)
    - hop1 KO-index alias match success: normalizes to KO canonical label
    - hop1 KO-index alias match failure: falls back to raw prev_answer

  _normalize_bridge:
    - KO prev_answer → EN canonical (alias match)
    - EN prev_answer → KO canonical (alias match)
    - no match → raw prev_answer unchanged
    - case-insensitive match
"""

import sys
import os
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("pilot05", os.path.join(ROOT, "pipeline/05_pilot_runner.py"))
pilot05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot05)

_build     = pilot05._build_dynamic_query
_normalize = pilot05._normalize_bridge

CTRL = "dynamic_subq"
ABL  = "dynamic_subq_qmatched"
NORM = "dynamic_subq_norm"


# ── 공통 fixture ──────────────────────────────────────────────────────────────

def hop_ko(sub_q="서울이 수도인 나라는?", bridge_label="서울", gold_entity="대한민국"):
    return {
        "gold_lang": "ko",
        "sub_q": sub_q,
        "sub_q_en": "What country has Seoul as its capital?",
        "sub_q_bridge_label": bridge_label,
        "sub_q_en_bridge_label": "Seoul",
        "gold_entity": gold_entity,
    }


def hop_en(sub_q="베이징시가 속한 나라는?",
           sub_q_en="What country is Beijing in?",
           bridge_label="베이징시",
           en_bridge_label="Beijing",
           gold_entity="China"):
    return {
        "gold_lang": "en",
        "sub_q": sub_q,
        "sub_q_en": sub_q_en,
        "sub_q_bridge_label": bridge_label,
        "sub_q_en_bridge_label": en_bridge_label,
        "gold_entity": gold_entity,
    }


def prev_ko(gold_entity="서울"):
    return {"gold_lang": "ko", "gold_entity": gold_entity}


def prev_en(gold_entity="Beijing"):
    return {"gold_lang": "en", "gold_entity": gold_entity}


# ── dynamic_subq (dyn_ctrl) ───────────────────────────────────────────────────

class TestDynCtrlHop0:
    def test_ko_index_uses_subq(self):
        assert _build(CTRL, 0, hop_ko(), "", None) == "서울이 수도인 나라는?"

    def test_en_index_uses_subq_not_subq_en(self):
        # controlled: EN-index hop0도 KO sub_q 사용 (의도적 mismatch)
        assert _build(CTRL, 0, hop_en(), "", None) == "베이징시가 속한 나라는?"

    def test_no_prev_answer_returns_template(self):
        assert _build(CTRL, 1, hop_ko(), "", prev_ko()) == "서울이 수도인 나라는?"


class TestDynCtrlHop1:
    def test_ko_index_bridge_label_substitution(self):
        h = hop_ko(sub_q="서울의 인구는 얼마인가?", bridge_label="서울")
        result = _build(CTRL, 1, h, "부산", prev_ko("서울"))
        assert result == "부산의 인구는 얼마인가?"

    def test_en_index_uses_ko_template(self):
        # controlled: hop1 EN-index도 KO template 사용 (의도적 mismatch)
        h = hop_en(sub_q="베이징시가 속한 나라는?", bridge_label="베이징시")
        result = _build(CTRL, 1, h, "Tokyo", prev_en("Beijing"))
        assert result == "Tokyo가 속한 나라는?"

    def test_fallback_to_prev_gold_when_no_bridge_label(self):
        h = hop_ko(sub_q="서울의 인구는 얼마인가?", bridge_label="없는라벨")
        result = _build(CTRL, 1, h, "부산", prev_ko("서울"))
        assert result == "부산의 인구는 얼마인가?"

    def test_fallback_returns_template_when_nothing_matches(self):
        h = {
            "gold_lang": "ko",
            "sub_q": "완전히 다른 질문",
            "sub_q_bridge_label": "",
            "gold_entity": "X",
        }
        result = _build(CTRL, 1, h, "부산", {"gold_entity": "서울"})
        assert result == "완전히 다른 질문"


# ── dynamic_subq_qmatched (dyn_abl_raw) ──────────────────────────────────────

class TestDynAblRawHop0:
    def test_ko_index_uses_subq(self):
        assert _build(ABL, 0, hop_ko(), "", None) == "서울이 수도인 나라는?"

    def test_en_index_uses_subq_en(self):
        assert _build(ABL, 0, hop_en(), "", None) == "What country is Beijing in?"

    def test_no_prev_answer_returns_template(self):
        assert _build(ABL, 1, hop_en(), "", prev_en()) == "What country is Beijing in?"


class TestDynAblRawHop1:
    def test_ko_index_bridge_label_substitution(self):
        h = hop_ko(sub_q="서울의 인구는 얼마인가?", bridge_label="서울")
        result = _build(ABL, 1, h, "부산", prev_ko("서울"))
        assert result == "부산의 인구는 얼마인가?"

    def test_en_index_uses_subq_en_with_en_bridge_label(self):
        # pattern d: hop0=EN, hop1=EN, sub_q_en_bridge_label="Beijing"
        h = hop_en(sub_q_en="What country is Beijing in?", en_bridge_label="Beijing")
        result = _build(ABL, 1, h, "Tokyo", prev_en("Beijing"))
        assert result == "What country is Tokyo in?"

    def test_en_index_pattern_c_uses_en_bridge_label(self):
        # pattern c: hop0=KO, hop1=EN, en_bridge_label != sub_q_bridge_label
        h = hop_en(
            sub_q="야마다 요지의 출생지는?",
            sub_q_en="Where was Yoji Yamada born?",
            bridge_label="야마다 요지",
            en_bridge_label="Yoji Yamada",
        )
        result = _build(ABL, 1, h, "야마다 요지", prev_ko("야마다 요지"))
        assert result == "Where was 야마다 요지 born?"

    def test_en_index_fallback_to_prev_gold_when_no_en_bridge_label(self):
        # Hugh Jackman 케이스: sub_q_en_bridge_label 없음, prev_gold(EN)이 sub_q_en에 존재
        h = {
            "gold_lang": "en",
            "sub_q": "휴 잭맨의 출생지는?",
            "sub_q_en": "Where was Hugh Jackman born?",
            "sub_q_bridge_label": "휴 잭맨",
            "sub_q_en_bridge_label": None,
            "gold_entity": "Australia",
        }
        result = _build(ABL, 1, h, "Hugh Grant", prev_en("Hugh Jackman"))
        assert result == "Where was Hugh Grant born?"

    def test_en_index_fallback_returns_template_when_nothing_matches(self):
        h = {
            "gold_lang": "en",
            "sub_q": "...",
            "sub_q_en": "What country is X in?",
            "sub_q_bridge_label": "...",
            "sub_q_en_bridge_label": None,
            "gold_entity": "Y",
        }
        result = _build(ABL, 1, h, "Tokyo", {"gold_entity": "Z"})
        assert result == "What country is X in?"


# ── _normalize_bridge ─────────────────────────────────────────────────────────

class TestNormalizeBridge:
    def _hop(self, ko="베이징시", en="Beijing"):
        return {"sub_q_bridge_label": ko, "sub_q_en_bridge_label": en}

    def test_ko_input_to_en_canonical(self):
        text, status, basis = _normalize("베이징시", self._hop(), "en")
        assert text == "Beijing"
        assert status == "success_gold_bridge_alias"
        assert basis == "prev_answer_alias_match"

    def test_en_input_to_ko_canonical(self):
        text, status, _ = _normalize("Beijing", self._hop(), "ko")
        assert text == "베이징시"
        assert status == "success_gold_bridge_alias"

    def test_case_insensitive_match(self):
        text, status, _ = _normalize("beijing", self._hop(), "ko")
        assert text == "베이징시"
        assert status == "success_gold_bridge_alias"

        text2, status2, _ = _normalize("BEIJING", self._hop(), "en")
        assert text2 == "Beijing"
        assert status2 == "success_gold_bridge_alias"

    def test_no_match_returns_raw(self):
        text, status, basis = _normalize("도쿄", self._hop(), "en")
        assert text == "도쿄"
        assert status == "failed_unlinked"
        assert basis is None

        text2, status2, _ = _normalize("London", self._hop(), "ko")
        assert text2 == "London"
        assert status2 == "failed_unlinked"

    def test_empty_prev_answer_returns_skipped(self):
        text, status, basis = _normalize("", self._hop(), "en")
        assert text == ""
        assert status == "skipped_abstain"
        assert basis is None


# ── dynamic_subq_norm (dyn_abl_norm) ─────────────────────────────────────────

class TestDynAblNormHop0:
    def test_ko_index_uses_subq(self):
        # hop0: identical to dyn_abl_raw
        assert _build(NORM, 0, hop_ko(), "", None) == "서울이 수도인 나라는?"

    def test_en_index_uses_subq_en(self):
        assert _build(NORM, 0, hop_en(), "", None) == "What country is Beijing in?"


class TestDynAblNormHop1:
    def test_en_index_alias_match_normalizes_ko_to_en(self):
        # pattern c/d: prev_answer가 KO label → EN canonical로 정규화
        h = hop_en(sub_q_en="What country is Beijing in?", en_bridge_label="Beijing")
        result = _build(NORM, 1, h, "베이징시", prev_ko("베이징시"))
        assert result == "What country is Beijing in?"

    def test_en_index_alias_match_normalizes_en_to_en(self):
        # prev_answer가 이미 EN label → 그대로 사용
        h = hop_en(sub_q_en="What country is Beijing in?", en_bridge_label="Beijing")
        result = _build(NORM, 1, h, "Beijing", prev_en("Beijing"))
        assert result == "What country is Beijing in?"

    def test_en_index_alias_no_match_falls_back_to_raw(self):
        # hallucination 케이스: alias match 실패 → raw prev_answer 그대로
        h = hop_en(sub_q_en="What country is Beijing in?", en_bridge_label="Beijing")
        result = _build(NORM, 1, h, "도쿄", prev_en("Beijing"))
        assert result == "What country is 도쿄 in?"

    def test_ko_index_alias_match_normalizes_en_to_ko(self):
        # KO-index hop1: prev_answer가 EN label → KO canonical로 정규화
        h = hop_ko(sub_q="서울의 인구는 얼마인가?", bridge_label="서울")
        result = _build(NORM, 1, h, "Seoul", prev_en("Seoul"))
        assert result == "서울의 인구는 얼마인가?"

    def test_ko_index_alias_no_match_falls_back_to_raw(self):
        h = hop_ko(sub_q="서울의 인구는 얼마인가?", bridge_label="서울")
        result = _build(NORM, 1, h, "런던", prev_en("Seoul"))
        assert result == "런던의 인구는 얼마인가?"
