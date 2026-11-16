"""
unit tests for pipeline/05_pilot_runner.py:
  - _select_prompts returns correct (sys, hop, max_tokens, temperature, top_p) per version
  - v2.2 and v22 alias to the same V2.2 constants
  - v1 and v2 remain unchanged
"""

import sys
import os
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("pilot05", os.path.join(ROOT, "pipeline/05_pilot_runner.py"))
pilot05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot05)

_select_prompts   = pilot05._select_prompts
SYSTEM_PROMPT_V1  = pilot05.SYSTEM_PROMPT_V1
HOP_PROMPT_V1     = pilot05.HOP_PROMPT_V1
SYSTEM_PROMPT_V2  = pilot05.SYSTEM_PROMPT_V2
HOP_PROMPT_V2     = pilot05.HOP_PROMPT_V2
SYSTEM_PROMPT_V22 = pilot05.SYSTEM_PROMPT_V22
HOP_PROMPT_V22    = pilot05.HOP_PROMPT_V22
SYSTEM_PROMPT_V23 = pilot05.SYSTEM_PROMPT_V23
HOP_PROMPT_V23    = pilot05.HOP_PROMPT_V23


def test_v1_prompts():
    sys_p, hop_t, max_tok, temp, top_p = _select_prompts("v1")
    assert sys_p is SYSTEM_PROMPT_V1
    assert hop_t is HOP_PROMPT_V1
    assert max_tok == 512
    assert temp is None
    assert top_p is None


def test_v2_prompts():
    sys_p, hop_t, max_tok, temp, top_p = _select_prompts("v2")
    assert sys_p is SYSTEM_PROMPT_V2
    assert hop_t is HOP_PROMPT_V2
    assert max_tok == 512
    assert temp is None
    assert top_p is None


def test_v22_prompts():
    sys_p, hop_t, max_tok, temp, top_p = _select_prompts("v22")
    assert sys_p is SYSTEM_PROMPT_V22
    assert hop_t is HOP_PROMPT_V22
    assert max_tok == 96
    assert temp == 0.0
    assert top_p == 1.0


def test_v22_dotted_alias():
    # "v2.2" and "v22" must resolve identically
    result_dot   = _select_prompts("v2.2")
    result_plain = _select_prompts("v22")
    assert result_dot == result_plain


def test_v22_sys_prompt_contains_key_rules():
    assert "답변:" in SYSTEM_PROMPT_V22
    assert "한 줄" in SYSTEM_PROMPT_V22
    assert "모르겠습니다" in SYSTEM_PROMPT_V22


def test_v22_hop_prompt_has_prev_answer_placeholder():
    assert "{prev_answer}" in HOP_PROMPT_V22
    assert "{passages}" in HOP_PROMPT_V22
    assert "{question}" in HOP_PROMPT_V22


def test_v1_unchanged():
    assert "생각:" in SYSTEM_PROMPT_V1
    assert "검색어:" in SYSTEM_PROMPT_V1
    assert "{hop_idx}" in HOP_PROMPT_V1


def test_unknown_version_defaults_to_v2():
    sys_p, hop_t, max_tok, _, _ = _select_prompts("unknown")
    assert sys_p is SYSTEM_PROMPT_V2
    assert hop_t is HOP_PROMPT_V2
    assert max_tok == 512


def test_v23_prompts():
    sys_p, hop_t, max_tok, temp, top_p = _select_prompts("v23")
    assert sys_p is SYSTEM_PROMPT_V23
    assert hop_t is HOP_PROMPT_V23
    assert max_tok == 128
    assert temp == 0.0
    assert top_p == 1.0


def test_v23_dotted_alias():
    result_dot   = _select_prompts("v2.3")
    result_plain = _select_prompts("v23")
    assert result_dot == result_plain


def test_v23_sys_prompt_characteristics():
    # 마커는 권장사항이므로 포함되어야 함
    assert "답변:" in SYSTEM_PROMPT_V23
    assert "모르겠습니다" in SYSTEM_PROMPT_V23
    # V2.2와 달리 한 줄 강제 없음
    assert "한 줄" not in SYSTEM_PROMPT_V23


def test_v23_hop_prompt_has_placeholders():
    assert "{passages}" in HOP_PROMPT_V23
    assert "{question}" in HOP_PROMPT_V23
    assert "{prev_answer}" in HOP_PROMPT_V23
