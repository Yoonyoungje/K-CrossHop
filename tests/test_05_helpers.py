"""
pipeline/05_pilot_runner.py 순수 함수(동기/배치 공유) 테스트.
배치 재구조화 시 프롬프트 포맷·prev_answer 파싱이 바뀌지 않도록 락.
"""
import os
import sys
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("runner05", os.path.join(ROOT, "pipeline/05_pilot_runner.py"))
runner05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner05)

build_prompt = runner05.build_prompt
extract_prev_answer = runner05.extract_prev_answer
build_gpt_body = runner05.build_gpt_body
build_claude_kwargs = runner05.build_claude_kwargs


def test_build_claude_kwargs_temperature_priority():
    k = build_claude_kwargs("claude-sonnet-4-6", "SYS", "P", 256, temperature=0.0, top_p=0.9)
    assert k["model"] == "claude-sonnet-4-6"
    assert k["max_tokens"] == 256
    assert k["system"] == "SYS"
    assert k["messages"] == [{"role": "user", "content": "P"}]
    assert k["temperature"] == 0.0
    assert "top_p" not in k   # temperature/top_p 동시 불가 → temperature 우선


def test_build_claude_kwargs_top_p_when_no_temperature():
    k = build_claude_kwargs("claude-sonnet-4-6", "S", "P", 128, top_p=0.9)
    assert k["top_p"] == 0.9 and "temperature" not in k


def test_build_claude_kwargs_no_sampling():
    k = build_claude_kwargs("claude-sonnet-4-6", "S", "P", 128)
    assert "temperature" not in k and "top_p" not in k


def test_build_gpt_body_gpt5_uses_completion_tokens_omits_temp():
    # gpt-5: max_completion_tokens + reasoning_effort + seed, temperature/top_p 미첨부
    b = build_gpt_body("gpt-5.4-2026-03-05", "SYS", "PROMPT", 256,
                       temperature=0.0, top_p=0.9, gen_seed=1, reasoning_effort="none")
    assert b["model"] == "gpt-5.4-2026-03-05"
    assert b["messages"] == [{"role": "system", "content": "SYS"},
                             {"role": "user", "content": "PROMPT"}]
    assert b["max_completion_tokens"] == 256
    assert b["reasoning_effort"] == "none"
    assert b["seed"] == 1
    assert "max_tokens" not in b and "temperature" not in b and "top_p" not in b


def test_build_gpt_body_non_gpt5_keeps_temp_omits_reasoning():
    b = build_gpt_body("gpt-4o", "SYS", "P", 128, temperature=0.0, top_p=0.9, gen_seed=2)
    assert b["max_tokens"] == 128
    assert b["temperature"] == 0.0   # 0.0도 첨부 (is not None)
    assert b["top_p"] == 0.9
    assert b["seed"] == 2
    assert "max_completion_tokens" not in b and "reasoning_effort" not in b


def test_build_prompt_exact_format():
    # hop_idx+1, 패시지 구분자, (없음) 기본값을 정확히 고정
    tmpl = "문서:\n{passages}\n\n질문:\n{question}\n\n이전:\n{prev_answer}\n[{hop_idx}/{total_hops}]"
    out = build_prompt(tmpl, "Q?", 0, 2, ["P1", "P2"], "")
    assert out == "문서:\nP1\n\n---\n\nP2\n\n질문:\nQ?\n\n이전:\n(없음)\n[1/2]"


def test_build_prompt_prev_answer_and_hop_index():
    tmpl = "{passages}|{question}|{prev_answer}|{hop_idx}"
    out = build_prompt(tmpl, "q", 1, 2, ["only"], "이전답")
    assert out == "only|q|이전답|2"   # hop_idx 1 → 표시 2, prev_answer 그대로


def test_extract_prev_answer_marker():
    assert extract_prev_answer("생각: ...\n답변: 서울특별시") == "서울특별시"


def test_extract_prev_answer_no_marker():
    # 마커 없으면 전체 strip (임의 절단 금지)
    assert extract_prev_answer("  경기도 용인시  ") == "경기도 용인시"


def test_extract_prev_answer_fullwidth_colon():
    assert extract_prev_answer("답변：모르겠습니다") == "모르겠습니다"
