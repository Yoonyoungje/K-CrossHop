"""
batch_openai.py 순수 함수 테스트 (비용 0).
make_batch_line(동기 body 재사용 + 봉투) / parse_output_line(동기 파싱 미러) 고정.
"""
import os
import sys
import json
import importlib.util

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "batch_openai", os.path.join(ROOT, "pipeline/batch_openai.py"))
bo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bo)

_RUNNER = bo.load_runner()   # 1회 로드 후 재사용


def test_make_batch_line_gpt5_envelope():
    line = bo.make_batch_line("kch_x-0-1", "gpt-5.4-2026-03-05", "SYS", "PROMPT", 256,
                              reasoning_effort="none", gen_seed=1, runner=_RUNNER)
    assert line["custom_id"] == "kch_x-0-1"
    assert line["method"] == "POST"
    assert line["url"] == "/v1/chat/completions"
    # body는 build_gpt_body 그대로 (byte-동일 보장)
    assert line["body"]["max_completion_tokens"] == 256
    assert line["body"]["reasoning_effort"] == "none"
    assert line["body"]["seed"] == 1
    assert "temperature" not in line["body"]


def test_make_batch_line_rejects_long_custom_id():
    with pytest.raises(ValueError):
        bo.make_batch_line("x" * 65, "gpt-5.4-2026-03-05", "S", "P", 256, runner=_RUNNER)


def test_parse_output_line_success():
    line = {"custom_id": "a-0-1", "error": None,
            "response": {"status_code": 200,
                         "body": {"choices": [{"message": {"content": "답변: 서울"},
                                                "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 10, "completion_tokens": 2}}}}
    cid, text, err, usage = bo.parse_output_line(line)
    assert cid == "a-0-1"
    assert text == "답변: 서울"
    assert err is False
    assert usage == {"prompt_tokens": 10, "completion_tokens": 2, "finish_reason": "stop"}


def test_parse_output_line_error_field():
    line = {"custom_id": "a-0-1", "error": {"message": "boom"}, "response": None}
    cid, text, err, usage = bo.parse_output_line(line)
    assert err is True and text == "" and usage is None


def test_parse_output_line_non_200():
    line = {"custom_id": "a-0-1", "error": None,
            "response": {"status_code": 500, "body": {}}}
    _, _, err, _ = bo.parse_output_line(line)
    assert err is True


# ─── poll/collect (디커플링) — get_batch/download_file 몽키패치 (HTTP 없이) ───────

def test_poll_done_when_completed(monkeypatch):
    monkeypatch.setattr(bo, "get_batch", lambda bid, key: {
        "status": "completed", "request_counts": {"completed": 3, "total": 3}})
    done, status = bo.poll({"batch_id": "batch_123"}, "key")
    assert done is True and "completed 3/3" in status


def test_poll_pending_when_in_progress(monkeypatch):
    monkeypatch.setattr(bo, "get_batch", lambda bid, key: {
        "status": "in_progress", "request_counts": {"completed": 1, "total": 3}})
    done, _ = bo.poll({"batch_id": "batch_123"}, "key")
    assert done is False


def test_collect_parses_output(monkeypatch):
    monkeypatch.setattr(bo, "get_batch", lambda bid, key: {
        "status": "completed", "output_file_id": "file_out"})
    out_lines = "\n".join([
        json.dumps({"custom_id": "a-0-1", "error": None,
                    "response": {"status_code": 200,
                                 "body": {"choices": [{"message": {"content": "답변: 서울"}}],
                                          "usage": {"prompt_tokens": 5, "completion_tokens": 1}}}}),
        json.dumps({"custom_id": "b-1-1", "error": {"message": "x"}, "response": None}),
    ])
    monkeypatch.setattr(bo, "download_file", lambda fid, key: out_lines)
    parsed = bo.collect({"batch_id": "batch_123"}, "key")
    assert parsed["a-0-1"][0] == "답변: 서울" and parsed["a-0-1"][1] is False
    assert parsed["b-1-1"][1] is True


def test_collect_raises_when_not_completed(monkeypatch):
    monkeypatch.setattr(bo, "get_batch", lambda bid, key: {"status": "failed"})
    with pytest.raises(RuntimeError):
        bo.collect({"batch_id": "batch_123"}, "key")
