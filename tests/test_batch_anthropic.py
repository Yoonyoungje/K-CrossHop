"""batch_anthropic.py 순수 함수 테스트 (비용 0)."""
import os
import sys
import types
import importlib.util

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location(
    "batch_anthropic", os.path.join(ROOT, "pipeline/batch_anthropic.py"))
ba = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ba)

_RUNNER = ba.load_runner()


def test_make_batch_request_uses_build_claude_kwargs():
    req = ba.make_batch_request("kch_x-0-1", "claude-sonnet-4-6", "SYS", "PROMPT", 256,
                                temperature=0.0, runner=_RUNNER)
    assert req["custom_id"] == "kch_x-0-1"
    p = req["params"]
    assert p["model"] == "claude-sonnet-4-6"
    assert p["max_tokens"] == 256
    assert p["system"] == "SYS"
    assert p["messages"] == [{"role": "user", "content": "PROMPT"}]
    assert p["temperature"] == 0.0


def test_make_batch_request_rejects_long_custom_id():
    with pytest.raises(ValueError):
        ba.make_batch_request("x" * 65, "claude-sonnet-4-6", "S", "P", 256, runner=_RUNNER)


def test_make_batch_request_rejects_pipe_custom_id():
    # Anthropic 패턴 ^[a-zA-Z0-9_-]$ 위반('|') → 로컬에서 fail-fast (400 회귀 방지)
    with pytest.raises(ValueError):
        ba.make_batch_request("kch_x|0|1", "claude-sonnet-4-6", "S", "P", 256, runner=_RUNNER)


def _fake_result(cid, rtype, text="답변: 서울", in_tok=10, out_tok=2):
    usage = types.SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok)
    block = types.SimpleNamespace(text=text)
    msg = types.SimpleNamespace(content=[block], usage=usage, stop_reason="end_turn")
    result = types.SimpleNamespace(type=rtype, message=msg)
    return types.SimpleNamespace(custom_id=cid, result=result)


def test_parse_result_succeeded():
    cid, text, err, usage = ba.parse_result(_fake_result("a-0-1", "succeeded"))
    assert cid == "a-0-1"
    assert text == "답변: 서울"
    assert err is False
    # finish_reason: 상한 도달 판정용 (동기 경로와 같은 키)
    assert usage == {"input_tokens": 10, "output_tokens": 2, "finish_reason": "end_turn"}


def test_parse_result_errored():
    cid, text, err, usage = ba.parse_result(_fake_result("a-0-1", "errored"))
    assert err is True and text == "" and usage is None


# ─── submit/poll/collect (디커플링) — 페이크 클라이언트 ──────────────────────────

class _FakeBatches:
    def __init__(self, status="ended", results=None):
        self._status = status
        self._results = results or []
        self.created = None

    def create(self, requests):
        self.created = requests
        return types.SimpleNamespace(id="batch_123")

    def retrieve(self, bid):
        counts = types.SimpleNamespace(succeeded=len(self._results), errored=0, processing=0)
        return types.SimpleNamespace(processing_status=self._status, request_counts=counts)

    def results(self, bid):
        return iter(self._results)


class _FakeClient:
    def __init__(self, batches):
        self.messages = types.SimpleNamespace(batches=batches)


def test_submit_returns_batch_id():
    fb = _FakeBatches()
    client = _FakeClient(fb)
    reqs = {"a-0-1": {"system_prompt": "S", "prompt": "P", "max_tokens": 64, "temperature": 0.0}}
    handle = ba.submit(client, "claude-sonnet-4-6", reqs, runner=_RUNNER)
    assert handle == {"batch_id": "batch_123"}
    assert fb.created[0]["custom_id"] == "a-0-1"      # build_claude_kwargs 경유 직렬화


def test_poll_done_when_ended():
    client = _FakeClient(_FakeBatches(status="ended"))
    done, status = ba.poll(client, {"batch_id": "batch_123"})
    assert done is True and "ended" in status


def test_poll_pending_when_in_progress():
    client = _FakeClient(_FakeBatches(status="in_progress"))
    done, _ = ba.poll(client, {"batch_id": "batch_123"})
    assert done is False


def test_collect_keys_by_custom_id():
    results = [_fake_result("a-0-1", "succeeded", text="답변: 서울"),
               _fake_result("b-1-1", "errored")]
    client = _FakeClient(_FakeBatches(results=results))
    parsed = ba.collect(client, {"batch_id": "batch_123"})
    assert parsed["a-0-1"] == ("답변: 서울", False,
                              {"input_tokens": 10, "output_tokens": 2, "finish_reason": "end_turn"})
    assert parsed["b-1-1"] == ("", True, None)
