"""batch_gemini.py 순수 함수 테스트 (비용 0)."""
import os
import sys
import types
import importlib.util

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location(
    "batch_gemini", os.path.join(ROOT, "pipeline/batch_gemini.py"))
bg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bg)


def test_make_inline_request_thinking_budget_zero():
    r = bg.make_inline_request("SYS", "PROMPT", 256)
    assert r["contents"] == [{"parts": [{"text": "PROMPT"}], "role": "user"}]
    cfg = r["config"]
    assert cfg["system_instruction"] == {"parts": [{"text": "SYS"}]}
    assert cfg["max_output_tokens"] == 256
    assert cfg["thinking_config"] == {"thinking_budget": 0}
    # 미지정 시 temp/top_p/seed 미첨부 (동기 분기와 동일 조건)
    assert "temperature" not in cfg and "top_p" not in cfg and "seed" not in cfg


def test_make_inline_request_conditional_sampling():
    r = bg.make_inline_request("S", "P", 128, temperature=0.0, top_p=0.9, gen_seed=7)
    cfg = r["config"]
    assert cfg["temperature"] == 0.0   # 0.0도 첨부 (is not None)
    assert cfg["top_p"] == 0.9
    assert cfg["seed"] == 7


def _fake_inline(text="답변: 서울", in_tok=10, out_tok=2, has_response=True, cid=None):
    """cid 지정 시 응답 metadata에 echo (실제 InlinedResponse.metadata 모사)."""
    metadata = {"cid": cid} if cid is not None else None
    if not has_response:
        return types.SimpleNamespace(response=None, metadata=metadata)
    um = types.SimpleNamespace(prompt_token_count=in_tok, candidates_token_count=out_tok,
                               thoughts_token_count=0)
    cand = types.SimpleNamespace(finish_reason="STOP")
    resp = types.SimpleNamespace(text=text, usage_metadata=um, candidates=[cand])
    return types.SimpleNamespace(response=resp, metadata=metadata)


def test_parse_inline_response_ok():
    text, err, usage = bg.parse_inline_response(_fake_inline())
    assert text == "답변: 서울"
    assert err is False
    assert usage == {"input_tokens": 10, "output_tokens": 2, "thoughts_tokens": 0,
                     "finish_reason": "STOP"}


def test_parse_inline_response_no_response():
    text, err, usage = bg.parse_inline_response(_fake_inline(has_response=False))
    assert err is True and text == "" and usage is None


# ─── submit/poll/collect (디커플링) — custom_id 미지원이라 metadata.cid 역매핑이 핵심 ──────

class _FakeBatchJob:
    def __init__(self, name, state, responses=None):
        self.name = name
        self.state = types.SimpleNamespace(name=state)
        self.dest = types.SimpleNamespace(inlined_responses=responses or [])


class _FakeBatches:
    def __init__(self, state="JOB_STATE_SUCCEEDED", responses=None):
        self._state = state
        self._responses = responses or []
        self.created_src = None

    def create(self, model, src, config):
        self.created_src = src
        return _FakeBatchJob("jobs/abc", "JOB_STATE_PENDING")

    def get(self, name):
        return _FakeBatchJob(name, self._state, self._responses)


class _FakeClient:
    def __init__(self, batches):
        self.batches = batches


def test_submit_embeds_cid_metadata():
    fb = _FakeBatches()
    client = _FakeClient(fb)
    reqs = {"a-0-1": {"system_prompt": "S", "prompt": "P1", "max_tokens": 64},
            "b-0-1": {"system_prompt": "S", "prompt": "P2", "max_tokens": 64}}
    handle = bg.submit(client, "gemini-3.5-flash", reqs)
    assert handle["job_name"] == "jobs/abc"
    assert handle["cids"] == ["a-0-1", "b-0-1"]            # 기대 집합(개수/누락 검증용)
    # cid가 각 요청 metadata에 심어져야 응답에서 역매핑 가능
    assert fb.created_src[0]["metadata"] == {"cid": "a-0-1"}
    assert fb.created_src[1]["metadata"] == {"cid": "b-0-1"}
    assert fb.created_src[0]["contents"][0]["parts"][0]["text"] == "P1"


def test_poll_done_when_succeeded():
    client = _FakeClient(_FakeBatches(state="JOB_STATE_SUCCEEDED"))
    done, status = bg.poll(client, {"job_name": "jobs/abc"})
    assert done is True and status == "JOB_STATE_SUCCEEDED"


def test_poll_pending_when_running():
    client = _FakeClient(_FakeBatches(state="JOB_STATE_RUNNING"))
    done, _ = bg.poll(client, {"job_name": "jobs/abc"})
    assert done is False


def test_collect_maps_by_metadata_cid():
    responses = [_fake_inline(text="답변: 서울", cid="a-0-1"),
                 _fake_inline(text="답변: 부산", cid="b-0-1")]
    client = _FakeClient(_FakeBatches(responses=responses))
    parsed = bg.collect(client, {"job_name": "jobs/abc", "cids": ["a-0-1", "b-0-1"]})
    assert parsed["a-0-1"][0] == "답변: 서울"
    assert parsed["b-0-1"][0] == "답변: 부산"


def test_collect_is_order_independent():
    """회귀: 응답이 제출 역순으로 와도 metadata.cid로 정확히 매핑 (1000 스케일 스크램블 방지)."""
    responses = [_fake_inline(text="답변: 부산", cid="b-0-1"),   # 역순
                 _fake_inline(text="답변: 서울", cid="a-0-1")]
    client = _FakeClient(_FakeBatches(responses=responses))
    parsed = bg.collect(client, {"job_name": "jobs/abc", "cids": ["a-0-1", "b-0-1"]})
    assert parsed["a-0-1"][0] == "답변: 서울"             # 순서가 아니라 cid 기준
    assert parsed["b-0-1"][0] == "답변: 부산"


def test_collect_raises_when_cid_metadata_missing():
    """cid echo가 안 되면 순서 역매핑은 스크램블 위험 → 즉시 예외(조용한 fallback 금지)."""
    responses = [_fake_inline(cid=None), _fake_inline(cid=None)]
    client = _FakeClient(_FakeBatches(responses=responses))
    try:
        bg.collect(client, {"job_name": "jobs/abc", "cids": ["a-0-1", "b-0-1"]})
        assert False, "cid metadata 없으면 예외 발생해야 함"
    except RuntimeError:
        pass


def test_collect_raises_on_count_mismatch():
    responses = [_fake_inline(cid="a-0-1")]               # 1개인데 cids 2개
    client = _FakeClient(_FakeBatches(responses=responses))
    try:
        bg.collect(client, {"job_name": "jobs/abc", "cids": ["a-0-1", "b-0-1"]})
        assert False, "count mismatch에서 예외 발생해야 함"
    except RuntimeError:
        pass
