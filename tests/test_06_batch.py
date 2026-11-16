"""pipeline/06_annotate.py Batch 디커플링(submit/collect) 테스트 (비용 0 — 페이크 OpenAI 클라이언트).

검증 대상:
  - _collect_batch_requests: 결정적 pending/parsed 도출
  - submit_batch_annotation: batch_id를 {output}.batch.json 체크포인트에 영속화
  - collect_batch_annotation: 미완료 시 False (재실행 가능) / 완료 시 annotated 출력 작성
"""
import os
import sys
import json
import types
import asyncio
import importlib.util

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("annotate06", os.path.join(ROOT, "pipeline/06_annotate.py"))
annotate06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(annotate06)

LLMJudge = annotate06.LLMJudge


def _make_judge():
    """실제 LLMJudge(_PROMPT/_SYSTEM/_OPENAI_SEED 재사용)를 만들고 _openai만 페이크로 교체."""
    judge = LLMJudge(model="gpt-5.4-mini-2026-03-17", api_key="sk-test")
    return judge


def _trajs():
    def hop(h, lang, text):
        return {
            "hop": h, "gold_lang": lang, "retrieval_success": True,
            "retrieved_passages": [f"[{lang}] passage about entity {h}"],
            "retrieved_langs": [lang], "generated_text": text,
        }
    return [
        {"qid": "kch_000001", "pattern": "a", "model": "gpt-5.4",
         "trajectory": [hop(0, "ko", "답변: 서울"), hop(1, "ko", "답변: 부산")],
         "final_answer": "부산"},
        {"qid": "kch_000002", "pattern": "b", "model": "gpt-5.4",
         "trajectory": [hop(0, "en", "답변: Tokyo"), hop(1, "ko", "답변: 오사카")],
         "final_answer": "오사카"},
    ]


# ─── 페이크 AsyncOpenAI ────────────────────────────────────────────────

class _FakeFiles:
    def __init__(self, output_text):
        self._output_text = output_text
        self.uploaded = []

    async def create(self, file, purpose):
        self.uploaded.append(purpose)
        return types.SimpleNamespace(id="file_in_123")

    async def content(self, fid):
        return types.SimpleNamespace(text=self._output_text)


class _FakeBatches:
    def __init__(self, status="completed"):
        self._status = status
        self.create_args = None

    async def create(self, input_file_id, endpoint, completion_window):
        self.create_args = (input_file_id, endpoint, completion_window)
        return types.SimpleNamespace(id="batch_abc")

    async def retrieve(self, bid):
        rc = types.SimpleNamespace(completed=4, failed=0, total=4)
        return types.SimpleNamespace(id=bid, status=self._status,
                                     request_counts=rc, output_file_id="file_out_123")


class _FakeOpenAI:
    def __init__(self, output_text="", status="completed"):
        self.files = _FakeFiles(output_text)
        self.batches = _FakeBatches(status)


def _output_text_for(pending):
    """pending custom_id마다 supported 1개를 반환하는 batch 출력 JSONL 텍스트."""
    lines = []
    for req in pending:
        body = {"choices": [{"message": {"content": '[{"label": "supported"}]'}}]}
        lines.append(json.dumps({
            "custom_id": req["custom_id"],
            "response": {"status_code": 200, "body": body},
        }))
    return "\n".join(lines)


def test_collect_batch_requests_deterministic():
    judge = _make_judge()
    pending, parsed = annotate06._collect_batch_requests(_trajs(), judge, {})
    # 4홉 전부 claim+passage 보유 → pending 4개
    assert len(pending) == 4
    cids = {r["custom_id"] for r in pending}
    assert cids == {"kch_000001__hop0", "kch_000001__hop1",
                    "kch_000002__hop0", "kch_000002__hop1"}
    # body 계약 확인
    b = pending[0]["body"]
    assert b["model"] == judge.model
    assert b["max_completion_tokens"] == 1024
    assert b["seed"] == LLMJudge._OPENAI_SEED
    assert len(parsed) == 2


def test_submit_writes_checkpoint(tmp_path):
    judge = _make_judge()
    judge._openai = _FakeOpenAI()
    out = tmp_path / "annotated_gpt.jsonl"
    bid = asyncio.run(annotate06.submit_batch_annotation(_trajs(), judge, {}, out))
    assert bid == "batch_abc"
    ckpt = tmp_path / "annotated_gpt.batch.json"
    assert ckpt.exists()
    info = json.loads(ckpt.read_text())
    assert info["batch_id"] == "batch_abc"
    assert info["judge_model"] == judge.model
    assert info["n_pending"] == 4
    # 출력 파일은 아직 생성되지 않음 (제출만)
    assert not out.exists()
    # batch 생성 인자 계약
    assert judge._openai.batches.create_args[1] == "/v1/chat/completions"
    assert judge._openai.batches.create_args[2] == "24h"


def test_collect_pending_returns_false_no_output(tmp_path):
    judge = _make_judge()
    judge._openai = _FakeOpenAI(status="in_progress")
    out = tmp_path / "annotated_gpt.jsonl"
    annotate06._write_batch_ckpt(out, "batch_abc", judge.model, 4)
    ok = asyncio.run(annotate06.collect_batch_annotation(
        _trajs(), judge, {}, {}, out, save_rationale=False, wait=False))
    assert ok is False
    assert not out.exists()


def test_collect_completed_writes_output(tmp_path):
    judge = _make_judge()
    pending, _ = annotate06._collect_batch_requests(_trajs(), judge, {})
    judge._openai = _FakeOpenAI(output_text=_output_text_for(pending), status="completed")
    out = tmp_path / "annotated_gpt.jsonl"
    annotate06._write_batch_ckpt(out, "batch_abc", judge.model, len(pending))
    ok = asyncio.run(annotate06.collect_batch_annotation(
        _trajs(), judge, {}, {}, out, save_rationale=False, wait=False))
    assert ok is True
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(rows) == 2
    for r in rows:
        for h in r["hops"]:
            # 모든 claim supported → faithfulness 1.0, 비환각
            assert h["hop_faithfulness"] == 1.0
            assert h["has_hallucination"] is False
            assert all(c["label"] == "supported" for c in h["claims"])


def test_collect_missing_checkpoint_raises(tmp_path):
    judge = _make_judge()
    judge._openai = _FakeOpenAI()
    out = tmp_path / "nope.jsonl"
    with pytest.raises(FileNotFoundError):
        asyncio.run(annotate06.collect_batch_annotation(
            _trajs(), judge, {}, {}, out, save_rationale=False, wait=False))


# ─── 실패 요청 (에러 파일에만 실림, 예: 429 insufficient_quota) ─────────────

class _FakeChatCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **body):
        self.calls.append(body)
        msg = types.SimpleNamespace(content='[{"label": "supported"}]')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _openai_with_failed(pending, fail_cid):
    """fail_cid 만 에러 파일로 가고 나머지는 supported 로 끝난 batch."""
    fake = _FakeOpenAI(output_text=_output_text_for([r for r in pending if r["custom_id"] != fail_cid]))
    texts = {"file_out_123": fake.files._output_text,
             "file_err_123": json.dumps({"custom_id": fail_cid, "error": None, "response": {
                 "status_code": 429, "body": {"error": {"code": "insufficient_quota"}}}})}

    async def content(fid):
        return types.SimpleNamespace(text=texts[fid])

    async def retrieve(bid):
        rc = types.SimpleNamespace(completed=len(pending) - 1, failed=1, total=len(pending))
        return types.SimpleNamespace(id=bid, status="completed", request_counts=rc,
                                     output_file_id="file_out_123", error_file_id="file_err_123")

    fake.files.content = content
    fake.batches.retrieve = retrieve
    fake.chat = types.SimpleNamespace(completions=_FakeChatCompletions())
    return fake


def test_collect_failed_request_stops_without_output(tmp_path):
    """결과 없는 요청을 neutral('batch error') → faith 0 으로 조용히 채우지 않고 멈춘다."""
    judge = _make_judge()
    pending, _ = annotate06._collect_batch_requests(_trajs(), judge, {})
    judge._openai = _openai_with_failed(pending, "kch_000002__hop0")
    out = tmp_path / "annotated_gpt.jsonl"
    annotate06._write_batch_ckpt(out, "batch_abc", judge.model, len(pending))
    with pytest.raises(RuntimeError, match="kch_000002__hop0"):
        asyncio.run(annotate06.collect_batch_annotation(
            _trajs(), judge, {}, {}, out, save_rationale=False, wait=False))
    assert not out.exists()
    assert judge._openai.chat.completions.calls == []              # 과금 호출 없음


def test_collect_failed_request_retried_with_same_body(tmp_path):
    judge = _make_judge()
    pending, _ = annotate06._collect_batch_requests(_trajs(), judge, {})
    judge._openai = _openai_with_failed(pending, "kch_000002__hop0")
    out = tmp_path / "annotated_gpt.jsonl"
    annotate06._write_batch_ckpt(out, "batch_abc", judge.model, len(pending))
    ok = asyncio.run(annotate06.collect_batch_annotation(
        _trajs(), judge, {}, {}, out, save_rationale=False, wait=False, retry_failed=True))
    assert ok is True
    want = next(r["body"] for r in pending if r["custom_id"] == "kch_000002__hop0")
    assert judge._openai.chat.completions.calls == [want]           # 실패 요청만, 같은 본문으로
    rows = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert all(c["label"] == "supported" for r in rows for h in r["hops"] for c in h["claims"])
