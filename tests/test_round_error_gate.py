"""
배치 라운드 부분 실패 게이트 (05_batch_runner._check_round_errors) 단위 테스트. API 호출 0회.

Gemini 배치는 요청 단위로 실패해도 job 이 SUCCEEDED 로 끝난다(code=13 INTERNAL). cid 누락
검사로는 잡히지 않아, 수집 직후 api_error 비율로 막는다.

  - 임계 초과면 중단하고 궤적을 만들지 않는다 (동기형·재개형 두 경로 모두)
  - 임계 이하면 그대로 진행한다
  - 중단 시 체크포인트가 남아 사람이 원인을 본 뒤 결정할 수 있다
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.test_self_translate import (  # noqa: E402
    QAS, StubAdapter, StubRetriever, _stub_generate_batch, batch, runner,
)

MODE = "dynamic_subq"


def _results(n_total, n_err):
    out = {}
    for i in range(n_total):
        out[f"cid{i}"] = ("", True, None) if i < n_err else ("답변: x", False, {})
    return out


def test_gate_passes_when_no_error():
    batch._check_round_errors(0, _results(100, 0))


def test_gate_passes_at_or_below_threshold():
    logged = []
    batch._check_round_errors(0, _results(100, 5), logged.append)     # 5% — 임계 이하
    assert "5/100" in logged[0]


def test_gate_raises_above_threshold():
    with pytest.raises(RuntimeError) as e:
        batch._check_round_errors(1, _results(100, 6))                # 6% — 초과
    assert "api_error 6/100" in str(e.value)
    assert "라운드 1" in str(e.value)


def test_gate_message_reports_observed_failure_shape():
    """실제 사고(gemini/ctrl 618/1000)를 그대로 넣으면 비율과 함께 중단한다."""
    with pytest.raises(RuntimeError, match=r"618/1000건 \(61\.8%"):
        batch._check_round_errors(0, _results(1000, 618))


def _failing_batch(n_err):
    """앞 n_err 건만 api_error 로 돌려주는 generate_batch."""
    def gen(model, requests):
        ok = _stub_generate_batch(model, requests)
        for i, cid in enumerate(sorted(ok)):
            if i < n_err:
                ok[cid] = ("", True, None)
        return ok
    return gen


def test_blocking_path_stops_before_building_trajectories():
    with pytest.raises(RuntimeError, match="제공자 장애"):
        batch.run_batch_trajectories(QAS, StubRetriever(), _failing_batch(4), "stub-model",
                                     retrieval_mode=MODE, prompt_version="v2.3", runner=runner)


def test_resume_path_stops_and_keeps_checkpoint(tmp_path):
    class FailingAdapter(StubAdapter):
        def collect(self, handle):
            return {cid: ("", True, None) for cid in StubAdapter.store[handle["key"]]}

    ckpt = tmp_path / "gate.ckpt.json"
    with pytest.raises(RuntimeError, match="제공자 장애"):
        batch.resume_batch_trajectories(
            ckpt, QAS, StubRetriever, FailingAdapter(), "stub-model", retrieval_mode=MODE,
            prompt_version="v2.3", block=True, runner=runner, log=lambda *a: None)
    assert ckpt.exists(), "중단 시 체크포인트가 남아야 사람이 원인을 보고 결정할 수 있다"
    assert json.loads(ckpt.read_text(encoding="utf-8"))["round"] == 0
