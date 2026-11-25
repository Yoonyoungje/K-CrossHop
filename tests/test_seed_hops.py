"""
hop0 공유(--seed-trajectories) 단위 테스트.

  - 같은 설정으로 만든 실행의 hop0 을 seed 로 넣으면, 궤적이 일반 실행과 byte-동일하다
    (run_metadata 의 출처 키 `seeded_hops_from` 만 추가된다)
  - 동기 · 배치 · 재개 세 경로가 같은 seed 로 같은 궤적을 만든다
  - seed 홉의 요청 문자열이 이 모드의 hop0 쿼리와 다르면 즉시 중단한다
  - prev_answer 는 06 파서 규칙(extract_prev_answer)으로 복원한다
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from tests.test_self_translate import (  # noqa: E402
    QAS, StubAdapter, StubRetriever, _dump, _run_sync, _stub_generate_batch, batch, runner,
)

TMPL = "dynamic_subq_qmatched_decoupled"
NORM = "dynamic_subq_norm_decoupled"


def _seed_from(trajs, upto_hop=0):
    return {qid: {h["hop"]: h for h in t["trajectory"] if h["hop"] <= upto_hop}
            for qid, t in trajs.items()}


def _strip_provenance(trajs):
    out = {}
    for qid, t in trajs.items():
        t = copy.deepcopy(t)
        t["run_metadata"] = {k: v for k, v in t["run_metadata"].items() if k != "seeded_hops_from"}
        out[qid] = t
    return out


def _run_seeded_sync(mode, seed_hops, seed_source="seed.jsonl"):
    def stub(question, hop_idx, total_hops, passages, prev_answer, api_key, model,
             local_endpoint=None, system_prompt="", hop_template="", **kw):
        return ("답변: Beijing" if hop_idx == 0 else "답변: STUB_h1"), False, None
    orig = runner.run_agent_hop
    runner.run_agent_hop = stub
    try:
        return {qa["qid"]: runner.run_pilot_trajectory(
            qa, StubRetriever(), "k", "stub-model", retrieval_mode=mode, prompt_version="v2.3",
            seed_hops=seed_hops, seed_source=seed_source) for qa in QAS}
    finally:
        runner.run_agent_hop = orig


def test_seeded_run_is_byte_identical_to_normal_run():
    """tmpl_dec 의 hop0 을 norm_dec 에 넣으면, norm_dec 를 그냥 돌린 것과 같은 궤적이 나온다.
    (두 조건의 hop0 요청은 문자열까지 같다 — 공유의 전제)"""
    tmpl = _run_sync(TMPL)
    normal = _run_sync(NORM)
    seeded = _run_seeded_sync(NORM, _seed_from(tmpl))
    assert _dump(_strip_provenance(seeded)) == _dump(normal)
    assert all(t["run_metadata"]["seeded_hops_from"] == "seed.jsonl" for t in seeded.values())
    assert "seeded_hops_from" not in normal[QAS[0]["qid"]]["run_metadata"]


def test_seeded_hop0_entry_is_reused_verbatim():
    tmpl = _run_sync(TMPL)
    seeded = _run_seeded_sync(NORM, _seed_from(tmpl))
    for qid, t in seeded.items():
        got, src = t["trajectory"][0], tmpl[qid]["trajectory"][0]
        assert got["retrieval_mode"] == NORM          # 조건 라벨만 현재 모드로
        assert {k: v for k, v in got.items() if k != "retrieval_mode"} == \
               {k: v for k, v in src.items() if k != "retrieval_mode"}


def test_seeded_prev_answer_follows_parser():
    tmpl = _run_sync(TMPL)
    seeded = _run_seeded_sync(NORM, _seed_from(tmpl))
    for qid, t in seeded.items():
        want = runner.extract_prev_answer(tmpl[qid]["trajectory"][0]["generated_text"])
        assert t["trajectory"][1]["prev_answer_raw"] == want


def test_seed_query_mismatch_raises():
    """controlled 의 hop0(KO sub_q)은 tmpl_dec 의 hop0 요청과 다르므로 seed 로 쓸 수 없다."""
    ctrl = _run_sync("dynamic_subq")
    try:
        _run_seeded_sync(TMPL, _seed_from(ctrl))
    except ValueError as e:
        assert "seed 홉 불일치" in str(e)
    else:
        raise AssertionError("불일치인데 예외가 나지 않았다")


def test_batch_and_resume_match_sync_when_seeded(tmp_path):
    tmpl = _run_sync(TMPL)
    seed = _seed_from(tmpl)
    sync = _dump(_run_seeded_sync(NORM, seed, seed_source="s.jsonl"))
    b = _dump(batch.run_batch_trajectories(
        QAS, StubRetriever(), _stub_generate_batch, "stub-model", retrieval_mode=NORM,
        prompt_version="v2.3", runner=runner, seed_hops=seed, seed_source="s.jsonl"))
    ckpt = tmp_path / "seeded.ckpt.json"
    trajs = None
    while trajs is None:
        trajs = batch.resume_batch_trajectories(
            ckpt, QAS, StubRetriever, StubAdapter(), "stub-model", retrieval_mode=NORM,
            prompt_version="v2.3", block=False, runner=runner, log=lambda *a: None,
            seed_hops=seed, seed_source="s.jsonl")
    assert sync == b == _dump(trajs)


def test_round_schedule_skips_seeded_hops():
    assert batch._round_schedule(NORM, 2, 1) == [(1, "gen")]
    assert batch._round_schedule("self_translate", 2, 1) == [(1, "rewrite"), (1, "gen")]


def _seed_via_loader(tmp_path, trajs):
    p = tmp_path / "t.jsonl"
    p.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in trajs.values()),
                 encoding="utf-8")
    return runner.load_seed_hops(p)


def test_load_seed_hops_carries_hop0_only(tmp_path):
    """hop1 까지 실으면 배치는 실행할 라운드가 사라지고, 동기는 쿼리가 우연히 같은 홉만
    재사용해 출처가 섞인다. 로더가 hop0 만 싣는 것이 그 방어선이다."""
    tmpl = _run_sync(TMPL)
    seed = _seed_via_loader(tmp_path, tmpl)
    assert set(seed) == set(tmpl)
    assert all(set(hops) == {0} for hops in seed.values())
    assert seed[QAS[0]["qid"]][0] == tmpl[QAS[0]["qid"]]["trajectory"][0]


def test_file_loaded_seed_runs_all_paths(tmp_path):
    """테스트 헬퍼가 아니라 실제 로더가 만든 seed 로도 세 경로가 끝까지 돌고 결과가 같다."""
    seed = _seed_via_loader(tmp_path, _run_sync(TMPL))
    sync = _dump(_run_seeded_sync(NORM, seed, seed_source="s.jsonl"))
    b = _dump(batch.run_batch_trajectories(
        QAS, StubRetriever(), _stub_generate_batch, "stub-model", retrieval_mode=NORM,
        prompt_version="v2.3", runner=runner, seed_hops=seed, seed_source="s.jsonl"))
    ckpt = tmp_path / "loaded.ckpt.json"
    trajs = None
    while trajs is None:
        trajs = batch.resume_batch_trajectories(
            ckpt, QAS, StubRetriever, StubAdapter(), "stub-model", retrieval_mode=NORM,
            prompt_version="v2.3", block=False, runner=runner, log=lambda *a: None,
            seed_hops=seed, seed_source="s.jsonl")
    assert sync == b == _dump(trajs)
    assert all(len(t["trajectory"]) == 2 for t in trajs.values())
