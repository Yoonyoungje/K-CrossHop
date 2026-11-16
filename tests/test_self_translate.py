"""
생성 질문 분리·self-translate 조건 (pipeline/05_pilot_runner.py) 단위 테스트.

  dynamic_subq_qmatched_decoupled / dynamic_subq_norm_decoupled / self_translate
    - 생성 질문(agent_question)은 모든 hop 에서 controlled(dynamic_subq)와 같다
    - 검색 쿼리만 기반 모드(qmatched / norm) 또는 모델 재작성 결과로 바뀐다
    - self_translate 재작성 입력 = controlled 쿼리, 시스템 프롬프트 = 인덱스 언어 지정
    - 동기 · 배치 · 재개(체크포인트 왕복) 세 경로의 궤적이 byte-동일
    - 기존 모드의 run 메타는 변하지 않는다
"""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("pilot05_st", "pipeline/05_pilot_runner.py")
batch = _load("batch05_st", "pipeline/05_batch_runner.py")

NEW_MODES = ["dynamic_subq_qmatched_decoupled", "dynamic_subq_norm_decoupled", "self_translate"]


def _qa(qid, pattern, h0_lang, h1_lang):
    return {
        "qid": qid, "pattern": pattern, "lang_sequence": ["ko", h0_lang, h1_lang],
        "hops": [
            {"hop": 0, "gold_lang": h0_lang, "gold_entity": "Beijing" if h0_lang == "en" else "베이징시",
             "sub_q": "올림픽 개최 도시는 어디인가?", "sub_q_en": "Which city hosted the Olympics?"},
            {"hop": 1, "gold_lang": h1_lang, "gold_entity": "China" if h1_lang == "en" else "중국",
             "sub_q": "베이징시가 속한 나라는?", "sub_q_en": "What country is Beijing in?",
             "sub_q_bridge_label": "베이징시", "sub_q_en_bridge_label": "Beijing"},
        ],
    }


QAS = [_qa("q_a", "a", "ko", "ko"), _qa("q_b", "b", "en", "ko"),
       _qa("q_c", "c", "ko", "en"), _qa("q_d", "d", "en", "en")]


class StubRetriever:
    def retrieve(self, query, lang, top_k=5):
        psgs = [f"PSG{i}|{lang}|{query}" for i in range(top_k)]
        return psgs, [lang] * len(psgs)


def _stub_text(system_prompt, prompt, hop_idx):
    """재작성 요청이면 목표 언어 표식 + 원문, 생성 요청이면 hop0 은 bridge 표면형(정규화 자극)."""
    if system_prompt.startswith("Rewrite"):
        target = "EN" if "in English" in system_prompt else "KO"
        return f"RW[{target}] {prompt}\n(extra line)"
    return "답변: Beijing" if hop_idx == 0 else "답변: STUB_h1"


def _run_sync(mode, calls=None):
    def stub(question, hop_idx, total_hops, passages, prev_answer, api_key, model,
             local_endpoint=None, system_prompt="", hop_template="", **kw):
        prompt = runner.build_prompt(hop_template, question, hop_idx, total_hops, passages, prev_answer)
        if calls is not None:
            calls.append({"hop": hop_idx, "system": system_prompt, "question": question})
        return _stub_text(system_prompt, prompt, hop_idx), False, None
    orig = runner.run_agent_hop
    runner.run_agent_hop = stub
    try:
        return {qa["qid"]: runner.run_pilot_trajectory(
            qa, StubRetriever(), "k", "stub-model", retrieval_mode=mode, prompt_version="v2.3")
            for qa in QAS}
    finally:
        runner.run_agent_hop = orig


def _stub_generate_batch(model, requests):
    return {cid: (_stub_text(r["system_prompt"], r["prompt"], int(cid.rsplit("-", 2)[1])), False, None)
            for cid, r in requests.items()}


class StubAdapter:
    store = {}

    def submit(self, model, requests):
        key = str(len(StubAdapter.store))
        StubAdapter.store[key] = _stub_generate_batch(model, requests)
        return {"key": key}

    def poll(self, handle):
        return True, "ended"

    def collect(self, handle):
        return StubAdapter.store[handle["key"]]


def _dump(trajs):
    return {k: json.dumps(v, ensure_ascii=False, sort_keys=False) for k, v in trajs.items()}


def test_self_translate_skips_ko_index_hops():
    """KO-index 홉: 재작성 호출 자체가 나가지 않는다 (호출 수로 확인)."""
    calls = []
    _run_sync("self_translate", calls)
    rw = [c for c in calls if c["system"].startswith("Rewrite")]
    n_en_hops = sum(1 for qa in QAS for h in qa["hops"] if h["gold_lang"] == "en")
    assert len(rw) == n_en_hops
    assert all("in English" in c["system"] for c in rw)


def test_generation_question_equals_controlled_and_query_follows_base():
    ctrl_calls = []
    _run_sync("dynamic_subq", ctrl_calls)
    ctrl_q = [c["question"] for c in ctrl_calls]
    for mode, base in [("dynamic_subq_qmatched_decoupled", "dynamic_subq_qmatched"),
                       ("dynamic_subq_norm_decoupled", "dynamic_subq_norm")]:
        calls = []
        new = _run_sync(mode, calls)
        base_trajs = _run_sync(base)
        assert [c["question"] for c in calls] == ctrl_q
        for qid in new:
            assert ([h["retrieval_query"] for h in new[qid]["trajectory"]]
                    == [h["retrieval_query"] for h in base_trajs[qid]["trajectory"]])
            for h in new[qid]["trajectory"]:
                assert h["generation_question"] in ctrl_q


def test_self_translate_rewrites_controlled_query_into_index_language():
    ctrl = _run_sync("dynamic_subq")
    calls = []
    st = _run_sync("self_translate", calls)
    gen_q = [c["question"] for c in calls if not c["system"].startswith("Rewrite")]
    ctrl_calls = []
    _run_sync("dynamic_subq", ctrl_calls)
    assert gen_q == [c["question"] for c in ctrl_calls]
    for qa in QAS:
        for h_st, h_ctrl, hop in zip(st[qa["qid"]]["trajectory"], ctrl[qa["qid"]]["trajectory"], qa["hops"]):
            assert h_st["self_translate_source_query"] == h_ctrl["retrieval_query"]
            assert h_st["retrieval_index_used"] == hop["gold_lang"]
            if hop["gold_lang"] == "en":                       # EN-index 홉만 재작성
                assert h_st["retrieval_query"] == f"RW[EN] {h_ctrl['retrieval_query']}"   # 첫 줄만
                assert h_st["self_translate_status"] == "ok"
                assert h_st["self_translate_n_lines"] == 2
                assert h_st["design_query_index_match"] is True
            else:                                              # KO-index 홉은 원문 그대로
                assert h_st["retrieval_query"] == h_ctrl["retrieval_query"]
                assert h_st["self_translate_status"] == "skipped_ko_index"
                assert h_st["self_translate_raw_output"] == ""


def test_apply_self_translate_fallbacks():
    hop = {"gold_lang": "en", "gold_entity": "China"}
    for raw, err, status in [("", False, "empty_fallback"), ("x", True, "api_error_fallback")]:
        state = {"query": "베이징시가 속한 나라는?", "query_template_field": "sub_q",
                 "query_template_lang": "ko"}
        q = runner.apply_self_translate(state, hop, raw, err, None)
        assert q == "베이징시가 속한 나라는?"
        assert state["self_translate_status"] == status
        assert state["design_query_index_match"] is False
    state = {"query": "베이징시가 속한 나라는?", "query_template_field": "sub_q", "query_template_lang": "ko"}
    q = runner.apply_self_translate(state, hop, "\n  What country is Beijing in China?  \nNote: x", False, None)
    assert q == "What country is Beijing in China?"
    assert state["query_contains_gold_answer"] is True
    assert state["actual_query_lang_detected"] == "en"


def test_self_translate_prompt_is_verbatim():
    """2차 절차 프롬프트 (요청 원문). {target_language} 자리만 채운다."""
    assert runner.build_self_translate_prompt("en") == (
        "Rewrite the following question in English so that it can be used as a search query over "
        "a English Wikipedia passage index. Write every entity name in its standard English form; "
        "translating or standardizing the name of an entity already in the question does not count "
        "as adding information. Do not answer the question, and do not add any entity or fact that "
        "is not in the question. If the question is already a suitable English query, or cannot be "
        "rewritten, output it unchanged. Output only the rewritten question on a single line.")


def test_sync_batch_resume_byte_identical(tmp_path):
    for mode in NEW_MODES:
        sync = _dump(_run_sync(mode))
        b = _dump(batch.run_batch_trajectories(QAS, StubRetriever(), _stub_generate_batch, "stub-model",
                                               retrieval_mode=mode, prompt_version="v2.3", runner=runner))
        ckpt = tmp_path / f"{mode}.ckpt.json"
        trajs = None
        while trajs is None:
            trajs = batch.resume_batch_trajectories(
                ckpt, QAS, StubRetriever, StubAdapter(), "stub-model", retrieval_mode=mode,
                prompt_version="v2.3", block=False, runner=runner, log=lambda *a: None)
        assert sync == b == _dump(trajs), mode


def test_round_schedule():
    assert batch._round_schedule("dynamic_subq", 2) == [(0, "gen"), (1, "gen")]
    assert batch._round_schedule("self_translate", 2) == [(0, "rewrite"), (0, "gen"), (1, "rewrite"), (1, "gen")]


def test_existing_mode_meta_unchanged():
    for mode in ("dynamic_subq", "dynamic_subq_qmatched", "dynamic_subq_norm"):
        assert "generation_question_policy" not in runner._TRAJ_MODE_META[mode]
    assert runner._TRAJ_MODE_META["self_translate"]["trajectory_mode"] == "dynamic_self_translate"
