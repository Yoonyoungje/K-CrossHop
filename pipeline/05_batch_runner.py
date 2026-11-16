#!/usr/bin/env python3
"""
pipeline/05_batch_runner.py
==============================
배치 오케스트레이터 (Phase 2). 동기 `run_pilot_trajectory`를 hop-레벨 배치로 재구조화한다.

구조: hop_idx 라운드마다 (모든 qid의 resolve_hop_query + retrieve + build_prompt) → 배치 1회 생성
→ extract_prev_answer로 다음 라운드 prev_answer 갱신 → build_traj_entry. 마지막에 assemble_trajectory.
모든 조립은 05의 순수함수 재사용 → 동기와 **byte-동일** (stub-골든 게이트로 증명, golden_capture_05 --check-batch).

retrieval(FAISS=GPU)과 generate_batch(API=비용)는 주입 의존성:
  - 운영: FAISSRetriever + make_openai_generate_batch(OpenAI Batch)
  - 게이트: StubRetriever + 결정적 stub (비용·GPU 0)

generate_batch(model, requests) 계약:
  requests: {custom_id: {system_prompt, prompt, max_tokens, reasoning_effort, gen_seed, temperature, top_p}}
  반환:     {custom_id: (text, api_error, usage)}
"""
import sys
import json
import time
import importlib.util
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_runner():
    spec = importlib.util.spec_from_file_location("runner05", ROOT / "pipeline/05_pilot_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_batch_module(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"pipeline/{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_batch_openai():
    return _load_batch_module("batch_openai")


# ─── 라운드 빌드/적용/조립 공유 헬퍼 (블로킹·재개 경로 공통 → 분기 간 거동 일치 보장) ──

def _init_state(qas, runner, seed_hops=None, retrieval_mode=None) -> dict:
    """qid → {prev_answer, entries, switches}. switches는 gold_lang에서 결정적 재계산 가능.

    seed_hops({qid: {hop: 엔트리}})가 오면 그 홉을 재생성하지 않고 그대로 채운 상태에서 시작한다
    (hop0 공유). prev_answer 는 동기 경로와 같은 파서로 복원한다."""
    state = {}
    for qa in qas:
        st = {"prev_answer": "", "entries": [],
              "switches": runner.compute_lang_switch([h["gold_lang"] for h in qa["hops"]])}
        for hop in range(len(qa["hops"])):
            e = (seed_hops or {}).get(qa["qid"], {}).get(hop)
            if e is None:
                break
            st["entries"].append({**e, "retrieval_mode": retrieval_mode})
            if not e.get("api_error"):
                st["prev_answer"] = runner.extract_prev_answer(e.get("generated_text", ""))
        state[qa["qid"]] = st
    return state


def _build_round(qbyid, state, hop_idx, retriever, *, retrieval_mode, prompt_version, sys_prompt,
                 hop_tmpl, max_tokens, temperature, top_p, gen_seed, reasoning_effort,
                 run_id, top_k, runner):
    """hop_idx 라운드의 (round_req, round_ctx). 동기 run_pilot_trajectory와 동일 순서."""
    round_req, round_ctx = {}, {}
    for qid, st in state.items():
        qa = qbyid[qid]
        if hop_idx >= len(qa["hops"]):
            continue
        hop_data = qa["hops"][hop_idx]
        lang_switch = st["switches"][hop_idx]
        q = runner.resolve_hop_query(retrieval_mode, qa, hop_idx, hop_data, st["prev_answer"])
        if q["hop_state"] is not None and q["hop_state"].get("self_translate_pending"):
            rw_text, rw_err, rw_usage = st.pop("rewrite")     # 직전 재작성 라운드 결과
            q["query"] = runner.apply_self_translate(q["hop_state"], hop_data,
                                                     rw_text, rw_err, rw_usage)
        query, lang, agent_question = q["query"], q["lang"], q["agent_question"]
        passages, retrieved_langs = retriever.retrieve(query, lang, top_k=top_k)
        if prompt_version == "v1":
            agent_question = qa["question"]
        prompt = runner.build_prompt(hop_tmpl, agent_question, hop_idx,
                                     len(qa["hops"]), passages, st["prev_answer"])
        cid = f"{qid}-{hop_idx}-{run_id}"   # 구분자 '-': Anthropic custom_id 패턴 ^[A-Za-z0-9_-]$ 충족
        round_req[cid] = {"system_prompt": sys_prompt, "prompt": prompt,
                          "max_tokens": max_tokens, "reasoning_effort": reasoning_effort,
                          "gen_seed": gen_seed, "temperature": temperature, "top_p": top_p}
        round_ctx[cid] = {"qid": qid, "hop_idx": hop_idx, "passages": passages,
                          "retrieved_langs": retrieved_langs, "lang_switch": lang_switch,
                          "hop_data": hop_data, "query": query, "lang": lang,
                          "hop_state": q["hop_state"], "bridge_entity": q["bridge_entity"],
                          "bridge_correct": q["bridge_correct"], "bridge_lang": q["bridge_lang"]}
    return round_req, round_ctx


def _build_rewrite_round(qbyid, state, hop_idx, retriever=None, *, retrieval_mode, max_tokens,
                         temperature, top_p, gen_seed, reasoning_effort, run_id, runner, **_):
    """self_translate 재작성 라운드: 검색 전에 controlled 쿼리를 인덱스 언어로 재작성 요청.
    동기 run_pilot_trajectory 와 같은 system/prompt/디코딩 파라미터. retriever 미사용."""
    round_req, round_ctx = {}, {}
    for qid, st in state.items():
        qa = qbyid[qid]
        if hop_idx >= len(qa["hops"]):
            continue
        hop_data = qa["hops"][hop_idx]
        q = runner.resolve_hop_query(retrieval_mode, qa, hop_idx, hop_data, st["prev_answer"])
        if not (q["hop_state"] or {}).get("self_translate_pending"):
            continue
        prompt = runner.build_prompt(runner.SELF_TRANSLATE_TEMPLATE, q["query"], hop_idx,
                                     len(qa["hops"]), [], "")
        cid = f"{qid}_rw-{hop_idx}-{run_id}"   # '-hop-run' 꼬리 유지 (stub·역매핑 호환)
        round_req[cid] = {"system_prompt": runner.build_self_translate_prompt(hop_data["gold_lang"]),
                          "prompt": prompt, "max_tokens": max_tokens,
                          "reasoning_effort": reasoning_effort, "gen_seed": gen_seed,
                          "temperature": temperature, "top_p": top_p}
        round_ctx[cid] = {"qid": qid, "hop_idx": hop_idx, "kind": "rewrite"}
    return round_req, round_ctx


def _round_schedule(retrieval_mode, n_hops, start_hop=0) -> list[tuple[int, str]]:
    """(hop_idx, kind) 라운드 순서. self_translate 만 hop 마다 재작성 라운드가 앞에 붙는다.
    start_hop 이전 홉은 seed 로 채워졌다고 보고 건너뛴다."""
    if retrieval_mode == "self_translate":
        return [(h, k) for h in range(start_hop, n_hops) for k in ("rewrite", "gen")]
    return [(h, "gen") for h in range(start_hop, n_hops)]


def _build_any_round(kind, qbyid, state, hop_idx, retriever, **bkw):
    builder = _build_rewrite_round if kind == "rewrite" else _build_round
    return builder(qbyid, state, hop_idx, retriever, **bkw)


def _apply_any_round(kind, state, round_ctx, results, retrieval_mode, runner):
    if kind == "rewrite":
        for cid, ctx in round_ctx.items():
            state[ctx["qid"]]["rewrite"] = list(results[cid])
        return
    _apply_round(state, round_ctx, results, retrieval_mode, runner)


MAX_ROUND_ERROR_RATE = 0.05   # 이 비율을 넘는 api_error 는 제공자 장애로 보고 중단한다


def _check_round_errors(round_label, results, log=None):
    """수집 결과의 api_error 비율이 임계를 넘으면 중단 — 부분 실패가 '완료' 파일로 굳는 것을 막는다.

    Gemini 배치는 요청 단위로 실패해도 job 은 SUCCEEDED 로 끝난다(code=13 INTERNAL 관측, 618/1000).
    cid 누락 검사로는 잡히지 않아 별도 게이트가 필요하다."""
    n_err = sum(1 for _, api_error, _ in results.values() if api_error)
    if not n_err:
        return
    rate = n_err / len(results)
    if rate > MAX_ROUND_ERROR_RATE:
        raise RuntimeError(
            f"라운드 {round_label} api_error {n_err}/{len(results)}건 ({rate:.1%} > "
            f"{MAX_ROUND_ERROR_RATE:.0%}) — 제공자 장애로 보고 중단한다. 궤적을 저장하지 않았다. "
            f"원인 확인 후 체크포인트를 지우고 이 조건을 다시 실행할 것.")
    if log:
        log(f"라운드 {round_label} api_error {n_err}/{len(results)}건 ({rate:.1%}, 임계 이하 — 계속)")


def _apply_round(state, round_ctx, results, retrieval_mode, runner):
    """배치 결과 → prev_answer 갱신 + traj entry 누적. round_ctx는 체크포인트 왕복 후일 수 있음."""
    for cid, ctx in round_ctx.items():
        gen_text, api_error, usage = results[cid]
        st = state[ctx["qid"]]
        if not api_error:
            st["prev_answer"] = runner.extract_prev_answer(gen_text)
        entry = runner.build_traj_entry(
            ctx["hop_idx"], ctx["passages"], ctx["retrieved_langs"], gen_text,
            ctx["lang_switch"], ctx["hop_data"], ctx["query"], ctx["lang"],
            retrieval_mode, api_error, ctx["hop_state"], ctx["bridge_entity"],
            ctx["bridge_correct"], ctx["bridge_lang"], usage)
        st["entries"].append(entry)


def _finalize(qbyid, state, model, retrieval_mode, runner, seed_source=None) -> dict:
    return {qid: runner.assemble_trajectory(qbyid[qid], st["entries"], st["prev_answer"],
                                            model, retrieval_mode, seed_source)
            for qid, st in state.items()}


def run_batch_trajectories(qas, retriever, generate_batch, model, *, run_id=1,
                           retrieval_mode="dynamic_subq", prompt_version="v2.3",
                           gen_seed=None, max_output_tokens=None, reasoning_effort=None,
                           top_k=5, runner=None, seed_hops=None, seed_source=None) -> dict:
    """qid → 최종 trajectory dict. 동기 run_pilot_trajectory와 동일 조립 (배치 across qid).

    generate_batch가 블로킹(제출+폴링+수집)인 경로. 디커플링은 resume_batch_trajectories 사용."""
    runner = runner or load_runner()
    sys_prompt, hop_tmpl, max_tokens, temperature, top_p = runner._select_prompts(prompt_version)
    if max_output_tokens is not None:
        max_tokens = max_output_tokens
    qbyid = {qa["qid"]: qa for qa in qas}
    state = _init_state(qas, runner, seed_hops, retrieval_mode)
    n_hops = max(len(qa["hops"]) for qa in qas)
    start_hop = min((len(st["entries"]) for st in state.values()), default=0)
    bkw = dict(retrieval_mode=retrieval_mode, prompt_version=prompt_version, sys_prompt=sys_prompt,
               hop_tmpl=hop_tmpl, max_tokens=max_tokens, temperature=temperature, top_p=top_p,
               gen_seed=gen_seed, reasoning_effort=reasoning_effort, run_id=run_id,
               top_k=top_k, runner=runner)
    for r_i, (hop_idx, kind) in enumerate(_round_schedule(retrieval_mode, n_hops, start_hop)):
        round_req, round_ctx = _build_any_round(kind, qbyid, state, hop_idx, retriever, **bkw)
        results = generate_batch(model, round_req)
        _check_round_errors(r_i, results)
        _apply_any_round(kind, state, round_ctx, results, retrieval_mode, runner)
    return _finalize(qbyid, state, model, retrieval_mode, runner, seed_source)


# ─── 재개형 오케스트레이터 (체크포인트 영속 → 배치 대기 중 프로세스 종료 허용) ──────

def _bkw_from_cfg(cfg, runner) -> dict:
    """체크포인트 cfg → _build_round 키워드. 재개 시 cfg가 라운드 빌드 파라미터의 권위."""
    sys_prompt, hop_tmpl, max_tokens, temperature, top_p = runner._select_prompts(cfg["prompt_version"])
    if cfg["max_output_tokens"] is not None:
        max_tokens = cfg["max_output_tokens"]
    return dict(retrieval_mode=cfg["retrieval_mode"], prompt_version=cfg["prompt_version"],
                sys_prompt=sys_prompt, hop_tmpl=hop_tmpl, max_tokens=max_tokens,
                temperature=temperature, top_p=top_p, gen_seed=cfg["gen_seed"],
                reasoning_effort=cfg["reasoning_effort"], run_id=cfg["run_id"],
                top_k=cfg["top_k"], runner=runner)


def _save_ckpt(ckpt_path, cfg, round_idx, handle, round_ctx, state, start_hop=0):
    """원자적 저장(tmp→replace): 라운드 경계 크래시에도 체크포인트 무손상. state는 prev_answer+entries만."""
    persist_state = {qid: {"prev_answer": st["prev_answer"], "entries": st["entries"],
                           **({"rewrite": st["rewrite"]} if "rewrite" in st else {})}
                     for qid, st in state.items()}
    payload = {"cfg": cfg, "round": round_idx, "handle": handle,
               "round_ctx": round_ctx, "state": persist_state, "start_hop": start_hop}
    tmp = ckpt_path.with_name(ckpt_path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ckpt_path)


def resume_batch_trajectories(ckpt_path, qas, retriever_factory, adapter, model, *, run_id=1,
                              retrieval_mode="dynamic_subq", prompt_version="v2.3", gen_seed=None,
                              max_output_tokens=None, reasoning_effort=None, top_k=5,
                              block=False, poll_interval=30, runner=None, log=print,
                              seed_hops=None, seed_source=None) -> dict | None:
    """체크포인트 기반 재개. 반환: 완료 시 {qid: traj}, 진행 중이면 None.

    라운드 경계(submit↔collect)에서 상태를 ckpt에 영속 → 배치 대기 중 프로세스 종료 후 같은
    명령으로 이어서 재개 가능. block=False(기본): 미완료 시 즉시 None(터미널 비점유). block=True:
    내부 폴링 루프로 끝까지 대기. retriever_factory는 라운드 빌드 시에만 호출(폴링-only 재개에서
    FAISS/GPU 로드 회피). adapter: .submit(model, requests)/.poll(handle)/.collect(handle)."""
    runner = runner or load_runner()
    qbyid = {qa["qid"]: qa for qa in qas}
    n_hops = max(len(qa["hops"]) for qa in qas)
    existed = ckpt_path.exists()

    if existed:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
        cfg = ckpt["cfg"]
        if cfg["model"] != model:
            raise RuntimeError(f"체크포인트 모델 불일치: ckpt={cfg['model']} vs 요청={model}")
        round_idx, handle, round_ctx = ckpt["round"], ckpt["handle"], ckpt["round_ctx"]
        state = {qid: {"prev_answer": s["prev_answer"], "entries": s["entries"],
                       **({"rewrite": s["rewrite"]} if "rewrite" in s else {}),
                       "switches": runner.compute_lang_switch(
                           [h["gold_lang"] for h in qbyid[qid]["hops"]])}
                 for qid, s in ckpt["state"].items()}
        log(f"체크포인트 재개: round={round_idx}/{len(_round_schedule(cfg['retrieval_mode'], n_hops))} "
            f"qids={len(state)} handle={handle}")
    else:
        cfg = {"model": model, "retrieval_mode": retrieval_mode, "prompt_version": prompt_version,
               "run_id": run_id, "gen_seed": gen_seed, "max_output_tokens": max_output_tokens,
               "reasoning_effort": reasoning_effort, "top_k": top_k, "seed_source": seed_source}
        round_idx, handle, round_ctx = 0, None, None
        state = _init_state(qas, runner, seed_hops, retrieval_mode)

    bkw = _bkw_from_cfg(cfg, runner)
    start_hop = ckpt.get("start_hop", 0) if existed else \
        min((len(st["entries"]) for st in state.values()), default=0)
    schedule = _round_schedule(cfg["retrieval_mode"], n_hops, start_hop)

    if not existed:
        hop0, kind0 = schedule[0]
        round_req, round_ctx = _build_any_round(kind0, qbyid, state, hop0, retriever_factory(), **bkw)
        handle = adapter.submit(cfg["model"], round_req)
        _save_ckpt(ckpt_path, cfg, 0, handle, round_ctx, state, start_hop)
        log(f"라운드 0 제출 (qids={len(state)}) handle={handle}")

    while True:
        done, status = adapter.poll(handle)
        log(f"라운드 {round_idx} [{status}]")
        if not done:
            if block:
                time.sleep(poll_interval)
                continue
            log("배치 처리 중 — 나중에 같은 --resume 명령으로 이어집니다 (터미널 종료 가능).")
            return None
        results = adapter.collect(handle)
        missing = set(round_ctx) - set(results)
        if missing:
            raise RuntimeError(f"라운드 {round_idx} 결과 누락 {len(missing)}건 (재제출 필요): "
                               f"{list(missing)[:5]}")
        _check_round_errors(round_idx, results, log)
        _apply_any_round(schedule[round_idx][1], state, round_ctx, results, cfg["retrieval_mode"], runner)
        round_idx += 1
        if round_idx >= len(schedule):
            ckpt_path.unlink(missing_ok=True)
            log("모든 라운드 완료 — 체크포인트 삭제.")
            return _finalize(qbyid, state, cfg["model"], cfg["retrieval_mode"], runner,
                             cfg.get("seed_source"))
        hop_i, kind = schedule[round_idx]
        round_req, round_ctx = _build_any_round(kind, qbyid, state, hop_i, retriever_factory(), **bkw)
        handle = adapter.submit(cfg["model"], round_req)
        _save_ckpt(ckpt_path, cfg, round_idx, handle, round_ctx, state, start_hop)
        log(f"라운드 {round_idx} 제출 handle={handle}")
        if not block:
            log("다음 라운드 제출 완료 — 나중에 같은 --resume 명령으로 이어집니다.")
            return None


def make_openai_generate_batch(api_key, *, window="24h", poll_interval=30,
                               workdir=None, batch_openai=None):
    """운영용 generate_batch (OpenAI Batch). 비용 발생 → 사용자 실행 경로.

    submit/poll/collect 합성(블로킹). 입력 jsonl은 submit이 업로드 후 삭제(누적 방지).
    """
    bo = batch_openai or _load_batch_openai()
    runner = load_runner()

    def generate_batch(model, requests: dict) -> dict:
        handle = bo.submit(model, requests, api_key, window=window, workdir=workdir, runner=runner)
        while True:
            done, status = bo.poll(handle, api_key)
            print(f"  [{status}]")
            if done:
                break
            time.sleep(poll_interval)
        parsed = bo.collect(handle, api_key)
        missing = set(requests) - set(parsed)
        if missing:
            raise RuntimeError(f"배치 결과 누락 {len(missing)}건 (재제출 필요): {list(missing)[:5]}")
        return parsed

    return generate_batch


def make_anthropic_generate_batch(api_key, *, poll_interval=30, batch_anthropic=None):
    """운영용 generate_batch (Anthropic Message Batches). 비용 발생 → 사용자 실행."""
    ba = batch_anthropic or _load_batch_module("batch_anthropic")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    runner = load_runner()

    def generate_batch(model, requests: dict) -> dict:
        return ba.submit_and_collect(client, model, requests,
                                     poll_interval=poll_interval, runner=runner)
    return generate_batch


def make_gemini_generate_batch(api_key, *, poll_interval=30, batch_gemini=None):
    """운영용 generate_batch (Gemini Batch API). 비용 발생 → 사용자 실행."""
    bg = batch_gemini or _load_batch_module("batch_gemini")
    from google import genai
    client = genai.Client(api_key=api_key)

    def generate_batch(model, requests: dict) -> dict:
        return bg.submit_and_collect(client, model, requests, poll_interval=poll_interval)
    return generate_batch


# ─── 디커플링용 어댑터 (.submit/.poll/.collect) ────────────────────────────────
# 재개형 오케스트레이터(resume_batch_trajectories)가 소비. 블로킹 factory와 같은 어댑터 모듈 재사용.

def make_openai_adapter(api_key, *, window="24h", workdir=None, batch_openai=None):
    bo = batch_openai or _load_batch_openai()
    runner = load_runner()
    return SimpleNamespace(
        submit=lambda model, requests: bo.submit(model, requests, api_key,
                                                  window=window, workdir=workdir, runner=runner),
        poll=lambda handle: bo.poll(handle, api_key),
        collect=lambda handle: bo.collect(handle, api_key))


def make_anthropic_adapter(api_key, *, batch_anthropic=None):
    ba = batch_anthropic or _load_batch_module("batch_anthropic")
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    runner = load_runner()
    return SimpleNamespace(
        submit=lambda model, requests: ba.submit(client, model, requests, runner=runner),
        poll=lambda handle: ba.poll(client, handle),
        collect=lambda handle: ba.collect(client, handle))


def make_gemini_adapter(api_key, *, batch_gemini=None):
    bg = batch_gemini or _load_batch_module("batch_gemini")
    from google import genai
    client = genai.Client(api_key=api_key)
    return SimpleNamespace(
        submit=lambda model, requests: bg.submit(client, model, requests),
        poll=lambda handle: bg.poll(client, handle),
        collect=lambda handle: bg.collect(client, handle))


# 모델 prefix → (env 키 변수, 블로킹 generate_batch 팩토리, 디커플링 어댑터 팩토리)
_PROVIDERS = {
    "gpt-":   ("OPENAI_API_KEY",    make_openai_generate_batch,    make_openai_adapter),
    "claude": ("ANTHROPIC_API_KEY", make_anthropic_generate_batch, make_anthropic_adapter),
    "gemini": ("GEMINI_API_KEY",    make_gemini_generate_batch,    make_gemini_adapter),
}


def _select_provider(model: str):
    for prefix, entry in _PROVIDERS.items():
        if model.startswith(prefix):
            return entry   # (env_var, blocking_factory, adapter_factory)
    return None, None, None


# ─── 드라이버 CLI (3-모델 batch, Phase 2 end-to-end) ──────────────────────────
# retrieval(FAISS=GPU) + 배치 생성(비용) 포함 → 사용자 실행.

def main():
    import os
    import argparse
    import random
    import logging
    from collections import defaultdict
    import yaml
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("batch_runner")

    ap = argparse.ArgumentParser(description="GPT 배치 트래젝토리 드라이버 (05 동기와 byte-동일 조립)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--agent-model", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-id", type=int, default=1, help="custom_id run 식별자 (run별 --seed와 함께 변경)")
    ap.add_argument("--output", default=None)
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--qa-pairs", default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--retrieval-mode", default=None,
                    choices=["gold_entity", "ko_subq", "sub_q",
                             "dynamic_subq", "dynamic_subq_qmatched", "dynamic_subq_norm",
                             "dynamic_subq_bridgenorm",
                             "dynamic_subq_qmatched_decoupled", "dynamic_subq_norm_decoupled",
                             "self_translate"])
    ap.add_argument("--seed-trajectories", default=None,
                    help="다른 실행의 trajectory JSONL. 거기 있는 홉(보통 hop0)은 재생성하지 않고 그대로 쓴다.")
    ap.add_argument("--prompt-version", default=None,
                    choices=["v1", "v2", "v2.2", "v22", "v2.3", "v23"])
    ap.add_argument("--per-pattern", type=int, default=None, help="패턴당 샘플 수 (검증엔 소량 권장)")
    ap.add_argument("--max-output-tokens", type=int, default=None)
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh"])
    ap.add_argument("--window", default="24h")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--resume", action="store_true",
                    help="디커플링 모드: 체크포인트({output}.ckpt.json) 기반 제출/재개. "
                         "배치 대기 중 터미널 종료 후 같은 명령으로 이어서 실행.")
    ap.add_argument("--block", action="store_true",
                    help="--resume과 함께: 내부 폴링으로 끝까지 대기(소규모/스모크). 미지정 시 단계별 종료.")
    args = ap.parse_args()

    runner = load_runner()
    cfg = yaml.safe_load(Path(args.config).read_text())
    pilot_cfg = cfg["pilot"]
    agent_model    = args.agent_model or pilot_cfg["agent_model"]
    seed           = args.seed if args.seed is not None else pilot_cfg["seed"]
    traj_file      = Path(args.output or cfg["output"]["trajectories"])
    top_k          = args.top_k          if args.top_k          is not None else pilot_cfg.get("top_k", 5)
    retrieval_mode = args.retrieval_mode if args.retrieval_mode is not None else pilot_cfg.get("retrieval_mode", "sub_q")
    prompt_version = args.prompt_version if args.prompt_version is not None else pilot_cfg.get("prompt_version", "v2.3")

    env_var, factory, adapter_factory = _select_provider(agent_model)
    if factory is None:
        raise SystemExit(f"지원 안 되는 모델: {agent_model} (gpt-/claude/gemini)")
    api_key = args.api_key or os.environ.get(env_var)
    if not api_key:
        raise SystemExit(f"{env_var} 필요 (--api-key 또는 .env)")
    if agent_model.startswith("gpt-5") and args.reasoning_effort is None:
        log.warning("gpt-5인데 --reasoning-effort 미지정 → API 기본값 적용. frontier 통제는 none 권장.")

    random.seed(seed)
    qa_file = Path(args.qa_pairs or cfg["output"]["qa_pairs"])
    traj_file.parent.mkdir(parents=True, exist_ok=True)

    done_qids: set[str] = set()
    if traj_file.exists():
        for line in traj_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_qids.add(json.loads(line)["qid"])
        if done_qids:
            log.info(f"Resume: 이미 완료된 {len(done_qids)}개 스킵")

    all_qa = [json.loads(l) for l in qa_file.read_text().splitlines() if l.strip()]
    by_pattern = defaultdict(list)
    for qa in all_qa:
        by_pattern[qa["pattern"]].append(qa)
    pilot_qas = []
    for pat in ["a", "b", "c", "d"]:
        pool = by_pattern.get(pat, [])
        sampled = random.sample(pool, min(args.per_pattern, len(pool))) if args.per_pattern else pool
        pilot_qas.extend(sampled)
    pilot_qas = [qa for qa in pilot_qas if qa["qid"] not in done_qids]
    log.info(f"총 {len(pilot_qas)}개 배치 실행 (run_id={args.run_id}, mode={retrieval_mode}, model={agent_model})")

    index_dir = Path(args.index_dir or cfg["output"]["faiss_index"])

    seed_hops = runner.load_seed_hops(args.seed_trajectories) if args.seed_trajectories else None
    if seed_hops:
        log.info(f"홉 재사용: {args.seed_trajectories} (qid {len(seed_hops)})")

    if args.resume:
        ckpt_path = Path(str(traj_file) + ".ckpt.json")
        if agent_model.startswith("gpt-"):
            adapter = adapter_factory(api_key, window=args.window, workdir=traj_file.parent)
        else:
            adapter = adapter_factory(api_key)
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
            target = set(ckpt["state"])
            pilot_qas = [qa for qa in all_qa if qa["qid"] in target]
            log.info(f"체크포인트 발견: {ckpt_path} (round={ckpt['round']}, qids={len(target)})")
        elif not pilot_qas:
            log.info("실행할 QA 없음"); return
        retriever_factory = lambda: runner.FAISSRetriever(index_dir)
        trajs = resume_batch_trajectories(
            ckpt_path, pilot_qas, retriever_factory, adapter, agent_model, run_id=args.run_id,
            retrieval_mode=retrieval_mode, prompt_version=prompt_version, gen_seed=seed,
            max_output_tokens=args.max_output_tokens, reasoning_effort=args.reasoning_effort,
            top_k=top_k, block=args.block, poll_interval=args.poll_interval,
            runner=runner, log=log.info, seed_hops=seed_hops,
            seed_source=args.seed_trajectories)
        if trajs is None:
            log.info("이번 단계 종료 (배치 진행 중). 같은 명령으로 나중에 재개하세요.")
            return
        with open(traj_file, "a", encoding="utf-8") as f:
            for qa in pilot_qas:
                f.write(json.dumps(trajs[qa["qid"]], ensure_ascii=False) + "\n")
        log.info(f"궤적 저장: {traj_file} ({len(pilot_qas)}개)")
        print(f"\n다음: python3 pipeline/06_annotate.py --trajectories {traj_file}")
        return

    if not pilot_qas:
        log.info("실행할 QA 없음"); return

    retriever = runner.FAISSRetriever(index_dir)
    if factory is make_openai_generate_batch:
        generate_batch = factory(api_key, window=args.window, poll_interval=args.poll_interval)
    else:
        generate_batch = factory(api_key, poll_interval=args.poll_interval)
    trajs = run_batch_trajectories(
        pilot_qas, retriever, generate_batch, agent_model, run_id=args.run_id,
        retrieval_mode=retrieval_mode, prompt_version=prompt_version, gen_seed=seed,
        max_output_tokens=args.max_output_tokens, reasoning_effort=args.reasoning_effort,
        top_k=top_k, runner=runner, seed_hops=seed_hops, seed_source=args.seed_trajectories)

    with open(traj_file, "a", encoding="utf-8") as f:
        for qa in pilot_qas:
            f.write(json.dumps(trajs[qa["qid"]], ensure_ascii=False) + "\n")
    log.info(f"궤적 저장: {traj_file} ({len(pilot_qas)}개)")
    print(f"\n다음: python3 pipeline/06_annotate.py --trajectories {traj_file}")


if __name__ == "__main__":
    main()
