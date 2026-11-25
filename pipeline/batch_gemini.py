#!/usr/bin/env python3
"""
pipeline/batch_gemini.py
===========================
Gemini Batch API 어댑터 (google-genai SDK, Phase 2b).

⚠ OpenAI/Anthropic과 달리 **custom_id 미지원**. inline 결과 순서는 **제출 순서가 보장되지 않는다**
(1000 스케일에서 순열 스크램블 발생 → 답변↔qid 오매핑 사고). 따라서 순서가 아니라
**요청 metadata에 cid를 심고 응답 metadata로 역매핑**한다(InlinedRequest.metadata ↔ InlinedResponse.metadata
echo). handle.cids는 기대 집합(개수/누락 검증) 용도로만 유지하며 역매핑 기준이 아니다.

inline config는 동기 Gemini 분기(05 run_agent_hop)와 **동일 의미**로 구성:
system_instruction + max_output_tokens + thinking_budget=0 (+ temp/top_p/seed 조건부).
(동기는 SDK 객체, 배치는 dict라 표현이 달라 build 함수 공유는 안 함 — 의미만 일치.)

순수 함수(make_inline_request / parse_inline_response)는 단위 테스트로 고정(비용 0).
실제 제출은 비용 발생 → 사용자 실행. 모델별 거동은 소규모 시험 배치로 검증.
"""
import time

_SUCCESS = "JOB_STATE_SUCCEEDED"
_TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


# ─── 순수 함수 (단위 테스트 대상) ──────────────────────────────────────────────

def make_inline_request(system_prompt: str, prompt: str, max_tokens: int,
                        *, temperature=None, top_p=None, gen_seed=None,
                        metadata: dict | None = None) -> dict:
    """Gemini inline batch request 1건. 동기 Gemini config와 동일 의미.

    thinking_budget=0 (옵션 ①). temp/top_p/seed는 지정 시에만 첨부(동기와 동일 조건).
    metadata 지정 시 InlinedRequest.metadata로 첨부 → 응답에 echo되어 순서 비의존 역매핑."""
    config: dict = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "max_output_tokens": max_tokens,
        "thinking_config": {"thinking_budget": 0},
    }
    if temperature is not None:
        config["temperature"] = temperature
    if top_p is not None:
        config["top_p"] = top_p
    if gen_seed is not None:
        config["seed"] = gen_seed
    req = {"contents": [{"parts": [{"text": prompt}], "role": "user"}], "config": config}
    if metadata is not None:
        req["metadata"] = metadata
    return req


def response_cid(inline_response) -> str | None:
    """응답 metadata에서 cid 추출 (dict/proto/namespace 모두 대응). 없으면 None."""
    md = getattr(inline_response, "metadata", None)
    if md is None:
        return None
    if isinstance(md, dict):
        return md.get("cid")
    return getattr(md, "cid", None)


def parse_inline_response(inline_response) -> tuple[str, bool, dict | None]:
    """inline 결과 1건 → (text, api_error, usage). response 없으면 api_error=True."""
    resp = getattr(inline_response, "response", None)
    if resp is None:
        return "", True, None
    text = resp.text or ""
    um = getattr(resp, "usage_metadata", None)
    cands = getattr(resp, "candidates", None) or []
    usage = {"input_tokens": um.prompt_token_count,
             "output_tokens": um.candidates_token_count,
             "thoughts_tokens": getattr(um, "thoughts_token_count", None)} if um else {}
    # 상한 도달 판정용 (동기 경로와 동일)
    usage["finish_reason"] = str(getattr(cands[0], "finish_reason", None)) if cands else None
    return text, False, usage


# ─── submit/poll/collect 분리 (디커플링: 제출↔수집 사이 프로세스 종료 허용) ──────
# ⚠ custom_id 미지원 → 요청 metadata에 cid를 심어 응답 metadata로 역매핑(순서 비의존).
#    handle.cids는 기대 집합(개수/누락 검증)용. handle은 JSON 직렬화 가능.

def submit(client, model, requests: dict) -> dict:
    """cid를 요청 metadata에 심어 제출 → handle {job_name, cids}. cids는 기대 집합."""
    cids = list(requests.keys())
    src = [make_inline_request(
        requests[cid]["system_prompt"], requests[cid]["prompt"], requests[cid]["max_tokens"],
        temperature=requests[cid].get("temperature"), top_p=requests[cid].get("top_p"),
        gen_seed=requests[cid].get("gen_seed"), metadata={"cid": cid})
        for cid in cids]
    job = client.batches.create(model=model, src=src, config={"display_name": "kcrosshop-frontier"})
    return {"job_name": job.name, "cids": cids}


def poll(client, handle: dict) -> tuple[bool, str]:
    """(종료 여부, 상태 문자열). 종료=_TERMINAL(SUCCEEDED/FAILED/CANCELLED/EXPIRED)."""
    job = client.batches.get(name=handle["job_name"])
    return job.state.name in _TERMINAL, job.state.name


def collect(client, handle: dict) -> dict:
    """완료 job → {cid: (text, api_error, usage)}. 응답 metadata.cid로 역매핑(순서 비의존).

    ⚠ 절대 제출 순서로 zip하지 않는다(과거 1000 스케일 순열 스크램블 사고). cid가 빠진 응답이
    하나라도 있으면 매핑 신뢰 불가로 즉시 예외 → 시험 배치에서 metadata echo를 먼저 검증할 것."""
    job = client.batches.get(name=handle["job_name"])
    if job.state.name != _SUCCESS:
        raise RuntimeError(f"Gemini batch 비정상 종료: {job.state.name}")
    responses = list(job.dest.inlined_responses)
    expected = set(handle["cids"])
    if len(responses) != len(expected):
        raise RuntimeError(f"Gemini 결과 수 불일치: {len(responses)} vs 요청 {len(expected)}")
    parsed = {}
    for inline_response in responses:
        cid = response_cid(inline_response)
        if cid is None:
            raise RuntimeError("Gemini 응답에 cid metadata 없음 → 순서 역매핑은 스크램블 위험. "
                               "시험 배치로 InlinedResponse.metadata echo부터 확인 필요.")
        if cid not in expected:
            raise RuntimeError(f"Gemini 응답 cid가 요청에 없음: {cid!r}")
        if cid in parsed:
            raise RuntimeError(f"Gemini 응답 cid 중복: {cid!r}")
        parsed[cid] = parse_inline_response(inline_response)
    if set(parsed) != expected:
        raise RuntimeError(f"Gemini cid 누락: 기대 {len(expected)} vs 매핑 {len(parsed)}")
    return parsed


# ─── API 왕복 (비용 발생 → 사용자 실행) ────────────────────────────────────────

def submit_and_collect(client, model, requests: dict, *, poll_interval=30) -> dict:
    """블로킹 편의: submit→poll(SUCCEEDED)→collect. 디커플링 경로는 submit/poll/collect 직접 사용."""
    handle = submit(client, model, requests)
    print(f"  Gemini batch job={handle['job_name']}")
    while True:
        done, status = poll(client, handle)
        print(f"  [{status}]")
        if done:
            break
        time.sleep(poll_interval)
    return collect(client, handle)
