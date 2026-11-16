#!/usr/bin/env python3
"""
pipeline/batch_anthropic.py
==============================
Anthropic Message Batches API 어댑터 (Claude 생성용, Phase 2b).

request params를 05의 `build_claude_kwargs`로 만들어 동기 Claude 분기와 **동일 구성**으로 유지.
custom_id 네이티브 지원 → 결과는 순서 무관, custom_id로 키잉(공식 권고).

순수 함수(make_batch_request / parse_result)는 단위 테스트로 고정(비용 0).
API 왕복(submit/poll/results)은 비용 발생 → 사용자 실행.
"""
import re
import sys
import time
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_TERMINAL = "ended"
_CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")   # Anthropic Batch custom_id 제약


def load_runner():
    spec = importlib.util.spec_from_file_location("runner05", ROOT / "pipeline/05_pilot_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── 순수 함수 (단위 테스트 대상) ──────────────────────────────────────────────

def make_batch_request(custom_id: str, model: str, system_prompt: str, prompt: str,
                       max_tokens: int, *, temperature=None, top_p=None, runner=None) -> dict:
    """배치 request 1건. params는 build_claude_kwargs 재사용 → 동기와 동일."""
    if not _CUSTOM_ID_RE.match(custom_id):
        raise ValueError(f"custom_id가 Anthropic 패턴 ^[a-zA-Z0-9_-]{{1,64}}$ 위반: {custom_id!r}")
    runner = runner or load_runner()
    params = runner.build_claude_kwargs(model, system_prompt, prompt, max_tokens,
                                        temperature, top_p)
    return {"custom_id": custom_id, "params": params}


def parse_result(r) -> tuple[str, str, bool, dict | None]:
    """배치 결과 1건(SDK 객체) → (custom_id, text, api_error, usage). 동기 Claude 파싱과 동일.

    result.type != 'succeeded'면 api_error=True (errored/canceled/expired)."""
    cid = r.custom_id
    if r.result.type != "succeeded":
        return cid, "", True, None
    msg = r.result.message
    text = msg.content[0].text
    usage = {"input_tokens": msg.usage.input_tokens,
             "output_tokens": msg.usage.output_tokens,
             "finish_reason": getattr(msg, "stop_reason", None)}   # 상한 도달 판정용 (동기 경로와 동일)
    return cid, text, False, usage


# ─── submit/poll/collect 분리 (디커플링: 제출↔수집 사이 프로세스 종료 허용) ──────
# handle은 JSON 직렬화 가능(배치 체크포인트에 영속).

def submit(client, model, requests: dict, *, runner=None) -> dict:
    """requests를 create → handle {batch_id}."""
    runner = runner or load_runner()
    batch_reqs = [make_batch_request(
        cid, model, r["system_prompt"], r["prompt"], r["max_tokens"],
        temperature=r.get("temperature"), top_p=r.get("top_p"), runner=runner)
        for cid, r in requests.items()]
    batch = client.messages.batches.create(requests=batch_reqs)
    return {"batch_id": batch.id}


def poll(client, handle: dict) -> tuple[bool, str]:
    """(종료 여부, 상태 문자열). 종료=processing_status=='ended'."""
    b = client.messages.batches.retrieve(handle["batch_id"])
    c = b.request_counts
    return (b.processing_status == _TERMINAL,
            f"{b.processing_status} ok={c.succeeded} err={c.errored} proc={c.processing}")


def collect(client, handle: dict) -> dict:
    """결과 → {cid: (text, api_error, usage)}. custom_id 키잉(순서 무관)."""
    parsed = {}
    for r in client.messages.batches.results(handle["batch_id"]):
        cid, text, err, usage = parse_result(r)
        parsed[cid] = (text, err, usage)
    return parsed


# ─── API 왕복 (비용 발생 → 사용자 실행) ────────────────────────────────────────

def submit_and_collect(client, model, requests: dict, *, poll_interval=30, runner=None) -> dict:
    """블로킹 편의: submit→poll(ended)→collect. 디커플링 경로는 submit/poll/collect 직접 사용."""
    handle = submit(client, model, requests, runner=runner)
    print(f"  Anthropic batch_id={handle['batch_id']}")
    while True:
        done, status = poll(client, handle)
        print(f"  [{status}]")
        if done:
            break
        time.sleep(poll_interval)
    parsed = collect(client, handle)
    missing = set(requests) - set(parsed)
    if missing:
        raise RuntimeError(f"배치 결과 누락 {len(missing)}건 (재제출 필요): {list(missing)[:5]}")
    return parsed
