#!/usr/bin/env python3
"""
pipeline/batch_openai.py
===========================
OpenAI Batch API 어댑터 (Phase 1, GPT 생성용).

배치 request line의 body를 05의 `build_gpt_body`로 만들어 **동기 경로와 byte-동일**하게 유지한다
(8B↔frontier 비교 성립 조건). 출력 파싱도 동기 `run_agent_hop`의 gpt 파싱과 동일 경로.

순수 함수(make_batch_line / parse_output_line)는 단위 테스트로 고정(비용 0).
API 왕복(submit/poll/download)은 비용 발생 → 사용자 실행(`--validate`).

흐름: build_input_jsonl → upload_input → create_batch → (poll) get_batch → download_file → parse.
custom_id = "qid|hop|run" (≤64자; OpenAI 제약).
"""
import os
import sys
import json
import time
import argparse
import importlib.util
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPENAI_BASE = "https://api.openai.com/v1"
_TERMINAL = {"completed", "failed", "expired", "cancelled"}


def load_runner():
    spec = importlib.util.spec_from_file_location("runner05", ROOT / "pipeline/05_pilot_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ─── 순수 함수 (단위 테스트 대상, 비용 0) ──────────────────────────────────────

def make_batch_line(custom_id: str, model: str, system_prompt: str, prompt: str,
                    max_tokens: int, *, reasoning_effort: str | None = None,
                    gen_seed: int | None = None, temperature: float | None = None,
                    top_p: float | None = None, runner=None) -> dict:
    """배치 request 1줄. body는 build_gpt_body 재사용 → 동기와 byte-동일."""
    if len(custom_id) > 64:
        raise ValueError(f"custom_id >64자 (OpenAI 제약): {custom_id!r}")
    runner = runner or load_runner()
    body = runner.build_gpt_body(model, system_prompt, prompt, max_tokens,
                                 temperature, top_p, gen_seed, reasoning_effort)
    return {"custom_id": custom_id, "method": "POST",
            "url": "/v1/chat/completions", "body": body}


def parse_output_line(line: dict) -> tuple[str, str, bool, dict | None]:
    """배치 출력 1줄 → (custom_id, text, api_error, usage). 동기 gpt 파싱과 동일.

    response.status_code != 200 또는 error 비어있지 않으면 api_error=True (집계 제외 대상)."""
    cid = line.get("custom_id")
    err = line.get("error")
    resp = line.get("response")
    if err is not None or resp is None or resp.get("status_code") != 200:
        return cid, "", True, None
    body = resp["body"]
    text = body["choices"][0]["message"]["content"]
    usage = dict(body.get("usage") or {})
    usage["finish_reason"] = body["choices"][0].get("finish_reason")   # 상한 도달 판정용
    return cid, text, False, usage


def write_input_jsonl(lines: list[dict], path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    return path


# ─── API 왕복 (비용 발생 → 사용자 실행) ────────────────────────────────────────

def upload_input(path: Path, api_key: str) -> str:
    with open(path, "rb") as f:
        resp = requests.post(
            f"{OPENAI_BASE}/files",
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (path.name, f, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()["id"]


def create_batch(input_file_id: str, api_key: str,
                 window: str = "24h", endpoint: str = "/v1/chat/completions") -> dict:
    resp = requests.post(
        f"{OPENAI_BASE}/batches",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input_file_id": input_file_id, "endpoint": endpoint,
              "completion_window": window},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_batch(batch_id: str, api_key: str) -> dict:
    resp = requests.get(f"{OPENAI_BASE}/batches/{batch_id}",
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_file(file_id: str, api_key: str) -> str:
    resp = requests.get(f"{OPENAI_BASE}/files/{file_id}/content",
                        headers={"Authorization": f"Bearer {api_key}"}, timeout=120)
    resp.raise_for_status()
    return resp.text


def poll_until_done(batch_id: str, api_key: str, interval: int = 30) -> dict:
    while True:
        b = get_batch(batch_id, api_key)
        counts = b.get("request_counts", {})
        print(f"  [{b['status']}] {counts.get('completed', 0)}/{counts.get('total', 0)} "
              f"(failed={counts.get('failed', 0)})")
        if b["status"] in _TERMINAL:
            return b
        time.sleep(interval)


# ─── submit/poll/collect 분리 (디커플링: 제출↔수집 사이 프로세스 종료 허용) ──────
# handle은 JSON 직렬화 가능(배치 체크포인트에 영속). 입력 jsonl은 업로드 후 즉시 삭제.

def submit(model: str, requests: dict, api_key: str, *, window: str = "24h",
           workdir=None, runner=None) -> dict:
    """requests 직렬화·업로드·배치 생성 → handle {batch_id}. 로컬 입력은 업로드 후 삭제."""
    runner = runner or load_runner()
    workdir = Path(workdir or (ROOT / "pipeline"))
    lines = [make_batch_line(
        cid, model, r["system_prompt"], r["prompt"], r["max_tokens"],
        reasoning_effort=r["reasoning_effort"], gen_seed=r["gen_seed"],
        temperature=r["temperature"], top_p=r["top_p"], runner=runner)
        for cid, r in requests.items()]
    inp = write_input_jsonl(lines, workdir / f"_batch_input_{int(time.time())}.jsonl")
    try:
        fid = upload_input(inp, api_key)
    finally:
        inp.unlink(missing_ok=True)
    batch = create_batch(fid, api_key, window=window)
    return {"batch_id": batch["id"]}


def poll(handle: dict, api_key: str) -> tuple[bool, str]:
    """(종료 여부, 상태 문자열). 종료=_TERMINAL(completed/failed/expired/cancelled)."""
    b = get_batch(handle["batch_id"], api_key)
    c = b.get("request_counts", {})
    return b["status"] in _TERMINAL, f"{b['status']} {c.get('completed', 0)}/{c.get('total', 0)}"


def collect(handle: dict, api_key: str) -> dict:
    """완료 배치 → {cid: (text, api_error, usage)}. completed 아니면 예외."""
    b = get_batch(handle["batch_id"], api_key)
    if b["status"] != "completed":
        raise RuntimeError(f"배치 비정상 종료: {b['status']} (batch_id={handle['batch_id']})")
    out_text = download_file(b["output_file_id"], api_key)
    parsed = {}
    for ln in out_text.splitlines():
        if ln.strip():
            cid, text, err, usage = parse_output_line(json.loads(ln))
            parsed[cid] = (text, err, usage)
    return parsed


# ─── CLI: 5 QA hop0 검증 (사용자 실행, 비용 발생) ──────────────────────────────

def _validate(args):
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY 필요 (--api-key 또는 환경변수)")
    runner = load_runner()
    qas = [json.loads(l) for l in Path(args.qa_pairs).read_text(encoding="utf-8").splitlines()
           if l.strip()][:args.n]

    lines = []
    for qa in qas:
        hop0 = qa["hops"][0]
        question = hop0.get("sub_q", "")
        passages = [f"[stub passage for {qa['qid']}]"]  # 검증: hop0 프롬프트 구성만 확인
        prompt = runner.build_prompt(runner.HOP_PROMPT, question, 0, len(qa["hops"]), passages, "")
        lines.append(make_batch_line(
            f"{qa['qid']}-0-1", args.model, runner.SYSTEM_PROMPT, prompt,
            args.max_output_tokens, reasoning_effort=args.reasoning_effort,
            gen_seed=args.seed, runner=runner))

    inp = write_input_jsonl(lines, ROOT / "pipeline/_batch_validate_input.jsonl")
    print(f"입력 {len(lines)}줄 작성: {inp}")
    fid = upload_input(inp, api_key)
    print(f"업로드 file_id={fid}")
    batch = create_batch(fid, api_key, window=args.window)
    print(f"배치 생성 batch_id={batch['id']} status={batch['status']}")
    done = poll_until_done(batch["id"], api_key, interval=args.poll_interval)
    if done["status"] != "completed":
        sys.exit(f"배치 비정상 종료: {done['status']}")
    out_text = download_file(done["output_file_id"], api_key)
    results = [parse_output_line(json.loads(l)) for l in out_text.splitlines() if l.strip()]
    ok = sum(1 for _, _, e, _ in results if not e)
    print(f"\n결과: {ok}/{len(results)} 성공")
    for cid, text, err, usage in results[:3]:
        print(f"  {cid}  err={err}  usage={usage}  text={text[:60]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="5 QA hop0 end-to-end 검증 (비용 발생)")
    ap.add_argument("--qa-pairs", default="data/qa_pairs_kch_smoke100.jsonl")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--model", default="gpt-5.4-2026-03-05")
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--max-output-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--window", default="24h")
    ap.add_argument("--poll-interval", type=int, default=30)
    ap.add_argument("--api-key", default=None)
    args = ap.parse_args()
    if args.validate:
        _validate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
