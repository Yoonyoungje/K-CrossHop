#!/usr/bin/env python3
"""
audit/verify_gemini_hop0_job.py
=====================================
hop1 제출(추가 비용) 전에, 살아있는 hop0 batch job을 read-only로 poll+collect 하여
답변↔qid 정렬을 검증하는 게이트.

배경: `--resume`은 hop0 done 시 곧바로 hop1을 제출한다(중간 정지점 없음). hop1 비용을 쓰기
전에 수정된 batch_gemini.collect()의 metadata-cid 역매핑이 **실제 1000 스케일**에서 작동하는지
확인한다. collect는 이미 결제된 hop0 결과 다운로드 → 추가 생성 비용 0. job/ckpt를 변경하지 않음.

검증 지표(verify_gemini_alignment와 동일): hop0 gold-hit rate = gold_entity가 hop0 답변에 surface 포함.
  - ≥ threshold(0.40) → PASS (정렬 정상 + metadata echo 1000-스케일 OK → hop1 진행 가능)
  - ~0.03~0.05 → FAIL (스크램블 재현)
  - job 미완료 → exit 2 (잠시 후 재실행)

실행:
  python3 audit/verify_gemini_hop0_job.py \\
    --ckpt data/trajectories/gemini_ecological.jsonl.ckpt.json \\
    --qa-pairs data/dataset/qa_pairs.jsonl
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_batch_gemini():
    spec = importlib.util.spec_from_file_location(
        "batch_gemini", str(ROOT / "pipeline/batch_gemini.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(s: str) -> str:
    return (s or "").lower().replace(" ", "")


def cid_to_qid(cid: str) -> str:
    # cid = f"{qid}-{hop_idx}-{run_id}" (05_batch_runner _build_round)
    return cid.rsplit("-", 2)[0]


def main():
    from dotenv import load_dotenv
    from google import genai

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY 필요 (.env)"); return 3

    bg = _load_batch_gemini()
    client = genai.Client(api_key=api_key)

    ckpt = json.loads(Path(args.ckpt).read_text(encoding="utf-8"))
    handle = ckpt["handle"]
    if ckpt.get("round") != 0:
        print(f"⚠ ckpt round={ckpt.get('round')} — 이 게이트는 hop0(round 0) 전용. handle을 그대로 검증함.")

    done, status = bg.poll(client, handle)
    print(f"job={handle['job_name']}  state={status}")
    if not done:
        print("  ⏳ 아직 미완료 — 잠시 후 같은 명령으로 재실행하세요. (hop1 미제출, 비용 0)")
        return 2

    # 수정된 collect: 응답 metadata.cid로 역매핑 (제출순서 의존 X). 실패 시 즉시 예외.
    parsed = bg.collect(client, handle)   # {cid: (text, api_error, usage)}
    print(f"  collect OK: {len(parsed)}건 (metadata-cid 역매핑 성공 = 1000-스케일 echo 정상)")

    qa = {r["qid"]: r for r in load_jsonl(args.qa_pairs)}
    hit = tot = 0
    samples = []
    for cid, (text, api_error, _usage) in parsed.items():
        qid = cid_to_qid(cid)
        if qid not in qa:
            continue
        h0 = qa[qid]["hops"][0]
        ge = h0.get("gold_entity", "")
        if not ge:
            continue
        ok = (not api_error) and norm(ge) in norm(text)
        tot += 1
        hit += ok
        if len(samples) < args.show:
            samples.append((qid, h0.get("sub_q", "")[:34], text[:34], ge[:20], ok))

    print("\n샘플 (qid | 질문 | hop0 답변 | gold_entity | hit):")
    for qid, q, a, ge, ok in samples:
        print(f"  {'✓' if ok else '✗'} {qid}  Q={q!r}  A={a!r}  gold={ge!r}")

    rate = hit / tot if tot else 0.0
    print(f"\nhop0 gold-hit: {hit}/{tot} = {rate:.1%}  (threshold={args.threshold:.0%})")
    print("  참고: 정렬됨 Sonnet 58.7% / 동기 호출 52.0% / 시험 배치 52.0% — 스크램블 3.6%")

    if rate >= args.threshold:
        print("  ✅ PASS — 1000-스케일 정렬 정상. hop1 제출(--resume --block) 진행 가능.")
        return 0
    print("  ❌ FAIL — 스크램블 의심. hop1 제출 금지. collect 매핑 재점검 필요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
