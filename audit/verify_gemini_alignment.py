#!/usr/bin/env python3
"""
audit/verify_gemini_alignment.py
======================================
batch_gemini 매핑 수정(메타데이터 cid 역매핑) 후 답변↔qid 정렬을 검증하는 read-only 게이트.

배경: Gemini Batch는 custom_id 미지원 → 과거 제출순서 zip 역매핑이 1000 스케일에서 순열
스크램블 되어 답변이 엉뚱한 qid에 붙는 사고가 있었다(hop0 gold-hit 3.6%). 정상 정렬이면
hop0 답변에 gold_entity(=hop0 정답 bridge)가 자주 포함된다(Sonnet 58.7%, 동기 호출 52.0%).

검증 지표: hop0 gold-hit rate = (gold_entity가 hop0 generated_text에 surface 포함된 비율).
  - ≥ threshold(기본 0.40) → PASS (정렬 정상, full run 진행 가능)
  - ~0.03~0.05 → FAIL (스크램블 재현 → 매핑 미수정/메타데이터 echo 실패)

실행:
  python3 audit/verify_gemini_alignment.py \\
    --trajectories data/trajectories/smoke50_gemini_ecological.jsonl \\
    --qa-pairs     data/qa_pairs_kch_smoke50.jsonl
"""
import argparse
import json
import sys


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def norm(s: str) -> str:
    return (s or "").lower().replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--threshold", type=float, default=0.40,
                    help="PASS 기준 hop0 gold-hit rate (기본 0.40)")
    ap.add_argument("--show", type=int, default=8, help="샘플 출력 개수")
    args = ap.parse_args()

    qa = {r["qid"]: r for r in load_jsonl(args.qa_pairs)}
    traj = load_jsonl(args.trajectories)

    hit = tot = 0
    samples = []
    for r in traj:
        qid = r["qid"]
        if qid not in qa:
            continue
        h0 = qa[qid]["hops"][0]
        ge = h0.get("gold_entity", "")
        step0 = next((s for s in r["trajectory"] if s["hop"] == 0), None)
        if step0 is None or not ge:
            continue
        ans = step0.get("generated_text", "")
        ok = norm(ge) in norm(ans)
        tot += 1
        hit += ok
        if len(samples) < args.show:
            samples.append((qid, h0.get("sub_q", "")[:34], ans[:34], ge[:20], ok))

    print(f"\n샘플 (qid | 질문 | 답변 | gold_entity | hit):")
    for qid, q, a, ge, ok in samples:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {qid}  Q={q!r}  A={a!r}  gold={ge!r}")

    rate = hit / tot if tot else 0.0
    print(f"\nhop0 gold-hit: {hit}/{tot} = {rate:.1%}  (threshold={args.threshold:.0%})")
    print(f"  참고: 정렬됨 Sonnet 58.7% / 동기 호출 52.0% — 스크램블 3.6%")

    if rate >= args.threshold:
        print("  ✅ PASS — 답변↔qid 정렬 정상. full run 진행 가능.")
        return 0
    print("  ❌ FAIL — 스크램블 의심. metadata cid echo / collect 역매핑 재확인 필요.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
