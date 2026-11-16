#!/usr/bin/env python3
"""
audit/verify_hop0_shared.py
=================================
결정성 게이트 (read-only). 두 모델의 dynamic trajectory에서 hop0 retrieved_passages가
qid별로 동일한지 확인한다.

hop0 query는 모델 무관(KO sub_q / lang-matched 템플릿, prev_answer 미개입)이고
FAISS IndexFlatIP는 exact search라, 같은 인덱스에서 재검색하면 hop0 passage가 동일해야 한다.
불일치 0이면 "hop0 retrieval은 결정적이며 모델 간 동일"을 확정 (논문 Appendix 1줄).
불일치 발생 시 규모를 보고하고 Q1 사전계산(2번)을 재검토한다.

실행:
  python3 audit/verify_hop0_shared.py \
      data/trajectories/sonnet_ecological.jsonl \
      data/trajectories/trajectories_gpt_dynamic_run1.jsonl
"""
import json
import sys
import argparse
from pathlib import Path


def load_hop0(path: Path) -> dict[str, list[str]]:
    """qid → hop0 retrieved_passages."""
    out: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for hop in row.get("trajectory", []):
            if hop.get("hop") == 0:
                out[row["qid"]] = hop.get("retrieved_passages", [])
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj_a")
    ap.add_argument("traj_b")
    ap.add_argument("--show", type=int, default=5, help="불일치 qid 최대 표시 개수")
    args = ap.parse_args()

    a = load_hop0(Path(args.traj_a))
    b = load_hop0(Path(args.traj_b))

    common = a.keys() & b.keys()
    only_a = a.keys() - b.keys()
    only_b = b.keys() - a.keys()

    mismatch = [qid for qid in sorted(common) if a[qid] != b[qid]]

    print(f"A: {Path(args.traj_a).name} ({len(a)} qids)")
    print(f"B: {Path(args.traj_b).name} ({len(b)} qids)")
    print(f"공통 qid: {len(common)} / A 단독: {len(only_a)} / B 단독: {len(only_b)}")
    print(f"hop0 passage 불일치: {len(mismatch)}")

    for qid in mismatch[:args.show]:
        print(f"  ✗ {qid}: A={len(a[qid])}개, B={len(b[qid])}개 패시지")

    if mismatch:
        print("\n결정성 게이트 실패 → 규모 보고 후 Q1 사전계산(2번) 재검토 필요.")
        sys.exit(1)
    print("\n✓ 결정성 게이트 통과: 공통 qid 전부 hop0 passage 동일.")


if __name__ == "__main__":
    main()
