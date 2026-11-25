#!/usr/bin/env python3
"""
audit/r1_match_rate.py

부록 parametric fallback 문단의 대조 수치: answer-bearing retrieval이 **성공한** 홉(R=1, 비기권·비ambiguous)에서
같은 매칭(생성문에 gold label·alias가 부분문자열로 들어 있는가)이 gold 엔티티를 찾는 비율이다.
PDR(R=0 홉)이 낮은 것이 매칭 결함 때문이 아님을 보이는 데 쓴다. 원고의 범위는 모델별 '전체' 값의 최소–최대다.

매칭은 analysis/13_bootstrap_ci.py §8(PDR)의 build_pdr_data·_pdr_counts를 그대로 쓴다. build_pdr_data는
R=0 홉만 유효로 잡으므로 hit 플래그만 뒤집어 넘긴다. 매칭·기권 처리·alias 집합은 PDR과 같다.

  python audit/r1_match_rate.py --annotated <ann> --retrieval-hits <hits> --trajectories <traj> \
      --alias-cache data/retrieval/alias_cache.json --qa-pairs data/dataset/qa_pairs.jsonl --label aya
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.qa_set import restrict

_spec = importlib.util.spec_from_file_location("bootstrap_ci", REPO / "analysis/13_bootstrap_ci.py")
_bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--retrieval-hits", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--alias-cache", required=True)
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    records = _bc.load_jsonl(args.annotated)
    hits = [{**r, "answer_bearing_hit": not r.get("answer_bearing_hit", True)}   # R=1 ↔ R=0 뒤집기
            for r in _bc.load_jsonl(args.retrieval_hits)]
    alias_cache = json.load(open(args.alias_cache, encoding="utf-8"))
    data = _bc.build_pdr_data(_bc.load_jsonl(args.trajectories), hits, records, alias_cache)
    records = restrict(records, args.qa_pairs, quiet=True)

    out = []
    for name, hop in (("hop0", 0), ("hop1", 1), ("전체", None)):
        sc, st, cc, ct = _bc._pdr_counts(records, data, hop)
        n, c = st + ct, sc + cc
        out.append(f"{name} {c}/{n} = {c / n:.1%}" if n else f"{name} N/A")
    print(f"{args.label or Path(args.annotated).stem:<8} R=1 매칭률  " + "  |  ".join(out))


if __name__ == "__main__":
    main()
