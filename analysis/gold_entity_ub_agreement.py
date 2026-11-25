#!/usr/bin/env python3
"""
analysis/gold_entity_ub_agreement.py — 두 인코더의 gold-entity UB 를 홉별로 맞대어 본다 (E5 부록). CPU, 무료.

입력은 analysis/gold_entity_ub_encoder.py 의 per-entity 출력 둘. 매처 두 가지를 각각 센다:
  진단 매처  = retrieval_success_gold_entity_ub (analysis/gold_entity_retrieval_diag.passage_has_entity)
  §4.1 매처 = hit_v3 (utils.retrieval_success.check_answer_bearing_hit, 원고 정본)

  python3 analysis/gold_entity_ub_agreement.py \\
      results/camera_ready/retriever_robustness/gold_entity_ub/bge_per_entity.jsonl \\
      results/camera_ready/retriever_robustness/gold_entity_ub/e5_per_entity.jsonl
"""
import json
import sys


def load(p):
    with open(p, encoding="utf-8") as f:
        return {(r["qid"], r["hop"]): r for r in map(json.loads, f)}


def main():
    bge, e5 = load(sys.argv[1]), load(sys.argv[2])
    keys = sorted(set(bge) & set(e5))
    print(f"공통 홉 {len(keys)} (BGE {len(bge)}, E5 {len(e5)})")
    for field, name in (("retrieval_success_gold_entity_ub", "진단 매처"), ("hit_v3", "§4.1 매처")):
        agree = sum(bge[k][field] == e5[k][field] for k in keys)
        bge_only = sum(bool(bge[k][field] and not e5[k][field]) for k in keys)
        e5_only = sum(bool(e5[k][field] and not bge[k][field]) for k in keys)
        rate = lambda d: sum(d[k][field] for k in keys) / len(keys)  # noqa: E731
        print(f"{name}: BGE {rate(bge):.4f}  E5 {rate(e5):.4f}  일치 {agree}/{len(keys)} ({agree/len(keys):.1%})"
              f"  불일치 {len(keys)-agree} = BGE만 적중 {bge_only} / E5만 적중 {e5_only}")


if __name__ == "__main__":
    main()
