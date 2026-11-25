#!/usr/bin/env python3
"""
analysis/truncation_subset.py — Ollama 문맥 상한(num_ctx 4096) 도달 홉이 있는 경로를 뺀 QA 부분집합.

05 는 num_ctx 를 지정하지 않아 Ollama 기본 4096 이 적용되고, 넘치면 앞 4토큰만 남기고 앞에서부터 잘린다
(시스템 지시문이 먼저 잘림). 궤적의 gen_usage.input_tokens(=prompt_eval_count) 가 상한 이상인 홉을 '잘림'으로
보고, 주어진 궤적들 중 하나라도 잘린 홉이 있는 qid 를 제외한 QA 파일을 쓴다 → yield_qmc_joint_ci 의
--qa-pairs 로 넘겨 민감도를 본다.

  python3 analysis/truncation_subset.py --qa-pairs data/dataset/qa_pairs.jsonl \
      --trajectories T_ecological.jsonl T_query_matched.jsonl T_normalized.jsonl --output qa_no_trunc.jsonl
"""
import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--trajectories", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=4096)
    args = ap.parse_args()

    bad, by_lang = set(), Counter()
    for p in args.trajectories:
        for line in open(p, encoding="utf-8"):
            t = json.loads(line)
            for h in t["trajectory"]:
                if ((h.get("gen_usage") or {}).get("input_tokens") or 0) >= args.limit:
                    bad.add(t["qid"])
                    by_lang[h.get("gold_lang")] += 1
    qa = [json.loads(l) for l in open(args.qa_pairs, encoding="utf-8") if l.strip()]
    keep = [q for q in qa if q["qid"] not in bad]
    with open(args.output, "w", encoding="utf-8") as f:
        for q in keep:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"잘림 홉(증거 언어별) {dict(by_lang)} · 제외 qid {len(bad)} · 남은 QA {len(keep)}/{len(qa)} → {args.output}")


if __name__ == "__main__":
    main()
