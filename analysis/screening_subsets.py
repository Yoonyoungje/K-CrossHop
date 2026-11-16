#!/usr/bin/env python3
"""
analysis/screening_subsets.py

전 홉 감사(dataset/screen_hop_answerability.py) 민감도용 QA 부분집합 (camera-ready).

FAIL 판정 홉이 하나라도 있는 entity path 를 네 패턴 모두 함께 뺀다 (matched 설계 유지).
  en  : 한국어 질문 × 영어 evidence 에서 FAIL — controlled 조건이 EN-index 홉에서 실제로 묻는 것
  any : 한국어 질문 × 어느 evidence 언어에서든 FAIL
지표는 기존 스크립트로 계산한다:
  python3 analysis/yield_qmc_joint_ci.py ... --qa-pairs <out-dir>/qa_screen_keep_en.jsonl --cluster-key path_cluster_id

  python3 analysis/screening_subsets.py --qa-pairs data/dataset/qa_pairs.jsonl \\
      --hops data/camera_ready/screening/answerability_hops_subq.jsonl --out-dir data/camera_ready/screening
"""
import argparse
import json
from collections import Counter
from pathlib import Path

GROUPS = {
    "en":  {"ko_question_en_evidence"},
    "any": {"ko_question_en_evidence", "ko_question_ko_evidence"},
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--hops", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    qa, hops = load(args.qa_pairs), load(args.hops)
    if any("oracle_verdict" not in h for h in hops):
        raise SystemExit("판정이 빠진 홉이 있다 — 스크리닝을 끝까지 돌렸는지 확인")
    paths = {r["path_cluster_id"] for r in qa}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"전체: entity path {len(paths)} / 궤적 {len(qa)}")
    for tag, groups in GROUPS.items():
        fail = [h for h in hops if h["group"] in groups and h["oracle_verdict"] != "PASS"]
        bad = {h["path_cluster_id"] for h in fail}
        cells = {(h["path_cluster_id"], h["hop"]) for h in fail}
        keep = [r for r in qa if r["path_cluster_id"] not in bad]
        assert len(qa) - len(keep) == 4 * len(bad)
        with (out / f"qa_screen_keep_{tag}.jsonl").open("w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rel = Counter(h["relation"] for h in fail if h["group"] in groups)
        print(f"  [{tag}] FAIL 홉 {len(fail)} · (경로, hop) {len(cells)} · 제외 경로 {len(bad)} "
              f"→ 남는 경로 {len(paths) - len(bad)}  | relation 상위 {rel.most_common(3)}")


if __name__ == "__main__":
    main()
