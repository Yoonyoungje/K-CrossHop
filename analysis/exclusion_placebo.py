#!/usr/bin/env python3
"""
analysis/exclusion_placebo.py

경로 제외 민감도의 위약 비교 (camera-ready). 실제로 뺀 entity path 집합의 yield-QMC 점추정을,
같은 수의 경로를 무작위로 뺀 위약 분포와 비교한다 — 성분이 약해진 것이 그 경로들의 성질 때문인지,
표본이 줄어서인지 가른다.

지표는 yield_qmc_joint_ci.metrics(정본 정의) 그대로 쓰고, 추첨마다 점추정만 계산한다(bootstrap 없음).
관찰값의 위약 분포 내 백분위(낮은 쪽)를 보고한다.

  python3 analysis/exclusion_placebo.py --controlled C --query-matched Q --normalized N \\
      --qa-pairs data/dataset/qa_pairs.jsonl --keep-qa <제외 후 QA 부분집합> --n 500 --seed 42 --label llama
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yield_qmc_joint_ci as yq  # noqa: E402

KEYS = ("YL_ctrl", "stage1", "stage2")


def paths_of(path):
    return {json.loads(l)["path_cluster_id"]: None for l in open(path, encoding="utf-8") if l.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlled", required=True)
    ap.add_argument("--query-matched", dest="query_matched", required=True)
    ap.add_argument("--normalized", required=True)
    ap.add_argument("--qa-pairs", required=True, help="전체 QA (위약 추첨 모집단)")
    ap.add_argument("--keep-qa", required=True, help="실제 제외 후 남은 QA 부분집합")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    qa = [json.loads(l) for l in open(args.qa_pairs, encoding="utf-8") if l.strip()]
    path_of = {r["qid"]: r["path_cluster_id"] for r in qa}
    all_paths = list(dict.fromkeys(r["path_cluster_id"] for r in qa))
    keep_paths = set(paths_of(args.keep_qa))
    k = len(all_paths) - len(keep_paths)

    c, d, n = (yq.load_by_qid(p) for p in (args.controlled, args.query_matched, args.normalized))
    by_path: dict[str, list] = {}
    for q in path_of:
        if q in c and q in d and q in n:
            by_path.setdefault(path_of[q], []).append((c[q], d[q], n[q]))

    def point(paths):
        return yq.metrics([t for p in paths for t in by_path.get(p, [])])

    full = point(all_paths)
    obs = point([p for p in all_paths if p in keep_paths])
    rng = random.Random(args.seed)
    draws = []
    for _ in range(args.n):
        drop = set(rng.sample(all_paths, k))
        m = point([p for p in all_paths if p not in drop])
        if m is not None:
            draws.append(m)

    print(f"[{args.label or 'model'}]  제외 {k} / 전체 {len(all_paths)} 경로 · 위약 {len(draws)}회 (seed={args.seed})")
    print(f"  {'지표':<9} {'전체':>8} {'관찰':>8} {'위약 중앙':>9}  {'위약 2.5–97.5%':>17}  관찰 백분위")
    for key in KEYS:
        vals = sorted(m[key] for m in draws)
        lo, med, hi = vals[int(0.025 * len(vals))], vals[len(vals) // 2], vals[int(0.975 * len(vals)) - 1]
        pct = sum(v <= obs[key] for v in vals) / len(vals)
        print(f"  {key:<9} {full[key]:>+8.3f} {obs[key]:>+8.3f} {med:>+9.3f}  [{lo:+.3f}, {hi:+.3f}]  {pct:>8.1%}")


if __name__ == "__main__":
    main()
