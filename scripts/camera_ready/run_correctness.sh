#!/usr/bin/env bash
# 정답 정확도·궤적 성공률을 원고 정본 CI 규칙(entity-path 250 재표집)으로 낸다. 무료, CPU.
#   bash scripts/camera_ready/run_correctness.sh [출력 디렉터리]
# 계산은 analysis/correctness_axis.py (정의·매처는 그 docstring), 입력은 그 안의 정본 표(_inputs.sh 와 같다).
# --resample path_cluster_id 는 analysis/yield_qmc_joint_ci.py --cluster-key path_cluster_id 와 같은 resample 을 뽑으므로
# stats_bootstrap.csv 의 qmc|yield_* 는 results/camera_ready/canonical_ci/qmc_yield.txt 와 같아야 한다.
# QID 재표집 판(data/correctness/all)은 그대로 둔다. hop 단위 원자료 hops.csv(약 13MB)는 지운다 — 다시 만들 수 있다.
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-results/camera_ready/correctness}"
mkdir -p "$OUT"
python3 analysis/correctness_axis.py --tag camera_ready --out-dir "$OUT" \
  --resample path_cluster_id --n-boot 10000 --seed 42
python3 analysis/correctness_tables.py "$OUT" > "$OUT/tables.md"
rm -f "$OUT/hops.csv"
echo "저장: $OUT"
