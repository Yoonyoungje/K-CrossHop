#!/usr/bin/env bash
# 모델별 정성 오류 분석 자료 — 유형별 비율 + cross-index 오류 유형별 사례. 무료, CPU.
#   bash scripts/camera_ready/run_qualitative.sh [출력 디렉터리] [사례 수 k]
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/qualitative}"
K="${2:-3}"
mkdir -p "$OUT"
for m in $MODELS; do
  python3 analysis/qualitative_cases.py --model "$m" --annotated "$(ann "$m")" --trajectories "$(traj "$m")" \
    --retrieval-hits "$(hits "$m")" --qa-pairs "$QA" --k "$K" \
    --correctness data/correctness/all/hops.csv > "$OUT/cases_${m}.md"
  echo "$m 완료"
done
echo "저장: $OUT"
