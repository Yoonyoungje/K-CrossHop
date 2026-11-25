#!/usr/bin/env bash
# E5 end-to-end QMC — llama·sonnet × {yield, faith}, entity-path(250) joint CI. 무료, CPU.
# 같은 모델의 정본 BGE-M3 분해를 바로 아래 줄에 나란히 찍는다 (같은 재표집 설정).
#   bash scripts/camera_ready/run_e5_qmc.sh [출력 디렉터리]
# norm judge 결과가 아직 없는 모델은 건너뛴다 (나중에 다시 실행하면 채워진다).
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/e5}"
A=data/camera_ready/annotated
mkdir -p "$OUT"

for axis in yield faith; do
  for m in llama sonnet; do
    if [ ! -f "$A/annotated_${m}_e5_normalized.jsonl" ]; then echo "  [건너뜀] $m — E5 norm judge 결과 없음" >&2; continue; fi
    python3 analysis/yield_qmc_joint_ci.py --controlled "$A/annotated_${m}_e5_ecological.jsonl" \
      --query-matched "$A/annotated_${m}_e5_query_matched.jsonl" --normalized "$A/annotated_${m}_e5_normalized.jsonl" \
      --qa-pairs "$QA" --cluster-key path_cluster_id --n-boot 10000 --seed 42 \
      --axis "$axis" --label "${m}/E5/${axis}"
    yqmc "$m" "$QA" --axis "$axis" --label "${m}/BGE/${axis}"
  done > "$OUT/qmc_e5_${axis}.txt"
done
echo "저장: $OUT"
