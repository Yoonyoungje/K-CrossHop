#!/usr/bin/env bash
# 템플릿 × bridge 2×2 — yield·faithfulness 축 (entity-path joint CI). 무료, CPU.
# 입력: 정본 controlled / query-matched / normalized + C 트랙 bridge-only(KO 템플릿 + 정규화 bridge) judge 결과.
#   template-first : stage1 = ctrl − qm (템플릿),  stage2 = qm − norm (bridge)
#   bridge-first   : b_stage1 = ctrl − bn (bridge), b_stage2 = bn − norm (템플릿)
#   interaction    = stage1 − b_stage2
#   bash scripts/camera_ready/run_2x2.sh [출력 디렉터리]
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/2x2}"
BN=data/camera_ready/annotated
mkdir -p "$OUT"

for m in $MODELS; do
  [ -f "$BN/annotated_${m}_bge_bridge_only.jsonl" ] || { echo "bridge-only judge 결과 없음: $m (run_judge.sh collect C 먼저)"; exit 1; }
done
for axis in yield faith; do
  for m in $MODELS; do
    yqmc "$m" "$QA" --bridge-only "$BN/annotated_${m}_bge_bridge_only.jsonl" --axis "$axis" --label "${m}/${axis}"
  done > "$OUT/qmc_2x2_${axis}.txt"
done
echo "저장: $OUT"
