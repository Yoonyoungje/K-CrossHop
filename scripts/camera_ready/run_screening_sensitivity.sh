#!/usr/bin/env bash
# 전 홉 감사 민감도: FAIL 홉이 있는 entity path 를 빼고 yield-QMC 를 다시 낸다. 무료, CPU.
# 입력: run_F_screening.sh 결과. 분할은 analysis/screening_subsets.py, 지표는 yield_qmc_joint_ci.py.
#   bash scripts/camera_ready/run_screening_sensitivity.sh [출력 디렉터리]
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/screening}"
SUB=data/camera_ready/screening
mkdir -p "$OUT"

python3 dataset/screen_hop_answerability.py --report-only --question-field sub_q \
  --prompt-version p2-hop-ko-1.0 --core $QA \
  --cache $SUB/answerability_cache_subq.jsonl --output $SUB/answerability_hops_subq.jsonl \
  > "$OUT/screening_report.txt" 2>&1
python3 analysis/screening_subsets.py --qa-pairs $QA --hops $SUB/answerability_hops_subq.jsonl \
  --out-dir $SUB > "$OUT/subsets.txt"

for tag in full keep_en keep_any; do
  qa=$QA; [ "$tag" != full ] && qa="$SUB/qa_screen_${tag}.jsonl"
  for m in $MODELS; do yqmc "$m" "$qa" --label "${m}/${tag}"; done > "$OUT/yield_qmc_${tag}.txt"
done
echo "저장: $OUT"
