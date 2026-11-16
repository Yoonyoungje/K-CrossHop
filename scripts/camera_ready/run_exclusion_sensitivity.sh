#!/usr/bin/env bash
# 경로 제외 민감도 + 위약 비교 (entity-path CI · 정본 yield-QMC 정의). 무료, CPU.
#   scope  : 템플릿 감사의 scope_mismatch 3쌍을 쓰는 44 경로 제외
#   keep_en / keep_any : 전 홉 감사 FAIL 경로 제외 (run_screening_sensitivity.sh 가 만든 부분집합)
#   bash scripts/camera_ready/run_exclusion_sensitivity.sh [출력 디렉터리]
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/exclusion}"
SUB=data/camera_ready/exclusion
mkdir -p "$OUT" "$SUB"

# scope 44 경로 제외 QA (템플릿 감사의 제외 qid 목록 그대로)
python3 - "$QA" data/correctness/scope_all3_exclude_qids.txt "$SUB/qa_keep_scope.jsonl" <<'EOF'
import json, sys
qa, excl, out = sys.argv[1:]
drop = {l.strip() for l in open(excl) if l.strip()}
rows = [l for l in open(qa, encoding="utf-8") if l.strip() and json.loads(l)["qid"] not in drop]
open(out, "w", encoding="utf-8").writelines(rows)
print(f"scope: 제외 qid {len(drop)} → 남은 궤적 {len(rows)}")
EOF

for tag in scope keep_en keep_any; do
  case $tag in scope) keep="$SUB/qa_keep_scope.jsonl" ;; *) keep="data/camera_ready/screening/qa_screen_${tag}.jsonl" ;; esac
  for m in $MODELS; do yqmc "$m" "$keep" --label "${m}/${tag}"; done > "$OUT/yield_qmc_${tag}.txt"
  for m in $MODELS; do
    python3 analysis/exclusion_placebo.py --controlled "$(ann "$m")" --query-matched "$(qm "$m")" \
      --normalized "$(nm "$m")" --qa-pairs "$QA" --keep-qa "$keep" --n 500 --seed 42 --label "${m}/${tag}"
  done > "$OUT/placebo_${tag}.txt"
done
echo "저장: $OUT"
