#!/usr/bin/env bash
# relation 구성 분석: bridge 유형별 yield-QMC (entity-path 재표집). 무료, CPU.
# 입력 분할은 analysis/bridge_type_subsets.py, 지표는 analysis/yield_qmc_joint_ci.py (정본 정의 그대로).
#   bash scripts/camera_ready/run_bridge_type_qmc.sh [출력 디렉터리]
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/bridge_type}"
SUB=data/camera_ready/bridge_type
# bridge 개체 P31 캐시(11/16 Wikidata 조회). 있으면 네트워크를 쓰지 않는다
P31=data/camera_ready/bridge_instance_of.json
mkdir -p "$OUT"

pairs=(); for m in $MODELS; do pairs+=(--pair "$m" "$(tqm $m)" "$(tnm $m)"); done
python3 analysis/bridge_type_subsets.py --qa-pairs $QA --out-dir $SUB --instance-of $P31 "${pairs[@]}" > "$OUT/subsets_and_change_rates.txt"

for t in person organization location; do
  for m in $MODELS; do yqmc "$m" "$SUB/qa_bridge_$t.jsonl" --label "${m}/${t}"; done > "$OUT/yield_qmc_${t}.txt"
done
# 개체 단위 확인: bridge 가 사람(P31=Q5)인지, 사람 bridge 는 hop1 관계별(교육기관·출생지·국적·수상)과
# educated_at 을 뺀 나머지(범위 불일치 템플릿과 겹치는 경로를 빼도 남는지)
for t in human nonhuman human_P69 human_P19 human_P27 human_P166 human_noP69; do
  for m in $MODELS; do yqmc "$m" "$SUB/qa_bridge_$t.jsonl" --label "${m}/${t}"; done > "$OUT/yield_qmc_${t}.txt"
done
echo "저장: $OUT"
