#!/usr/bin/env bash
# hop별 QMC 분해를 원고 정본 CI 규칙(entity-path 250 재표집)으로 낸다. 무료, CPU.
#   bash scripts/camera_ready/run_hop_qmc.sh [출력 디렉터리]
# analysis/yield_qmc_joint_ci.py --hop 0|1 : 그 hop 위치의 same/cross 홉만 집계한다(다른 정의는 원고 QMC 표와 같다).
# hop0 에는 옮겨 오는 bridge 가 없어 bridge-surface 성분은 설계상 0 이어야 하지만, 조건마다 hop0 을 따로 생성했으므로
# 생성 비결정성만큼 0 에서 벗어날 수 있다(그 크기를 실행 간 노이즈 눈금으로 읽는다).
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/hop_qmc}"
mkdir -p "$OUT"
for axis in yield faith; do
  for hop in 0 1; do
    for m in $MODELS; do
      python3 analysis/yield_qmc_joint_ci.py --controlled "$(ann "$m")" --query-matched "$(qm "$m")" \
        --normalized "$(nm "$m")" --qa-pairs "$QA" --cluster-key path_cluster_id --n-boot 10000 --seed 42 \
        --axis "$axis" --hop "$hop" --label "${m}/${axis}/hop${hop}"
    done > "$OUT/qmc_${axis}_hop${hop}.txt"
    echo "${axis} hop${hop} 완료"
  done
done
echo "저장: $OUT"
