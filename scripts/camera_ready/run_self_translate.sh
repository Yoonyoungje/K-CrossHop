#!/usr/bin/env bash
# self-translate 결과를 원고 정본 CI 규칙(entity-path 250 재표집)으로 다시 낸다. 무료, CPU.
#   bash scripts/camera_ready/run_self_translate.sh [출력 디렉터리]
# 입력: data/self_translate/ (6모델 × ctrl / tmpl_dec / norm_dec / self_tr, 앞서 생성·판정).
# QID 재표집 판은 data/self_translate/{analysis_tables.md,pipeline/} 에 그대로 둔다.
#   13_<m>_<cond>.txt      조건별 faith·yield gap (§1·§1-B). QID 재표집 판과 같이 hit 입력 없이 돌린다
#   qmc_{yield,faith}.txt  전체 hop 분해 (ctrl → tmpl_dec → norm_dec)
#   analysis_tables.md     회복률 등 파이프라인 밖 지표 (analysis/self_translate_analysis.py)
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/self_translate}"
ST=data/self_translate
# 조건 이름으로 파일명을 조립한다 — 배포 빌더의 파일명 치환("_ecological.jsonl" 등)은 camera-ready 추가 실행용이라
# self-translate 파일(이름을 바꾸지 않고 싣는다)에 걸리면 안 된다.
st() { echo "$ST/annotated_$1_$2.jsonl"; }
mkdir -p "$OUT"

for m in $MODELS; do
  for c in ctrl tmpl_dec norm_dec self_tr; do
    python3 analysis/13_bootstrap_ci.py --annotated "$(st "$m" "$c")" --qa-pairs "$QA" \
      --cluster-bootstrap --cluster-key path_cluster_id > "$OUT/13_${m}_${c}.txt"
  done
  echo "13 $m 완료"
done

for axis in yield faith; do
  for m in $MODELS; do
    python3 analysis/yield_qmc_joint_ci.py --controlled "$(st "$m" ctrl)" \
      --query-matched "$(st "$m" tmpl_dec)" --normalized "$(st "$m" norm_dec)" \
      --qa-pairs "$QA" --cluster-key path_cluster_id --n-boot 10000 --seed 42 --axis "$axis" --label "${m}/${axis}"
  done > "$OUT/qmc_${axis}.txt"
done
echo "QMC 완료"

python3 analysis/self_translate_analysis.py --cluster-key path_cluster_id > "$OUT/analysis_tables.md"
echo "저장: $OUT"
