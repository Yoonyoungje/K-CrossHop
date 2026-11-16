#!/usr/bin/env bash
# 원고 정본 CI: 원고에 실린 bootstrap CI 전부를 entity-path(250) 재표집으로 다시 낸다. 무료, CPU.
#   bash scripts/camera_ready/run_canonical_ci.sh [출력 디렉터리]
# 명령은 tools/verify_baseline.sh 의 정본 명령과 같고, 재표집 단위 플래그만 더한다.
#   13_<m>.txt        Table 3(§1·§5·§6), Table 9 R=1 yield(§6-B), Fig 3a(§1·§1-B·§1-C)
#   qmc_{yield,faith}.txt  Table 6·7 (yield_qmc_joint_ci, 세 성분 + residual CI)
#   pooled_clepr.txt  Table 13 (모델별 ΔCLEPR + pooled), pooled_clepr_8b.txt  §5.6 본문 8B pooled
#   (부록 parametric fallback ΔPDR 은 13_<m>.txt §8 [전체])
#   parser_fp_impact.txt  부록 parser 감사(적대적 보정 후 증폭 CI), table8_purity_audit.txt  Table 8
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/canonical_ci}"
ALIAS=data/retrieval/alias_cache.json
mkdir -p "$OUT"

for m in $MODELS; do
  python3 analysis/13_bootstrap_ci.py --annotated "$(ann "$m")" --qa-pairs "$QA" \
    --retrieval-hits "$(hits "$m")" --trajectories "$(traj "$m")" --alias-cache "$ALIAS" \
    --cluster-bootstrap --cluster-key path_cluster_id > "$OUT/13_${m}.txt"
  echo "13 $m 완료"
done

for axis in yield faith; do
  for m in $MODELS; do yqmc "$m" "$QA" --axis "$axis" --label "${m}/${axis}"; done > "$OUT/qmc_${axis}.txt"
done
echo "QMC 완료"

python3 analysis/pooled_clepr_meta.py --qa-pairs "$QA" --cluster-bootstrap --cluster-key path_cluster_id \
  --annotated $(for m in $MODELS; do ann "$m"; done) > "$OUT/pooled_clepr.txt"
# §5.6 본문의 8B 4종 pooled
python3 analysis/pooled_clepr_meta.py --qa-pairs "$QA" --cluster-bootstrap --cluster-key path_cluster_id \
  --annotated $(for m in aya exaone llama qwen; do ann "$m"; done) > "$OUT/pooled_clepr_8b.txt"
# 부록 parser 감사: 특정된 FP 82홉을 뒤집은 Δy·증폭 (적대적 보정 후 증폭 CI)
python3 audit/parser_fp_yield_impact.py --cluster-key path_cluster_id > "$OUT/parser_fp_impact.txt"
# Table 8 정규화 감사 (CI 없음 — 원고 표의 출처를 같은 디렉터리에 둔다)
python3 audit/stage2_purity_audit.py --qa-pairs "$QA" $(for m in $MODELS; do echo --traj "$(tnm "$m")"; done) \
  > "$OUT/table8_purity_audit.txt"
echo "저장: $OUT"
