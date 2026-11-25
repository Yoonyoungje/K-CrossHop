#!/usr/bin/env bash
# matched 분석: 양쪽 R=1 triple 짝으로 본 hop0·hop1 matched Δf/Δy(entity-path 재표집 CI),
# 세 조건 비교와 검색 품질 공변량. 무료, CPU.
#   bash scripts/camera_ready/run_matched.sh [출력 디렉터리]
#   matched.txt                       matched_subset.py — hop0·hop1 matched Δf/Δy (§1–§3), 요약 표
#   matched_subset_pairs.csv          짝 단위 값
#   three_conditions.txt              matched_subset_followup.py 절 1·1d — 세 조건, 조건 간 차이, 공변량
#   matched_subset_retrieval_cov.csv  절 1d 의 홉 단위 검색 품질
# followup 절 2(CI 재표집 단위 비교)는 canonical_ci 묶음이 정본 CI 를 대신 낸다.
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-results/camera_ready/matched}"
mkdir -p "$OUT"
python3 analysis/matched_subset.py --out-dir "$OUT" > "$OUT/matched.txt"
python3 analysis/matched_subset_followup.py --sections 1 1d --out-dir "$OUT" \
  > "$OUT/three_conditions.txt"
echo "저장: $OUT"
