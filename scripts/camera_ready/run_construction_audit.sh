#!/usr/bin/env bash
# 구축 부록: 게이트 패턴별 통과율과 수집 회차별 단계 수(funnel.txt), 수동 검수 표본 62개(위험 32 + 무작위 30)의
# 항목별 표와 집계, 중간 단계 ID 목록(410·263), 기록된 cap 결과의 규칙 점검과 순서 무관 cap 출력. 무료, CPU.
#   bash scripts/camera_ready/run_construction_audit.sh [출력 디렉터리]
# 원고의 패턴별 통과율(424 기준)·한 패턴만 FAIL 수·4차 수집 21→18 은 funnel.txt 다.
# 원고의 "무작위 표본 중 수동 제거 판단까지 남은 24개 가운데 7개가 품질 결함"은 sample62.txt 의 RAND 줄이다.
# 항목별 제거 사유는 검수자(저자)가 남긴 기록(2026-06-17)을 audit/construction_audit.py 의 MANUAL13 에 옮긴 것이다.
# 입력(게이트 판정·410 ID 목록·cap 기록·62 표본 원문·answer_type)은 공개 데이터에 그대로 싣는다(11/16 결정).
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-results/camera_ready/construction}"
mkdir -p "$OUT"
python3 audit/construction_audit.py --out-dir "$OUT"
python3 dataset/apply_caps.py --output "$OUT/cap_order_independent.json" > /dev/null
echo "저장: $OUT/cap_order_independent.json"
