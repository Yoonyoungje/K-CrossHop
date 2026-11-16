#!/usr/bin/env bash
# judge 관대함 규모: 정답 매처로는 오답인데 judge 가 faithful 로 본 hop 중, 응답이 gold 와 다른 엔티티를 가리키는 비율.
# 원고 Limitations 의 7.6–14.5% 는 §2 의 모델별 비율이다. 무료, CPU.
#   bash scripts/camera_ready/run_judge_leniency.sh [출력 디렉터리]
#   judge_leniency.md   §1 모델×조건×same/cross(entity-path 재표집 CI) · §2 모델별 비율 · §3 표기 차이 유형 · §4 사례
# 입력 entity_check.jsonl 은 gpt-5.4 엔티티 동일성 판정 기록이다(과금 실행 — 생성기는 싣지 않는다).
# §5(사후 정의 매처로 정확도 축 재계산)는 원고에 쓰이지 않아 돌리지 않는다.
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-results/camera_ready/judge_leniency}"
mkdir -p "$OUT"
python3 analysis/judge_leniency_report.py --skip-posthoc --out "$OUT/judge_leniency.md"
