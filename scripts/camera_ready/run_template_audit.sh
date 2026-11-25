#!/usr/bin/env bash
# 템플릿 감사 부록: 공개 익명 라벨 표만으로 일치도(보고값)와 재라벨 전 값을 계산한다. 무료, CPU.
#   bash scripts/camera_ready/run_template_audit.sh [출력 디렉터리]
# 재라벨 기록(template_audit_relabels.csv): calibration 뒤 3인 논의를 일정상 건너뛰었고, 라벨 정의를
# 사후에 설명한 뒤 Annotator A 가 main 3건(t07·t09·t21)을 unnatural_ko → scope_mismatch 로 직접 고쳤다
# (2026-09-18). 보고값은 수정 후이고 수정 전 값을 함께 밝힌다. 어노테이터가 쓴 판정 사유는 공개하지 않는다.
set -eu
cd "$(dirname "$0")/../.."
REL=human_audit
S=audit/template_audit_score.py
OUT="${1:-results/camera_ready/template_audit}"
mkdir -p "$OUT"
python3 $S --labels-csv $REL/template_audit_labels.csv \
  --out-json "$OUT/score.json" --out-disagreements "$OUT/disagreements.csv"
python3 $S --labels-csv $REL/template_audit_labels.csv --revert-relabels $REL/template_audit_relabels.csv \
  --out-json "$OUT/score_before_relabel.json" --out-disagreements "$OUT/disagreements_before_relabel.csv"
