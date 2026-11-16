#!/usr/bin/env bash
# E5 부록 truncation 문단: E5(512 토큰)에서 잘리는 패시지 비율과 근거 표면형이 잘리는 위험. 무료, CPU(토크나이저만).
#   bash scripts/camera_ready/run_e5_truncation.sh [출력 디렉터리]
#   token_audit.json        코퍼스 전체 'passage: ' + text 의 512 토큰 초과 비율 (analysis/e5_token_audit.py)
#   truncation_diag.json    근거 패시지 기준 잘림 손실·판정 영향 (analysis/e5_truncation_diag.py)
#   at_risk_passages.jsonl  근거 표면형이 512 토큰 밖에 있는 패시지
# 토크나이저(intfloat/multilingual-e5-large)는 HF 캐시에 있어야 한다 (오프라인이면 HF_HUB_OFFLINE=1).
set -eu
cd "$(dirname "$0")/../.."
OUT="${1:-results/camera_ready/retriever_robustness/e5_truncation}"
mkdir -p "$OUT"

# 초과 비율이 10% 를 넘으면 e5_token_audit.py 는 STOP 을 알리며 3 으로 끝난다 (원고가 보고하는 상태 그대로).
python3 analysis/e5_token_audit.py --out-json "$OUT/token_audit.json" || [ $? -eq 3 ]
python3 analysis/e5_truncation_diag.py --out-dir "$OUT"
rm -f "$OUT/evidence_passages.jsonl"   # 중간 산출물 (근거 패시지 본문 덤프)
echo "저장: $OUT"
