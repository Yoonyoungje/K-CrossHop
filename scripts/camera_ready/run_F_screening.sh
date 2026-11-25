#!/usr/bin/env bash
# 전 홉 감사 — 실험의 한국어 질문 sub_q 를 gold evidence(KO·EN)로 판정한다.
#   KO evidence 456 단위 = 앞서 만든 한국어 기준선 캐시 재사용 (호출 0)
#   EN evidence 456 단위 = 신규 호출 (gpt-5.4-mini, 약 $0.5)
# 데이터셋은 고정한다. 결과는 플래그 비율과 제외 민감도로만 보고한다.
#
#   bash scripts/camera_ready/run_F_screening.sh dry      # 호출 수·프롬프트 확인 (무료)
#   bash scripts/camera_ready/run_F_screening.sh run      # 실제 판정 (과금)
#   bash scripts/camera_ready/run_F_screening.sh report   # 캐시만으로 재집계
set -eu
cd "$(dirname "$0")/../.."

ARGS=(--question-field sub_q --prompt-version p2-hop-ko-1.0
      --core data/dataset/qa_pairs.jsonl
      --cache data/camera_ready/screening/answerability_cache_subq.jsonl
      --output data/camera_ready/screening/answerability_hops_subq.jsonl)

case "${1:-dry}" in
  dry)    python3 dataset/screen_hop_answerability.py "${ARGS[@]}" --dry-run ;;
  run)    python3 dataset/screen_hop_answerability.py "${ARGS[@]}" ;;
  report) python3 dataset/screen_hop_answerability.py "${ARGS[@]}" --report-only ;;
  *)      echo "사용법: $0 {dry|run|report}"; exit 2 ;;
esac
