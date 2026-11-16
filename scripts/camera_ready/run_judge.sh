#!/usr/bin/env bash
# camera-ready judge (06, OpenAI Batch, 과금). **사전 점검(audit/pre_judge_check.py)을 통과한 궤적만** 넣는다.
#
#   bash scripts/camera_ready/run_judge.sh dry     C   # 판정 대상 홉 수·비용 추정만 (무료)
#   bash scripts/camera_ready/run_judge.sh submit  C   # 배치 제출 → {출력}.batch.json
#   bash scripts/camera_ready/run_judge.sh collect C   # 결과 수집 (완료까지 대기). 실패 요청이 있으면 출력 없이 멈춤
#   bash scripts/camera_ready/run_judge.sh retry   C   # 수집 + 실패 요청만 같은 본문으로 동기 재판정 (과금, 소량)
#   MODELS="aya sonnet gemini" bash ...               # 일부 모델만 (생성이 덜 끝난 모델은 자동으로 건너뜀)
#   bash scripts/camera_ready/run_judge.sh submit  A   # E5 end-to-end (llama·sonnet; MODELS=sonnet 이면 sonnet 만)
#   bash scripts/camera_ready/run_judge.sh submit  B   # Qwen3 사다리 (B_MODELS 로 태그 선택, 기본 4태그)
#
# 트랙 C: hop0 은 정본 controlled 의 라벨을 그대로 재사용하고(--reuse-hop-labels) hop1 만 판정한다.
# 트랙 A: ctrl·qm 은 두 홉 모두 판정. norm 은 hop0 을 qm 에서 seed 했으므로 qm 의 hop0 라벨을 재사용한다
#         → qm 결과를 collect 한 뒤에 norm 을 submit 한다 (그 전에는 [보류]).
# 트랙 B: A 와 같은 규칙 (ctrl·qm 전부, norm 은 qm hop0 라벨 재사용).
set -u
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh

GEN=data/trajectories/camera_ready
OUT=data/camera_ready/annotated
LOG=logs/camera_ready
mkdir -p "$OUT" "$LOG"

judge() {   # judge <action> <이름> [06 추가 인자...]
  local action=$1 name=$2; shift 2
  local traj="$GEN/trajectories_${name}.jsonl" out="$OUT/annotated_${name}.jsonl"
  [ -f "$traj" ] || { echo "  [없음] $traj"; return 0; }
  local have; have=$(grep -c . "$traj")
  if [ "$have" -lt "$(grep -c . "$QA")" ]; then echo "  [미완] $name 궤적 $have — 생성이 끝난 뒤 실행"; return 0; fi
  # 06 은 이미 낸 배치를 막지 않는다 → 같은 명령을 다시 써도 이중 제출(이중 과금)이 없게 여기서 막는다
  if [ "$action" = submit ] && { [ -f "${out%.jsonl}.batch.json" ] || [ -f "$out" ]; }; then
    echo "  [제출됨] $name — ${out%.jsonl}.batch.json 또는 결과가 이미 있음 (다시 내려면 그 파일을 치운 뒤)"; return 0
  fi
  if { [ "$action" = collect ] || [ "$action" = retry ]; } && [ ! -f "${out%.jsonl}.batch.json" ]; then
    echo "  [미제출] $name — 제출한 배치가 없음"; return 0
  fi
  # 이미 받은 결과는 다시 받지 않는다 (retry 로 채운 결과를 collect 가 다시 실패로 멈추거나, retry 가 재과금하지 않게)
  if { [ "$action" = collect ] || [ "$action" = retry ]; } && [ -f "$out" ]; then
    echo "  [수집됨] $name — 결과가 이미 있음 (다시 받으려면 그 파일을 치운 뒤)"; return 0
  fi
  local mode
  case "$action" in
    dry)     mode=(--dry-run) ;;
    submit)  mode=(--batch-submit -y) ;;
    collect) mode=(--batch-collect --batch-wait) ;;
    retry)   mode=(--batch-collect --batch-wait --batch-retry-failed) ;;
  esac
  echo "=== $action $name  $(date '+%F %T')"
  python3 pipeline/06_annotate.py --trajectories "$traj" --output "$out" --qa-pairs "$QA" \
    "$@" "${mode[@]}" 2>&1 | tee -a "$LOG/judge_${name}.log" | grep -avE 'HTTP Request|it/s\]|s/it\]' | tail -12
}

three() {   # three <action> <접두어>: ctrl·qm 은 전부 판정, norm 은 qm 의 hop0 라벨 재사용 (qm collect 후)
  local action=$1 p=$2
  judge "$action" "${p}_ctrl"
  judge "$action" "${p}_qm"
  if [ -f "$OUT/annotated_${p}_query_matched.jsonl" ]; then
    judge "$action" "${p}_norm" \
      --reuse-hop-labels "$OUT/annotated_${p}_query_matched.jsonl" "$GEN/trajectories_${p}_query_matched.jsonl"
  else
    echo "  [보류] ${p}_norm — qm judge 결과를 collect 한 뒤 실행 (hop0 라벨 재사용)"
  fi
}

action="${1:-}"; track="${2:-}"
case "$action" in dry|submit|collect|retry) ;; *) sed -n '2,15p' "$0"; exit 2 ;; esac
case "$track" in
  C)
    for m in $MODELS; do
      judge "$action" "${m}_bge_bn" --reuse-hop-labels "$(ann "$m")" "$(traj "$m")"
    done ;;
  A)
    for m in $MODELS; do
      case "$m" in llama|sonnet) ;; *) continue ;; esac     # A 트랙 모델만 (MODELS=sonnet 로 하나만)
      three "$action" "${m}_e5"
    done ;;
  B)
    for tag in ${B_MODELS:-qwen3:1.7b qwen3:4b-q4_K_M qwen3:14b qwen3:8b-fp16}; do
      three "$action" "${tag//:/-}_bge"
    done ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
