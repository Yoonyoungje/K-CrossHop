#!/usr/bin/env bash
# camera-ready frontier 생성 (Message Batches / Gemini Batch, 과금). 조건마다 프로세스 1개를 띄우고
# 각 프로세스가 스스로 라운드를 이어간다 (--resume --block). 로컬 생성과 동시에 돌려도 된다.
#
# 트랙
#   C  bridge-only: sonnet·gemini × dynamic_subq_bridgenorm. hop0 = 정본 controlled 궤적 재사용 → hop1 1라운드
#   A  E5 end-to-end: sonnet × {ctrl, qm, norm}. norm 은 qm 이 끝난 뒤 다시 실행하면 띄운다 (hop0 = qm 재사용)
#      E5 쿼리 인코딩은 CPU 로 돌린다 (GPU 는 Ollama 몫)
#
# 실행:  bash scripts/camera_ready/run_frontier.sh C
#        bash scripts/camera_ready/run_frontier.sh A
#        bash scripts/camera_ready/run_frontier.sh status
# 중단:  프로세스를 죽여도 체크포인트({출력}.ckpt.json)로 같은 명령에서 이어진다.
# 비용(배치 추정): C 약 $3 (hop1 2,000홉) · A 약 $19 (Sonnet 3조건)
set -u
cd "$(dirname "$0")/../.."

QA=data/dataset/qa_pairs.jsonl
OUT=data/trajectories/camera_ready
LOG=logs/camera_ready
BGE=data/faiss_index
E5=data/camera_ready/index_e5
STAGGER="${STAGGER:-60}"
mkdir -p "$OUT" "$LOG"
n_qa=$(grep -c . "$QA")

declare -A API=( [sonnet]="claude-sonnet-4-6" [gemini]="gemini-3.5-flash" )
declare -A CANON_CTRL=( [sonnet]="data/trajectories/sonnet_ecological.jsonl"
                        [gemini]="data/trajectories/gemini_ecological.jsonl" )
declare -A MODE=( [ctrl]="dynamic_subq" [qm]="dynamic_subq_qmatched"
                  [norm]="dynamic_subq_norm" [bn]="dynamic_subq_bridgenorm" )

# launch <이름> <모델 키> <조건> <인덱스> <seed 파일 또는 ->
launch() {
  local name=$1 m=$2 cond=$3 idx=$4 seed=$5
  local out="$OUT/trajectories_${name}.jsonl" log="$LOG/gen_${name}.log"
  local have; have=$( [ -f "$out" ] && grep -c . "$out" || echo 0 )
  if [ "$have" -ge "$n_qa" ]; then echo "  [skip] $name 이미 $have/$n_qa"; return 0; fi
  local seed_args=()
  if [ "$seed" != "-" ]; then
    if [ ! -f "$seed" ] || [ "$(grep -c . "$seed")" -lt "$n_qa" ]; then
      echo "  [보류] $name — hop0 seed 미완 ($seed)"; return 0
    fi
    seed_args=(--seed-trajectories "$seed")
  fi
  echo "=== 띄움: $name (${API[$m]}, ${MODE[$cond]}, $(basename "$idx"))  $(date '+%F %T')  ${seed_args[*]}"
  nohup python3 pipeline/05_batch_runner.py \
    --qa-pairs "$QA" --index-dir "$idx" \
    --agent-model "${API[$m]}" --retrieval-mode "${MODE[$cond]}" "${seed_args[@]}" \
    --prompt-version v2.3 --max-output-tokens 256 \
    --resume --block --poll-interval 60 \
    --output "$out" >> "$log" 2>&1 &
  echo "    pid=$!  로그=$log"
  sleep "$STAGGER"
}

case "${1:-}" in
  C)
    for m in sonnet gemini; do launch "${m}_bge_bn" "$m" bn "$BGE" "${CANON_CTRL[$m]}"; done ;;
  A)
    [ -f "$E5/retriever.json" ] || { echo "E5 인덱스 없음 → bash scripts/camera_ready/build_e5_index.sh"; exit 1; }
    export CUDA_VISIBLE_DEVICES=""
    launch sonnet_e5_ctrl sonnet ctrl "$E5" -
    launch sonnet_e5_qm   sonnet qm   "$E5" -
    launch sonnet_e5_norm sonnet norm "$E5" "$OUT/trajectories_sonnet_e5_query_matched.jsonl" ;;
  status)
    # 궤적 파일은 첫 라운드가 끝나야 생기므로 체크포인트만 있는 진행 중 실행도 함께 보인다
    for f in $(ls "$OUT"/trajectories_{sonnet,gemini}_*.jsonl{,.ckpt.json} 2>/dev/null \
               | sed 's/\.ckpt\.json$//' | sort -u); do
      ck="$f.ckpt.json"
      r=$( [ -f "$ck" ] && python3 -c "import json;print(json.load(open('$ck'))['round'])" || echo - )
      have=$( [ -f "$f" ] && grep -c . "$f" || echo 0 )
      printf '%-45s %5s/%s  round=%s\n' "$(basename "$f")" "$have" "$n_qa" "$r"
    done ;;
  *)
    sed -n '2,15p' "$0"; exit 2 ;;
esac
