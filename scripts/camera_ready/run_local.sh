#!/usr/bin/env bash
# camera-ready 로컬 생성 (Ollama, GPU). 비용 $0. judge 는 부르지 않는다 — 생성 후 사전 점검이 먼저다.
#
# 트랙
#   C  bridge-only: 8B 4종 × dynamic_subq_bridgenorm. hop0 = 정본 controlled 궤적 재사용 → hop1 만 생성
#   A  E5 end-to-end: llama × {ctrl, qm, norm}. 인덱스 data/camera_ready/index_e5 (build_e5_index.sh 먼저)
#   B  Qwen3 크기 사다리 + 정밀도 대조: MODELS 의 각 태그 × {ctrl, qm, norm}. BGE-M3 인덱스.
#      기본 = 1.7B·4B·14B (Q4_K_M) + 8B fp16. 8B Q4 는 정본(qwen) 재사용. fp16 은 BGE-M3 를 CPU 로 돌린다
#   P  fp16 파일럿: qwen3:8b-fp16 × ctrl, 패턴당 5 = 20 궤적. 속도로 본 실행 여부를 정한다
#   R  버전 replay: 정본 controlled hop0 을 재사용해 hop1 만 현재 Ollama 로 다시 생성한다.
#      정본 hop1 출력과 문자열이 같은 비율로 Ollama 버전(정본 0.23.1) 차이의 영향을 잰다
#
# 실행:  bash scripts/camera_ready/run_local.sh C                 # 4모델
#        bash scripts/camera_ready/run_local.sh C llama qwen      # 일부만
#        bash scripts/camera_ready/run_local.sh A
#        bash scripts/camera_ready/run_local.sh B                   # 기본 4태그 (약 5시간)
#        MODELS="qwen3:1.7b qwen3:4b-q4_K_M" bash scripts/camera_ready/run_local.sh B   # 일부만
#        bash scripts/camera_ready/run_local.sh P
# 재개:  같은 명령을 다시 실행하면 끝난 조건은 건너뛰고, 진행 중이던 조건은 05 가 이어 쓴다.
# 주의:  로컬끼리 병렬로 돌리지 말 것 (Ollama 가 직렬화한다). norm 은 qm 이 끝나야 시작한다.
set -u
cd "$(dirname "$0")/../.."

QA=data/dataset/qa_pairs.jsonl
OUT=data/trajectories/camera_ready
LOG=logs/camera_ready
BGE=data/faiss_index
E5=data/camera_ready/index_e5
mkdir -p "$OUT" "$LOG"

declare -A OLLAMA=( [aya]="aya-expanse:8b" [exaone]="exaone3.5:latest"
                    [llama]="llama3.1:8b"  [qwen]="qwen3:8b" )
declare -A CANON_CTRL=( [aya]="data/trajectories/aya_ecological.jsonl"
                        [exaone]="data/trajectories/exaone_ecological.jsonl"
                        [llama]="data/trajectories/llama_ecological.jsonl"
                        [qwen]="data/trajectories/qwen_ecological.jsonl" )
declare -A MODE=( [ctrl]="dynamic_subq" [qm]="dynamic_subq_qmatched"
                  [norm]="dynamic_subq_norm" [bn]="dynamic_subq_bridgenorm" )

# gen <이름> <ollama 태그> <조건> <인덱스> <seed 파일 또는 -> [05 추가 인자...]
gen() {
  local name=$1 tag=$2 cond=$3 idx=$4 seed=$5; shift 5
  local out="$OUT/trajectories_${name}.jsonl" log="$LOG/gen_${name}.log"
  local want; want=$(if [ "$*" = "--per-pattern 5" ]; then echo 20; else grep -c . "$QA"; fi)
  local have; have=$( [ -f "$out" ] && grep -c . "$out" || echo 0 )
  if [ "$have" -ge "$want" ]; then echo "  [skip] $name 이미 $have/$want"; return 0; fi
  local seed_args=()
  if [ "$seed" != "-" ]; then
    if [ ! -f "$seed" ] || [ "$(grep -c . "$seed")" -lt "$want" ]; then
      echo "  [보류] $name — hop0 seed 미완 ($seed)"; return 0
    fi
    seed_args=(--seed-trajectories "$seed")
  fi
  echo "=== $name ($tag, ${MODE[$cond]}, $(basename "$idx"))  시작 $(date '+%F %T')  기존 $have ${seed_args[*]}"
  python3 pipeline/05_pilot_runner.py \
    --qa-pairs "$QA" --index-dir "$idx" \
    --agent-model "$tag" --retrieval-mode "${MODE[$cond]}" "${seed_args[@]}" \
    --prompt-version v2.3 --output "$out" "$@" > "$log" 2>&1
  echo "    exit=$?  궤적=$(grep -c . "$out")/$want  끝 $(date '+%F %T')  로그=$log"
}

track="${1:-}"; shift || true
case "$track" in
  C)
    models=("$@"); [ ${#models[@]} -eq 0 ] && models=(aya exaone llama qwen)
    for m in "${models[@]}"; do
      gen "${m}_bge_bn" "${OLLAMA[$m]}" bn "$BGE" "${CANON_CTRL[$m]}"
    done ;;
  A)
    [ -f "$E5/retriever.json" ] || { echo "E5 인덱스 없음 → bash scripts/camera_ready/build_e5_index.sh"; exit 1; }
    gen llama_e5_ctrl llama3.1:8b ctrl "$E5" -
    gen llama_e5_qm   llama3.1:8b qm   "$E5" -
    gen llama_e5_norm llama3.1:8b norm "$E5" "$OUT/trajectories_llama_e5_query_matched.jsonl" ;;
  B)
    # qwen3:4b 는 2507 thinking 판이라 원본 4B 태그를 쓴다
    for tag in ${MODELS:-qwen3:1.7b qwen3:4b-q4_K_M qwen3:14b qwen3:8b-fp16}; do
      n="${tag//:/-}"
      (   # fp16(약 17GB)은 VRAM 을 다 쓰므로 BGE-M3 를 CPU 로 (파일럿에서 hop0 패시지가 GPU 인코딩과 20/20 같음)
        case "$tag" in *fp16*) export CUDA_VISIBLE_DEVICES="" ;; esac
        gen "${n}_bge_ctrl" "$tag" ctrl "$BGE" -
        gen "${n}_bge_qm"   "$tag" qm   "$BGE" -
        gen "${n}_bge_norm" "$tag" norm "$BGE" "$OUT/trajectories_${n}_bge_query_matched.jsonl"
      )
    done ;;
  R)
    models=("$@"); [ ${#models[@]} -eq 0 ] && models=(llama)
    for m in "${models[@]}"; do
      gen "${m}_bge_ctrl_replay" "${OLLAMA[$m]}" ctrl "$BGE" "${CANON_CTRL[$m]}"
    done ;;
  P)
    # BGE-M3 를 CPU 로 돌려 VRAM 을 Ollama 에 모두 준다 (fp16 8B 는 약 16.4GB).
    export CUDA_VISIBLE_DEVICES=""
    gen qwen3-8b-fp16_bge_ctrl_pilot20 qwen3:8b-fp16 ctrl "$BGE" - --per-pattern 5 ;;
  *)
    sed -n '2,17p' "$0"; exit 2 ;;
esac

echo
echo "생성 후 (무료): python3 audit/pre_judge_check.py --trajectories $OUT/trajectories_*.jsonl"
