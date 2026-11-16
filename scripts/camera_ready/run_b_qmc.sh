#!/usr/bin/env bash
# within-family (Qwen3 크기 사다리 + 정밀도) QMC — {yield, faith}, entity-path(250) joint CI. 무료, CPU.
# 기준점 qwen3:8b Q4_K_M 은 정본(qwen) 분해를 같은 설정으로 함께 찍는다.
#   bash scripts/camera_ready/run_b_qmc.sh [출력 디렉터리]
# norm judge 결과가 아직 없는 태그는 건너뛴다 (나중에 다시 실행하면 채워진다).
# 이어서 크기·정밀도 쌍의 paired 차이(--ref-*)를 qmc_qwen_ladder_pairs_{yield,faith}.txt 에 쓰고,
# 문맥 상한 민감도(잘린 홉이 있는 경로 제외)를 qmc_<tag>_no_trunc_yield.txt 에 쓴다.
set -eu
cd "$(dirname "$0")/../.."
source scripts/camera_ready/_inputs.sh
OUT="${1:-results/camera_ready/qwen_ladder}"
A=data/camera_ready/annotated
mkdir -p "$OUT"

for axis in yield faith; do
  {
    yqmc qwen "$QA" --axis "$axis" --label "qwen3:8b Q4 (정본)/${axis}"
    for tag in ${B_MODELS:-qwen3:1.7b qwen3:4b-q4_K_M qwen3:14b qwen3:8b-fp16}; do
      n="${tag//:/-}_bge"
      if [ ! -f "$A/annotated_${n}_normalized.jsonl" ]; then echo "  [건너뜀] $tag — norm judge 결과 없음" >&2; continue; fi
      python3 analysis/yield_qmc_joint_ci.py --controlled "$A/annotated_${n}_ecological.jsonl" \
        --query-matched "$A/annotated_${n}_query_matched.jsonl" --normalized "$A/annotated_${n}_normalized.jsonl" \
        --qa-pairs "$QA" --cluster-key path_cluster_id --n-boot 10000 --seed 42 \
        --axis "$axis" --label "${tag}/${axis}"
    done
  } > "$OUT/qmc_qwen_ladder_${axis}.txt"
done

# 크기·정밀도 간 paired 차이 (같은 경로 재표집 draw 에서 두 모델을 함께 평가). qwen3:8b = 정본 Q4.
files() { if [ "$1" = qwen3:8b ]; then echo "$(ann qwen) $(qm qwen) $(nm qwen)"
          else local n="${1//:/-}_bge"; echo "$A/annotated_${n}_ecological.jsonl $A/annotated_${n}_query_matched.jsonl $A/annotated_${n}_normalized.jsonl"; fi; }
pair() { local a=$1 b=$2 axis=$3; set -- $(files "$a") $(files "$b")
  if [ ! -f "$3" ] || [ ! -f "$6" ]; then echo "  [건너뜀] $a − $b — norm judge 결과 없음" >&2; return 0; fi
  python3 analysis/yield_qmc_joint_ci.py --controlled "$1" --query-matched "$2" --normalized "$3" \
    --ref-controlled "$4" --ref-query-matched "$5" --ref-normalized "$6" --ref-label "$b" \
    --qa-pairs "$QA" --cluster-key path_cluster_id --n-boot 10000 --seed 42 --axis "$axis" --label "${a}/${axis}"; }
PAIRS="qwen3:1.7b,qwen3:4b-q4_K_M qwen3:4b-q4_K_M,qwen3:8b qwen3:14b,qwen3:8b qwen3:4b-q4_K_M,qwen3:14b qwen3:1.7b,qwen3:8b qwen3:8b-fp16,qwen3:8b"
for axis in yield faith; do
  for p in $PAIRS; do pair "${p%,*}" "${p#*,}" "$axis"; done > "$OUT/qmc_qwen_ladder_pairs_${axis}.txt"
done

# 문맥 상한(Ollama 기본 num_ctx 4096) 민감도: 상한에 닿은 홉이 있는 경로를 빼고 yield 분해를 다시 낸다.
SUB=data/camera_ready/qwen_no_trunc
G=data/trajectories/camera_ready
mkdir -p "$SUB"
for tag in ${B_MODELS:-qwen3:1.7b qwen3:4b-q4_K_M qwen3:14b qwen3:8b-fp16}; do
  n="${tag//:/-}"
  if [ ! -f "$A/annotated_${n}_bge_normalized.jsonl" ]; then continue; fi
  python3 analysis/truncation_subset.py --qa-pairs "$QA" --output "$SUB/qa_no_trunc_${n}.jsonl" \
    --trajectories "$G/trajectories_${n}_bge_ecological.jsonl" "$G/trajectories_${n}_bge_query_matched.jsonl" \
    "$G/trajectories_${n}_bge_normalized.jsonl" > "$OUT/no_trunc_subset_${n}.txt"
  python3 analysis/yield_qmc_joint_ci.py --controlled "$A/annotated_${n}_bge_ecological.jsonl" \
    --query-matched "$A/annotated_${n}_bge_query_matched.jsonl" --normalized "$A/annotated_${n}_bge_normalized.jsonl" \
    --qa-pairs "$SUB/qa_no_trunc_${n}.jsonl" --cluster-key path_cluster_id --n-boot 10000 --seed 42 \
    --axis yield --label "${tag}/yield/잘림 경로 제외" > "$OUT/qmc_${n}_no_trunc_yield.txt"
done
echo "저장: $OUT"
