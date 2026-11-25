#!/usr/bin/env bash
# gold-entity 상한: Table 12(UB·controlled hit@5·gap)와 E5 부록 상한 문장(BGE-M3 와 E5 의 UB 를 홉별로 맞댐). 무료, CPU.
#   bash scripts/camera_ready/run_gold_entity_ub.sh [출력 디렉터리]
# 입력 per-entity 는 analysis/gold_entity_ub_encoder.py(GPU)로 만들어 커밋해 둔 것이다 (11/15, 정본 core QA):
#   D=results/camera_ready/retriever_robustness/gold_entity_ub
#   python3 analysis/gold_entity_ub_encoder.py --model BAAI/bge-m3 --query-prefix "" --index-dir data/faiss_index \
#     --label bge --output $D/bge_per_entity.jsonl --summary $D/bge_summary.json
#   python3 analysis/gold_entity_ub_encoder.py --label e5
set -eu
cd "$(dirname "$0")/../.."
D=results/camera_ready/retriever_robustness/gold_entity_ub
OUT="${1:-$D}"
mkdir -p "$OUT"
python3 analysis/gold_entity_ub_agreement.py "$D/bge_per_entity.jsonl" "$D/e5_per_entity.jsonl" > "$OUT/agreement.txt"
# Table 12: BGE-M3 gold-entity UB 와 controlled hit@5, gap (§4.1 매처)
python3 analysis/gold_entity_ub_table.py > "$OUT/table12.txt"
echo "저장: $OUT/agreement.txt, $OUT/table12.txt"
