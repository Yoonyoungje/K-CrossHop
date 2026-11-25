#!/usr/bin/env bash
# A. E5 인덱스를 정본 04 로 재생성한다 (GPU 약 45분, 비용 $0).
#   부록 H 의 인덱스(work/e5_gate/index_e5)와 같은 설정: multilingual-e5-large, fp32, batch 32,
#   "passage: " 접두사, L2 정규화 + IndexFlatIP. 04 가 retriever.json 을 써서 05 가 인코더를 맞춘다.
# 끝나면 Claude 가 부록 H 인덱스와 벡터를 대조한다 (무료, CPU).
set -eu
cd "$(dirname "$0")/../.."
mkdir -p logs/camera_ready

python3 pipeline/04_build_faiss.py \
  --passages data/corpus/passages.jsonl \
  --index-dir data/camera_ready/index_e5 \
  --embed-model intfloat/multilingual-e5-large \
  --passage-prefix "passage: " --query-prefix "query: " \
  --batch-size 32 2>&1 | tee logs/camera_ready/build_e5_index.log
