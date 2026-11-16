#!/usr/bin/env bash
# pipeline/run_trajectories.sh
# 4 models × 3 modes = 12 runs (aya controlled은 별도 실행 중이므로 나머지 11개)
# 출력 파일이 이미 존재하면 스킵 (중단 후 재개 가능)

set -euo pipefail

QA=data/dataset/qa_pairs.jsonl
IDX=data/faiss_index
OUT=data/trajectories
LOG=data/trajectories/run_log.txt

mkdir -p "$OUT"

run() {
    local model="$1" mode="$2" out="$3"
    if [[ -f "$out" ]]; then
        echo "[SKIP] $out 이미 존재" | tee -a "$LOG"
        return
    fi
    echo "" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    echo "[START] $(date '+%Y-%m-%d %H:%M:%S')  model=$model  mode=$mode" | tee -a "$LOG"
    echo "========================================" | tee -a "$LOG"
    conda run -n clh python pipeline/05_pilot_runner.py \
        --qa-pairs "$QA" \
        --index-dir "$IDX" \
        --agent-model "$model" \
        --retrieval-mode "$mode" \
        --output "$out"
    echo "[DONE]  $(date '+%Y-%m-%d %H:%M:%S')  → $out" | tee -a "$LOG"
}

# ── aya-expanse:8b ──────────────────────────────────────────────
# controlled은 별도 실행 중 → 스킵 로직이 처리
run "aya-expanse:8b"   ko_subq      "$OUT/trajectories_aya_controlled.jsonl"
run "aya-expanse:8b"   sub_q        "$OUT/trajectories_aya_ablation.jsonl"
run "aya-expanse:8b"   dynamic_subq "$OUT/trajectories_aya_ecological.jsonl"
echo "[VRAM] ollama stop aya-expanse:8b" | tee -a "$LOG"
ollama stop aya-expanse:8b

# ── exaone3.5:latest ────────────────────────────────────────────
run "exaone3.5:latest" ko_subq      "$OUT/trajectories_exaone_controlled.jsonl"
run "exaone3.5:latest" sub_q        "$OUT/trajectories_exaone_ablation.jsonl"
run "exaone3.5:latest" dynamic_subq "$OUT/exaone_ecological.jsonl"
echo "[VRAM] ollama stop exaone3.5:latest" | tee -a "$LOG"
ollama stop exaone3.5:latest

# ── llama3.1:8b ─────────────────────────────────────────────────
run "llama3.1:8b"      ko_subq      "$OUT/trajectories_llama_controlled.jsonl"
run "llama3.1:8b"      sub_q        "$OUT/trajectories_llama_ablation.jsonl"
run "llama3.1:8b"      dynamic_subq "$OUT/llama_ecological.jsonl"
echo "[VRAM] ollama stop llama3.1:8b" | tee -a "$LOG"
ollama stop llama3.1:8b

# ── qwen3:8b ────────────────────────────────────────────────────
run "qwen3:8b"         ko_subq      "$OUT/trajectories_qwen_controlled.jsonl"
run "qwen3:8b"         sub_q        "$OUT/trajectories_qwen_ablation.jsonl"
run "qwen3:8b"         dynamic_subq "$OUT/qwen_ecological.jsonl"
echo "[VRAM] ollama stop qwen3:8b" | tee -a "$LOG"
ollama stop qwen3:8b

echo "" | tee -a "$LOG"
echo "========================================"  | tee -a "$LOG"
echo "전체 완료: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"
echo "========================================"  | tee -a "$LOG"
