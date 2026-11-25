#!/usr/bin/env bash
# 배포 준비(재편·어휘 정규화) 회귀 검증 겸 배포용 지표 출력 생성기.
# 커밋된 results/*.txt 27건과 camera-ready 산출물(results/camera_ready/, 아래 CR 블록)을 정답지로 삼아
# 현재 코드가 같은 수치를 내는지 확인하고,
# 같은 실행 결과를 출력 디렉터리에 남긴다 (tools/build_release.py 가 이 디렉터리를 --metrics-dir 로 받는다).
#
# 사용법:  [MODE=numeric] [CR=0] bash tools/verify_baseline.sh [출력디렉터리]
#          CR=0 camera-ready 블록 생략 (기본 1, 이 블록이 약 70분, 기존 27건이 약 17분 걸린다)
#          MODE=strict (기본) 전체 byte 비교 / MODE=numeric 지표 숫자만 비교(라벨 개명 후 검증용)
# 실패 시 exit 1 + diff 출력.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/base}"
cd "$ROOT"
mkdir -p "$OUT"

# QA 와 모델별 정본 입력 도우미(ann/qm/nm/traj/hits) — camera-ready 스크립트와 같은 파일을 쓴다.
# 8B 정본은 annotated/, frontier 정본은 annotated/*_run1.jsonl.
# 13/16 의 answer-bearing hit 경로는 hop1 이 있어야 한다 (8B 는 _all_hops 파일).
source scripts/camera_ready/_inputs.sh
ALIAS=data/retrieval/alias_cache.json
EXP=results

# MODE=strict : 출력 전체를 byte 비교 (재편처럼 텍스트가 안 바뀌어야 하는 변경용)
# MODE=numeric: 출력에서 숫자 시퀀스만 뽑아 비교 (어휘 개명처럼 라벨만 바뀌는 변경용)
MODE="${MODE:-strict}"

# 부기 라인(로드 경로·필터 안내)을 버리고, 라벨에 붙은 숫자를 지운 뒤 남은 지표 숫자만 뽑는다 —
# 어휘 개명이 수치 비교를 오염시키지 않게.
#   1) 설계문서 참조 "(v1.0 §9.3)" 통째로 제거   2) 식별자에 붙은 숫자(hop0 / answer_bearing_hit
#   …) 제거
nums() { grep -vE '^로드:|pooled meta-analysis|^[[:space:]]*\[[^]]*\.jsonl\]' "$1" \
         | sed -E 's/\(?v[0-9]+\.[0-9]+ §[0-9.]+(, §[0-9.]+)*\)?//g; s/§[0-9.]+//g' \
         | sed 's/[A-Za-z_][A-Za-z_]*[0-9][0-9]*//g' \
         | grep -oE '[-+]?[0-9]+\.?[0-9]*'; }

fail=0
run() {  # run <name> <expected.txt> -- <command...>
  local name="$1" exp="$2"; shift 3
  printf '%-24s' "$name"
  "$@" > "$OUT/$name.txt" 2>&1
  if [ "$MODE" = numeric ]; then
    if diff -q <(nums "$OUT/$name.txt") <(nums "$exp") > /dev/null; then
      echo "OK(수치)"; return
    fi
    echo "DIFF(수치)"
    diff <(nums "$OUT/$name.txt") <(nums "$exp") | head -20
    fail=1; return
  fi
  if diff -q "$OUT/$name.txt" "$exp" > /dev/null; then
    echo "OK"
  else
    echo "DIFF"
    diff "$OUT/$name.txt" "$exp" | head -30
    fail=1
  fi
}

ALL="aya exaone llama qwen sonnet gemini"

# ── 07: 핵심 지표 (frontier 2종만 커밋돼 있음) ────────────────────────────────
for m in sonnet gemini; do
  run "07_${m}_ecological" "$EXP/07_${m}_ecological.txt" -- \
    python3 analysis/07_analyze.py --annotated "$(ann $m)" --qa-pairs $QA
done

# ── 11: QMC 2단계 분해 (6모델) ────────────────────────────────────────────────
for m in $ALL; do
  run "11_${m}_qmc_stage1" "$EXP/11_${m}_qmc_stage1.txt" -- \
    python3 analysis/11_ablation_compare.py --main-annotated "$(ann $m)" \
      --ablation-annotated "$(qm $m)" --qa-pairs $QA
  run "11_${m}_qmc_stage2" "$EXP/11_${m}_qmc_stage2.txt" -- \
    python3 analysis/11_ablation_compare.py --main-annotated "$(qm $m)" \
      --ablation-annotated "$(nm $m)" --qa-pairs $QA
done

# ── 12: retrieval 조건부 (frontier 2종) ───────────────────────────────────────
for m in sonnet gemini; do
  run "12_${m}_conditioned" "$EXP/12_${m}_conditioned.txt" -- \
    python3 analysis/12_conditioned_analysis.py --annotated "$(ann $m)" \
      --retrieval-hits "$(hits $m)"
done

# ── 13: bootstrap CI (6모델) ──────────────────────────────────────────────────
for m in $ALL; do
  run "13_${m}_ecological_ci" "$EXP/13_${m}_ecological_ci.txt" -- \
    python3 analysis/13_bootstrap_ci.py --annotated "$(ann $m)" --qa-pairs $QA \
      --retrieval-hits "$(hits $m)" --trajectories "$(traj $m)" --alias-cache $ALIAS
done

# ── 14: 모델 비교 ─────────────────────────────────────────────────────────────
run 14_all6_comparison "$EXP/14_all6_comparison.txt" -- \
  python3 analysis/14_model_comparison.py --qa-pairs $QA --annotated \
    "$(ann aya)" "$(ann exaone)" "$(ann llama)" "$(ann qwen)" \
    "$(ann sonnet)" "$(ann gemini)"

run 14_8b_vs_sonnet "$EXP/14_8b_vs_sonnet.txt" -- \
  python3 analysis/14_model_comparison.py --qa-pairs $QA --annotated \
    "$(ann aya)" "$(ann exaone)" "$(ann llama)" "$(ann qwen)" \
    "$(ann sonnet)"

# ── 16: PDR + pooled ΔCLEPR ───────────────────────────────────────────────────
for m in sonnet gemini; do
  run "16_${m}_pdr" "$EXP/16_${m}_pdr.txt" -- \
    python3 analysis/16_pdr_analysis.py --trajectories "$(traj $m)" \
      --retrieval-hits "$(hits $m)" --alias-cache $ALIAS --annotated "$(ann $m)"
done

run 16_pooled_clepr_meta_all6 "$EXP/16_pooled_clepr_meta_all6.txt" -- \
  python3 analysis/pooled_clepr_meta.py --qa-pairs $QA --annotated \
    "$(ann aya)" "$(ann exaone)" "$(ann llama)" "$(ann qwen)" \
    "$(ann sonnet)" "$(ann gemini)"

# ── camera-ready (EACL 2027) 산출물 ───────────────────────────────────────────
# 실행 스크립트(scripts/camera_ready/run_*.sh)를 $OUT/camera_ready/<이름> 에 돌리고, 커밋된 기대 디렉터리의
# 같은 이름 파일과 비교한다(MODE 규칙 동일). CR=0 이면 건너뛴다(빠른 점검용, 이 블록이 대부분의 시간을 쓴다).
# GPU·API 가 필요한 실행(05 생성·06 judge·F 판정·analysis/retriever_robustness.py)은 하네스 밖이다.
crdir() {  # crdir <이름> <기대 디렉터리> <파일 glob> <실행 스크립트>
  local name="$1" exp_dir="$2" pat="$3" script="$4"
  local got_dir="$OUT/camera_ready/$name" n=0 bad=() exp b
  mkdir -p "$got_dir"
  printf '%-24s' "cr/$name"
  if ! bash "$script" "$got_dir" > "$OUT/camera_ready/$name.log" 2>&1; then
    echo "실행 실패 — $OUT/camera_ready/$name.log"; fail=1; return
  fi
  for exp in "$exp_dir"/$pat; do
    n=$((n + 1))
    b="$(basename "$exp")"
    if [ "$MODE" = numeric ]; then
      diff -q <(nums "$got_dir/$b") <(nums "$exp") > /dev/null 2>&1 || bad+=("$b")
    else
      diff -q "$got_dir/$b" "$exp" > /dev/null 2>&1 || bad+=("$b")
    fi
  done
  if [ ${#bad[@]} -eq 0 ]; then echo "OK (${n}개 파일)"; return; fi
  echo "DIFF ${bad[*]}"
  for b in "${bad[@]}"; do diff "$got_dir/$b" "$exp_dir/$b" | head -10; done
  fail=1
}

if [ "${CR:-1}" = 1 ]; then
  CRR=results/camera_ready
  CRS=scripts/camera_ready
  crdir canonical_ci "$CRR/canonical_ci" '*.txt'      "$CRS/run_canonical_ci.sh"
  crdir qwen_ladder  "$CRR/qwen_ladder"  '*.txt'      "$CRS/run_b_qmc.sh"
  crdir 2x2          "$CRR/2x2"          '*.txt'      "$CRS/run_2x2.sh"
  crdir e5           "$CRR/e5"           '*.txt'      "$CRS/run_e5_qmc.sh"
  crdir bridge_type  "$CRR/bridge_type"  '*.txt'      "$CRS/run_bridge_type_qmc.sh"
  crdir screening    "$CRR/screening"    '*.txt'      "$CRS/run_screening_sensitivity.sh"
  crdir exclusion    "$CRR/exclusion"    '*.txt'      "$CRS/run_exclusion_sensitivity.sh"  # screening 부분집합을 쓴다
  crdir qualitative  "$CRR/qualitative"  'cases_*.md' "$CRS/run_qualitative.sh"
  crdir self_translate "$CRR/self_translate" '*'      "$CRS/run_self_translate.sh"
  crdir e5_truncation "$CRR/retriever_robustness/e5_truncation" '*' "$CRS/run_e5_truncation.sh"  # E5 토크나이저 캐시 필요
  crdir gold_entity_ub "$CRR/retriever_robustness/gold_entity_ub" '*.txt' "$CRS/run_gold_entity_ub.sh"
  crdir matched      "$CRR/matched"      '*'          "$CRS/run_matched.sh"
  crdir judge_leniency "$CRR/judge_leniency" '*'      "$CRS/run_judge_leniency.sh"
  crdir construction "$CRR/construction" '*'          "$CRS/run_construction_audit.sh"
  crdir template_audit "$CRR/template_audit" '*'      "$CRS/run_template_audit.sh"
  crdir correctness  "$CRR/correctness"  '*'          "$CRS/run_correctness.sh"   # 정답 정확도·궤적 성공률
  crdir hop_qmc      "$CRR/hop_qmc"      '*.txt'      "$CRS/run_hop_qmc.sh"       # hop별 QMC 분해
fi

echo
if [ $fail -eq 0 ]; then echo "전부 일치"; else echo "차분 발생 — 위 diff 확인"; fi
exit $fail
