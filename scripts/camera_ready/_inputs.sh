# camera-ready 분석 스크립트와 tools/verify_baseline.sh 가 공유하는 정본 입력 경로. source 해서 쓴다.
# 8B 정본 = annotated/, frontier 정본 = annotated/*_run1.jsonl.
# 경로를 모델별로 그대로 적는다 — 배포 빌더(tools/build_release.py)가 경로 문자열을 배포 이름으로
# 바꾸므로, 변수로 조립한 파일명은 배포본에서 깨진다.
QA=data/dataset/qa_pairs.jsonl
MODELS="${MODELS:-aya exaone llama qwen sonnet gemini}"

# ann/qm/nm = 판정(dynamic / query-matched / normalized), tqm/tnm = 궤적(query-matched / normalized)
# traj/hits = 13·12·정성 분석의 dynamic 궤적 / answer-bearing hit (8B 는 hop1 이 있는 _all_hops)
_p() { case "$1:$2" in
  ann:aya) echo data/annotated/aya_ecological.jsonl ;;
  ann:exaone) echo data/annotated/exaone_ecological.jsonl ;;
  ann:llama) echo data/annotated/llama_ecological.jsonl ;;
  ann:qwen) echo data/annotated/qwen_ecological.jsonl ;;
  ann:sonnet) echo data/annotated/sonnet_ecological.jsonl ;;
  ann:gemini) echo data/annotated/gemini_ecological.jsonl ;;
  qm:aya) echo data/annotated/aya_query_matched.jsonl ;;
  qm:exaone) echo data/annotated/exaone_query_matched.jsonl ;;
  qm:llama) echo data/annotated/llama_query_matched.jsonl ;;
  qm:qwen) echo data/annotated/qwen_query_matched.jsonl ;;
  qm:sonnet) echo data/annotated/sonnet_query_matched.jsonl ;;
  qm:gemini) echo data/annotated/gemini_query_matched.jsonl ;;
  nm:aya) echo data/annotated/aya_normalized.jsonl ;;
  nm:exaone) echo data/annotated/exaone_normalized.jsonl ;;
  nm:llama) echo data/annotated/llama_normalized.jsonl ;;
  nm:qwen) echo data/annotated/qwen_normalized.jsonl ;;
  nm:sonnet) echo data/annotated/sonnet_normalized.jsonl ;;
  nm:gemini) echo data/annotated/gemini_normalized.jsonl ;;
  tqm:aya) echo data/trajectories/aya_query_matched.jsonl ;;
  tqm:exaone) echo data/trajectories/exaone_query_matched.jsonl ;;
  tqm:llama) echo data/trajectories/llama_query_matched.jsonl ;;
  tqm:qwen) echo data/trajectories/qwen_query_matched.jsonl ;;
  tqm:sonnet) echo data/trajectories/sonnet_query_matched.jsonl ;;
  tqm:gemini) echo data/trajectories/gemini_query_matched.jsonl ;;
  tnm:aya) echo data/trajectories/aya_normalized.jsonl ;;
  tnm:exaone) echo data/trajectories/exaone_normalized.jsonl ;;
  tnm:llama) echo data/trajectories/llama_normalized.jsonl ;;
  tnm:qwen) echo data/trajectories/qwen_normalized.jsonl ;;
  tnm:sonnet) echo data/trajectories/sonnet_normalized.jsonl ;;
  tnm:gemini) echo data/trajectories/gemini_normalized.jsonl ;;
  traj:aya) echo data/trajectories/aya_ecological.jsonl ;;
  traj:exaone) echo data/trajectories/exaone_ecological.jsonl ;;
  traj:llama) echo data/trajectories/llama_ecological.jsonl ;;
  traj:qwen) echo data/trajectories/qwen_ecological.jsonl ;;
  traj:sonnet) echo data/trajectories/sonnet_ecological.jsonl ;;
  traj:gemini) echo data/trajectories/gemini_ecological.jsonl ;;
  hits:aya) echo data/retrieval/aya_ecological_hits.jsonl ;;
  hits:exaone) echo data/retrieval/exaone_ecological_hits.jsonl ;;
  hits:llama) echo data/retrieval/llama_ecological_hits.jsonl ;;
  hits:qwen) echo data/retrieval/qwen_ecological_hits.jsonl ;;
  hits:sonnet) echo data/retrieval/sonnet_ecological_hits.jsonl ;;
  hits:gemini) echo data/retrieval/gemini_ecological_hits.jsonl ;;
  *) echo "_inputs.sh: 정의되지 않은 입력 $1 $2" >&2; return 1 ;;
esac; }
ann()  { _p ann  "$1"; }
qm()   { _p qm   "$1"; }
nm()   { _p nm   "$1"; }
tqm()  { _p tqm  "$1"; }
tnm()  { _p tnm  "$1"; }
traj() { _p traj "$1"; }
hits() { _p hits "$1"; }

# yield-QMC (entity-path 재표집) 한 모델: yqmc <model> <qa 부분집합> [추가 인자...]
yqmc() { local m=$1 qa=$2; shift 2
  python3 analysis/yield_qmc_joint_ci.py --controlled "$(ann "$m")" --query-matched "$(qm "$m")" \
    --normalized "$(nm "$m")" --qa-pairs "$qa" --cluster-key path_cluster_id \
    --n-boot 10000 --seed 42 "$@"; }
