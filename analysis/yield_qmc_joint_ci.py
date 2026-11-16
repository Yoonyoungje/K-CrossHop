#!/usr/bin/env python3
"""
analysis/yield_qmc_joint_ci.py

Yield-QMC 2단계 분해의 joint CI — YL_ctrl / stage1 / stage2 / residual(YL_norm)을
**하나의 QID-cluster paired-bootstrap pass**에서 동시에 뽑는다.

매 draw에서 controlled/query-matched/normalized를 같은 resampled QID 집합으로 평가하므로
항등 YL_ctrl = stage1 + stage2 + residual 이 per-draw로 유지되어 네 CI가 jointly coherent하다.
(별도 bootstrap이면 marginal로는 맞아도 항등·정합성이 보장 안 됨.)

11_ablation_compare.bootstrap_qmc와 동일 resampling(QID 단위 random.choices, k=N) · seed.
Yield-LSHD는 utils.metrics.hop_yield/lshd 공유함수 사용 (07/11/13/14와 동일 정의).

--bridge-only (camera-ready): KO 템플릿 + 정규화 bridge 조건(05 dynamic_subq_bridgenorm)을 주면
bridge-first 순서의 분해를 같은 draw 에서 함께 낸다.
  b_stage1 = YL_ctrl − YL_bn   (bridge 정규화 먼저, KO 템플릿에서)
  b_stage2 = YL_bn − YL_norm   (템플릿 정합을 나중에, 정규화 bridge 에서)
  interaction = stage1 − b_stage2 = (YL_ctrl − YL_qm) − (YL_bn − YL_norm)
두 순서의 합은 모두 YL_ctrl − YL_norm 이다.

--cluster-key path_cluster_id: QID 대신 entity path(triple) 250개를 재표집 단위로 쓴다 (네 패턴을 함께
뽑는다, --qa-pairs 필요). 기본(qid)은 기존 출력과 byte 동일하다.

--ref-controlled/--ref-query-matched/--ref-normalized (camera-ready): 비교 모델의 세 조건을 주면
매 draw 에서 같은 재표집 집합으로 두 모델을 함께 평가해 (이 모델 − 비교 모델) 차이의 CI 를 덧붙인다
(같은 경로 위에서 크기·정밀도 차이 검정). 주지 않으면 출력이 기존과 같다.

사용:
  python analysis/yield_qmc_joint_ci.py \
    --controlled data/annotated/exaone_ecological.jsonl \
    --query-matched data/annotated/exaone_query_matched.jsonl \
    --normalized    data/annotated/exaone_normalized.jsonl \
    --qa-pairs      data/dataset/qa_pairs.jsonl \
    --n-boot 10000 --seed 42 --label exaone
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import hop_faith, hop_yield, lshd
from utils.qa_set import load_qids


def load_by_qid(path: str) -> dict:
    return {r["qid"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


VALUE_FN = {"yield": hop_yield, "faith": hop_faith}
_value_fn = hop_yield            # --axis faith 면 main 이 hop_faith 로 바꾼다
_hop_filter = None               # --hop 이면 그 hop 위치만 집계한다


def yl(records: list[dict]) -> float:
    """Yield-LSHD = mean(same yield) − mean(cross yield). utils 공유 정의 (--axis faith 면 FaithRate-LSHD)."""
    return lshd(records, _value_fn, _hop_filter)[0]


def metrics(triples: list[tuple]) -> dict:
    yc = yl([t[0] for t in triples])
    yd = yl([t[1] for t in triples])
    yn = yl([t[2] for t in triples])
    if any(math.isnan(x) for x in (yc, yd, yn)):
        return None
    out = {"YL_ctrl": yc, "stage1": yc - yd, "stage2": yd - yn, "residual": yn}
    if len(triples[0]) > 3:                         # --bridge-only: bridge-first 순서
        yb = yl([t[3] for t in triples])
        if math.isnan(yb):
            return None
        out.update({"b_stage1": yc - yb, "b_stage2": yb - yn, "interaction": (yc - yd) - (yb - yn)})
    return out


def ci(vals: list[float], alpha: float) -> tuple:
    s = sorted(vals)
    n = len(s)
    return s[int(alpha/2 * n)], s[int((1 - alpha/2) * n) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controlled", required=True)
    ap.add_argument("--query-matched", dest="query_matched", required=True)
    ap.add_argument("--normalized", required=True)
    ap.add_argument("--qa-pairs", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label", default="")
    ap.add_argument("--bridge-only", dest="bridge_only", default=None,
                    help="KO 템플릿 + 정규화 bridge 조건 annotated (bridge-first 분해 + 교호작용)")
    ap.add_argument("--axis", default="yield", choices=sorted(VALUE_FN),
                    help="yield(기본) 또는 faith (FaithRate-LSHD, 기권·애매 홉은 분모 제외)")
    ap.add_argument("--hop", type=int, choices=[0, 1], default=None,
                    help="한 hop 위치(0/1)만 집계한다(hop별 분해). 기본은 두 hop 모두 — 기존 출력과 같다")
    ap.add_argument("--cluster-key", default="qid", choices=["qid", "path_cluster_id"],
                    help="재표집 단위 (path_cluster_id = entity path 250개, --qa-pairs 필요)")
    ap.add_argument("--ref-controlled", dest="ref_controlled", default=None,
                    help="비교 모델 controlled annotated — 주면 같은 재표집에서 (이 모델 − 비교 모델) 차이 CI 를 낸다")
    ap.add_argument("--ref-query-matched", dest="ref_query_matched", default=None)
    ap.add_argument("--ref-normalized", dest="ref_normalized", default=None)
    ap.add_argument("--ref-label", dest="ref_label", default="ref")
    args = ap.parse_args()
    ref_paths = (args.ref_controlled, args.ref_query_matched, args.ref_normalized)
    if any(ref_paths) and not all(ref_paths):
        ap.error("--ref-controlled/--ref-query-matched/--ref-normalized 는 함께 줘야 한다")
    global _value_fn, _hop_filter
    _value_fn = VALUE_FN[args.axis]
    if args.hop is not None:
        _hop_filter = lambda h, k=args.hop: h.get("hop") == k  # noqa: E731

    keep = load_qids(args.qa_pairs) if args.qa_pairs else None
    c = load_by_qid(args.controlled); d = load_by_qid(args.query_matched); n = load_by_qid(args.normalized)
    b = load_by_qid(args.bridge_only) if args.bridge_only else None
    ref = [load_by_qid(p) for p in ref_paths] if all(ref_paths) else None
    common = [q for q in c if q in d and q in n and (b is None or q in b)
              and (ref is None or all(q in r for r in ref))
              and (keep is None or q in keep)]
    triples = [(c[q], d[q], n[q]) + ((b[q],) if b else ()) for q in common]
    ref_triples = [tuple(r[q] for r in ref) for q in common] if ref else None

    groups = None
    if args.cluster_key != "qid":
        if not args.qa_pairs:
            ap.error("--cluster-key path_cluster_id 는 --qa-pairs 필요")
        key_of = {r["qid"]: r[args.cluster_key] for r in
                  (json.loads(l) for l in open(args.qa_pairs, encoding="utf-8") if l.strip())}
        groups = {}
        for i, q in enumerate(common):
            groups.setdefault(key_of[q], []).append(i)
        group_ids = list(groups)

    point = metrics(triples)
    if point is None:
        print("계산 불가 (빈 same/cross)"); return
    ref_point = metrics(ref_triples) if ref else None
    if ref and ref_point is None:
        print("비교 모델 계산 불가 (빈 same/cross)"); return
    diff_keys = [k for k in point if ref_point and k in ref_point]

    random.seed(args.seed)
    boot = {k: [] for k in point}
    dboot = {k: [] for k in diff_keys}
    for _ in range(args.n_boot):
        # 인덱스로 뽑는다 — random.choices(triples) 와 같은 난수 소비라 기존 출력이 그대로다.
        if groups is None:
            sel = random.choices(range(len(triples)), k=len(triples))
        else:
            sel = [i for g in random.choices(group_ids, k=len(group_ids)) for i in groups[g]]
        m = metrics([triples[i] for i in sel])
        if m is None:
            continue
        if ref:
            mr = metrics([ref_triples[i] for i in sel])
            if mr is None:
                continue
            for k in diff_keys:
                dboot[k].append(m[k] - mr[k])
        for k, v in m.items():
            boot[k].append(v)

    unit = "QID-cluster joint" if groups is None else f"{args.cluster_key} {len(groups)} joint"
    if args.axis != "yield":
        unit += f", axis={args.axis}"
    if args.hop is not None:
        unit += f", hop={args.hop}"
    print(f"[{args.label or 'model'}]  공통 QID {len(triples)}  (n_boot={args.n_boot}, α={args.alpha}, "
          f"seed={args.seed}, {unit})")
    print(f"  {'지표':<10} {'점추정':>8}   95% CI")
    keys = ["YL_ctrl", "stage1", "stage2", "residual"]
    if b is not None:
        keys += ["b_stage1", "b_stage2", "interaction"]
    for k in keys:
        lo, hi = ci(boot[k], args.alpha)
        sig = " (*)" if (lo > 0 or hi < 0) else ""
        print(f"  {k:<10} {point[k]:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{sig}")
    # 항등 점검 (점추정)
    ident = point["stage1"] + point["stage2"] + point["residual"]
    print(f"  항등 stage1+stage2+residual = {ident:+.3f}  (= YL_ctrl {point['YL_ctrl']:+.3f})")
    if ref:
        print(f"  ── paired 차이 [{args.label or 'model'}] − [{args.ref_label}] (같은 재표집 draw)")
        for k in diff_keys:
            lo, hi = ci(dboot[k], args.alpha)
            sig = " (*)" if (lo > 0 or hi < 0) else ""
            print(f"  Δ{k:<9} {point[k] - ref_point[k]:>+8.3f}   [{lo:+.3f}, {hi:+.3f}]{sig}"
                  f"   ({args.ref_label} {ref_point[k]:+.3f})")


if __name__ == "__main__":
    main()
