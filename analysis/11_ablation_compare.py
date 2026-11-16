#!/usr/bin/env python3
"""
10_ablation_compare.py
======================
Controlled protocol vs Ablation 비교 분석.

  Controlled (ko_subq):  KO sub_q → KO/EN index  (retrieval_query_lang=ko)
  Ablation   (sub_q):    sub_q/sub_q_en → 각 gold_lang index (retrieval_query_lang=gold_lang)

지표:
  - LSHD_controlled, LSHD_ablation
  - QMC = LSHD_controlled - LSHD_ablation
  - EN hop answer_bearing_hit 회복량 (ablation에서 KO→EN mismatch 제거 후)
  - 패턴별 cross-index hop FaithRate 변화 (b/d hop0, c/d hop1)

입력:
  --main-annotated      data/annotated/annotated_{model}_dynamic.jsonl
  --ablation-annotated  data/annotated/annotated_{model}_query_matched.jsonl
  --qa-pairs            data/dataset/qa_pairs.jsonl
  --main-hits / --ablation-hits   answer-bearing hit 파일 (선택)

실행:
  python 10_ablation_compare.py \\
    --main-annotated      data/annotated/aya_ecological.jsonl \\
    --ablation-annotated  data/annotated/aya_query_matched.jsonl \\
    --qa-pairs            data/dataset/qa_pairs.jsonl
"""

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

DEFAULT_N_BOOT = 10000
DEFAULT_ALPHA  = 0.05

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.metrics import (clepr_by_pattern, FAITH_HALLU_THRESHOLD as THRESHOLD,
                           lshd, hop_faith, hop_yield)
from utils.qa_set import restrict


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def fmt(val, n=None) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "  N/A  "
    s = f"{val:.3f}"
    if n is not None:
        s += f" (n={n})"
    return s


def section(title: str):
    print(f"\n{'='*66}")
    print(title)
    print("=" * 66)


# ─── LSHD 계산 ───────────────────────────────────────────────────────────────

def compute_lshd(records: list[dict]) -> tuple[float, float, int, int]:
    """
    (lshd, same_FR, cross_FR, n_same, n_cross) 반환.
    gold_lang 기준 same-index(ko) / cross-index(en). FaithRate 축. utils.lshd 단일 소스.
    """
    return lshd(records, hop_faith)


def compute_yield_gap(records: list[dict]) -> tuple[float, float, float, int, int]:
    """
    Yield 축 LSHD: (gap, same_YR, cross_YR, n_same, n_cross). utils.hop_yield 단일 소스.
    yield = 1[¬기권 ∧ ¬애매 ∧ faith≥0.5] (기권·무claim→0, ambiguous→분모 제외).
    """
    return lshd(records, hop_yield)


# ─── QMC bootstrap CI (QID-level paired resampling) ──────────────────────────

def _ci(values: list[float], alpha: float) -> tuple[float, float]:
    lo, hi = alpha / 2, 1 - alpha / 2
    s = sorted(values)
    n = len(s)
    return s[int(lo * n)], s[int(hi * n) - 1]


def fmt_ci(est: float, lo: float | None, hi: float | None) -> str:
    if est is None or math.isnan(est):
        return "  N/A  "
    if lo is None:
        return f"{est:+.3f}  [CI N/A]"
    sig = "(*)" if (lo > 0 or hi < 0) else "   "
    return f"{est:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}"


def bootstrap_qmc(main_ann: list[dict], abl_ann: list[dict], gap_fn,
                  n_boot: int, alpha: float) -> tuple[float | None, float | None, float | None]:
    """QMC = gap(main) − gap(ablation)의 QID-level paired bootstrap CI.

    같은 QID 집합을 한 번 resample해 main/ablation 양쪽에 동일 적용 → 두 모드의
    QID별 상관을 반영한 QMC 차이의 percentile CI. gap_fn ∈ {compute_lshd, compute_yield_gap}.
    """
    main_by = {r["qid"]: r for r in main_ann}
    abl_by  = {r["qid"]: r for r in abl_ann}
    pairs = [(main_by[q], abl_by[q]) for q in main_by if q in abl_by]
    if not pairs:
        return None, None, None

    def qmc_of(sample_pairs):
        mg = gap_fn([p[0] for p in sample_pairs])[0]
        ag = gap_fn([p[1] for p in sample_pairs])[0]
        if math.isnan(mg) or math.isnan(ag):
            return None
        return mg - ag

    point = qmc_of(pairs)
    if point is None:
        return None, None, None
    boot = []
    for _ in range(n_boot):
        v = qmc_of(random.choices(pairs, k=len(pairs)))
        if v is not None:
            boot.append(v)
    if len(boot) < n_boot * 0.9:
        return point, None, None
    lo, hi = _ci(boot, alpha)
    return point, lo, hi


# ─── 패턴×hop 별 FaithRate ───────────────────────────────────────────────────

def faithrate_by_pat_hop(records: list[dict]) -> dict[tuple, float]:
    """(pattern, hop_idx) → mean FaithRate."""
    buckets: dict[tuple, list] = defaultdict(list)
    for r in records:
        for h in r["hops"]:
            if h.get("abstained") or h.get("ambiguous_answer_marker"):
                continue
            f = h.get("hop_faithfulness")
            if f is None:
                continue
            buckets[(r["pattern"], h["hop"])].append(f)
    return {k: statistics.mean(v) for k, v in buckets.items()}


# ─── hop0 answer_bearing_hit 비교 ─────────────────────────────────────────

def retrieval_success_by_pattern(hit_records: list[dict],
                                 ann_records: list[dict]) -> dict[str, dict]:
    """패턴별 hop0 answer_bearing_hit 비율."""
    qid_pat = {r["qid"]: r["pattern"] for r in ann_records}
    by_pat: dict[str, list] = defaultdict(list)
    for r in hit_records:
        pat = qid_pat.get(r["qid"])
        if pat:
            by_pat[pat].append(r["answer_bearing_hit"])
    return {
        pat: {"rate": sum(v) / len(v), "n": len(v)}
        for pat, v in by_pat.items()
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-annotated",      required=True,
                        help="Main protocol annotated JSONL (ko_subq 모드)")
    parser.add_argument("--ablation-annotated",  required=True,
                        help="Ablation 1 annotated JSONL (lang_matched/sub_q 모드)")
    parser.add_argument("--main-hits",             default=None,
                        help="Main hop0 answer_bearing_hit JSONL (선택)")
    parser.add_argument("--ablation-hits",         default=None,
                        help="Ablation hop0 answer_bearing_hit JSONL (선택)")
    parser.add_argument("--qa-pairs",            default=None,
                        help="분석 대상 QA JSONL (지정 시 이 파일의 qid 로 제한)")
    parser.add_argument("--n-boot", type=int,   default=DEFAULT_N_BOOT,
                        help="QMC bootstrap 반복수 (0이면 CI 생략)")
    parser.add_argument("--alpha",  type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--out", default=None,
                        help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    args = parser.parse_args()
    from utils.report import start_tee
    start_tee(args.out)
    random.seed(args.seed)

    main_ann = load_jsonl(args.main_annotated)
    abl_ann  = load_jsonl(args.ablation_annotated)

    # ── 분석 대상 QA 로 제한 (analysis/07_analyze.py와 동일 기준) ────────────────────────
    main_ann = restrict(main_ann, args.qa_pairs, label="main")
    abl_ann  = restrict(abl_ann,  args.qa_pairs, label="ablation")

    model = Path(args.main_annotated).stem.replace("annotated_", "")
    print(f"\n모델: {model}")
    print(f"  Main      (ko_subq):      {len(main_ann)}건")
    print(f"  Ablation1 (lang_matched): {len(abl_ann)}건")

    # ── 1. LSHD_main / LSHD_ablation / QMC ──────────────────────────────────
    section("1. LSHD_end2end — Main vs Ablation")
    print(f"\n{'조건':<22} {'same-index FR':>16} {'cross-index FR':>16} {'LSHD':>8}")
    print("-" * 66)

    m_lshd, m_sm, m_cr, m_ns, m_nc = compute_lshd(main_ann)
    a_lshd, a_sm, a_cr, a_ns, a_nc = compute_lshd(abl_ann)

    print(f"  {'Main (ko_subq)':<20} {m_sm:.3f} (n={m_ns:4d})   {m_cr:.3f} (n={m_nc:4d})   {m_lshd:+.3f}")
    print(f"  {'Ablation1 (matched)':<20} {a_sm:.3f} (n={a_ns:4d})   {a_cr:.3f} (n={a_nc:4d})   {a_lshd:+.3f}")

    qmc = m_lshd - a_lshd if not (math.isnan(m_lshd) or math.isnan(a_lshd)) else float("nan")
    print()
    print(f"  QMC = LSHD_main − LSHD_ablation = {m_lshd:+.3f} − {a_lshd:+.3f} = {qmc:+.3f}")
    if args.n_boot > 0:
        _, qlo, qhi = bootstrap_qmc(main_ann, abl_ann, compute_lshd, args.n_boot, args.alpha)
        print(f"  QMC 95% CI (QID paired bootstrap, n_boot={args.n_boot}): {fmt_ci(qmc, qlo, qhi)}")
    print()
    if not math.isnan(qmc):
        if qmc > 0.05:
            print("  → QMC 양수·큼: cross-index 손실의 주요 원인은 KO query → EN index mismatch")
        elif qmc > 0:
            print("  → QMC 양수·소: query mismatch 기여 있으나 residual grounding gap도 존재")
        else:
            print("  → QMC ≤ 0: query mismatch보다 EN index/evidence grounding 자체가 주 원인")

    # ── 1-B. Yield-QMC (yield 축 — FaithRate 착시에서 실제 손실 분해) ──────────
    section("1-B. Yield-QMC — Main vs Ablation  (yield 축, 기권→0 분모 포함)")
    print(f"\n{'조건':<22} {'same-index YR':>16} {'cross-index YR':>16} {'Yield-LSHD':>11}")
    print("-" * 66)
    m_gap, m_ysm, m_ycr, m_yns, m_ync = compute_yield_gap(main_ann)
    a_gap, a_ysm, a_ycr, a_yns, a_ync = compute_yield_gap(abl_ann)
    print(f"  {'Main (ko_subq)':<20} {m_ysm:.3f} (n={m_yns:4d})   {m_ycr:.3f} (n={m_ync:4d})   {m_gap:+.3f}")
    print(f"  {'Ablation1 (matched)':<20} {a_ysm:.3f} (n={a_yns:4d})   {a_ycr:.3f} (n={a_ync:4d})   {a_gap:+.3f}")
    yqmc = m_gap - a_gap if not (math.isnan(m_gap) or math.isnan(a_gap)) else float("nan")
    print()
    print(f"  Yield-QMC = Yield-LSHD_main − Yield-LSHD_ablation = {m_gap:+.3f} − {a_gap:+.3f} = {yqmc:+.3f}")
    if args.n_boot > 0:
        _, ylo, yhi = bootstrap_qmc(main_ann, abl_ann, compute_yield_gap, args.n_boot, args.alpha)
        print(f"  Yield-QMC 95% CI (QID paired bootstrap, n_boot={args.n_boot}): {fmt_ci(yqmc, ylo, yhi)}")
    print()
    if not math.isnan(yqmc):
        if yqmc > 0.05:
            print("  → Yield-QMC 양수·큼: cross-index yield 손실의 주원인은 KO query → EN index mismatch")
        elif yqmc > 0:
            print("  → Yield-QMC 양수: query mismatch가 yield 손실에 기여 (matched 시 yield 회복)")
        else:
            print("  → Yield-QMC ≤ 0: yield 손실이 query mismatch로 설명되지 않음")

    # ── 2. 패턴×hop FaithRate 변화 ───────────────────────────────────────────
    section("2. 패턴별 cross-index hop FaithRate 변화  (Main → Ablation)")
    print()
    m_fr = faithrate_by_pat_hop(main_ann)
    a_fr = faithrate_by_pat_hop(abl_ann)

    # cross-index hops: b-hop0, c-hop1, d-hop0, d-hop1
    cross_hops = [("b", 0), ("c", 1), ("d", 0), ("d", 1)]
    print(f"  {'hop':<12} {'Main FR':>10} {'Ablation FR':>12} {'Δ (abl−main)':>14}")
    print("  " + "-" * 50)
    for pat, hop in cross_hops:
        m = m_fr.get((pat, hop), float("nan"))
        a = a_fr.get((pat, hop), float("nan"))
        delta = a - m if not (math.isnan(m) or math.isnan(a)) else float("nan")
        delta_s = f"{delta:+.3f}" if not math.isnan(delta) else "  N/A"
        print(f"  {pat}-hop{hop} (cross)  {m:>10.3f}   {a:>10.3f}   {delta_s:>12}")

    # same-index 기준 (변화 없어야 함)
    print()
    same_hops = [("a", 0), ("a", 1), ("b", 1), ("c", 0)]
    print(f"  (same-index hops — 변화 없어야 함)")
    print(f"  {'hop':<12} {'Main FR':>10} {'Ablation FR':>12} {'Δ':>14}")
    print("  " + "-" * 50)
    for pat, hop in same_hops:
        m = m_fr.get((pat, hop), float("nan"))
        a = a_fr.get((pat, hop), float("nan"))
        delta = a - m if not (math.isnan(m) or math.isnan(a)) else float("nan")
        delta_s = f"{delta:+.3f}" if not math.isnan(delta) else "  N/A"
        print(f"  {pat}-hop{hop} (same)   {m:>10.3f}   {a:>10.3f}   {delta_s:>12}")

    # ── 3. hop0 answer_bearing_hit 비교 ────────────────────────────────────
    if args.main_hits and args.ablation_hits:
        section("3. hop0 answer_bearing_hit — Main vs Ablation")
        main_hits = load_jsonl(args.main_hits)
        abl_v3  = load_jsonl(args.ablation_hits)

        m_ret = retrieval_success_by_pattern(main_hits,  main_ann)
        a_ret = retrieval_success_by_pattern(abl_v3,   abl_ann)

        print(f"\n  {'패턴':<6} {'Main succ%':>12} {'Ablation succ%':>16} {'회복량':>10}")
        print("  " + "-" * 46)
        for pat in "abcd":
            m = m_ret.get(pat, {})
            a = a_ret.get(pat, {})
            mr = m.get("rate", float("nan"))
            ar = a.get("rate", float("nan"))
            delta = ar - mr if not (math.isnan(mr) or math.isnan(ar)) else float("nan")
            delta_s = f"{delta:+.1%}" if not math.isnan(delta) else "  N/A"
            mr_s = f"{mr:.1%} (n={m.get('n','?')})" if not math.isnan(mr) else "  N/A"
            ar_s = f"{ar:.1%} (n={a.get('n','?')})" if not math.isnan(ar) else "  N/A"
            print(f"  {pat}     {mr_s:>14} {ar_s:>16} {delta_s:>10}")
        print()
        print("  (b/d hop0에서 회복량이 크면 → KO query→EN index mismatch가 retrieval의 주 원인)")

    # ── 4. CLEPR 비교 ─────────────────────────────────────────────────────────
    section("4. CLEPR — Main vs Ablation")
    m_cr_bp = clepr_by_pattern(main_ann)
    a_cr_bp = clepr_by_pattern(abl_ann)

    print(f"\n  {'패턴':<6} {'Main CLEPR':>14} {'Ablation CLEPR':>16} {'Δ':>10}")
    print("  " + "-" * 48)
    for pat in "abcd":
        mr = m_cr_bp[pat]["rate"]
        ar = a_cr_bp[pat]["rate"]
        delta = ar - mr if not (math.isnan(mr) or math.isnan(ar)) else float("nan")
        delta_s = f"{delta:+.3f}" if not math.isnan(delta) else "  N/A"
        print(f"  {pat}     {fmt(mr, m_cr_bp[pat]['n']):>14} {fmt(ar, a_cr_bp[pat]['n']):>16} {delta_s:>10}")

    # ── 5. 요약 ───────────────────────────────────────────────────────────────
    section("5. 요약")
    print(f"  LSHD_main      = {m_lshd:+.3f}  (n_same={m_ns}, n_cross={m_nc})")
    print(f"  LSHD_ablation  = {a_lshd:+.3f}  (n_same={a_ns}, n_cross={a_nc})")
    if args.n_boot > 0:
        print(f"  QMC            = {fmt_ci(qmc, qlo, qhi)}  (FaithRate 축)")
    else:
        print(f"  QMC            = {qmc:+.3f}  (query-index mismatch contribution, FaithRate 축)")
    print(f"  Yield-LSHD_main     = {m_gap:+.3f}  (n_same={m_yns}, n_cross={m_ync})")
    print(f"  Yield-LSHD_ablation = {a_gap:+.3f}  (n_same={a_yns}, n_cross={a_ync})")
    if args.n_boot > 0:
        print(f"  Yield-QMC      = {fmt_ci(yqmc, ylo, yhi)}  (yield 축)")
    else:
        print(f"  Yield-QMC      = {yqmc:+.3f}  (query-index mismatch contribution, yield 축)")
    print()
    m_sw = m_cr_bp["switch"]; m_sa = m_cr_bp["same"]
    a_sw = a_cr_bp["switch"]; a_sa = a_cr_bp["same"]
    if not math.isnan(m_sw["rate"]) and not math.isnan(m_sa["rate"]):
        print(f"  ΔCLEPR_main     = {m_sw['rate'] - m_sa['rate']:+.3f}")
    if not math.isnan(a_sw["rate"]) and not math.isnan(a_sa["rate"]):
        print(f"  ΔCLEPR_ablation = {a_sw['rate'] - a_sa['rate']:+.3f}")
    print()


if __name__ == "__main__":
    main()
