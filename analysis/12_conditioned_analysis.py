#!/usr/bin/env python3
"""
analysis/12_conditioned_analysis.py
==========================
hop0/hop1 answer_bearing_hit 조건부 H1/H2 분해 분석.

입력:
  --annotated      data/annotated/{model}_controlled.jsonl
  --retrieval-hits        data/retrieval/retrieval_v3_{model}_controlled.jsonl
                   (hop0 + hop1 두 홉 모두 포함된 파일)

출력 (stdout):
  1. hop0/hop1 answer_bearing_hit 분포 (패턴별)
  2. H1: LSHD_conditioned hop0 (hop0 v3 조건부)
  3. H1: LSHD_conditioned hop1 (hop1 v3 조건부)
  4. H2/CLEPR: hop0 retrieval_success 조건부 분해

실행:
  python analysis/12_conditioned_analysis.py \\
    --annotated data/annotated/aya_controlled.jsonl \\
    --retrieval-hits   data/retrieval/retrieval_v3_aya_controlled.jsonl
"""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.metrics import (clepr_by_pattern, FAITH_HALLU_THRESHOLD as THRESHOLD,
                           index_class, hop_faith, hop_yield)


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
    print(f"\n{'='*62}")
    print(title)
    print("=" * 62)


# ─── 1. answer_bearing_hit × 패턴 분포 ────────────────────────────────────

def print_retrieval_distribution(hit_hop0_map: dict[str, bool], hit_hop1_map: dict[str, bool], ann_records: list[dict]):
    section("1. hop0/hop1 answer_bearing_hit 분포 (패턴별)")
    print(f"{'패턴':<6} {'N':>5} {'hop0 succ%':>12} {'hop1 succ%':>12}")
    print("-" * 38)
    all_total = 0; all_succ0 = 0; all_succ1 = 0
    for pat in "abcd":
        recs = [r for r in ann_records if r["pattern"] == pat]
        total = len(recs)
        succ0 = sum(1 for r in recs if hit_hop0_map.get(r["qid"], False))
        succ1 = sum(1 for r in recs if hit_hop1_map.get(r["qid"], False))
        all_total += total; all_succ0 += succ0; all_succ1 += succ1
        print(f"  {pat}    {total:>5} {succ0/total:>12.1%} {succ1/total:>12.1%}")
    print(f"  전체  {all_total:>5} {all_succ0/all_total:>12.1%} {all_succ1/all_total:>12.1%}")


# ─── 2. H1: LSHD_conditioned (hop0 positions) ──────────────────

def _hop_faithrates_by_index(ann_records, hit_map, hop_idx, success_filter):
    """
    hop_idx 위치 홉을 gold_lang 기준 same-index/cross-index로 분류해 FaithRate 반환.
    success_filter: True(v3 succ만) / False(v3 fail만) / None(전체)
    """
    same_f, cross_f = [], []
    for r in ann_records:
        if len(r["hops"]) <= hop_idx:
            continue
        h = r["hops"][hop_idx]
        f = hop_faith(h)
        if f is None:
            continue
        if success_filter is not None:
            if hit_map.get(r["qid"], False) != success_filter:
                continue
        (cross_f if index_class(h) == "cross" else same_f).append(f)
    return same_f, cross_f


def _print_lshd_conditioned(hit_map: dict, ann_records: list[dict], hop_idx: int, section_num: int, section_title: str):
    section(f"{section_num}. H1: LSHD_conditioned — {section_title}")
    print(f"\n{'조건':<22} {'same-index FR':>16} {'cross-index FR':>16} {'LSHD':>8}")
    print("-" * 66)

    for label, filt in [("전체 (무조건)", None), ("v3 success=True", True), ("v3 success=False", False)]:
        sm, cr = _hop_faithrates_by_index(ann_records, hit_map, hop_idx, filt)
        sm_fr = statistics.mean(sm) if sm else float("nan")
        cr_fr = statistics.mean(cr) if cr else float("nan")
        lshd  = sm_fr - cr_fr if not (math.isnan(sm_fr) or math.isnan(cr_fr)) else float("nan")
        sm_s  = f"{sm_fr:.3f} (n={len(sm)})" if sm else "  N/A  "
        cr_s  = f"{cr_fr:.3f} (n={len(cr)})" if cr else "  N/A  "
        lshd_s = f"{lshd:+.3f}" if not math.isnan(lshd) else "  N/A  "
        print(f"  {label:<20} {sm_s:>16} {cr_s:>16} {lshd_s:>8}")

    print()
    print("  해석: LSHD_conditioned > 0 → retrieval 확보 후에도 cross-index grounding 취약 (H1b)")
    print("        LSHD_conditioned ≈ 0 → end2end 손실의 주 원인은 retrieval failure (H1a)")


def _hop_yields_by_index(ann_records, hit_map, hop_idx, success_filter):
    """hop_idx 홉을 gold_lang 기준 same/cross로 분류해 yield(기권·무효→0) 반환."""
    same_y, cross_y = [], []
    for r in ann_records:
        if len(r["hops"]) <= hop_idx:
            continue
        h = r["hops"][hop_idx]
        if success_filter is not None and hit_map.get(r["qid"], False) != success_filter:
            continue
        y = hop_yield(h)
        if y is None:        # ambiguous → 분모 제외
            continue
        (cross_y if index_class(h) == "cross" else same_y).append(y)
    return same_y, cross_y


def _print_yield_conditioned(hit_map: dict, ann_records: list[dict], hop_idx: int, section_num: int, section_title: str):
    section(f"{section_num}. H1: Yield-LSHD_conditioned — {section_title}  (yield 축, 착시 보정)")
    print(f"\n{'조건':<22} {'same-index YR':>16} {'cross-index YR':>16} {'Yield-LSHD':>11}")
    print("-" * 69)
    for label, filt in [("전체 (무조건)", None), ("v3 success=True", True), ("v3 success=False", False)]:
        sm, cr = _hop_yields_by_index(ann_records, hit_map, hop_idx, filt)
        sm_y = statistics.mean(sm) if sm else float("nan")
        cr_y = statistics.mean(cr) if cr else float("nan")
        gap  = sm_y - cr_y if not (math.isnan(sm_y) or math.isnan(cr_y)) else float("nan")
        sm_s = f"{sm_y:.3f} (n={len(sm)})" if sm else "  N/A  "
        cr_s = f"{cr_y:.3f} (n={len(cr)})" if cr else "  N/A  "
        gap_s = f"{gap:+.3f}" if not math.isnan(gap) else "  N/A  "
        print(f"  {label:<20} {sm_s:>16} {cr_s:>16} {gap_s:>11}")
    print()
    print("  해석: Yield-LSHD_conditioned > 0 → retrieval 확보 후에도 cross-index 기권 잔존 (yield 손실)")
    print("        ≈ 0 → cross yield 손실이 retrieval failure로 설명됨")


# ─── 3. hop1 LSHD_conditioned ─────────────────────────────────────────────


# ─── 4. H2: CLEPR 패턴별 direction + retrieval_success 조건부 분해 ──────────

def _subset_records(ann_records, hit_map, success_filter):
    if success_filter is None:
        return ann_records
    return [r for r in ann_records if hit_map.get(r["qid"], False) == success_filter]


def print_h2_conditioned(hit_map: dict, ann_records: list[dict], section_num: int = 4):
    section(f"{section_num}. H2/CLEPR — 패턴별 direction + answer_bearing_hit 조건부")
    print("  primary: 패턴별 direction  |  secondary: switch vs same aggregate\n")

    for label, filt in [("전체 (무조건)", None), ("v3 success=True", True), ("v3 success=False", False)]:
        subset = _subset_records(ann_records, hit_map, filt)
        cr = clepr_by_pattern(subset)
        print(f"  [{label}]")
        print(f"  {'패턴':>4} {'direction':>10} {'CLEPR':>12} {'n':>5}")
        print("  " + "-" * 34)
        for pat in "abcd":
            r = cr[pat]
            rate_s = fmt(r["rate"], r["n"])
            print(f"    {pat}   {r['label']:>10}  {rate_s:>12}")
        sw = cr["switch"]
        sm = cr["same"]
        delta_s = f"{sw['rate'] - sm['rate']:+.3f}" if not (math.isnan(sw["rate"]) or math.isnan(sm["rate"])) else "N/A"
        print(f"  switch(b+c)           {fmt(sw['rate'], sw['n']):>12}")
        print(f"  same  (a+d)           {fmt(sm['rate'], sm['n']):>12}")
        print(f"  ΔCLEPR = {delta_s}  {'← H2 지지' if not math.isnan(sw['rate']) and sw['rate'] > sm['rate'] else ''}")
        print()


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", required=True)
    parser.add_argument("--retrieval-hits",   required=True)
    parser.add_argument("--out", default=None,
                        help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    args = parser.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    ann_records = load_jsonl(args.annotated)
    hit_records  = load_jsonl(args.retrieval_hits)

    # qid → hop별 answer_bearing_hit 매핑 (hop0/hop1 분리)
    hit_hop0_map: dict[str, bool] = {r["qid"]: r["answer_bearing_hit"] for r in hit_records if r.get("hop", 0) == 0}
    hit_hop1_map: dict[str, bool] = {r["qid"]: r["answer_bearing_hit"] for r in hit_records if r.get("hop", 0) == 1}

    model = Path(args.annotated).stem.replace("annotated_", "")
    print(f"\n모델: {model}  (annotated={len(ann_records)}, hit_hop0={len(hit_hop0_map)}, hit_hop1={len(hit_hop1_map)})")

    print_retrieval_distribution(hit_hop0_map, hit_hop1_map, ann_records)
    _print_lshd_conditioned(hit_hop0_map, ann_records, hop_idx=0, section_num=2,
                            section_title="hop0 same-index vs cross-index")
    _print_lshd_conditioned(hit_hop1_map, ann_records, hop_idx=1, section_num=3,
                            section_title="hop1 same-index vs cross-index")
    _print_yield_conditioned(hit_hop0_map, ann_records, hop_idx=0, section_num="2-B",
                             section_title="hop0 same-index vs cross-index")
    _print_yield_conditioned(hit_hop1_map, ann_records, hop_idx=1, section_num="3-B",
                             section_title="hop1 same-index vs cross-index")
    print_h2_conditioned(hit_hop0_map, ann_records, section_num=4)


if __name__ == "__main__":
    main()
