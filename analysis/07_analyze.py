#!/usr/bin/env python3
"""
analysis/07_analyze.py
=============
annotated.jsonl에서 핵심 지표를 계산하고 출력한다.
별도 API 호출 없이 기존 어노테이션 결과만 사용.

지표:
  - FaithRate   : supported / total claims
  - PDR         : unsupported / total claims  (파라메트릭 의존도 프록시)
  - ContraRate  : contradicted / total claims
  - LSHD        : FaithRate(비전환) - FaithRate(전환)
  - ΔPDR        : PDR(전환) - PDR(비전환)
  - CLEPR_xling : P(k+1 unfaithful | k unfaithful, lang_switch@k)
  - CLEPR_same  : P(k+1 unfaithful | k unfaithful, ~lang_switch@k)
  - AFR         : 기권율

실행:
  python analysis/07_analyze.py [--annotated data/processed/annotated_full.jsonl]
"""

import json
import math
import argparse
import statistics
from collections import defaultdict
from pathlib import Path
import yaml

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.metrics import (clepr_by_pattern, FAITH_HALLU_THRESHOLD,
                           index_class, hop_yield, hop_claim_rates)
from utils.qa_set import restrict


LABELS = {"a": "KO→KO→KO", "b": "KO→EN→KO", "c": "KO→KO→EN", "d": "KO→EN→EN"}
# FAITH_HALLU_THRESHOLD은 utils.metrics에서 import


# ─── 필터링 ──────────────────────────────────────────────────────────────────

def build_excluded_qids(
    qa_pairs_path: Path | None,
    trajectories_path: Path | None,
    multi_annotated_paths: list[Path] | None,
    abstain_threshold: int,
) -> tuple[set[str], dict[str, str]]:
    """제외할 QID 집합과 이유를 반환한다."""
    reasons: dict[str, str] = {}

    # 필터 1: gold answer가 retrieved passages에 없는 QA
    if qa_pairs_path and qa_pairs_path.exists() and trajectories_path and trajectories_path.exists():
        gold = {}
        for l in qa_pairs_path.read_text().splitlines():
            if l.strip():
                q = json.loads(l)
                gold[q["qid"]] = q["answer"].lower()

        for l in trajectories_path.read_text().splitlines():
            if not l.strip():
                continue
            t = json.loads(l)
            qid = t["qid"]
            answer = gold.get(qid, "")
            if not answer:
                continue
            all_passages = " ".join(
                p for step in t["trajectory"]
                for p in step.get("retrieved_passages", [])
            ).lower()
            if answer not in all_passages:
                reasons[qid] = "retrieval_miss"

    # 필터 2: N개 이상 모델에서 최종 기권한 QA
    if multi_annotated_paths:
        abstain_count: dict[str, int] = defaultdict(int)
        for path in multi_annotated_paths:
            if not path.exists():
                continue
            for l in path.read_text().splitlines():
                if l.strip():
                    r = json.loads(l)
                    if r.get("abstained"):
                        abstain_count[r["qid"]] += 1
        for qid, cnt in abstain_count.items():
            if cnt >= abstain_threshold and qid not in reasons:
                reasons[qid] = f"abstain_{cnt}/{len(multi_annotated_paths)}_models"

    return set(reasons.keys()), reasons


# ─── 홉 레벨 지표 ────────────────────────────────────────────────────────────

# ─── 출력 헬퍼 ───────────────────────────────────────────────────────────────

def fmt(val: float, n: int | None = None) -> str:
    if math.isnan(val):
        return "  N/A  "
    s = f"{val:.3f}"
    if n is not None:
        s += f" (n={n})"
    return s


def section(title: str):
    print(f"\n{'='*62}")
    print(title)
    print('='*62)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--annotated",    default=None,
                        help="어노테이션 JSONL (미지정 시 config output.annotated 사용)")
    parser.add_argument("--multi-annotated", nargs="+", default=None,
                        help="여러 모델의 annotated JSONL (다중 모델 공통 기권 필터용)")
    parser.add_argument("--trajectories", default=None,
                        help="궤적 JSONL (retrieval miss 필터용)")
    parser.add_argument("--qa-pairs",     default=None,
                        help="QA pairs JSONL (retrieval miss 필터용, 미지정 시 config 사용)")
    parser.add_argument("--filter-abstain-threshold", type=int, default=4,
                        help="N개 이상 모델 기권 시 제외 (기본값: 4=전 모델)")
    parser.add_argument("--no-filter",    action="store_true",
                        help="필터링 없이 전체 데이터 분석")
    parser.add_argument("--out", default=None,
                        help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    args = parser.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    cfg = yaml.safe_load(Path(args.config).read_text())
    annotated_path = Path(args.annotated or cfg["output"]["annotated"])

    if not annotated_path.exists():
        print(f"파일 없음: {annotated_path}")
        return

    results = [json.loads(l) for l in annotated_path.read_text().splitlines() if l.strip()]
    print(f"로드: {annotated_path}  ({len(results)}개 궤적)")

    # ── 분석 대상 QA 로 제한 ────────────────────────────────────────────────
    results = restrict(results, args.qa_pairs, label="궤적")

    # ── 필터링 ─────────────────────────────────────────────────────────────
    if not args.no_filter and (args.trajectories or args.multi_annotated):
        qa_pairs_path    = Path(args.qa_pairs) if args.qa_pairs else Path(cfg["output"]["qa_pairs"])
        trajectories_path = Path(args.trajectories) if args.trajectories else None
        multi_paths = [Path(p) for p in args.multi_annotated] if args.multi_annotated else None

        excluded, reasons = build_excluded_qids(
            qa_pairs_path, trajectories_path, multi_paths, args.filter_abstain_threshold
        )

        section("필터링 결과")
        print(f"  제외 QID: {len(excluded)}개 / {len(results)}개 ({len(excluded)/len(results):.1%})")
        by_reason = defaultdict(list)
        for qid, reason in sorted(reasons.items()):
            by_reason[reason].append(qid)
        for reason, qids in sorted(by_reason.items()):
            print(f"  [{reason}] {len(qids)}개: {', '.join(sorted(qids)[:5])}"
                  + ("..." if len(qids) > 5 else ""))

        before = len(results)
        results = [r for r in results if r["qid"] not in excluded]
        print(f"  → 분석 대상: {before}개 → {len(results)}개")
    else:
        if not args.no_filter:
            print("  (필터 미적용 — --trajectories 또는 --multi-annotated 미지정)")

    # 패턴별 집계 — gold_lang 기준 same-index(KO) / cross-index(EN)
    pat_data: dict[str, dict] = {
        p: {"same": [], "cross": [], "abstained": [], "same_yield": [], "cross_yield": []}
        for p in ["a", "b", "c", "d"]
    }

    for r in results:
        pat = r["pattern"]
        for h in r["hops"]:
            if h.get("ambiguous_answer_marker"):
                continue
            pat_data[pat]["abstained"].append(h.get("abstained", False))
            key = index_class(h)
            # Yield (기권·무claim→0, 분모 포함): utils.hop_yield 단일 소스
            y = hop_yield(h)
            if y is not None:
                pat_data[pat][f"{key}_yield"].append(y)
            # FaithRate/PDR/Contra 표: claim 기반 (기권·무claim → None → 제외)
            m = hop_claim_rates(h)
            if m is not None:
                pat_data[pat][key].append(m)

    # ── 1. FaithRate / PDR / ContraRate ─────────────────────────────────────
    section("패턴별 FaithRate / PDR  (gold_lang 기준)")
    print(f"  same-index  = KO retrieval query → KO index  (gold_lang=ko)")
    print(f"  cross-index = KO retrieval query → EN index  (gold_lang=en)")
    print(f"\n{'패턴':<6} {'인덱스유형':<13} {'FaithRate':>10} {'PDR':>10} {'Contra':>8} {'n':>5}")
    print("-" * 54)

    for pat in ["a", "b", "c", "d"]:
        for key, label in [("same", "same-index"), ("cross", "cross-index")]:
            ms = pat_data[pat][key]
            if not ms:
                continue
            faith  = statistics.mean(m["faith"]  for m in ms)
            pdr    = statistics.mean(m["pdr"]    for m in ms)
            contra = statistics.mean(m["contra"] for m in ms)
            print(f"{pat}({LABELS[pat][:6]}) {label:<13} {faith:>10.3f} {pdr:>10.3f} {contra:>8.3f} {len(ms):>5}")
        sm = pat_data[pat]["same"]
        cr = pat_data[pat]["cross"]
        if sm and cr:
            lshd = statistics.mean(m["faith"] for m in sm) - statistics.mean(m["faith"] for m in cr)
            dpdr = statistics.mean(m["pdr"]   for m in cr) - statistics.mean(m["pdr"]   for m in sm)
            print(f"  └ 패턴내 LSHD={lshd:+.3f}  ΔPDR={dpdr:+.3f}")

    # LSHD_end2end (§9.3): 모든 패턴 same-index 홉 vs cross-index 홉 집계
    all_same  = [m for p in "abcd" for m in pat_data[p]["same"]]
    all_cross = [m for p in "abcd" for m in pat_data[p]["cross"]]
    all_same_yield  = [y for p in "abcd" for y in pat_data[p]["same_yield"]]
    all_cross_yield = [y for p in "abcd" for y in pat_data[p]["cross_yield"]]
    lshd_e2e = dpdr_e2e = float("nan")
    print()
    if all_same and all_cross:
        lshd_e2e = statistics.mean(m["faith"] for m in all_same) - statistics.mean(m["faith"] for m in all_cross)
        dpdr_e2e = statistics.mean(m["pdr"]   for m in all_cross) - statistics.mean(m["pdr"]   for m in all_same)
        print(f"  LSHD_end2end  = same FR - cross FR    = {lshd_e2e:+.3f}"
              f"  (n_same={len(all_same)}, n_cross={len(all_cross)})")
        print(f"  ΔPDR_end2end  = cross PDR - same PDR  = {dpdr_e2e:+.3f}")
    if all_same_yield and all_cross_yield:
        yield_lshd = statistics.mean(all_same_yield) - statistics.mean(all_cross_yield)
        print(f"  Yield-LSHD    = same YR - cross YR    = {yield_lshd:+.3f}"
              f"  (n_same={len(all_same_yield)}, n_cross={len(all_cross_yield)}, 기권 포함)")

    # ── 2. CLEPR ─────────────────────────────────────────────────────────────
    section("CLEPR 패턴별 direction  (H2 primary)")
    print(f"  P(hop1 hallu | hop0 hallu) per transition direction\n")
    cr_bp = clepr_by_pattern(results)
    print(f"{'패턴':>4} {'direction':>10} {'CLEPR':>16} {'n(분모)':>8}")
    print("-" * 42)
    for pat in "abcd":
        r = cr_bp[pat]
        print(f"  {pat}   {r['label']:>10}   {fmt(r['rate'], r['n']):>16}")
    print("-" * 42)
    sw = cr_bp["switch"]
    sm_cr = cr_bp["same"]
    print(f"  switch (b+c)              {fmt(sw['rate'], sw['n']):>16}")
    print(f"  same   (a+d)              {fmt(sm_cr['rate'], sm_cr['n']):>16}")
    if not math.isnan(sw["rate"]) and not math.isnan(sm_cr["rate"]):
        diff = sw["rate"] - sm_cr["rate"]
        print(f"\n  ΔCLEPR (switch - same) = {diff:+.3f}  "
              f"{'← H2 지지' if diff > 0 else '← H2 불지지'}")

    # ── 3. AFR ───────────────────────────────────────────────────────────────
    section("AFR (기권율)")
    print(f"{'패턴':<20} {'AFR':>8} {'기권홉':>8} {'전체홉':>8}")
    print("-" * 46)
    for pat in ["a", "b", "c", "d"]:
        ab = pat_data[pat]["abstained"]
        if ab:
            n_ab = sum(ab)
            afr  = n_ab / len(ab)
            print(f"{pat}({LABELS[pat]})  {afr:>8.1%} {n_ab:>8} {len(ab):>8}")

    # ── 4. 요약 ──────────────────────────────────────────────────────────────
    section("요약 (H1 / H2)")
    if not math.isnan(lshd_e2e):
        print(f"  H1 LSHD_end2end = {lshd_e2e:+.3f}  "
              f"{'← H1 지지 (same > cross)' if lshd_e2e > 0 else '← H1 불지지'}")
        print(f"     ΔPDR_end2end = {dpdr_e2e:+.3f}")
    if not math.isnan(sw["rate"]) and not math.isnan(sm_cr["rate"]):
        diff = sw["rate"] - sm_cr["rate"]
        print(f"  H2 ΔCLEPR (switch - same) = {diff:+.3f}  "
              f"{'← H2 지지' if diff > 0 else '← H2 불지지'}")
    print()


if __name__ == "__main__":
    main()
