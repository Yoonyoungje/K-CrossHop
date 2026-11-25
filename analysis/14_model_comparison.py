#!/usr/bin/env python3
"""
12_model_comparison.py
======================
모델 간 LSHD / CLEPR / AFR 비교 (RQ4).

입력:
  --annotated  data/annotated/annotated_*_dynamic.jsonl  (복수 지정)

출력 (stdout):
  1. 모델별 LSHD_end2end
  2. 모델별 CLEPR_direction (패턴별 + ΔCLEPR)
  3. 모델별 AFR (패턴별)
  4. 모델별 behavioral regime 분류

실행:
  python 12_model_comparison.py \\
    --annotated data/annotated/aya_ecological.jsonl \\
                data/annotated/exaone_ecological.jsonl \\
                data/annotated/llama_ecological.jsonl \\
                data/annotated/qwen_ecological.jsonl
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.metrics import (clepr_by_pattern, FAITH_HALLU_THRESHOLD as THRESHOLD,
                           lshd, hop_faith, hop_yield)
from utils.qa_set import restrict

LABELS = {"a": "KO→KO→KO", "b": "KO→EN→KO", "c": "KO→KO→EN", "d": "KO→EN→EN"}


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def fmt(val, n=None) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "  N/A  "
    s = f"{val:.3f}"
    if n is not None:
        s += f"({n})"
    return s


def section(title: str):
    print(f"\n{'='*70}")
    print(title)
    print("=" * 70)


# ─── 지표 계산 ────────────────────────────────────────────────────────────────

def compute_lshd(records: list[dict]) -> tuple[float, float, float, int, int]:
    """FaithRate 축 LSHD. utils.lshd 단일 소스."""
    return lshd(records, hop_faith)


def compute_yield_lshd(records: list[dict]) -> tuple[float, float, float, int, int]:
    """Yield 축 LSHD: (gap, same_YR, cross_YR, n_same, n_cross). utils.hop_yield 단일 소스."""
    return lshd(records, hop_yield)


def compute_afr(records: list[dict]) -> dict[str, float]:
    """패턴별 AFR 및 전체 AFR. 분모 = non-ambiguous 홉 (07과 동일)."""
    by_pat: dict[str, list] = defaultdict(list)
    for r in records:
        for h in r["hops"]:
            if h.get("ambiguous_answer_marker"):
                continue
            by_pat[r["pattern"]].append(h.get("abstained", False))
    result = {}
    all_hops = []
    for pat in "abcd":
        hops = by_pat.get(pat, [])
        result[pat] = sum(hops) / len(hops) if hops else float("nan")
        all_hops.extend(hops)
    result["overall"] = sum(all_hops) / len(all_hops) if all_hops else float("nan")
    return result


def classify_regime(lshd: float, delta_clepr: float, afr: float) -> str:
    """behavioral regime 분류."""
    if math.isnan(lshd) or math.isnan(delta_clepr) or math.isnan(afr):
        return "unknown"
    if afr > 0.30:
        return "abstain-heavy"
    if lshd > 0.05 or delta_clepr > 0.10:
        return "hallucination-heavy"
    return "mixed/balanced"


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated", nargs="+", required=True,
                        help="모델별 annotated JSONL (복수 지정)")
    parser.add_argument("--out", default=None,
                        help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    parser.add_argument("--qa-pairs", default=None,
                        help="분석 대상 QA JSONL — 여기 실린 qid 로 궤적을 제한한다")
    args = parser.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    models: list[tuple[str, list[dict]]] = []
    for path in args.annotated:
        name = Path(path).stem.replace("annotated_", "")
        records = restrict(load_jsonl(path), args.qa_pairs, label=f"{name} 궤적")
        models.append((name, records))

    print(f"\n분석 모델: {[m for m, _ in models]}")

    # ── 1. LSHD_end2end ──────────────────────────────────────────────────────
    section("1. LSHD_end2end")
    print(f"  {'모델':<24} {'same-index FR':>16} {'cross-index FR':>16} {'LSHD':>8}")
    print("  " + "-" * 68)

    lshd_results = {}
    for model, records in models:
        lshd, sm, cr, ns, nc = compute_lshd(records)
        lshd_results[model] = (lshd, sm, cr, ns, nc)
        sm_s = f"{sm:.3f}(n={ns})" if not math.isnan(sm) else "  N/A  "
        cr_s = f"{cr:.3f}(n={nc})" if not math.isnan(cr) else "  N/A  "
        lshd_s = f"{lshd:+.3f}" if not math.isnan(lshd) else "  N/A"
        print(f"  {model:<24} {sm_s:>16} {cr_s:>16} {lshd_s:>8}")

    # ── 1-B. Yield-LSHD (yield 축 — FaithRate 착시에서 실제 손실) ──────────────
    section("1-B. Yield-LSHD  (same YR − cross YR, 기권→0; 증폭 = Yield−FR)")
    print(f"  {'모델':<24} {'same-index YR':>16} {'cross-index YR':>16} {'Yield-LSHD':>11} {'증폭':>8}")
    print("  " + "-" * 80)
    yield_results = {}
    for model, records in models:
        gap, ysm, ycr, yns, ync = compute_yield_lshd(records)
        yield_results[model] = (gap, ysm, ycr, yns, ync)
        fr_lshd = lshd_results[model][0]
        amp = gap - fr_lshd if not (math.isnan(gap) or math.isnan(fr_lshd)) else float("nan")
        ysm_s = f"{ysm:.3f}(n={yns})" if not math.isnan(ysm) else "  N/A  "
        ycr_s = f"{ycr:.3f}(n={ync})" if not math.isnan(ycr) else "  N/A  "
        gap_s = f"{gap:+.3f}" if not math.isnan(gap) else "  N/A"
        amp_s = f"{amp:+.3f}" if not math.isnan(amp) else "  N/A"
        print(f"  {model:<24} {ysm_s:>16} {ycr_s:>16} {gap_s:>11} {amp_s:>8}")

    # ── 2. CLEPR_direction ───────────────────────────────────────────────────
    section("2. CLEPR_direction  (primary)")
    print(f"  {'모델':<24} {'a KO→KO':>10} {'b EN→KO':>10} {'c KO→EN':>10} {'d EN→EN':>10} {'ΔCLEPR':>8}")
    print("  " + "-" * 76)

    clepr_results = {}
    for model, records in models:
        cr = clepr_by_pattern(records)
        clepr_results[model] = cr
        sw = cr["switch"]; sa = cr["same"]
        delta = sw["rate"] - sa["rate"] if not (math.isnan(sw["rate"]) or math.isnan(sa["rate"])) else float("nan")
        delta_s = f"{delta:+.3f}" if not math.isnan(delta) else "  N/A"
        vals = [fmt(cr[p]["rate"]) for p in "abcd"]
        print(f"  {model:<24} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {delta_s:>8}")

    # ── 3. AFR ───────────────────────────────────────────────────────────────
    section("3. AFR — 기권율")
    print(f"  {'모델':<24} {'a':>8} {'b':>8} {'c':>8} {'d':>8} {'전체':>8}")
    print("  " + "-" * 60)

    afr_results = {}
    for model, records in models:
        afr = compute_afr(records)
        afr_results[model] = afr
        # 반올림 주의: Python 서식(:.1%)은 정확한 동점을 짝수 쪽으로 반올림한다(round-half-even). 논문은
        # 사사오입(half-up)이라 동점 값은 끝자리가 다를 수 있다. 예: Qwen3 전체 AFR 205/2000 = 10.25%
        # → 여기서는 10.2%, 논문 부록 AFR 표는 10.3%. 출력은 정본 산출물과 byte 비교하므로 서식을 바꾸지 않는다.
        vals = [f"{afr.get(p, float('nan')):.1%}" if not math.isnan(afr.get(p, float('nan'))) else " N/A" for p in "abcd"]
        ov = f"{afr['overall']:.1%}" if not math.isnan(afr["overall"]) else " N/A"
        print(f"  {model:<24} {vals[0]:>8} {vals[1]:>8} {vals[2]:>8} {vals[3]:>8} {ov:>8}")

    # ── 4. Behavioral Regime ─────────────────────────────────────────────────
    section("4. Behavioral Regime  (RQ4)")
    print(f"\n  {'모델':<24} {'FR-LSHD':>8} {'Yield-LSHD':>11} {'ΔCLEPR':>8} {'AFR':>8} {'Regime':<20}")
    print("  " + "-" * 84)

    for model in [m for m, _ in models]:
        lshd = lshd_results[model][0]
        ygap = yield_results[model][0]
        cr   = clepr_results[model]
        sw   = cr["switch"]; sa = cr["same"]
        delta = sw["rate"] - sa["rate"] if not (math.isnan(sw["rate"]) or math.isnan(sa["rate"])) else float("nan")
        afr  = afr_results[model]["overall"]
        regime = classify_regime(lshd, delta, afr)

        lshd_s  = f"{lshd:+.3f}"  if not math.isnan(lshd)  else "  N/A"
        ygap_s  = f"{ygap:+.3f}"  if not math.isnan(ygap)  else "  N/A"
        delta_s = f"{delta:+.3f}" if not math.isnan(delta)  else "  N/A"
        afr_s   = f"{afr:.1%}"    if not math.isnan(afr)    else "  N/A"
        print(f"  {model:<24} {lshd_s:>8} {ygap_s:>11} {delta_s:>8} {afr_s:>8} {regime:<20}")

    print()


if __name__ == "__main__":
    main()
