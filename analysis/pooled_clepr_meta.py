#!/usr/bin/env python3
"""
analysis/pooled_clepr_meta.py
=======================
4-model ΔCLEPR pooled meta-analysis (H2 aggregate).

각 모델을 하나의 "study"로 보고 모델별 ΔCLEPR(pattern 기준 switch(b+c) − same(a+d))을
inverse-variance meta-analysis로 통합한다.

  - 모델별 ΔCLEPR 점추정 + bootstrap SE (QID-level resample)
  - Fixed-effect pooled estimate + 95% CI
  - Random-effects (DerSimonian-Laird) pooled estimate + 95% CI
  - 이질성: Cochran Q, df, I²

ΔCLEPR 정의·resampling·분석대상 제한은 analysis/13_bootstrap_ci.py 로직을 재사용한다.

실행:
  python analysis/pooled_clepr_meta.py \\
    --annotated data/annotated/annotated_{aya,exaone,llama,qwen}_dynamic.jsonl \\
    --qa-pairs  data/dataset/qa_pairs.jsonl \\
    [--n-boot 10000] [--alpha 0.05] [--seed 42]
"""

import argparse
import importlib.util
import math
import random
import statistics
from collections import defaultdict
import sys
from pathlib import Path

# ── analysis/13_bootstrap_ci.py 재사용 (모듈명이 숫자로 시작 → importlib) ──────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))  # utils 패키지 import 경로 확보
from utils.qa_set import load_qids

_spec = importlib.util.spec_from_file_location("bootstrap_ci", _ROOT / "analysis/13_bootstrap_ci.py")
_bc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bc)

load_jsonl        = _bc.load_jsonl
delta_clepr_full  = _bc.delta_clepr_full
ci                = _bc.ci


def boot_delta_clepr(records, n_boot, alpha, clusters=None):
    """ΔCLEPR 점추정 + percentile CI + bootstrap SE.

    clusters(=[[record,...], ...])가 주어지면 그 그룹 단위로 재표집한다 (기본은 QID 단위).
    """
    point = delta_clepr_full(records)
    if point is None:
        return None
    boot = []
    for _ in range(n_boot):
        resample = ([r for g in random.choices(clusters, k=len(clusters)) for r in g]
                    if clusters is not None else random.choices(records, k=len(records)))
        v = delta_clepr_full(resample)
        if v is not None:
            boot.append(v)
    lo, hi = ci(boot, alpha)
    se = statistics.pstdev(boot)
    return {"point": point, "lo": lo, "hi": hi, "se": se, "n": len(records)}


def meta_analysis(studies):
    """studies: [{name, point, se}] → fixed/random-effects pooled + 이질성."""
    theta = [s["point"] for s in studies]
    var   = [s["se"] ** 2 for s in studies]
    k     = len(studies)

    # Fixed-effect (inverse-variance)
    w      = [1.0 / v for v in var]
    sw     = sum(w)
    fe     = sum(wi * ti for wi, ti in zip(w, theta)) / sw
    fe_se  = math.sqrt(1.0 / sw)

    # 이질성 (Cochran Q)
    Q  = sum(wi * (ti - fe) ** 2 for wi, ti in zip(w, theta))
    df = k - 1
    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0

    # Random-effects (DerSimonian-Laird)
    C    = sw - sum(wi ** 2 for wi in w) / sw
    tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    wr   = [1.0 / (v + tau2) for v in var]
    swr  = sum(wr)
    re   = sum(wi * ti for wi, ti in zip(wr, theta)) / swr
    re_se = math.sqrt(1.0 / swr)

    z = 1.959963985
    return {
        "fe": fe, "fe_lo": fe - z * fe_se, "fe_hi": fe + z * fe_se,
        "re": re, "re_lo": re - z * re_se, "re_hi": re + z * re_se,
        "Q": Q, "df": df, "I2": I2, "tau2": tau2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", nargs="+", required=True,
                    help="모델별 annotated_*_dynamic.jsonl (2개 이상)")
    ap.add_argument("--qa-pairs", required=True,
                    help="분석 대상 QA JSONL (이 파일의 qid 로 제한)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--alpha",  type=float, default=0.05)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--cluster-bootstrap", action="store_true",
                    help="--qa-pairs 의 --cluster-key 단위로 재표집 (기본은 QID 단위)")
    ap.add_argument("--cluster-key", default="bridge_cluster_id",
                    choices=["bridge_cluster_id", "path_cluster_id"],
                    help="재표집 단위. path_cluster_id = entity path(triple) 250개 (camera-ready 정본 CI)")
    ap.add_argument("--out", default=None,
                    help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    args = ap.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    random.seed(args.seed)
    qa_qids = load_qids(args.qa_pairs)
    qid_to_cluster = {}
    if args.cluster_bootstrap:
        qid_to_cluster = {r["qid"]: r[args.cluster_key] for r in load_jsonl(args.qa_pairs)}

    print(f"\nΔCLEPR pooled meta-analysis  (boot={args.n_boot:,}, α={args.alpha}, "
          f"seed={args.seed}, n={len(qa_qids):,} QA)")
    print("=" * 70)
    print(f"  {'모델':<10} {'ΔCLEPR':>9} {'95% CI':>20} {'SE':>8} {'n':>6}")
    print("  " + "-" * 56)

    studies = []
    for path in args.annotated:
        name = Path(path).stem.replace("annotated_", "").replace("_dynamic", "")
        recs = [r for r in load_jsonl(path) if r["qid"] in qa_qids]
        clusters = None
        if args.cluster_bootstrap:
            by_c = defaultdict(list)
            for r in recs:
                by_c[qid_to_cluster[r["qid"]]].append(r)
            clusters = [by_c[c] for c in sorted(by_c)]
        res = boot_delta_clepr(recs, args.n_boot, args.alpha, clusters)
        if res is None:
            print(f"  {name:<10}  ΔCLEPR 계산 불가 (분모 0)")
            continue
        sig = "(*)" if res["lo"] > 0 or res["hi"] < 0 else ""
        print(f"  {name:<10} {res['point']:>+9.3f} "
              f"[{res['lo']:+.3f}, {res['hi']:+.3f}] {res['se']:>8.3f} {res['n']:>6} {sig}")
        studies.append({"name": name, "point": res["point"], "se": res["se"]})

    if len(studies) < 2:
        print("\n  pooled 불가: study 2개 미만")
        return

    m = meta_analysis(studies)
    print("\n" + "=" * 70)
    print("Pooled (inverse-variance meta-analysis)")
    print("=" * 70)
    fe_sig = "(*)" if m["fe_lo"] > 0 or m["fe_hi"] < 0 else ""
    re_sig = "(*)" if m["re_lo"] > 0 or m["re_hi"] < 0 else ""
    print(f"  Fixed-effect   ΔCLEPR = {m['fe']:+.3f}  "
          f"[{m['fe_lo']:+.3f}, {m['fe_hi']:+.3f}]  {fe_sig}")
    print(f"  Random-effects ΔCLEPR = {m['re']:+.3f}  "
          f"[{m['re_lo']:+.3f}, {m['re_hi']:+.3f}]  {re_sig}   (DerSimonian-Laird)")
    print(f"\n  이질성: Q={m['Q']:.3f} (df={m['df']}), I²={m['I2']:.1f}%, τ²={m['tau2']:.5f}")
    if m["I2"] >= 50:
        print("  → I² ≥ 50%: 모델 간 이질성 큼 → random-effects를 primary로 보고.")
    else:
        print("  → I² < 50%: 이질성 작음 → fixed/random 추정 유사.")


if __name__ == "__main__":
    main()
