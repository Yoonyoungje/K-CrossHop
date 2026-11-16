#!/usr/bin/env python3
"""
audit/parser_fp_yield_impact.py

parser 감사에서 **전수 조사된 층**(no_abstain_kw / gold_in_passage / long)에서 human=answer로
판정된 82홉 = 위치가 특정된 parser FP. 이 홉들의 abstained를 뒤집었을 때 Δy(yield-LSHD)와
amplification(Δy − Δf)이 얼마나 움직이는지 실측한다.

정본 데이터는 읽기만 한다 — 뒤집기는 메모리 내 사본에서만 한다.

faithfulness 결측 처리
  82홉은 전부 hop_faithfulness=None, claims=0이다(06이 기권 홉의 claim 추출을 건너뜀).
  utils.metrics.hop_yield는 faith 무효 시 0.0을 돌려주므로,
    하한 시나리오(g=0) = abstained만 뒤집기 → yield 변화가 항등적으로 0
    상한 시나리오(g=1) = abstained 뒤집기 + faith=1.0 부여 → 이동폭 최대
  두 시나리오가 참값을 감싼다.

  python audit/parser_fp_yield_impact.py
  python audit/parser_fp_yield_impact.py --cluster-key path_cluster_id   # camera-ready 정본 CI (entity path 250)
"""
import argparse
import csv
import glob
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.metrics import index_class, hop_faith, hop_yield
from utils.qa_set import load_qids

QA_PAIRS = "data/dataset/qa_pairs.jsonl"
FILLED = "human_audit/parser_majority/*.csv"
CENSUS_STRATA = ("no_abstain_kw", "gold_in_passage", "long")

ANNOTATED = {
    "aya":    "data/annotated/aya_ecological.jsonl",
    "exaone": "data/annotated/exaone_ecological.jsonl",
    "llama":  "data/annotated/llama_ecological.jsonl",
    "qwen":   "data/annotated/qwen_ecological.jsonl",
    "sonnet": "data/annotated/sonnet_ecological.jsonl",
    "gemini": "data/annotated/gemini_ecological.jsonl",
}


def fp_hops():
    """→ {config: {(qid, hop)}}  전수층에서 human=answer인 특정된 FP."""
    out = defaultdict(set)
    for f in sorted(glob.glob(FILLED)):
        for r in csv.DictReader(open(f, encoding="utf-8")):
            if (r["parser_class"] == "abstain" and r["stratum"] in CENSUS_STRATA
                    and (r.get("human_label") or "").strip() == "answer"):
                out[r["config"]].add((r["qid"], int(r["hop"])))
    return out


def load(cfg, keep):
    return [json.loads(l) for l in open(ANNOTATED[cfg], encoding="utf-8")
            if l.strip() and json.loads(l)["qid"] in keep]


def apply_flip(records, targets, faith):
    """사본에 FP 뒤집기 적용. faith=None이면 abstained만 해제(하한), 1.0이면 부여(상한)."""
    out = []
    for r in records:
        rr = {**r, "hops": []}
        for h in r["hops"]:
            if (r["qid"], h["hop"]) in targets:
                hh = {**h, "abstained": False}
                if faith is not None:
                    hh["hop_faithfulness"] = faith
                rr["hops"].append(hh)
            else:
                rr["hops"].append(h)
        out.append(rr)
    return out


def gap(records, value_fn):
    same, cross = [], []
    for r in records:
        for h in r["hops"]:
            v = value_fn(h)
            if v is None:
                continue
            (cross if index_class(h) == "cross" else same).append(v)
    if not same or not cross:
        return float("nan")
    return sum(same) / len(same) - sum(cross) / len(cross)


def metrics(records):
    dy = gap(records, hop_yield)
    df = gap(records, hop_faith)
    return dy, df, dy - df


def boot_paired(base, alt, n_boot, seed, groups=None):
    """paired bootstrap: 같은 재표집으로 base·alt를 동시에 계산.
    groups 가 없으면 QID 단위, 있으면 {cluster: [base 인덱스]} 의 클러스터 단위로 재표집한다."""
    idx = {r["qid"]: i for i, r in enumerate(base)}
    qids = sorted(idx)
    gids = sorted(groups) if groups else None
    rng = random.Random(seed)
    acc = {k: [] for k in ("dy_b", "dy_a", "d_dy", "amp_b", "amp_a", "d_amp")}
    for _ in range(n_boot):
        if gids is None:
            pick = [idx[q] for q in (rng.choice(qids) for _ in qids)]
        else:
            pick = [i for g in (rng.choice(gids) for _ in gids) for i in groups[g]]
        b = [base[i] for i in pick]
        a = [alt[i] for i in pick]
        dyb, dfb, ampb = metrics(b)
        dya, dfa, ampa = metrics(a)
        if any(math.isnan(x) for x in (dyb, dya, ampb, ampa)):
            continue
        acc["dy_b"].append(dyb); acc["dy_a"].append(dya); acc["d_dy"].append(dya - dyb)
        acc["amp_b"].append(ampb); acc["amp_a"].append(ampa); acc["d_amp"].append(ampa - ampb)
    return {k: sorted(v) for k, v in acc.items()}


def ci(v):
    if not v:
        return float("nan"), float("nan")
    return v[int(0.025 * len(v))], v[min(int(0.975 * len(v)), len(v) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cluster-key", default="qid", choices=["qid", "path_cluster_id"],
                    help="재표집 단위. path_cluster_id = entity path 250개 (camera-ready 정본 CI). 기본 qid 는 기존 출력과 같다")
    args = ap.parse_args()

    keep = load_qids(QA_PAIRS)
    cluster_of = ({r["qid"]: r[args.cluster_key] for r in map(json.loads, open(QA_PAIRS, encoding="utf-8"))}
                  if args.cluster_key != "qid" else None)
    fps = fp_hops()
    total = sum(len(v) for v in fps.values())
    if cluster_of:
        print(f"[재표집 단위] {args.cluster_key} {len(set(cluster_of.values()))}개 클러스터")
    print(f"특정된 parser FP {total}홉 (전수층 {', '.join(CENSUS_STRATA)})")
    print(f"  모델별: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(fps.items())))

    print(f"\n{'model':<8}{'Δy 정본':>10}{'Δy 하한':>10}{'Δy 상한':>10}{'최대변화':>10}"
          f"   {'Δ변화 95%CI (paired)':<26}{'amp 정본':>10}{'amp 상한':>10}{'amp 변화':>10}"
          f"   {'amp 정본 95%CI':<22}{'amp 상한 95%CI':<22}")
    rows = []
    for cfg in ANNOTATED:
        base = load(cfg, keep)
        tgt = fps.get(cfg, set())
        lo = apply_flip(base, tgt, faith=None)     # 하한: abstained만 해제
        hi = apply_flip(base, tgt, faith=1.0)      # 상한: faith=1 부여
        dyb, dfb, ampb = metrics(base)
        dyl, _, _ = metrics(lo)
        dyh, dfh, amph = metrics(hi)
        groups = None
        if cluster_of:
            groups = defaultdict(list)
            for i, r in enumerate(base):
                groups[cluster_of[r["qid"]]].append(i)
        bs = boot_paired(base, hi, args.n_boot, args.seed, groups)
        clo, chi = ci(bs["d_dy"])
        ablo, abhi = ci(bs["amp_b"])
        ahlo, ahhi = ci(bs["amp_a"])
        mx = max(abs(dyl - dyb), abs(dyh - dyb))
        rows.append((cfg, dyb, dyl, dyh, mx, ampb, amph))
        print(f"{cfg:<8}{dyb:>+10.4f}{dyl:>+10.4f}{dyh:>+10.4f}{mx:>10.4f}"
              f"   [{clo:+.4f}, {chi:+.4f}]{'':<6}{ampb:>+10.4f}{amph:>+10.4f}"
              f"{amph-ampb:>+10.4f}"
              f"   [{ablo:+.4f}, {abhi:+.4f}]{'':<3}[{ahlo:+.4f}, {ahhi:+.4f}]")

    print(f"\n  하한 = abstained만 해제. 82홉 전부 faith 결측이라 hop_yield가 0을 유지 →")
    print(f"         정본과 항등적으로 동일하다 (Δ변화 = 0).")
    print(f"  상한 = 뒤집힌 홉을 전부 faithful(g=1)로 간주 — 물리적 최대 이동폭.")
    worst = max(r[4] for r in rows)
    worst_m = [r[0] for r in rows if r[4] == worst][0]
    print(f"\n  ⇒ 모든 모델·모든 시나리오에서 Δy 최대 변화폭 = {worst:.4f} ({worst_m})")
    wa = max(abs(r[6] - r[5]) for r in rows)
    print(f"  ⇒ amplification 최대 변화폭 = {wa:.4f}")


if __name__ == "__main__":
    main()
