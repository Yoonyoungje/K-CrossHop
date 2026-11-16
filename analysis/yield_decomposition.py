#!/usr/bin/env python3
"""
analysis/yield_decomposition.py

Yield-LSHD를 "답을 냈는가(answer term)" × "낸 답이 근거 있는가(faith term)"로
정확 분해한다. §5.3의 conditioned yield 잔차가 기권율 차이에서 오는지 grounding
차이에서 오는지 판별하는 용도.

항등식 (utils.metrics 정의에서 직접 따라옴):
    hop_yield(h) = 1[hop_faith(h) is not None] · 1[hop_faith(h) >= 0.5]
  → y = u · g          u = 답을 낸 비율(=1−a),  g = 낸 답 중 faith>=0.5 비율
  → Δy = ḡ·Δu + ū·Δg   (ū,ḡ = same/cross 평균; 잔차 항등적으로 0)
         ^answer term   ^faith term

u·g의 두 성분은 hop_yield / hop_faith 로만 유도한다 (인라인 재구현 금지 규칙 준수):
hop_faith는 기권·ambiguous·무채점에서 None이고, hop_yield는 그 경우 0(또는 ambiguous면
None)이므로 위 항등식이 성립한다.

  python analysis/yield_decomposition.py \
    --annotated data/annotated/sonnet_ecological.jsonl \
    --retrieval-hits data/retrieval/sonnet_ecological_hits.jsonl --hop 0

  # 8B는 --qa-pairs 로 분석 대상을 제한
  python analysis/yield_decomposition.py \
    --annotated data/annotated/llama_ecological.jsonl \
    --retrieval-hits data/retrieval/llama_ecological_hits.jsonl --hop 0 \
    --qa-pairs data/dataset/qa_pairs.jsonl
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import (FAITH_HALLU_THRESHOLD as THRESHOLD,
                           index_class, hop_faith, hop_yield)
from utils.qa_set import restrict


def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def collect(ann_records, hit_map, hop_idx, cond):
    """→ {qid: (cls, u_i, g_i|None, abstained)}  cond=True/False/None (v3 조건)."""
    out = {}
    for r in ann_records:
        if len(r["hops"]) <= hop_idx:
            continue
        if cond is not None and hit_map.get(r["qid"], False) != cond:
            continue
        h = r["hops"][hop_idx]
        if hop_yield(h) is None:          # ambiguous → 분모 제외
            continue
        f = hop_faith(h)
        out[r["qid"]] = (index_class(h),
                         0.0 if f is None else 1.0,                       # u_i: 답을 냈는가
                         None if f is None else (1.0 if f >= THRESHOLD else 0.0),
                         bool(h.get("abstained")))                        # 진단용 구성비
    return out


def stats(items):
    """items: [(cls,u,g,abst)] → cls별 (u, g, y, n, n_ans, n_abst)."""
    out = {}
    for cls in ("same", "cross"):
        v = [x for x in items if x[0] == cls]
        if not v:
            out[cls] = None
            continue
        n = len(v)
        u = sum(x[1] for x in v) / n
        ans = [x[2] for x in v if x[2] is not None]
        g = sum(ans) / len(ans) if ans else float("nan")
        out[cls] = {"u": u, "g": g, "y": u * g if ans else float("nan"),
                    "n": n, "n_ans": len(ans),
                    "n_abst": sum(1 for x in v if x[3]),
                    "n_noclaim": sum(1 for x in v if x[1] == 0.0 and not x[3])}
    return out


def decompose(s):
    """Δy = ḡ·Δu + ū·Δg  (잔차 항등적 0)."""
    a, b = s["same"], s["cross"]
    du, dg = a["u"] - b["u"], a["g"] - b["g"]
    ubar, gbar = (a["u"] + b["u"]) / 2, (a["g"] + b["g"]) / 2
    dy = a["y"] - b["y"]
    ans_term, faith_term = gbar * du, ubar * dg
    return dy, ans_term, faith_term, dy - (ans_term + faith_term)


def boot(items_by_qid, B, seed=42):
    """qid 단위 resample → (Δy, answer term, faith term) 분포."""
    qids = sorted(items_by_qid)
    rng = random.Random(seed)
    acc = [[], [], []]
    for _ in range(B):
        smp = [items_by_qid[q] for q in rng.choices(qids, k=len(qids))]
        s = stats(smp)
        if not s["same"] or not s["cross"] or math.isnan(s["same"]["y"]) or math.isnan(s["cross"]["y"]):
            continue
        dy, at, ft, _ = decompose(s)
        acc[0].append(dy); acc[1].append(at); acc[2].append(ft)
    return [sorted(a) for a in acc]


def ci(sorted_vals):
    if not sorted_vals:
        return float("nan"), float("nan")
    n = len(sorted_vals)
    return sorted_vals[int(0.025 * n)], sorted_vals[min(int(0.975 * n), n - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--retrieval-hits", required=True)
    ap.add_argument("--hop", type=int, default=0)
    ap.add_argument("--qa-pairs", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    ann = restrict(load_jsonl(args.annotated), args.qa_pairs)
    v3 = load_jsonl(args.retrieval_hits)
    hit_map = {r["qid"]: r["answer_bearing_hit"] for r in v3 if r.get("hop", 0) == args.hop}

    name = args.label or Path(args.annotated).stem.replace("annotated_", "")
    print(f"\n{'='*78}\n{name}  hop{args.hop}   (annotated={len(ann)})\n{'='*78}")

    for cond, clabel in [(None, "전체 (무조건)"), (True, "v3=True (conditioned)"), (False, "v3=False")]:
        by_qid = collect(ann, hit_map, args.hop, cond)
        s = stats(list(by_qid.values()))
        if not s["same"] or not s["cross"]:
            print(f"\n[{clabel}] 한쪽 그룹이 비어 계산 불가")
            continue
        dy, at, ft, res = decompose(s)
        print(f"\n[{clabel}]")
        print(f"  {'':<8}{'n':>6}{'u=답변율':>11}{'g=답변중faith':>14}{'y=u·g':>9}"
              f"{'기권':>7}{'무claim':>8}")
        for cls in ("same", "cross"):
            c = s[cls]
            print(f"  {cls:<8}{c['n']:>6}{c['u']:>11.3f}{c['g']:>14.3f}{c['y']:>9.3f}"
                  f"{c['n_abst']:>7}{c['n_noclaim']:>8}")
        print(f"  Δy = {dy:+.4f}   =   answer term {at:+.4f}  +  faith term {ft:+.4f}"
              f"   (잔차 {res:+.2e})")
        share = abs(at) / (abs(at) + abs(ft)) if (abs(at) + abs(ft)) > 0 else float("nan")
        print(f"  answer term 기여 비중 {share:.1%}")

        if cond is True and args.n_boot:
            b = boot(by_qid, args.n_boot)
            for lab, a in zip(["Δy        ", "answer term", "faith term "], b):
                lo, hi = ci(a)
                sig = "✅" if (lo > 0 or hi < 0) else "❌"
                print(f"    {lab}  [{lo:+.4f}, {hi:+.4f}] {sig}  (qid bootstrap B={len(a)})")


if __name__ == "__main__":
    main()
