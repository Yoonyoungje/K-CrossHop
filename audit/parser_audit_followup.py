#!/usr/bin/env python3
"""
audit/parser_audit_followup.py

parser 기권 감사(parser_abstain_audit_prep.py) 후속 검증. 점추정은 건드리지 않고
CI 방식·조건 분해·재가중 granularity·감도만 재계산한다. 정본 데이터는 읽기만 한다.

서브커맨드
  ci      A-1  영건수 층 CI 퇴화 확인 + 유한모집단 베이지안 CI 재계산
  split   A-2  FP/FN의 same/cross 분해 (조건별 P/R, 층×조건, AFR 편향)
  weights A-4  모델×층 재가중 내역 + 층 단위 단순 재가중과의 차이
  sens    A-5  unclear·tie 양극단 감도 + unclear의 실제 층 배분
  shards  A-8  어노테이터×층 배분 + 층 구성 보정 후 라벨러별 P/R

CI 방법 (A-1)
  A(현행) 층화 비모수 부트스트랩: 셀 (class,config,stratum) 내 복원추출 → 재가중.
          셀의 오류가 0건이면 모든 재표본이 0 → 그 셀 불확실성이 CI에 전혀 안 들어간다.
  B(제안) 유한모집단 베이지안: 셀별 p ~ Beta(k+0.5, n-k+0.5) [Jeffreys],
          미표집분 K_rest ~ Binomial(N-n, p), K = k + K_rest.
          n=N(전수)이면 K=k로 분산 0 → 전수 층에 가짜 불확실성을 넣지 않는다.
          k=0 층에서도 상한이 열린다.
  점추정은 두 방법 모두 k/n 재가중 그대로다.

  python audit/parser_audit_followup.py ci \
    --filled 'human_audit/parser_majority/*.csv'
"""
import argparse
import csv
import glob
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "audit"))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "prep", REPO / "audit" / "parser_abstain_audit_prep.py")
prep_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep_mod)

from utils.qa_set import load_qids  # noqa: E402

import numpy as np

QA_PAIRS = "data/dataset/qa_pairs.jsonl"

# 정본 레시피: 8B=기권 판정 일원화 반영본, frontier=원본
MODELS = {
    "aya":    ("data/annotated/aya_ecological.jsonl",
               "data/trajectories/aya_ecological.jsonl"),
    "exaone": ("data/annotated/exaone_ecological.jsonl",
               "data/trajectories/exaone_ecological.jsonl"),
    "llama":  ("data/annotated/llama_ecological.jsonl",
               "data/trajectories/llama_ecological.jsonl"),
    "qwen":   ("data/annotated/qwen_ecological.jsonl",
               "data/trajectories/qwen_ecological.jsonl"),
    "sonnet": ("data/annotated/sonnet_ecological.jsonl",
               "data/trajectories/sonnet_ecological.jsonl"),
    "gemini": ("data/annotated/gemini_ecological.jsonl",
               "data/trajectories/gemini_ecological.jsonl"),
}


# ─── 공통 ────────────────────────────────────────────────────────────────────
def load_filled(patterns):
    rows = []
    for pat in patterns:
        for f in sorted(glob.glob(pat)):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                r["_file"] = Path(f).name          # id_map 대조용 (basename, idx)
                rows.append(r)
    return rows


def valid_rows(rows):
    return [r for r in rows if (r.get("human_label") or "").strip() in ("abstain", "answer")]


def cond_of(r):
    """same/cross. gold_lang 기준 (utils.metrics.index_class와 동일 정의)."""
    return "cross" if (r.get("gold_lang") or "").strip() == "en" else "same"


def cond_from_pattern(pattern, hop):
    """논문 §4.1의 패턴×홉 규칙. gold_lang과 일치하는지 교차검증용."""
    pattern, hop = (pattern or "").strip(), int(hop)
    if pattern == "a":
        return "same"
    if pattern == "d":
        return "cross"
    if pattern == "b":
        return "cross" if hop == 0 else "same"
    if pattern == "c":
        return "same" if hop == 0 else "cross"
    return "?"


def cells_of(rows, extra_key=None):
    """→ {(class, config, stratum[, extra]): {"n":, "err":, "pop":}}  err=소수(=오류) 라벨."""
    out = defaultdict(lambda: {"n": 0, "err": 0, "pop": 0})
    for r in rows:
        key = (r["parser_class"], r["config"], r["stratum"])
        if extra_key:
            key = key + (extra_key(r),)
        c = out[key]
        c["n"] += 1
        lab = r["human_label"].strip()
        minor = "answer" if r["parser_class"] == "abstain" else "abstain"
        c["err"] += (lab == minor)
    return out


def prf(TP, FP, FN):
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = (2 * prec * rec / (prec + rec)
          if prec and rec and not (math.isnan(prec) or math.isnan(rec)) else float("nan"))
    return prec, rec, f1


def compose(cells, pops, err_counts):
    """err_counts: {cell: 모집단 오류 홉수} → (TP, FP, FN, TN)."""
    TP = FP = FN = TN = 0.0
    for cell, N in pops.items():
        e = err_counts[cell]
        if cell[0] == "abstain":
            FP += e
            TP += N - e
        else:
            FN += e
            TN += N - e
    return TP, FP, FN, TN


def cp_upper_zero(n, alpha=0.05):
    """k=0일 때 Clopper-Pearson 상한 = 1 - (alpha/2)^(1/n)."""
    return 1.0 - (alpha / 2) ** (1.0 / n)


# ─── A-1 ─────────────────────────────────────────────────────────────────────
def method_a(cells, pops, n_boot, seed):
    """현행: 층화 비모수 부트스트랩 (셀 내 복원추출)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n_boot):
        err = {}
        for cell, c in cells.items():
            n, k = c["n"], c["err"]
            draws = sum(1 for _ in range(n) if rng.random() < k / n) if n else 0
            err[cell] = draws / n * pops[cell] if n else 0.0
        out.append(prf(*compose(cells, pops, err)[:3]))
    return out


def method_b(cells, pops, n_boot, seed, split_prior=False):
    """제안: 유한모집단 베이지안. n=N이면 분산 0, k=0이어도 상한이 열린다.

    split_prior=True면 Jeffreys 질량 0.5를 층 안의 셀 수로 나눈다. 셀마다 0.5씩 넣으면
    (config 6개 × 층) 가짜 오류가 누적돼 추정 중심이 점추정에서 밀린다."""
    rng = np.random.default_rng(seed)
    keys = list(cells)
    n = np.array([cells[c]["n"] for c in keys], dtype=float)
    k = np.array([cells[c]["err"] for c in keys], dtype=float)
    N = np.array([pops[c] for c in keys])
    rest = np.maximum(N - n.astype(int), 0)
    if split_prior:
        per_str = Counter((c[0], c[2]) for c in keys)
        a0 = np.array([0.5 / per_str[(c[0], c[2])] for c in keys])
    else:
        a0 = np.full(len(keys), 0.5)
    p = rng.beta(k + a0, n - k + a0, size=(n_boot, len(keys)))
    K = k + rng.binomial(np.broadcast_to(rest, p.shape), p)
    out = []
    for i in range(n_boot):
        err = {c: float(K[i, j]) for j, c in enumerate(keys)}
        out.append(prf(*compose(cells, pops, err)[:3]))
    return out


def pct(vals, idx):
    v = sorted(x[idx] for x in vals if not math.isnan(x[idx]))
    if not v:
        return float("nan"), float("nan")
    return v[int(0.025 * len(v))], v[min(int(0.975 * len(v)), len(v) - 1)]


def pool_by_stratum(cells, pops):
    """(class, config, stratum) → (class, stratum)로 합산. 사전분포 질량을 층당 1개로 줄인다."""
    c2 = defaultdict(lambda: {"n": 0, "err": 0})
    p2 = defaultdict(int)
    for c, v in cells.items():
        k = (c[0], "_pooled", c[2])
        c2[k]["n"] += v["n"]; c2[k]["err"] += v["err"]
        p2[k] += pops[c]
    return dict(c2), dict(p2)


def cmd_ci(args):
    rows = valid_rows(load_filled(args.filled))
    cells = cells_of(rows)
    pops = {}
    for r in rows:
        pops[(r["parser_class"], r["config"], r["stratum"])] = int(float(r["stratum_pop"]))
    point_err = {c: cells[c]["err"] / cells[c]["n"] * pops[c] for c in cells}
    TP, FP, FN, TN = compose(cells, pops, point_err)
    pt = prf(TP, FP, FN)

    print("[CI 계산 방식 — parser_abstain_audit_prep.py:246-267]")
    print("  층화 비모수 부트스트랩. 셀=(parser_class, config, stratum, stratum_pop),")
    print("  셀 내 복원추출 후 _reweighted_cells()로 재가중. 연속성 보정·사전분포 없음.")
    print("  → 셀의 오류가 0건이면 f_abstain=0/n=0이 모든 재표본에서 재현된다(:201).\n")
    print(f"[점추정] N̂ TP={TP:.0f} FP={FP:.0f} FN={FN:.0f} TN={TN:.0f}"
          f"  P={pt[0]:.3f} R={pt[1]:.3f} F1={pt[2]:.3f}")

    a_all = method_a(cells, pops, args.n_boot, args.seed)
    b_all = method_b(cells, pops, args.n_boot, args.seed)
    b_spl = method_b(cells, pops, args.n_boot, args.seed, split_prior=True)
    pc, pp = pool_by_stratum(cells, pops)
    b_pool = method_b(pc, pp, args.n_boot, args.seed)
    print(f"\n[전체 CI]")
    print(f"  {'지표':<11}{'점추정':>8}   {'A 현행':<20}{'B 셀별':<20}{'B′ 층정규화':<20}{'B 층pooled'}")
    for i, nm in enumerate(["Precision", "Recall", "F1"]):
        la, ha = pct(a_all, i); lb, hb = pct(b_all, i)
        ls, hs = pct(b_spl, i); lp, hp = pct(b_pool, i)
        print(f"  {nm:<11}{pt[i]:>8.3f}   [{la:.3f}, {ha:.3f}]{'':<6}[{lb:.3f}, {hb:.3f}]{'':<6}"
              f"[{ls:.3f}, {hs:.3f}]{'':<6}[{lp:.3f}, {hp:.3f}]")
    print("  B′ = 셀별 granularity(점추정과 동일) + Jeffreys 질량을 층당 0.5로 정규화 → 권고안")

    # 층별 기여도: 해당 층 셀만 재표집, 나머지는 점추정 고정
    print("\n[층별 CI 기여도 — 해당 층만 흔들고 나머지는 점추정 고정]")
    print(f"  {'층':<28}{'A 현행 Recall CI':<26}{'B 셀별 Recall CI':<26}{'A Precision CI'}")
    strata = ["short", "normal", "clean", "gold_in_passage", "long", "no_abstain_kw"]
    for st in strata:
        act = {c: v for c, v in cells.items() if c[2] == st}
        if not act:
            continue
        fixed = {c: point_err[c] for c in cells if c[2] != st}

        def run(method):
            out = []
            sub_p = {c: pops[c] for c in act}
            draws = method(act, sub_p, args.n_boot, args.seed)
            return draws
        # 부분 재표집: act 셀만 무작위, 나머지 고정
        def partial(method):
            res = []
            if method == "A":
                rng = random.Random(args.seed)
                for _ in range(args.n_boot):
                    err = dict(fixed)
                    for c, v in act.items():
                        n, k = v["n"], v["err"]
                        d = sum(1 for _ in range(n) if rng.random() < k / n)
                        err[c] = d / n * pops[c]
                    res.append(prf(*compose(cells, pops, err)[:3]))
            else:
                rng = np.random.default_rng(args.seed)
                keys = list(act)
                n = np.array([act[c]["n"] for c in keys]); k = np.array([act[c]["err"] for c in keys])
                N = np.array([pops[c] for c in keys]); rest = np.maximum(N - n, 0)
                p = rng.beta(k + 0.5, n - k + 0.5, size=(args.n_boot, len(keys)))
                K = k + rng.binomial(np.broadcast_to(rest, p.shape), p)
                for i in range(args.n_boot):
                    err = dict(fixed)
                    for j, c in enumerate(keys):
                        err[c] = float(K[i, j])
                    res.append(prf(*compose(cells, pops, err)[:3]))
            return res
        ra, rb = partial("A"), partial("B")
        lar, har = pct(ra, 1); lbr, hbr = pct(rb, 1); lap, hap = pct(ra, 0)
        cls = "nonabstain" if st in ("short", "normal") else "abstain"
        n_ = sum(v["n"] for v in act.values()); k_ = sum(v["err"] for v in act.values())
        print(f"  {cls+'/'+st+f' ({k_}/{n_})':<28}[{lar:.3f}, {har:.3f}]{'':<10}"
              f"[{lbr:.3f}, {hbr:.3f}]{'':<10}[{lap:.3f}, {hap:.3f}]")

    # 영건수 층 닫힌형 상한
    print("\n[영건수 층 상한 — 닫힌형]")
    sub = {c: v for c, v in cells.items() if c[2] == "short"}
    n = sum(v["n"] for v in sub.values()); k = sum(v["err"] for v in sub.values())
    N = sum(pops[c] for c in sub)
    cp, r3 = cp_upper_zero(n), 3.0 / n
    print(f"  short: FN {k}/{n}, 모집단 {N} (전체 {sum(pops.values())}의 {N/sum(pops.values()):.0%})")
    print(f"    Clopper-Pearson 97.5% 상한 = {cp:.4f} → 최대 {cp*N:.0f} 홉")
    print(f"    rule of three (3/n)        = {r3:.4f} → 최대 {r3*N:.0f} 홉")
    for nm, ub in [("CP", cp * N), ("rule of three", r3 * N)]:
        r_worst = TP / (TP + FN + ub)
        print(f"    → short FN이 {nm} 상한이면 Recall = {TP:.0f}/({TP:.0f}+{FN:.0f}+{ub:.0f})"
              f" = {r_worst:.3f}")


# ─── A-2 ─────────────────────────────────────────────────────────────────────
def population_rows(configs, restrict=True):
    """collect()로 표집 전 전수 모집단 재구성 → 조건별 층 모집단."""
    keep = load_qids(QA_PAIRS) if restrict else None
    out = []
    for cfg in configs:
        ann, traj = MODELS[cfg]
        for pc in ("abstain", "nonabstain"):
            a = argparse.Namespace(annotated=ann, trajectory=traj, config=cfg, parser_class=pc)
            out += prep_mod.collect(a, keep)
    return out


def cmd_split(args):
    rows = valid_rows(load_filled(args.filled))

    # 조건 정의 교차검증
    mism = [r for r in rows if cond_of(r) != cond_from_pattern(r["pattern"], r["hop"])]
    print(f"[조건 정의 교차검증] gold_lang 기준 vs 논문 §4.1 패턴×홉 규칙: 불일치 {len(mism)}/{len(rows)}")

    pop_rows = population_rows(sorted({r["config"] for r in rows}))
    pop = defaultdict(int)
    for r in pop_rows:
        pop[(r["parser_class"], r["config"], r["stratum"], cond_of(r))] += 1

    # 재구성 모집단이 CSV의 stratum_pop과 맞는지 검증
    csv_pop, recon = {}, defaultdict(int)
    for r in rows:
        csv_pop[(r["parser_class"], r["config"], r["stratum"])] = int(float(r["stratum_pop"]))
    for (pc, cfg, st, cond), v in pop.items():
        recon[(pc, cfg, st)] += v
    bad = {k: (csv_pop[k], recon[k]) for k in csv_pop if csv_pop[k] != recon.get(k)}
    print(f"[모집단 재구성 검증] CSV stratum_pop과 불일치 셀: {len(bad)}" +
          (f"  {bad}" if bad else "  (전부 일치)"))

    cells = cells_of(rows, extra_key=cond_of)
    pops = {c: pop[c] for c in cells}

    print("\n[출력 2 — 층 × 조건 교차]")
    print(f"  {'stratum':<17}{'same pop':>9}{'same n':>8}{'same FP/FN':>12}"
          f"{'cross pop':>10}{'cross n':>9}{'cross FP/FN':>13}")
    strata = ["no_abstain_kw", "gold_in_passage", "long", "clean", "normal", "short"]
    for st in strata:
        line = {}
        for cond in ("same", "cross"):
            sub = {c: v for c, v in cells.items() if c[2] == st and c[3] == cond}
            line[cond] = (sum(pops[c] for c in sub), sum(v["n"] for v in sub.values()),
                          sum(v["err"] for v in sub.values()))
        if line["same"][0] or line["cross"][0]:
            print(f"  {st:<17}{line['same'][0]:>9}{line['same'][1]:>8}{line['same'][2]:>12}"
                  f"{line['cross'][0]:>10}{line['cross'][1]:>9}{line['cross'][2]:>13}")

    print("\n[출력 1 — 조건별 오류]")
    res = {}
    for cond in ("same", "cross"):
        sub = {c: v for c, v in cells.items() if c[3] == cond}
        sp = {c: pops[c] for c in sub}
        err = {c: sub[c]["err"] / sub[c]["n"] * sp[c] for c in sub}
        TP, FP, FN, TN = compose(sub, sp, err)
        b = method_b(sub, sp, args.n_boot, args.seed, split_prior=True)   # A-1 권고안 B′
        pt = prf(TP, FP, FN)
        res[cond] = (TP, FP, FN, TN, pt, b)
        lp, hp = pct(b, 0)
        lr, hr = pct(b, 1)
        print(f"  {cond:<6} pop={sum(sp.values()):>6}  N̂ TP={TP:>7.0f} FP={FP:>6.0f} "
              f"FN={FN:>6.0f}  P={pt[0]:.3f} [{lp:.3f},{hp:.3f}]  R={pt[1]:.3f} [{lr:.3f},{hr:.3f}]")
    dp = res["cross"][4][0] - res["same"][4][0]
    dr = res["cross"][4][1] - res["same"][4][1]
    print(f"  {'차이':<6} (cross − same)                FP차={res['cross'][1]-res['same'][1]:+.0f}"
          f" FN차={res['cross'][2]-res['same'][2]:+.0f}  ΔP={dp:+.3f}  ΔR={dr:+.3f}")

    print("\n[출력 3 — 모델별 AFR 편향 추정]")
    print("  FP는 AFR을 부풀리고(기권 아닌데 기권), FN은 낮춘다(기권인데 답변).")
    print(f"  {'model':<8}{'cond':<7}{'홉수':>7}{'AFR 관측':>10}{'N̂FP':>7}{'N̂FN':>7}"
          f"{'보정AFR':>9}{'편향(pp)':>10}")
    gap = {}
    denom = defaultdict(int)
    abst = defaultdict(int)
    for r in pop_rows:
        key = (r["config"], cond_of(r))
        denom[key] += 1
        if r["parser_class"] == "abstain":
            abst[key] += 1
    for cfg in sorted({c[1] for c in cells}):
        for cond in ("same", "cross"):
            sub = {c: v for c, v in cells.items() if c[1] == cfg and c[3] == cond}
            if not sub:
                continue
            fp = sum(sub[c]["err"] / sub[c]["n"] * pops[c] for c in sub if c[0] == "abstain")
            fn = sum(sub[c]["err"] / sub[c]["n"] * pops[c] for c in sub if c[0] == "nonabstain")
            d = denom[(cfg, cond)]
            afr = abst[(cfg, cond)] / d if d else float("nan")
            corr = (abst[(cfg, cond)] - fp + fn) / d if d else float("nan")
            print(f"  {cfg:<8}{cond:<7}{d:>7}{afr:>10.3f}{fp:>7.1f}{fn:>7.1f}"
                  f"{corr:>9.3f}{(corr-afr)*100:>+10.2f}")
            gap.setdefault(cfg, {})[cond] = (afr, corr)
    print(f"\n  [same/cross AFR 격차의 변화]  격차 = cross − same")
    print(f"  {'model':<8}{'관측 격차':>11}{'보정 격차':>11}{'변화(pp)':>11}")
    for cfg, v in gap.items():
        if "same" in v and "cross" in v:
            go = v["cross"][0] - v["same"][0]
            gc = v["cross"][1] - v["same"][1]
            print(f"  {cfg:<8}{go:>11.3f}{gc:>11.3f}{(gc-go)*100:>+11.2f}")


# ─── A-4 ─────────────────────────────────────────────────────────────────────
def cmd_weights(args):
    rows = valid_rows(load_filled(args.filled))
    cells = cells_of(rows)
    pops = {}
    for r in rows:
        pops[(r["parser_class"], r["config"], r["stratum"])] = int(float(r["stratum_pop"]))

    print("[재가중 코드 경로]")
    print("  parser_abstain_audit_prep.py:189 _reweighted_cells()")
    print("  셀 키 = (parser_class, config, stratum, stratum_pop) → granularity = 모델 × 층")
    print("  §5 층별 표는 config를 합산해 출력하므로 층 단위로 보이지만 계산은 모델×층이다.\n")

    print(f"  {'class':<11}{'config':<9}{'stratum':<17}{'pop':>6}{'n':>5}{'err':>5}"
          f"{'err율':>8}{'N̂오류':>9}")
    tot = defaultdict(float)
    for c in sorted(cells):
        v = cells[c]
        rate = v["err"] / v["n"]
        contrib = rate * pops[c]
        tot[c[0]] += contrib
        print(f"  {c[0]:<11}{c[1]:<9}{c[2]:<17}{pops[c]:>6}{v['n']:>5}{v['err']:>5}"
              f"{rate:>8.3f}{contrib:>9.1f}")
    print(f"\n  모델×층 재가중 합계:  N̂FP={tot['abstain']:.1f}  N̂FN={tot['nonabstain']:.1f}")

    # 층 단위 단순 재가중
    st_cells = defaultdict(lambda: {"n": 0, "err": 0, "pop": 0})
    for c, v in cells.items():
        s = st_cells[(c[0], c[2])]
        s["n"] += v["n"]; s["err"] += v["err"]; s["pop"] += pops[c]
    simple = defaultdict(float)
    print(f"\n  [층 단위 단순 재가중]")
    print(f"  {'class':<11}{'stratum':<17}{'pop':>6}{'n':>5}{'err':>5}{'err율':>8}{'N̂오류':>9}")
    for c in sorted(st_cells):
        v = st_cells[c]
        rate = v["err"] / v["n"]
        contrib = rate * v["pop"]
        simple[c[0]] += contrib
        print(f"  {c[0]:<11}{c[1]:<17}{v['pop']:>6}{v['n']:>5}{v['err']:>5}"
              f"{rate:>8.3f}{contrib:>9.1f}")
    print(f"\n  층 단순 재가중 합계:  N̂FP={simple['abstain']:.1f}  N̂FN={simple['nonabstain']:.1f}")
    print(f"  차이(모델×층 − 층단순): FP {tot['abstain']-simple['abstain']:+.1f}"
          f"  FN {tot['nonabstain']-simple['nonabstain']:+.1f}")
    print("  원인: 층 내에서 모델별 err율과 pop 비중이 다르면 Simpson 효과로 두 값이 갈린다.")


# ─── A-5 ─────────────────────────────────────────────────────────────────────
def cmd_sens(args):
    all_rows = load_filled(args.filled)
    labeled = [r for r in all_rows if (r.get("human_label") or "").strip()]
    unclear = [r for r in labeled if r["human_label"].strip() == "unclear"]
    unl = [r for r in all_rows if not (r.get("human_label") or "").strip()]

    print(f"[라벨 상태] 행 {len(all_rows)}  라벨 {len(labeled)}"
          f"  (valid {len(labeled)-len(unclear)}, unclear {len(unclear)})  미라벨(tie) {len(unl)}")
    print("\n[unclear 9 + tie 1의 실제 층 배분]")
    print(f"  {'상태':<9}{'class':<11}{'stratum':<17}{'config':<9}{'조건':<6}{'idx'}")
    for r in unclear:
        print(f"  {'unclear':<9}{r['parser_class']:<11}{r['stratum']:<17}{r['config']:<9}"
              f"{cond_of(r):<6}{r['idx']}")
    for r in unl:
        print(f"  {'tie':<9}{r['parser_class']:<11}{r['stratum']:<17}{r['config']:<9}"
              f"{cond_of(r):<6}{r['idx']}")
    print("\n  요약:", dict(Counter((r["parser_class"], r["stratum"])
                                   for r in unclear + unl)))

    pops = {}
    for r in all_rows:
        pops[(r["parser_class"], r["config"], r["stratum"])] = int(float(r["stratum_pop"]))

    print(f"\n[감도] {'시나리오':<26}{'Precision':>10}{'Recall':>9}{'F1':>8}   정본 대비")
    base = None
    for tag, fill in [("정본 (unclear·tie 제외)", None),
                      ("A: 10건 전부 human=abstain", "abstain"),
                      ("B: 10건 전부 human=answer", "answer")]:
        rows = valid_rows(all_rows)[:]
        if fill:
            for r in unclear + unl:
                rr = dict(r); rr["human_label"] = fill
                rows.append(rr)
        cells = cells_of(rows)
        err = {c: cells[c]["err"] / cells[c]["n"] * pops[c] for c in cells}
        pt = prf(*compose(cells, {c: pops[c] for c in cells}, err)[:3])
        if base is None:
            base = pt
            print(f"  {tag:<26}{pt[0]:>10.3f}{pt[1]:>9.3f}{pt[2]:>8.3f}   —")
        else:
            print(f"  {tag:<26}{pt[0]:>10.3f}{pt[1]:>9.3f}{pt[2]:>8.3f}   "
                  f"P{pt[0]-base[0]:+.3f} R{pt[1]-base[1]:+.3f} F1{pt[2]-base[2]:+.3f}")


# ─── A-8 ─────────────────────────────────────────────────────────────────────
def cmd_shards(args):
    per = {}
    for d in sorted(glob.glob("reports/parser_audit/filled_*")):
        name = Path(d).name.replace("filled_", "")
        if name == "majority":
            continue
        per[name] = valid_rows(load_filled([f"{d}/*.csv"]))
    strata = ["no_abstain_kw", "gold_in_passage", "long", "clean", "normal", "short"]
    print("[어노테이터 × 층 배분]")
    print(f"  {'어노테이터':<14}" + "".join(f"{s:>17}" for s in strata) + f"{'계':>6}")
    for n, rows in per.items():
        c = Counter(r["stratum"] for r in rows)
        print(f"  {n:<14}" + "".join(f"{c.get(s,0):>17}" for s in strata) + f"{len(rows):>6}")

    pops = {}
    for rows in per.values():
        for r in rows:
            pops[(r["parser_class"], r["config"], r["stratum"])] = int(float(r["stratum_pop"]))

    print("\n[라벨러별 P/R — 각자 shard 전체 (현행 §4)]")
    print(f"  {'어노테이터':<14}{'n':>5}{'Precision':>11}{'Recall':>9}{'F1':>8}")
    for n, rows in per.items():
        cells = cells_of(rows)
        err = {c: cells[c]["err"] / cells[c]["n"] * pops[c] for c in cells}
        pt = prf(*compose(cells, {c: pops[c] for c in cells}, err)[:3])
        print(f"  {n:<14}{len(rows):>5}{pt[0]:>11.3f}{pt[1]:>9.3f}{pt[2]:>8.3f}")

    # 층 구성 보정: overlap 80건(3인 공통)만으로 비교 → 층 구성 정의상 동일
    ov = set(json.load(open("data/human_audit/parser/parser_overlap_ids.json",
                            encoding="utf-8"))["overlap_ids"])
    idmap = json.load(open("data/human_audit/parser/parser_id_map.json", encoding="utf-8"))
    ov_key = {(Path(idmap[p]["csv"]).name, str(idmap[p]["idx"])) for p in ov if p in idmap}
    print(f"\n[층 구성 보정 — 3인 공통 overlap {len(ov_key)}건만 (층 구성 정의상 동일)]")
    print(f"  {'어노테이터':<14}{'n':>5}{'Precision':>11}{'Recall':>9}{'F1':>8}"
          f"   층분포(no_kw/gold/long/clean/normal/short)")
    for n, rows in per.items():
        sub = [r for r in rows if (r["_file"], r["idx"]) in ov_key]
        if not sub:
            print(f"  {n:<14}  overlap 매칭 0건 — id_map 대조 실패")
            continue
        cells = cells_of(sub)
        err = {c: cells[c]["err"] / cells[c]["n"] * pops[c] for c in cells}
        pt = prf(*compose(cells, {c: pops[c] for c in cells}, err)[:3])
        dist = Counter(r["stratum"] for r in sub)
        print(f"  {n:<14}{len(sub):>5}{pt[0]:>11.3f}{pt[1]:>9.3f}{pt[2]:>8.3f}   "
              + "/".join(str(dist.get(s, 0)) for s in strata))
    print("\n  주: overlap 80건은 3인이 동일 항목을 보므로 층 구성이 완전히 같다.")
    print("      여기서도 차이가 남으면 원인은 shard 구성이 아니라 판정 임계값이다.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("ci", cmd_ci), ("split", cmd_split), ("weights", cmd_weights),
                     ("sens", cmd_sens), ("shards", cmd_shards)]:
        p = sub.add_parser(name)
        p.add_argument("--filled", nargs="+",
                       default=["human_audit/parser_majority/*.csv"])
        p.add_argument("--n-boot", type=int, default=10000)
        p.add_argument("--seed", type=int, default=42)
        p.set_defaults(fn=fn)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
