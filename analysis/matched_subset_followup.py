#!/usr/bin/env python3
"""
analysis/matched_subset_followup.py

matched_subset.py 후속. 기존 결과만 읽는다 (새 생성·채점 없음). 출력은 reports/rebuttal/ 아래.

--section 1  세 조건(controlled / template_matched / surface_normalized) hop0 matched 분석
             a) 조건별 자기 R=1 집합   b) 세 조건 교집합 고정 집합
             c) 조건 간 차이(ctrl−tmpl, tmpl−norm) joint paired bootstrap
             d) 검색 품질 공변량 — R=1 hop0의 top-5 중 gold 포함 패시지 수·첫 포함 순위
--section 2  부트스트랩 단위 민감도 — triple 재표집으로 CI 재계산 (점추정 불변)
             지표 함수는 analysis/13_bootstrap_ci.py 를 그대로 import 해서 쓴다.
--emit-triple-qa PATH
             qa_pairs_core 의 bridge_cluster_id 를 triple id 로 바꾼 사본을 쓴다
             (13_bootstrap_ci.py --cluster-bootstrap 를 triple 단위로 돌리기 위한 입력)

R=1 파일: controlled 는 data/retrieval/ 정본, query_matched/norm 은
data/correctness/retrieval_v3/ (analysis/10_retrieval_proxy.py 재실행본, §7 참조)
"""
import argparse
import importlib.util
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import matched_subset as ms                                        # noqa: E402
from utils.qa_set import restrict                                  # noqa: E402
from utils.retrieval_success import check_answer_bearing_hit       # noqa: E402

_spec = importlib.util.spec_from_file_location("bs13", ROOT / "analysis" / "13_bootstrap_ci.py")
bs13 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bs13)

CONDS = ["controlled", "template_matched", "surface_normalized"]
SHORT = {"controlled": "ctrl", "template_matched": "tmpl", "surface_normalized": "norm"}
ALIAS = ROOT / "data/retrieval/alias_cache.json"

# 원고 값 (비교용). 출처: Table 3 / §5.3 / tab:qmc-yield / tab:qmc-faith / 커밋된 13 출력
PAPER = {
    "aya":    {"df": "+0.058 [+0.019, +0.095]", "dy": "+0.073 [+0.034, +0.111]",
               "amp": "+0.015 [+0.005, +0.026]", "y1": "+0.061 [+0.027, +0.094]",
               "y2": "+0.017 [+0.001, +0.034]", "yr": "-0.005 [-0.039, +0.029]",
               "f1": "+0.048 [+0.014, +0.082]", "f2": "+0.013 [-0.003, +0.030]", "fr": "-0.004",
               "c0f": "+0.025 [-0.013, +0.064]", "c1f": "+0.002 [-0.046, +0.051]",
               "c0y": "+0.037 [-0.004, +0.079]", "c1y": "-0.004 [-0.055, +0.046]"},
    "exaone": {"df": "+0.035 [+0.001, +0.068]", "dy": "+0.085 [+0.043, +0.127]",
               "amp": "+0.050 [+0.029, +0.072]", "y1": "+0.098 [+0.067, +0.131]",
               "y2": "+0.020 [+0.006, +0.035]", "yr": "-0.033 [-0.070, +0.003]",
               "f1": "+0.027 [-0.003, +0.057]", "f2": "+0.005 [-0.010, +0.019]", "fr": "+0.004",
               "c0f": "+0.028 [-0.006, +0.064]", "c1f": "-0.001 [-0.046, +0.044]",
               "c0y": "+0.034 [-0.011, +0.080]", "c1y": "-0.036 [-0.096, +0.024]"},
    "llama":  {"df": "+0.100 [+0.062, +0.139]", "dy": "+0.115 [+0.075, +0.155]",
               "amp": "+0.015 [+0.001, +0.030]", "y1": "+0.086 [+0.053, +0.118]",
               "y2": "+0.031 [+0.014, +0.048]", "yr": "-0.002 [-0.036, +0.032]",
               "f1": "+0.075 [+0.043, +0.107]", "f2": "+0.027 [+0.010, +0.044]", "fr": "-0.001",
               "c0f": "+0.026 [-0.011, +0.063]", "c1f": "+0.025 [-0.018, +0.068]",
               "c0y": "+0.050 [+0.010, +0.090]", "c1y": "+0.034 [-0.012, +0.080]"},
    "qwen":   {"df": "+0.066 [+0.028, +0.105]", "dy": "+0.110 [+0.070, +0.150]",
               "amp": "+0.044 [+0.024, +0.065]", "y1": "+0.062 [+0.028, +0.095]",
               "y2": "+0.030 [+0.009, +0.050]", "yr": "+0.018 [-0.019, +0.056]",
               "f1": "+0.025 [-0.010, +0.060]", "f2": "+0.028 [+0.007, +0.048]", "fr": "+0.014",
               "c0f": "+0.004 [-0.033, +0.040]", "c1f": "+0.012 [-0.040, +0.065]",
               "c0y": "+0.022 [-0.016, +0.060]", "c1y": "+0.039 [-0.016, +0.095]"},
    "sonnet": {"df": "-0.002 [-0.027, +0.024]", "dy": "+0.093 [+0.054, +0.133]",
               "amp": "+0.095 [+0.062, +0.128]", "y1": "+0.075 [+0.046, +0.104]",
               "y2": "+0.027 [+0.012, +0.043]", "yr": "-0.009 [-0.043, +0.027]",
               "f1": "-0.002 [-0.025, +0.021]", "f2": "+0.005 [-0.009, +0.018]", "fr": "≈0",
               "c0f": "-0.003 [-0.026, +0.021]", "c1f": "-0.007 [-0.037, +0.024]",
               "c0y": "+0.046 [+0.011, +0.082]", "c1y": "-0.015 [-0.067, +0.037]"},
    "gemini": {"df": "+0.026 [-0.002, +0.055]", "dy": "+0.117 [+0.079, +0.157]",
               "amp": "+0.091 [+0.058, +0.125]", "y1": "+0.069 [+0.041, +0.098]",
               "y2": "+0.036 [+0.021, +0.051]", "yr": "+0.012 [-0.022, +0.046]",
               "f1": "+0.012 [-0.011, +0.035]", "f2": "+0.001 [-0.010, +0.012]", "fr": "≈0",
               "c0f": "-0.003 [-0.028, +0.021]", "c1f": "+0.005 [-0.025, +0.036]",
               "c0y": "+0.036 [+0.001, +0.070]", "c1y": "+0.028 [-0.020, +0.076]"},
}


def fmt(p, lo, hi, dec=3) -> str:
    """값 [CI]. 0 제외면 *, 한쪽 경계가 정확히 0 이면 '경계'."""
    s = f"{p:+.{dec}f} [{lo:+.{dec}f}, {hi:+.{dec}f}]"
    if lo == 0.0 or hi == 0.0:
        return s + " 경계"
    return s + ("*" if (lo > 0 or hi < 0) else "")


def boot(vals_fn, keys: list, n_boot: int, seed: int) -> dict:
    """keys(=triple id) 재표집 → vals_fn(resampled keys) 이 돌려주는 dict 의 통계별 CI."""
    point = vals_fn(keys)
    rng = random.Random(seed)
    acc = defaultdict(list)
    for _ in range(n_boot):
        for k, v in vals_fn(rng.choices(keys, k=len(keys))).items():
            if v is not None:
                acc[k].append(v)
    out = {}
    for k, v in point.items():
        b = sorted(acc[k])
        out[k] = (v, b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1], len(b)) if b and v is not None else None
    return out


# ═══ §1. 세 조건 hop0 matched ═════════════════════════════════════════════════

def per_triple_values(model: str, qa: dict, triples: list[str]) -> dict:
    """cond → {triple: (df, dy)}  (hop0, 양쪽 R=1 아니면 None) + cond → 양쪽 R=1 triple 집합."""
    vals, r1sets, cells_by_cond = {}, {}, {}
    for cond in CONDS:
        cells = ms.load_model(model, qa, cond)
        cells_by_cond[cond] = cells
        rows = {r["triple"]: r for r in ms.hop0_rows(cells, triples)}
        vals[cond] = {t: (r["hop0_df"], r["hop0_dy"]) for t, r in rows.items()}
        r1sets[cond] = {t for t, r in rows.items() if r["hop0_both_R1"]}
    return vals, r1sets, cells_by_cond


def section1(models, qa, triples, n_boot, seed):
    print(f"\n{'#'*78}\n§1. 세 조건 hop0 matched 분석\n{'#'*78}")
    for model in models:
        vals, r1sets, cells_by_cond = per_triple_values(model, qa, triples)
        fixed = sorted(set.intersection(*(r1sets[c] for c in CONDS)))
        print(f"\n=== {ms.NAMES[model]}")
        print(f"[1b] 양쪽 R=1 triple: " + " / ".join(f"{SHORT[c]} {len(r1sets[c])}" for c in CONDS)
              + f" → 세 조건 교집합 {len(fixed)}")

        print("\n[1a/1b] 조건별 matched Δf·Δy (자기 R=1 집합 vs 고정 교집합)")
        print(f"  {'조건':<6} {'자기집합 n':>10} {'자기 Δf':>26} {'자기 Δy':>26} "
              f"{'고정 n':>7} {'고정 Δf':>26} {'고정 Δy':>26}")
        for cond in CONDS:
            # 통계량별 독립 RNG(ms.boot_mean) — matched_subset.py §1·§2 와 같은 값이 나온다
            own = sorted(r1sets[cond])
            vv = lambda ks, i, c=cond: [vals[c][t][i] for t in ks if vals[c][t][i] is not None]
            o_f, o_y = ms.boot_mean(vv(own, 0), n_boot, seed), ms.boot_mean(vv(own, 1), n_boot, seed)
            x_f, x_y = ms.boot_mean(vv(fixed, 0), n_boot, seed), ms.boot_mean(vv(fixed, 1), n_boot, seed)
            print(f"  {SHORT[cond]:<6} {o_f[3]:>10} {fmt(*o_f[:3]):>26} "
                  f"{fmt(*o_y[:3]):>26} {x_f[3]:>7} {fmt(*x_f[:3]):>26} {fmt(*x_y[:3]):>26}")

        # 1c. joint paired bootstrap — 고정 교집합에서 세 조건 + 차이
        def joint(ks):
            out = {}
            for c in CONDS:
                out[f"df_{SHORT[c]}"] = _mean(vals[c], ks, 0)
                out[f"dy_{SHORT[c]}"] = _mean(vals[c], ks, 1)
            for ax in ("df", "dy"):
                a, b, d = (out[f"{ax}_{SHORT[c]}"] for c in CONDS)
                out[f"{ax}_ctrl_minus_tmpl"] = None if a is None or b is None else a - b
                out[f"{ax}_tmpl_minus_norm"] = None if b is None or d is None else b - d
            return out

        res = boot(joint, fixed, n_boot, seed)
        paired = [t for t in fixed if all(vals[c][t][0] is not None for c in CONDS)]
        res_p = boot(lambda ks: {k: v for k, v in joint(ks).items()
                                 if k in ("df_ctrl_minus_tmpl", "df_tmpl_minus_norm")},
                     paired, n_boot, seed)
        print(f"\n[1c] 고정 교집합 {len(fixed)} triple, joint paired bootstrap (B={n_boot:,}, seed={seed})")
        for k in ("df_ctrl_minus_tmpl", "df_tmpl_minus_norm", "dy_ctrl_minus_tmpl", "dy_tmpl_minus_norm"):
            print(f"  {k:<22} {fmt(*res[k][:3])}")
        print(f"  완전 페어(세 조건 모두 Δf 정의) n={len(paired)}")
        for k in ("df_ctrl_minus_tmpl", "df_tmpl_minus_norm"):
            print(f"  {k+' [완전페어]':<22} {fmt(*res_p[k][:3])}")


def _mean(d: dict, keys: list, idx: int):
    v = [d[t][idx] for t in keys if d[t][idx] is not None]
    return statistics.mean(v) if v else None


# ═══ §1d. 검색 품질 공변량 ════════════════════════════════════════════════════

def section1d(models, qa, triples, cache):
    """R=1 hop0 의 top-5 중 gold 포함 패시지 수와 첫 포함 순위. hop0 검색은 모델 무관."""
    print(f"\n{'#'*78}\n§1d. 검색 품질 공변량 (hop0, R=1)\n{'#'*78}")
    # hop0 패시지·R 이 모델과 무관한지 먼저 확인
    base = {}
    for cond in CONDS:
        ref, diff = None, 0
        for m in models:                       # 메모리 절약: 모델마다 hop0만 남기고 버린다
            cells = ms.load_model(m, qa, cond)
            hop0 = {(t, p): (cells[(t, p, 0)]["qid"], cells[(t, p, 0)]["passages"],
                             cells[(t, p, 0)]["R"]) for t in triples for p in "abcd"}
            del cells
            if ref is None:
                ref = hop0
            else:
                diff += sum(1 for k in ref if hop0[k] != ref[k])
        print(f"  [{SHORT[cond]}] hop0 패시지·R 의 모델 간 불일치: {diff}건 "
              f"(0 이면 모델 무관 → 대표 1개로 집계)")
        base[cond] = ref

    rows = []
    for cond in CONDS:
        for t in triples:
            for p in "abcd":
                qid, passages, R = base[cond][(t, p)]
                if R is not True:
                    continue
                qh = next(h for h in qa[qid]["hops"] if h["hop"] == 0)
                hits = [i + 1 for i, ps in enumerate(passages)
                        if check_answer_bearing_hit([ps], qh["gold_entity_qid"], cache,
                                                    qh["gold_lang"])["answer_bearing_hit"]]
                rows.append({"cond": cond, "side": "KO" if p in "ac" else "EN",
                             "triple": t, "pattern": p, "qid": qid,
                             "n_gold_passages": len(hits), "first_rank": hits[0] if hits else None})

    print(f"\n  {'조건':<6} {'index':<6} {'R=1 홉':>7} {'gold 포함 패시지 수 평균':>22} "
          f"{'첫 포함 순위 평균':>18} {'중앙값':>7} {'순위 1 비율':>11}")
    for cond in CONDS:
        for side in ("KO", "EN"):
            sub = [r for r in rows if r["cond"] == cond and r["side"] == side]
            ranks = [r["first_rank"] for r in sub if r["first_rank"]]
            print(f"  {SHORT[cond]:<6} {side:<6} {len(sub):>7} "
                  f"{statistics.mean(r['n_gold_passages'] for r in sub):>22.2f} "
                  f"{statistics.mean(ranks):>18.2f} {statistics.median(ranks):>7.1f} "
                  f"{sum(1 for x in ranks if x == 1)/len(ranks):>11.1%}")

    print("\n  [EN index] controlled vs template_matched — 순위별 분포")
    print(f"  {'조건':<6} {'R=1 홉':>7} " + " ".join(f"{'순위'+str(i):>7}" for i in range(1, 6))
          + f" {'패시지 수 1':>10} {'2':>5} {'3+':>5}")
    for cond in ("controlled", "template_matched"):
        sub = [r for r in rows if r["cond"] == cond and r["side"] == "EN"]
        rc = Counter(r["first_rank"] for r in sub)
        nc = Counter(r["n_gold_passages"] for r in sub)
        print(f"  {SHORT[cond]:<6} {len(sub):>7} " + " ".join(f"{rc[i]:>7}" for i in range(1, 6))
              + f" {nc[1]:>10} {nc[2]:>5} {sum(v for k, v in nc.items() if k >= 3):>5}")
    return rows


# ═══ §2. 부트스트랩 단위 민감도 ═══════════════════════════════════════════════

def section2(models, qa, triples, n_boot, seed):
    print(f"\n{'#'*78}\n§2b. triple 재표집 CI (지표 함수는 analysis/13_bootstrap_ci.py 그대로)\n{'#'*78}")
    qid_triple = {q: r["triple"] for q, r in qa.items()}
    for model in models:
        recs, hit_map = {}, None
        for cond in CONDS:
            rs = restrict(ms.load_jsonl(ms.ann_path(model, cond)), str(ms.QA))
            recs[cond] = {r["qid"]: r for r in rs}
        raw = ms.load_jsonl(ms.hits_path(model, "controlled"))
        hit_map = {"hop0": {r["qid"]: r.get("answer_bearing_hit") for r in raw if r.get("hop", 0) == 0},
                   "hop1": {r["qid"]: r.get("answer_bearing_hit") for r in raw if r.get("hop", 0) == 1}}
        by_triple = defaultdict(list)
        for q in recs["controlled"]:
            by_triple[qid_triple[q]].append(q)
        keys = sorted(by_triple)
        assert all(len(v) == 4 for v in by_triple.values())

        def stats(ks):
            qids = [q for k in ks for q in by_triple[k]]
            s = {c: [recs[c][q] for q in qids] for c in CONDS}
            c, t, n = (s[x] for x in CONDS)
            out = {"df": bs13.lshd_end2end(c), "dy": bs13.yield_lshd_end2end(c),
                   "amp": bs13.yield_amplification(c)}
            fc, ft, fn = (bs13.lshd_end2end(x) for x in (c, t, n))
            yc, yt, yn = (bs13.yield_lshd_end2end(x) for x in (c, t, n))
            out.update({"f1": fc - ft, "f2": ft - fn, "fr": fn,
                        "y_gap": yc, "y1": yc - yt, "y2": yt - yn, "yr": yn})
            out["c0f"] = bs13.lshd_conditioned_hop(c, hit_map["hop0"], 0, True)
            out["c1f"] = bs13.lshd_conditioned_hop(c, hit_map["hop1"], 1, True)
            out["c0y"] = bs13.yield_lshd_conditioned_hop(c, hit_map["hop0"], 0, True)
            out["c1y"] = bs13.yield_lshd_conditioned_hop(c, hit_map["hop1"], 1, True)
            return out

        res = boot(stats, keys, n_boot, seed)
        print(f"\n=== {ms.NAMES[model]}  (triple {len(keys)}개 재표집, B={n_boot:,}, seed={seed})")
        print(f"  {'지표':<26} {'triple 재표집 (점추정 [CI])':<34} 원고 (QID 재표집)")
        for key, label in [("df", "Δf 무조건"), ("dy", "Δy 무조건"), ("amp", "증폭 Δy−Δf"),
                           ("y_gap", "yield gap(ctrl)"), ("y1", "yield template-language"),
                           ("y2", "yield bridge-surface"), ("yr", "yield residual"),
                           ("f1", "faith template-language"), ("f2", "faith bridge-surface"),
                           ("fr", "faith residual"),
                           ("c0f", "Table 3 hop0 Δf (R=1)"), ("c1f", "Table 3 hop1 Δf (R=1)"),
                           ("c0y", "hop0 Δy (R=1)"), ("c1y", "hop1 Δy (R=1)")]:
            paper = PAPER[model].get(key, "—")
            if key == "y_gap":
                paper = PAPER[model]["dy"]
            print(f"  {label:<26} {fmt(*res[key][:3]):<34} {paper}")


# ═══ §3. 0c 생성 불일치 — 프롬프트 동일성 확인 ════════════════════════════════

def section3(models, qa, triples):
    """hop0 generated_text 가 a–c / b–d 에서 다른 triple 의 프롬프트 전체 문자열을 비교한다.

    프롬프트는 저장돼 있지 않으므로 pipeline/05_pilot_runner.py 의 함수로 재구성한다
    (build_prompt + _select_prompts). hop0 은 agent_question == retrieval_query == sub_q 라
    저장된 retrieval_query 와 재구성 질의가 같은지 assert 로 확인한다.
    """
    spec = importlib.util.spec_from_file_location("p05", ROOT / "pipeline" / "05_pilot_runner.py")
    p05 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p05)
    sys_prompt, hop_tmpl, mt, temp, tp = p05._select_prompts("v2.3")
    print(f"\n{'#'*78}\n§3. 0c 생성 불일치 — 프롬프트 재구성 비교\n{'#'*78}")
    print(f"  prompt_version v2.3 → max_tokens={mt}, temperature={temp}, top_p={tp}")
    print(f"  SYSTEM_PROMPT_V23 {len(sys_prompt)}자 / HOP_PROMPT_V23 {len(hop_tmpl)}자 "
          f"(모델·패턴·언어 분기 없음: build_prompt 인자에 pattern/gold_lang 없음)")

    for model in models:
        traj = {r["qid"]: r for r in ms.load_jsonl(ms.traj_path(model, "controlled"))}
        step0 = {q: next(s for s in r["trajectory"] if int(s["hop"]) == 0) for q, r in traj.items()}
        pairs, same_prompt, diffs, tok = [], 0, [], []
        for t in triples:
            for p1, p2 in (("a", "c"), ("b", "d")):
                qx = next(q for q, r in qa.items() if r["triple"] == t and r["pattern"] == p1)
                qy = next(q for q, r in qa.items() if r["triple"] == t and r["pattern"] == p2)
                sx, sy = step0[qx], step0[qy]
                if (sx.get("generated_text") or "") == (sy.get("generated_text") or ""):
                    continue
                pr = []
                for q, s in ((qx, sx), (qy, sy)):
                    sub_q = next(h for h in qa[q]["hops"] if h["hop"] == 0)["sub_q"]
                    assert sub_q == (s.get("retrieval_query") or ""), (model, q)
                    pr.append(p05.build_prompt(hop_tmpl, sub_q, 0, len(qa[q]["hops"]),
                                               list(s.get("retrieved_passages") or []), ""))
                pairs.append((t, p1 + p2, qx, qy, len(pr[0])))
                if pr[0] == pr[1]:
                    same_prompt += 1
                else:
                    diffs.append((t, p1 + p2, pr[0], pr[1]))
                ux, uy = sx.get("gen_usage"), sy.get("gen_usage")
                if ux and uy:
                    tok.append((ux.get("input_tokens"), uy.get("input_tokens"),
                                ux.get("output_tokens"), uy.get("output_tokens")))
        print(f"\n=== {ms.NAMES[model]}  생성 불일치 짝 {len(pairs)}개")
        print(f"    프롬프트 전체 문자열 동일: {same_prompt}/{len(pairs)}  (다름 {len(diffs)})")
        for t, tag, a, b in diffs[:5]:
            print(f"    [다름] {t} {tag}")
            for i, (la, lb) in enumerate(zip(a.splitlines(), b.splitlines())):
                if la != lb:
                    print(f"      line {i}: A={la[:80]!r}")
                    print(f"              B={lb[:80]!r}")
        if tok:
            eq_in = sum(1 for a, b, _, _ in tok if a == b)
            print(f"    gen_usage input_tokens 동일: {eq_in}/{len(tok)} "
                  f"(output_tokens 동일 {sum(1 for _, _, c, d in tok if c == d)}/{len(tok)})")
        elif not pairs:
            print("    gen_usage 비교: 불일치 짝이 없어 생략")
        else:
            print("    gen_usage: 없음 (8B 구 스키마 — 토큰 수 기록 자체가 없음)")
        s = step0[next(iter(step0))]
        print(f"    hop0 스텝에 있는 호출 기록성 필드: "
              f"{[k for k in s if k in ('gen_usage', 'api_error', 'run_metadata')] or '없음'}"
              f" / 레코드 run_metadata: {'있음' if traj[next(iter(traj))].get('run_metadata') else '없음/null'}"
              f" / 시각·요청ID 필드: "
              f"{[k for k in s if any(w in k.lower() for w in ('timestamp', 'created', 'request_id', 'custom_id', 'batch', 'order'))] or '없음'}")

        # 같은 패턴 안 중복 요청(분석 대상 밖 질문까지 포함한 파일 전체)에서의 분기 — 비결정성 직접 증거
        grp = defaultdict(list)
        for q, s in step0.items():
            grp[(s.get("retrieval_query"), tuple(s.get("retrieved_passages") or []))].append(
                (traj[q]["pattern"], q, s.get("generated_text") or ""))
        dup = {k: v for k, v in grp.items() if len(v) > 1}
        split_any = [v for v in dup.values() if len({x[2] for x in v}) > 1]
        split_same_pat = [v for v in split_any
                          if any(len({x[2] for x in v if x[0] == p}) > 1 for p in {y[0] for y in v})]
        print(f"    전체 파일({len(step0)} 궤적) 동일 요청 그룹 {len(dup)}개 중 출력 분기 {len(split_any)}개, "
              f"그중 같은 패턴 내부 분기 {len(split_same_pat)}개")
        print(f"    동일 요청 그룹 크기 분포: {dict(sorted(Counter(len(v) for v in dup.values()).items()))}"
              f" (크기 2 = a+c 또는 b+d 뿐 → 같은 패턴 내부 분기가 구조적으로 불가)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ms.MODELS)
    ap.add_argument("--sections", nargs="+", default=["1", "1d", "2"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--emit-triple-qa", default=None,
                    help="qa_pairs_core 의 bridge_cluster_id 를 triple id 로 바꾼 사본 경로")
    ap.add_argument("--out-dir", default=str(ms.OUT / "correctness"),
                    help="matched_subset_retrieval_cov.csv(절 1d)를 쓸 디렉터리")
    args = ap.parse_args()

    qa = {}
    for r in ms.load_jsonl(ms.QA):
        r["triple"] = r["source_path_id"].rsplit("_pat_", 1)[0]
        qa[r["qid"]] = r
    triples = sorted({r["triple"] for r in qa.values()})

    if args.emit_triple_qa:
        p = Path(args.emit_triple_qa)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for r in ms.load_jsonl(ms.QA):
                r["bridge_cluster_id"] = r["source_path_id"].rsplit("_pat_", 1)[0]
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"저장: {p} (bridge_cluster_id ← triple id, {len(qa)}행)")

    print(f"[입력] QA {len(qa)} 궤적 / triple {len(triples)}개")
    if "1" in args.sections:
        section1(args.models, qa, triples, args.n_boot, args.seed)
    if "1d" in args.sections:
        cache = json.loads(ALIAS.read_text(encoding="utf-8"))
        rows = section1d(args.models, qa, triples, cache)
        out = Path(args.out_dir) / "matched_subset_retrieval_cov.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n저장: {out.name} ({len(rows)}행)")
    if "3" in args.sections:
        section3(args.models, qa, triples)
    if "2" in args.sections:
        section2(args.models, qa, triples, args.n_boot, args.seed)


if __name__ == "__main__":
    main()
