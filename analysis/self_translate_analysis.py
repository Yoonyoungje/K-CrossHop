#!/usr/bin/env python3
"""
analysis/self_translate_analysis.py

self-translate 묶음(6모델 × ctrl/tmpl_dec/norm_dec/self_tr) 분석 중 **기존 파이프라인에 없는 지표만** 계산한다.
모델 호출 0회. 명세: data/self_translate.md §2.6 (3)(5)(6), 원 요청의 R_tmpl·R_full 정의. hit@5 축은 보류.

기존 파이프라인으로 계산하는 항목 (이 스크립트에서 하지 않는다, 출력은 self_translate/pipeline/)
  조건별 faithfulness·yield gap + CI     analysis/13_bootstrap_ci.py §1·§1-B
  전체 hop QMC (template·bridge) + CI    analysis/11_ablation_compare.py (stage1·stage2), analysis/yield_qmc_joint_ci.py
  패턴별 AFR                             analysis/07_analyze.py

이 스크립트가 계산하는 항목 (파이프라인에 없음)
  회복률 R_tmpl·R_full + Δsame·Δcross (주 분석·민감도 1·2·hop0 정답 부분집합), hop 별 QMC 재분해(QID-cluster),
  생성 질문 효과(hop0 일치 짝 + 잡음 기준선), 원문에 없는 개체명 추가, 재작성 변경률, hop0 기권 bridge 집계

지표 (전부 utils.metrics 공유 함수)
  yield   = hop_yield,  faith = hop_faith,  same/cross = index_class,  gap = lshd (same − cross)
  hop 정답 = 정답 정확도 축 주 매처 (analysis/16_pdr_analysis build_alias_surfaces + entity_in_text),
            기권 → 0, ambiguous → None

회복률 (yield 축, 같은 resample)
  R_tmpl = (gap_ctrl − gap_self) / (gap_ctrl − gap_tmpl)
  R_full = (gap_ctrl − gap_self) / (gap_ctrl − gap_norm)
  Δsame = same(self) − same(ctrl),  Δcross = cross(self) − cross(ctrl),  분자 = Δcross − Δsame (매 draw 항등 점검)
  분모 CI 가 0 을 포함하면 비율은 보고하지 않고 분자·분모만 적는다.

변형 (같은 resample 안에서 함께 계산)
  주 분석      모든 hop
  민감도 1     self_tr 재작성 hop 중 누출 V3(leak_v3=1) 인 (qid, hop) 을 네 조건 모두에서 제외
  민감도 2     self_tr 재작성 hop 중 답 서술문 또는 주석 섞임(explanation_mixed=1) 인 (qid, hop) 을 네 조건 모두에서 제외
               (메타 문장은 판정 규칙이 구현돼 있지 않아 포함하지 않았다)
  hop0 정답(주)   네 조건 모두에서 hop0 이 정답인 QID 교집합, hop1 만 (2026-09-17 변경, self_translate.md §2.6 (6-b) 참조)
  hop0 정답(부록) 각 조건에서 자기 hop0 이 정답인 궤적, 전체 hop (변경 전 방식, 그대로 유지)

탐색 분석 (사전 명세 외): hop0·hop1 별 회복률, tmpl_dec·norm_dec 의 Δsame(ctrl 대비)

통계: QID-cluster paired bootstrap, B=10,000, seed 42. analysis/yield_qmc_joint_ci.py 와 같은 재표집
      (random.seed(42) 후 매 draw random.choices, k=N; QID 순서 = 이번 ctrl annotated 파일 순서).
      CI = 정렬 후 [int(0.025·n)], [int(0.975·n) − 1].
      --cluster-key path_cluster_id: QID 대신 entity path 250개를 재표집한다 (camera-ready 원고 정본 CI 규칙).

  python3 analysis/self_translate_analysis.py > data/self_translate/analysis_tables.md
"""
import argparse
import csv
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.metrics import hop_faith, hop_yield, index_class, lshd   # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pdr16 = _load("pdr16", "analysis/16_pdr_analysis.py")
corr = _load("corr_axis", "analysis/correctness_axis.py")       # 정본 입력 경로(paths)
form_of = _load("st_compare", "analysis/self_translate_compare.py").form
PREAMBLE = _load("st_check", "analysis/self_translate_check.py").PREAMBLE   # 머리말·레이블 판정

RS = ROOT / "data/self_translate"
CSVD = ROOT / "data/self_translate"
QA_PATH = ROOT / "data/dataset/qa_pairs.jsonl"
ALIAS = ROOT / "data/retrieval/alias_cache.json"
MODELS = ["aya", "exaone", "llama", "qwen", "sonnet", "gemini"]
CONDS = ["ctrl", "tmpl_dec", "norm_dec", "self_tr"]
VARIANTS = ["주 분석", "민감도 1 (누출 V3 제외)", "민감도 2 (답 서술문·주석 섞임 제외)",
            "부록: hop0 정답 — 조건별 자기 hop0 정답 궤적 (전체 hop)"]
H0_INTER = "hop0 정답 (주) — 네 조건 모두 hop0 정답인 QID 교집합, hop1 만"
SENS2B = "[탐색 — 사전 명세 외] 민감도 2-b (답 서술문 ∪ 첫 줄이 비질문인 주석 섞임 제외)"


def jl(p):
    return [json.loads(l) for l in Path(p).open(encoding="utf-8") if l.strip()]


def ci(vals, alpha=0.05):
    s = sorted(v for v in vals if not math.isnan(v))
    n = len(s)
    return (s[int(alpha / 2 * n)], s[int((1 - alpha / 2) * n) - 1]) if n else (float("nan"), float("nan"))


def fmt_ci(p, lo, hi):
    star = " *" if (lo > 0 or hi < 0) else ""
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]{star}"


def hop_correct(ah, st, qh, cache):
    if ah.get("ambiguous_answer_marker"):
        return None
    if ah.get("abstained"):
        return 0
    text = st.get("generated_text") or ""
    aliases = pdr16.build_alias_surfaces(qh["gold_entity_qid"], cache, qh["gold_entity"])
    return int(bool(text) and pdr16.entity_in_text(aliases, text))


# ── 입력 ─────────────────────────────────────────────────────────────────────

def load_model(m, qa, cache):
    ann = {c: jl(RS / f"annotated_{m}_{c}.jsonl") for c in CONDS}
    trj = {c: {t["qid"]: t for t in jl(RS / f"trajectories_{m}_{c}.jsonl")} for c in CONDS}
    order = [r["qid"] for r in ann["ctrl"]]
    by = {c: {r["qid"]: r for r in ann[c]} for c in CONDS}
    for c in CONDS:
        assert set(by[c]) == set(order) == set(qa) == set(trj[c]), (m, c)
    h0c = {c: {} for c in CONDS}
    for c in CONDS:
        for q in order:
            ah = next(h for h in by[c][q]["hops"] if h["hop"] == 0)
            st = next(s for s in trj[c][q]["trajectory"] if int(s["hop"]) == 0)
            h0c[c][q] = hop_correct(ah, st, qa[q]["hops"][0], cache)
    rows = [r for r in csv.DictReader((CSVD / f"rewrites_{m}.csv").open(encoding="utf-8-sig"))
            if r["status"] != "skipped_ko_index"]
    sens1 = {(r["qid"], int(r["hop"])) for r in rows if r["leak_v3"] == "1"}
    sens2 = {(r["qid"], int(r["hop"])) for r in rows
             if form_of(r["retrieval_query"]) == "declarative" or r["explanation_mixed"] == "1"}
    sens2b = {(r["qid"], int(r["hop"])) for r in rows
              if form_of(r["retrieval_query"]) == "declarative"
              or (r["explanation_mixed"] == "1" and first_line_nonquestion(r))}
    return order, by, trj, h0c, rows, sens1, sens2, sens2b


HIT5 = CSVD / "hit5"


def attach_hits(m, by):
    """analysis/10_retrieval_proxy.py 출력(answer_bearing_hit)을 annotated hop 에 `_hit5` 로 붙인다."""
    for c in CONDS:
        hits = {(r["qid"], int(r["hop"])): float(bool(r["answer_bearing_hit"]))
                for r in jl(HIT5 / f"retrieval_v3_{m}_{c}.jsonl")}
        for q, rec in by[c].items():
            for h in rec["hops"]:
                h["_hit5"] = hits[(q, h["hop"])]


def hop_hit5(h):
    return h["_hit5"]


def first_line_nonquestion(r):
    """실제 검색 쿼리(첫 줄)가 질문이 아닌가: 물음표 없음(self_translate_compare.form) 또는 머리말·레이블로 시작."""
    q = r["retrieval_query"]
    return form_of(q) != "question" or bool(PREAMBLE.match(q))


# ── 통계량 배열 (QID × (sum_a, n_a, sum_b, n_b)) ──────────────────────────────

def cell(hops, value_fn, keep=lambda h: True, split=index_class, a="same"):
    out = [0.0, 0, 0.0, 0]
    for h in hops:
        if not keep(h):
            continue
        v = value_fn(h)
        if v is None:
            continue
        j = 0 if split(h) == a else 2
        out[j] += v
        out[j + 1] += 1
    return out


CLUSTER = None   # --cluster-key path_cluster_id 이면 {qid: path_cluster_id} (main 에서 채운다)
UNIT = "QID-cluster"


class Joint:
    """QID 단위 (합, 개수) 배열을 쌓아 두고, 같은 resample 로 모든 통계량을 한 번에 계산한다.

    CLUSTER 가 있으면 QID 대신 entity path(재표집 단위)를 뽑고 그 path 의 QID 전부에 가중치를 준다
    (analysis/yield_qmc_joint_ci.py --cluster-key 와 같은 방식: path 순서 = order 안 첫 등장 순서)."""

    def __init__(self, order):
        self.order, self.keys, self.cols = order, [], []

    def add(self, key, per_qid):
        self.keys.append(key)
        self.cols.append([per_qid[q] for q in self.order])

    def run(self, n_boot, seed):
        A = np.asarray(self.cols, dtype=float)                  # (K, N, 4)
        N = len(self.order)
        point = A.sum(axis=1)
        groups = None
        if CLUSTER is not None:
            groups = {}
            for i, q in enumerate(self.order):
                groups.setdefault(CLUSTER[q], []).append(i)
            gids = list(groups)
        random.seed(seed)
        draws = np.empty((n_boot, len(self.keys), 4))
        for b in range(n_boot):
            if groups is None:
                idx = random.choices(range(N), k=N)
            else:
                idx = [i for g in random.choices(gids, k=len(gids)) for i in groups[g]]
            w = np.bincount(idx, minlength=N).astype(float)
            draws[b] = np.einsum("kni,n->ki", A, w)
        return point, draws

    def idx(self, key):
        return self.keys.index(key)


def means(S):
    """(…, 4) → (same_mean, cross_mean, gap)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        sm, cr = S[..., 0] / S[..., 1], S[..., 2] / S[..., 3]
    return sm, cr, sm - cr


# ── 1. 조건별 gap · QMC · 회복률 ─────────────────────────────────────────────

def analyse_model(m, qa, cache, n_boot, seed, out):
    order, by, trj, h0c, rows, sens1, sens2, sens2b = load_model(m, qa, cache)
    attach_hits(m, by)
    fired = {(q, 1) for q in order
             for st in trj["norm_dec"][q]["trajectory"]
             if int(st["hop"]) == 1 and st["gold_lang"] == "en" and st.get("normalization_applied")}
    added = added_entities(m, qa, cache, by, trj, h0c, rows)
    other_keys = {(r["qid"], r["hop"]) for r in added if is_other(r)}
    J = Joint(order)
    hops = lambda c, q: by[c][q]["hops"]                                  # noqa: E731
    filt = {"주 분석": lambda q: (lambda h: True),
            VARIANTS[1]: lambda q: (lambda h: (q, h["hop"]) not in sens1),
            VARIANTS[2]: lambda q: (lambda h: (q, h["hop"]) not in sens2),
            SENS2B: lambda q: (lambda h: (q, h["hop"]) not in sens2b)}
    for c in CONDS:
        for v in VARIANTS:
            if v == VARIANTS[3]:
                per = {q: (cell(hops(c, q), hop_yield) if h0c[c][q] == 1 else [0.0, 0, 0.0, 0]) for q in order}
            else:
                per = {q: cell(hops(c, q), hop_yield, filt[v](q)) for q in order}
            J.add(("y", c, v), per)
        for k in (0, 1):
            J.add(("y", c, f"hop{k}"), {q: cell(hops(c, q), hop_yield, lambda h, k=k: h["hop"] == k) for q in order})
            J.add(("f", c, f"hop{k}"), {q: cell(hops(c, q), hop_faith, lambda h, k=k: h["hop"] == k) for q in order})
    inter = {q for q in order if all(h0c[c][q] == 1 for c in CONDS)}
    keep_fired = lambda q: (lambda h: h["hop"] == 1 and (index_class(h) == "same" or (q, 1) in fired))  # noqa: E731
    for c in CONDS:
        for v in VARIANTS[:3]:
            J.add(("h", c, v), {q: cell(hops(c, q), hop_hit5, filt[v](q)) for q in order})
        J.add(("h", c, H0_INTER), {q: (cell(hops(c, q), hop_hit5, lambda h: h["hop"] == 1) if q in inter
                                       else [0.0, 0, 0.0, 0]) for q in order})
        J.add(("h", c, VARIANTS[3]), {q: (cell(hops(c, q), hop_hit5) if h0c[c][q] == 1 else [0.0, 0, 0.0, 0])
                                      for q in order})
        for ax, fn in (("y", hop_yield), ("h", hop_hit5)):
            J.add((ax, c, "fired"), {q: cell(hops(c, q), fn, keep_fired(q)) for q in order})
        J.add(("y", c, SENS2B), {q: cell(hops(c, q), hop_yield, filt[SENS2B](q)) for q in order})
        J.add(("y", c, "hop1_ex_other"), {q: cell(hops(c, q), hop_yield,
                                                   lambda h, q=q: h["hop"] == 1 and (q, 1) not in other_keys)
                                          for q in order})
        J.add(("y", c, H0_INTER), {q: (cell(hops(c, q), hop_yield, lambda h: h["hop"] == 1) if q in inter
                                       else [0.0, 0, 0.0, 0]) for q in order})

    # 점추정이 공유 함수 lshd 와 같은지 확인
    point, draws = J.run(n_boot, seed)
    for c in CONDS:
        recs = [by[c][q] for q in order]
        g = lshd(recs, hop_yield)[0]
        assert abs(means(point[J.idx(("y", c, "주 분석"))])[2] - g) < 1e-12, (m, c)
        for k in (0, 1):
            g = lshd(recs, hop_faith, hop_filter=lambda h, k=k: h["hop"] == k)[0]
            assert abs(means(point[J.idx(("f", c, f"hop{k}"))])[2] - g) < 1e-12, (m, c, k)
        g = lshd([r for r in recs if h0c[c][r["qid"]] == 1], hop_yield)[0]
        assert abs(means(point[J.idx(("y", c, VARIANTS[3]))])[2] - g) < 1e-12, (m, c)
        g = lshd(recs, hop_hit5)[0]
        assert abs(means(point[J.idx(("h", c, "주 분석"))])[2] - g) < 1e-12, (m, c)
        g = lshd([r for r in recs if r["qid"] in inter], hop_yield, hop_filter=lambda h: h["hop"] == 1)[0]
        assert abs(means(point[J.idx(("y", c, H0_INTER))])[2] - g) < 1e-12, (m, c)

    res = {"n_sens1": len(sens1), "n_sens2": len(sens2), "n_sens2b": len(sens2b),
           "n_other_hop1": sum(1 for k in other_keys if k[1] == 1), "rows": rows,
           "n_h0c": {c: sum(1 for q in order if h0c[c][q] == 1) for c in CONDS}, "n_inter": len(inter)}

    # 참고: 조건별 자기 hop0 정답 부분집합의 hop0 yield (점추정)
    res["h0c_hop0_yield"] = {}
    for c in CONDS:
        hs = [h for q in order if h0c[c][q] == 1 for h in by[c][q]["hops"] if h["hop"] == 0]
        vals = {k: [v for h in hs if (k == "전체" or index_class(h) == k) and (v := hop_yield(h)) is not None]
                for k in ("전체", "same", "cross")}
        res["h0c_hop0_yield"][c] = {k: (len(v), sum(v) / len(v) if v else float("nan")) for k, v in vals.items()}

    # hop 별 QMC 재분해 (분리판 기준). 전체 hop 분해는 11·yield_qmc_joint_ci 가 낸다.
    res["qmc"] = {}
    for ax in ("y", "f"):
        for scope in ("hop0", "hop1"):
            G = {c: (means(point[J.idx((ax, c, scope))])[2], means(draws[:, J.idx((ax, c, scope))])[2])
                 for c in ("ctrl", "tmpl_dec", "norm_dec")}
            comps = {"gap(ctrl)": (G["ctrl"][0], G["ctrl"][1]),
                     "template": (G["ctrl"][0] - G["tmpl_dec"][0], G["ctrl"][1] - G["tmpl_dec"][1]),
                     "bridge": (G["tmpl_dec"][0] - G["norm_dec"][0], G["tmpl_dec"][1] - G["norm_dec"][1]),
                     "residual": (G["norm_dec"][0], G["norm_dec"][1])}
            res["qmc"][(ax, scope)] = {k: (p, *ci(d.tolist())) for k, (p, d) in comps.items()}

    # 회복률 (yield 축)
    def recovery(key):
        P = {c: means(point[J.idx(key(c))]) for c in CONDS}
        D = {c: means(draws[:, J.idx(key(c))]) for c in CONDS}
        num_p = P["ctrl"][2] - P["self_tr"][2]
        num_d = D["ctrl"][2] - D["self_tr"][2]
        ds_p, dc_p = P["self_tr"][0] - P["ctrl"][0], P["self_tr"][1] - P["ctrl"][1]
        ds_d, dc_d = D["self_tr"][0] - D["ctrl"][0], D["self_tr"][1] - D["ctrl"][1]
        viol = int(np.sum(np.abs(num_d - (dc_d - ds_d)) > 1e-9))
        ent = {"분자": (num_p, *ci(num_d.tolist())), "Δsame": (ds_p, *ci(ds_d.tolist())),
               "Δcross": (dc_p, *ci(dc_d.tolist())), "항등 위반 draw": viol}
        for name, base in (("R_tmpl", "tmpl_dec"), ("R_full", "norm_dec")):
            den_p = P["ctrl"][2] - P[base][2]
            den_d = D["ctrl"][2] - D[base][2]
            lo, hi = ci(den_d.tolist())
            with np.errstate(invalid="ignore", divide="ignore"):
                r_d = num_d / den_d
            r_lo, r_hi = ci(r_d.tolist())
            ent[name] = {"분모": (den_p, lo, hi), "den_lo_not_pos": not lo > 0,     # 비율은 분모 CI 하한 > 0 일 때만
                         "비율": (num_p / den_p if den_p else float("nan"), r_lo, r_hi)}
        return ent

    res["R"] = {v: recovery(lambda c, v=v: ("y", c, v)) for v in VARIANTS}
    res["R"][H0_INTER] = recovery(lambda c: ("y", c, H0_INTER))
    # 탐색 (사전 명세 외): hop 별 회복률, 조건별 Δsame
    res["R_hop"] = {k: recovery(lambda c, k=k: ("y", c, f"hop{k}")) for k in (0, 1)}
    res["R"][SENS2B] = recovery(lambda c: ("y", c, SENS2B))
    res["R_hit"] = {v: recovery(lambda c, v=v: ("h", c, v)) for v in (*VARIANTS[:3], H0_INTER, VARIANTS[3])}
    res["R_fired"] = {ax: recovery(lambda c, ax=ax: (ax, c, "fired")) for ax in ("y", "h")}
    en1_all = [st for q in order for st in trj["norm_dec"][q]["trajectory"]
               if int(st["hop"]) == 1 and st["gold_lang"] == "en"]
    res["fire_all"] = (sum(1 for st in en1_all if st.get("normalization_applied")), len(en1_all))
    res["n_fired"] = len(fired)
    # self_tr vs norm_dec hop 별 hit@5 불일치 (한쪽만 hit)
    res["mismatch"] = []
    for q in order:
        for k in (0, 1):
            a = next(h for h in by["self_tr"][q]["hops"] if h["hop"] == k)
            b = next(h for h in by["norm_dec"][q]["hops"] if h["hop"] == k)
            if a["_hit5"] != b["_hit5"]:
                sa = next(x for x in trj["self_tr"][q]["trajectory"] if int(x["hop"]) == k)
                sb = next(x for x in trj["norm_dec"][q]["trajectory"] if int(x["hop"]) == k)
                res["mismatch"].append({"model": m, "qid": q, "hop": k, "index_lang": a["gold_lang"],
                                        "self_tr_hit": int(a["_hit5"]), "norm_dec_hit": int(b["_hit5"]),
                                        "self_tr_query": sa["retrieval_query"], "norm_dec_query": sb["retrieval_query"]})
    res["R_hop1_ex_other"] = recovery(lambda c: ("y", c, "hop1_ex_other"))
    # 교집합 QID 의 EN-index hop1 에서 norm_dec 정규화가 실제로 발동한 비율 (점추정)
    en1 = [next(s_ for s_ in trj["norm_dec"][q]["trajectory"] if int(s_["hop"]) == 1) for q in inter]
    en1 = [s_ for s_ in en1 if s_["gold_lang"] == "en"]
    res["norm_applied"] = (sum(1 for s_ in en1 if s_.get("normalization_applied")), len(en1),
                           dict(sorted(Counter(s_.get("normalization_status") for s_ in en1).items())))
    res["dsame"] = {}
    for scope in ("주 분석", "hop0", "hop1"):
        base_p, base_d = means(point[J.idx(("y", "ctrl", scope))])[0], means(draws[:, J.idx(("y", "ctrl", scope))])[0]
        for c in ("tmpl_dec", "norm_dec", "self_tr"):
            d = means(draws[:, J.idx(("y", c, scope))])[0] - base_d
            res["dsame"][(scope, c)] = (means(point[J.idx(("y", c, scope))])[0] - base_p, *ci(d.tolist()))

    # 재작성 변경률 (EN-index hop)
    ws = lambda s: " ".join((s or "").split())                            # noqa: E731
    changed = sum(1 for r in rows if ws(r["retrieval_query"]) != ws(r["source_query"]))
    res["changed"] = (changed, len(rows))

    # hop0 기권 bridge hop (별도 집계, 점추정)
    res["abst_bridge"] = {}
    for c in CONDS:
        ys = []
        for q in order:
            hs = {h["hop"]: h for h in by[c][q]["hops"]}
            if hs[0].get("abstained"):
                v = hop_yield(hs[1])
                if v is not None:
                    ys.append(v)
        res["abst_bridge"][c] = (len(ys), sum(ys) / len(ys) if ys else float("nan"))

    out[m] = res
    return order, by, trj, h0c, rows, added


# ── 논문 결합판 hop 별 QMC (정본 controlled / template-matched / surface-normalized) ──

def canon_hop_qmc(m, qa, n_boot, seed):
    """분리판 hop 별 QMC 와 같은 정의·같은 QID-cluster 재표집(QID 순서 = 정본 controlled 파일 순서)."""
    anns = {}
    for cond in ("controlled", "template_matched", "surface_normalized"):
        ann_p, _ = corr.paths(m, cond)
        anns[cond] = {r["qid"]: r for r in jl(ann_p) if r["qid"] in qa}
    ctrl_p, _ = corr.paths(m, "controlled")
    order = [r["qid"] for r in jl(ctrl_p) if r["qid"] in qa]
    for cond in anns:
        assert set(anns[cond]) == set(order) == set(qa), (m, cond)
    J = Joint(order)
    for cond in anns:
        for ax, fn in (("y", hop_yield), ("f", hop_faith)):
            for k in (0, 1):
                J.add((ax, cond, k), {q: cell(anns[cond][q]["hops"], fn, lambda h, k=k: h["hop"] == k) for q in order})
    point, draws = J.run(n_boot, seed)
    out = {}
    for ax in ("y", "f"):
        for k in (0, 1):
            G = {c: (means(point[J.idx((ax, c, k))])[2], means(draws[:, J.idx((ax, c, k))])[2]) for c in anns}
            comps = {"gap(ctrl)": G["controlled"],
                     "template": (G["controlled"][0] - G["template_matched"][0], G["controlled"][1] - G["template_matched"][1]),
                     "bridge": (G["template_matched"][0] - G["surface_normalized"][0],
                                G["template_matched"][1] - G["surface_normalized"][1]),
                     "residual": G["surface_normalized"]}
            out[(ax, f"hop{k}")] = {n: (pp, *ci(d.tolist())) for n, (pp, d) in comps.items()}
    return out


# ── 2. 생성 질문 효과 (hop0 일치 짝, 결합판 hop1 − 분리판 hop1) ─────────────────

def gen_question_effect(m, qa, order, by, trj, n_boot, seed):
    res = {}
    comps = [("template (정본 query_matched − tmpl_dec)", "template_matched", "tmpl_dec"),
             ("surface (정본 norm − norm_dec)", "surface_normalized", "norm_dec"),
             ("잡음 기준선 (정본 controlled − 이번 ctrl)", "controlled", "ctrl")]
    for label, canon_cond, new_cond in comps:
        ann_p, trj_p = corr.paths(m, canon_cond)
        c_ann = {r["qid"]: r for r in jl(ann_p) if r["qid"] in qa}
        c_trj = {r["qid"]: r for r in jl(trj_p) if r["qid"] in qa}
        g0 = lambda t: next(s for s in t["trajectory"] if int(s["hop"]) == 0).get("generated_text")  # noqa: E731
        paired = [q for q in order if g0(c_trj[q]) == g0(trj[new_cond][q])]
        excluded = [q for q in order if q not in set(paired)]

        def per(value_fn):
            d = {}
            for q in paired:
                a = next(h for h in c_ann[q]["hops"] if h["hop"] == 1)
                b = next(h for h in by[new_cond][q]["hops"] if h["hop"] == 1)
                va, vb = value_fn(a), value_fn(b)
                d[q] = [va or 0.0, int(va is not None), vb or 0.0, int(vb is not None)]
            return d

        J = Joint(paired)
        J.add("y", per(hop_yield))
        J.add("f", per(hop_faith))
        point, draws = J.run(n_boot, seed)
        ent = {"n_pair": len(paired), "n_total": len(order)}
        for ax in ("y", "f"):
            i = J.idx(ax)
            p = means(point[i])[2]
            ent[ax] = (p, *ci(means(draws[:, i])[2].tolist()))
        ctrl_y = lambda qs: (                                                           # noqa: E731
            sum(v for q in qs for h in by["ctrl"][q]["hops"] if (v := hop_yield(h)) is not None)
            / max(1, sum(1 for q in qs for h in by["ctrl"][q]["hops"] if hop_yield(h) is not None)))
        ctrl_h = lambda qs: (sum(h["_hit5"] for q in qs for h in by["ctrl"][q]["hops"])        # noqa: E731
                             / max(1, sum(len(by["ctrl"][q]["hops"]) for q in qs)))
        ent["ctrl_hit_paired"] = ctrl_h(paired)
        ent["ctrl_hit_excluded"] = ctrl_h(excluded) if excluded else float("nan")
        ent["ctrl_yield_paired"] = ctrl_y(paired)
        ent["ctrl_yield_excluded"] = ctrl_y(excluded) if excluded else float("nan")
        ent["n_excluded"] = len(excluded)
        res[label] = ent
    return res


# ── 3. 원문에 없는 개체명 추가 (self_tr EN-index 재작성 hop) ─────────────────────

STOP_EN = {"the", "of", "in", "is", "was", "what", "which", "who", "whom", "where", "when", "how", "does",
           "did", "and", "or", "for", "to", "by", "on", "at", "an", "as", "are", "be", "its", "that", "with",
           "from", "this", "were", "has", "have", "do", "located", "country"}
KO_PARTICLE = re.compile(r"(은|는|이|가|을|를|의|에|에서|으로|로|와|과|도|만|이다|인가|입니까|인가요)$")


def content_tokens(s):
    toks = []
    for t in re.findall(r"[0-9A-Za-z가-힣]+", s or ""):
        t = KO_PARTICLE.sub("", t.lower())
        if len(t) >= 2 and t not in STOP_EN:
            toks.append(t)
    return toks


def is_other(r):
    """개체 추가 hop 중 '3자 이하 alias 만' 도 '같은 개체가 원문에 다른 표면형으로 있음' 도 아닌 hop."""
    return bool(r["added"] and not r["added_same_entity_in_source"] and not r["added_short_only"])


def added_entities(m, qa, cache, by, trj, h0c, rows):
    out = []
    for r in rows:
        q, k = r["qid"], int(r["hop"])
        path = qa[q]["hops"]
        src, qry = pdr16.normalize(r["source_query"]), pdr16.normalize(r["retrieval_query"])
        added, translated = [], False
        for ent_hop in path:                                           # bridge(hop0 gold) · 정답(hop1 gold)
            al = pdr16.build_alias_surfaces(ent_hop["gold_entity_qid"], cache, ent_hop["gold_entity"])
            new = sorted(a for a in al if a and a in qry and a not in src)
            if new:
                added += new
                if any(a and a in src for a in al):
                    translated = True
        a0 = next(h for h in by["self_tr"][q]["hops"] if h["hop"] == 0)
        state = ("애매" if a0.get("ambiguous_answer_marker") else "기권" if a0.get("abstained")
                 else "정답" if h0c["self_tr"][q] == 1 else "오답")
        new_tok = [t for t in content_tokens(r["retrieval_query"]) if t not in src]
        out.append({"model": m, "qid": q, "hop": k, "index_lang": r["index_lang"], "hop0_state": state,
                    "added": int(bool(added)), "added_same_entity_in_source": int(bool(added) and translated),
                    "added_short_only": int(bool(added) and all(len(a) <= 3 for a in added)),
                    "added_surfaces": "|".join(added), "n_new_tokens": len(new_tok),
                    "source_query": r["source_query"], "retrieval_query": r["retrieval_query"]})
    return out


# ── 출력 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--cluster-key", default="qid", choices=["qid", "path_cluster_id"],
                    help="재표집 단위 (path_cluster_id = entity path 250개, 원고 정본 CI 규칙)")
    args = ap.parse_args()

    qa = {r["qid"]: r for r in jl(QA_PATH)}
    global CLUSTER, UNIT
    if args.cluster_key != "qid":
        CLUSTER = {q: r[args.cluster_key] for q, r in qa.items()}
        UNIT = f"entity-path({args.cluster_key} {len(set(CLUSTER.values()))})"
    cache = json.loads(ALIAS.read_text(encoding="utf-8"))
    res, gq, cq, added = {}, {}, {}, []
    for m in args.models:
        print(f"[진행] {m}", file=sys.stderr, flush=True)
        order, by, trj, h0c, rows, added_m = analyse_model(m, qa, cache, args.n_boot, args.seed, res)
        gq[m] = gen_question_effect(m, qa, order, by, trj, args.n_boot, args.seed)
        cq[m] = canon_hop_qmc(m, qa, args.n_boot, args.seed)
        added += added_m

    P = print
    P("# self-translate 분석 표 (자동 생성)\n")
    P(f"입력: `data/self_translate/` annotated·trajectories (6모델 × 4조건). "
      f"{UNIT} paired bootstrap B={args.n_boot}, seed {args.seed}. `*` = 95% CI 가 0 을 제외. hit@5 축은 보류.\n")
    P("조건별 gap·전체 hop QMC·패턴별 AFR 은 기존 파이프라인 출력(`self_translate/pipeline/`)을 본다.\n")

    def r_table(get):
        P("| 모델 | 분자 | Δsame | Δcross | 분모 tmpl (gap_ctrl − gap_tmpl) | R_tmpl | 분모 full (gap_ctrl − gap_norm) | R_full | 항등 위반 draw |")
        P("|-|-|-|-|-|-|-|-|-|")
        for m in args.models:
            e = get(m)
            cells = [fmt_ci(*e["분자"]), fmt_ci(*e["Δsame"]), fmt_ci(*e["Δcross"])]
            for name in ("R_tmpl", "R_full"):
                cells.append(fmt_ci(*e[name]["분모"]))
                cells.append("분모 CI 하한 ≤ 0 — 비율 미보고" if e[name]["den_lo_not_pos"]
                             else f"{e[name]['비율'][0]:.3f} [{e[name]['비율'][1]:.3f}, {e[name]['비율'][2]:.3f}]")
            P(f"| {m} | " + " | ".join(cells) + f" | {e['항등 위반 draw']} |")
        P()

    P("## 1. 회복률 — yield 축 (분자 = gap_ctrl − gap_self = Δcross − Δsame)\n")
    P("비율 규칙 (2026-09-17 변경, self_translate.md §2.6 (5-a) 기록): R 은 **분모 95% CI 하한 > 0** 일 때만 보고한다. "
      "그 밖에는 분자·분모 값만 적는다. 이 절·부분집합·탐색 표 전부에 같은 규칙을 쓴다.\n")
    for v in (VARIANTS[0], VARIANTS[1], VARIANTS[2], SENS2B, H0_INTER, VARIANTS[3]):
        P(f"### {v}\n")
        r_table(lambda m, v=v: res[m]["R"][v])
    P("### hit@5 축 (answer-bearing hit@5, `analysis/10_retrieval_proxy.py` 출력 — 사전 명세 (5-a) (6-b))\n")
    for v in (VARIANTS[0], VARIANTS[1], VARIANTS[2], H0_INTER, VARIANTS[3]):
        P(f"#### hit@5 · {v}\n")
        r_table(lambda m, v=v: res[m]["R_hit"][v])
    P("변형별 제외·포함 규모\n")
    P("| 모델 | 민감도 1 제외 (qid, hop) | 민감도 2 제외 (qid, hop) | [탐색] 민감도 2-b 제외 (qid, hop) | hop0 정답 교집합 QID (주) | 조건별 hop0 정답 궤적 ctrl / tmpl_dec / norm_dec / self_tr (부록) |")
    P("|-|-|-|-|-|-|")
    for m in args.models:
        h = res[m]["n_h0c"]
        P(f"| {m} | {res[m]['n_sens1']} | {res[m]['n_sens2']} | {res[m]['n_sens2b']} | {res[m]['n_inter']} | "
          + " / ".join(str(h[c]) for c in CONDS) + " |")
    P()
    P("참고: 부록 방식(조건별 자기 hop0 정답 궤적)에서 부분집합의 hop0 yield (점추정, n / 평균)\n")
    P("| 모델 | 구분 | ctrl | tmpl_dec | norm_dec | self_tr |")
    P("|-|-|-|-|-|-|")
    for m in args.models:
        for k in ("전체", "same", "cross"):
            P(f"| {m} | {k} | " + " | ".join(
                f"{res[m]['h0c_hop0_yield'][c][k][0]} / {res[m]['h0c_hop0_yield'][c][k][1]:.3f}" for c in CONDS) + " |")
    P()

    P("## 2. [탐색 — 사전 명세 외] hop 별 회복률과 조건별 Δsame (yield 축)\n")
    for k in (0, 1):
        P(f"### hop{k} 만\n")
        r_table(lambda m, k=k: res[m]["R_hop"][k])
    P("### hop1 만 — 개체 추가 '둘 다 아님' hop 을 네 조건 모두에서 제외\n")
    P("제외 = self_tr EN-index 재작성 hop1 중 §8 의 '둘 다 아님' 판정 (qid, 1). "
      "제외 수: " + ", ".join(f"{m} {res[m]['n_other_hop1']}" for m in args.models) + "\n")
    r_table(lambda m: res[m]["R_hop1_ex_other"])
    P("### hop1 만 — EN-index hop1 은 norm_dec 정규화가 발동한 hop 만 (KO-index hop1 은 전부), 네 조건 모두 같은 (qid, hop)\n")
    P("정규화 발동 = 이번 norm_dec trajectory hop1 의 `normalization_applied`. 발동 hop 수: "
      + ", ".join(f"{m} {res[m]['n_fired']}" for m in args.models) + "\n")
    for ax, axn in (("y", "yield"), ("h", "hit@5")):
        P(f"#### {axn}\n")
        r_table(lambda m, ax=ax: res[m]["R_fired"][ax])
    P("### 전체 표본 EN-index hop1 의 정규화 발동률 — 이번 norm_dec vs 논문 surface-normalized 정본 실행\n")
    P("| 모델 | 이번 norm_dec | 정본 surface-normalized (trajectory) |")
    P("|-|-|-|")
    for m in args.models:
        a, n = res[m]["fire_all"]
        _, tp = corr.paths(m, "surface_normalized")
        st = [h for r in jl(tp) if r["qid"] in qa for h in r["trajectory"] if int(h["hop"]) == 1 and h.get("gold_lang") == "en"]
        ca = sum(1 for h in st if h.get("normalization_applied"))
        P(f"| {m} | {a}/{n} ({a/n:.1%}) | {ca}/{len(st)} ({ca/len(st):.1%}) — `{tp.name}` |")
    P()
    P("### hop0 정답 교집합 QID 의 EN-index hop1 에서 norm_dec 정규화가 실제로 발동한 비율 (점추정)\n")
    P("| 모델 | 발동 (`normalization_applied`) | normalization_status 분포 |")
    P("|-|-|-|")
    for m in args.models:
        a, n, dist = res[m]["norm_applied"]
        P(f"| {m} | {a}/{n} ({a/n:.1%}) | " + ", ".join(f"{k} {v}" for k, v in dist.items()) + " |")
    P()
    P("### Δsame (각 조건 same-index yield − ctrl same-index yield)\n")
    P("| 모델 | 범위 | tmpl_dec | norm_dec | self_tr |")
    P("|-|-|-|-|-|")
    for m in args.models:
        for scope in ("주 분석", "hop0", "hop1"):
            P(f"| {m} | {'전체 hop' if scope == '주 분석' else scope} | "
              + " | ".join(fmt_ci(*res[m]["dsame"][(scope, c)]) for c in ("tmpl_dec", "norm_dec", "self_tr")) + " |")
    P()

    P("## 3. hop 별 QMC 재분해 — 분리판 기준 (gap(ctrl) = template + bridge + residual)\n")
    P("template = gap_ctrl − gap_tmpl_dec, bridge = gap_tmpl_dec − gap_norm_dec, residual = gap_norm_dec. "
      "hop0 은 tmpl_dec·norm_dec 가 같은 hop0 을 공유하므로 bridge 성분이 0 이다. 전체 hop 분해는 §5 와 pipeline/ 출력.\n")
    for ax, name in (("y", "yield"), ("f", "faithfulness")):
        for scope in ("hop0", "hop1"):
            P(f"### {name} · {scope}\n")
            P("| 모델 | gap(ctrl) | template | bridge | residual |")
            P("|-|-|-|-|-|")
            for m in args.models:
                q = res[m]["qmc"][(ax, scope)]
                P(f"| {m} | " + " | ".join(fmt_ci(*q[k]) for k in ("gap(ctrl)", "template", "bridge", "residual")) + " |")
            P()

    P("### [탐색 — 사전 명세 외] hop 별 QMC 성분 — 분리판(이번) vs 결합판(논문 정본)\n")
    P("결합판 = 정본 controlled / template-matched / surface-normalized annotated (8B 는 기권 판정 일원화본, frontier 는 1회차 실행), "
      f"core 1,000 QID. 같은 정의·같은 {UNIT} 재표집(B, seed 동일; QID 순서는 각 묶음의 controlled 파일 순서).\n")
    P("| 모델 | hop | 축 | 성분 | 결합판 (논문) | 분리판 (이번) |")
    P("|-|-|-|-|-|-|")
    for m in args.models:
        for scope in ("hop0", "hop1"):
            for ax, axn in (("y", "yield"), ("f", "faithfulness")):
                for comp in ("gap(ctrl)", "template", "bridge", "residual"):
                    P(f"| {m} | {scope} | {axn} | {comp} | {fmt_ci(*cq[m][(ax, scope)][comp])} | "
                      f"{fmt_ci(*res[m]['qmc'][(ax, scope)][comp])} |")
    P()

    P("## 4. 생성 질문 효과 — 결합판 hop1 − 분리판 hop1 (hop0 생성 일치 짝)\n")
    P("부호: template = 정본 template-matched(query_matched) hop1 − 이번 tmpl_dec hop1, "
      "surface = 정본 surface-normalized(norm) hop1 − 이번 norm_dec hop1, "
      "잡음 기준선 = 정본 controlled hop1 − 이번 ctrl hop1. 양수 = 정본 쪽 값이 크다.\n")
    P("짝 = 정본 결합판 trajectory 와 이번 분리판 trajectory 의 hop0 `generated_text` 가 같은 QID. "
      "잡음 기준선은 같은 규칙으로 정본 controlled 와 이번 ctrl 을 짝지었다. 비교마다 짝 QID 목록 위에서 seed 42 로 따로 재표집했다.\n")
    P("### 전체 (모델 × 성분 × 축, 18개 비교 × 2축)\n")
    P("| 모델 | 성분 | 축 | 짝 / 전체 | 결합판 − 분리판 hop1 |")
    P("|-|-|-|-|-|")
    for m in args.models:
        for label, e in gq[m].items():
            for ax, axn in (("y", "yield"), ("f", "faithfulness")):
                P(f"| {m} | {label} | {axn} | {e['n_pair']} / {e['n_total']} | {fmt_ci(*e[ax])} |")
    P()
    P("### 모델별 잡음 기준선\n")
    P("| 모델 | 짝 / 전체 | yield | faithfulness |")
    P("|-|-|-|-|")
    for m in args.models:
        e = gq[m]["잡음 기준선 (정본 controlled − 이번 ctrl)"]
        P(f"| {m} | {e['n_pair']} / {e['n_total']} | {fmt_ci(*e['y'])} | {fmt_ci(*e['f'])} |")
    P()
    P("### 선택 편향 점검 — 이번 ctrl 의 yield·hit@5 (전체 hop 평균), 짝 / 제외\n")
    P("| 모델 | 성분 | yield 짝 | yield 제외 | hit@5 짝 | hit@5 제외 | 제외 수 |")
    P("|-|-|-|-|-|-|-|")
    for m in args.models:
        for label, e in gq[m].items():
            P(f"| {m} | {label} | {e['ctrl_yield_paired']:.3f} | {e['ctrl_yield_excluded']:.3f} | "
              f"{e['ctrl_hit_paired']:.3f} | {e['ctrl_hit_excluded']:.3f} | {e['n_excluded']} |")
    P()

    P("## 5. 전체 hop QMC 성분 — 분리판(이번, 11 stage1·stage2) vs 결합판(논문 정본, results)\n")
    P("같은 `analysis/11_ablation_compare.py` 출력에서 읽었다. 분리판 stage1 = ctrl→tmpl_dec, stage2 = tmpl_dec→norm_dec; "
      "결합판 stage1 = controlled→template-matched, stage2 = template-matched→surface-normalized.\n")
    P("| 모델 | 축 | 성분 | 결합판 (논문) | 분리판 (이번) |")
    P("|-|-|-|-|-|")
    pat = {"faithfulness": r"^\s+QMC 95% CI[^:]*: (.+)$", "yield": r"^\s+Yield-QMC 95% CI[^:]*: (.+)$"}

    def q11(path, axn):
        mm = re.search(pat[axn], Path(path).read_text(encoding="utf-8"), re.M)
        return re.sub(r"\s+", " ", mm.group(1)).strip() if mm else "없음"
    PIPE = CSVD / "pipeline"
    for m in args.models:
        for axn in ("yield", "faithfulness"):
            for st, newf in (("template (stage1)", f"11_{m}_stage1_ctrl_vs_tmpl_dec.txt"),
                             ("bridge (stage2)", f"11_{m}_stage2_tmpl_dec_vs_norm_dec.txt")):
                oldf = ROOT / "results" / f"11_{m}_qmc_{'stage1' if 'stage1' in st else 'stage2'}.txt"
                P(f"| {m} | {axn} | {st} | {q11(oldf, axn)} | {q11(PIPE / newf, axn)} |")
    P()

    P("## 6. 재작성이 원문과 달라진 비율 (EN-index hop, 공백 정규화 후 비교)\n")
    P("| 모델 | 달라짐 | 원문 그대로 |")
    P("|-|-|-|")
    for m in args.models:
        ch, n = res[m]["changed"]
        P(f"| {m} | {ch}/{n} ({ch/n:.1%}) | {n - ch} |")
    P()

    P("## 7. hop0 기권 → bridge 가 기권 문자열인 hop1 (별도 집계, 점추정)\n")
    P("| 모델 | ctrl n / yield | tmpl_dec | norm_dec | self_tr |")
    P("|-|-|-|-|-|")
    for m in args.models:
        a = res[m]["abst_bridge"]
        P(f"| {m} | " + " | ".join(f"{a[c][0]} / {a[c][1]:.3f}" for c in CONDS) + " |")
    P()

    P("## 8. 원문에 없는 개체명 추가 (self_tr EN-index 재작성 hop)\n")
    P("주 판정: 경로의 bridge(hop0 gold)·정답(hop1 gold) alias 중 재작성 쿼리에는 있고 원문 KO 쿼리에는 없는 표면형이 있으면 '추가'. "
      "PDR 매처 정규화(NFC·소문자·부분문자열). 보조: 원문에 부분문자열로 없는 내용 토큰 수(2자 이상, 조사·불용어 제거). "
      "'같은 개체가 원문에 있음' = 추가된 표면형의 개체가 다른 표면형으로 원문에 이미 있는 경우(번역·표준화). "
      "'둘 다 아님' = 추가 hop 중 '3자 이하 alias 만' 도 '같은 개체가 원문에 있음' 도 아닌 hop.\n")
    P("| 모델 | hop0 상태 | hop 수 | 추가 | 그중 같은 개체가 원문에 있음 | 그중 추가 표면형이 전부 3자 이하 | 둘 다 아님 | 보조: 새 토큰 수 평균 |")
    P("|-|-|-|-|-|-|-|-|")
    grp = defaultdict(list)
    for r in added:
        grp[(r["model"], r["hop0_state"])].append(r)
    other = is_other
    for m in args.models:
        for st in ("정답", "오답", "기권", "애매"):
            g = grp.get((m, st), [])
            if not g:
                continue
            a = sum(r["added"] for r in g)
            t = sum(r["added_same_entity_in_source"] for r in g)
            sh = sum(r["added_short_only"] for r in g)
            o = sum(1 for r in g if other(r))
            P(f"| {m} | {st} | {len(g)} | {a} ({a/len(g):.1%}) | {t} | {sh} | {o} | {sum(r['n_new_tokens'] for r in g)/len(g):.2f} |")
    P()

    def examples(cands, title):
        random.seed(args.seed)
        ex = sorted(random.sample(cands, min(10, len(cands))), key=lambda r: (MODELS.index(r["model"]), r["qid"]))
        P(f"{title}\n")
        P("| 모델 | qid · hop | hop0 | 원문 | 재작성 쿼리 | 추가된 표면형 |")
        P("|-|-|-|-|-|-|")
        for r in ex:
            P(f"| {r['model']} | {r['qid']} · {r['hop']} | {r['hop0_state']} | {r['source_query']} | "
              f"{r['retrieval_query']} | {r['added_surfaces']} |")
        P()
    examples([r for r in added if r["added"]], "사례 10개 — 추가 판정 hop 전체 중 무작위, seed 42")
    examples([r for r in added if other(r)], "사례 10개 — '둘 다 아님' hop 중 무작위, seed 42")

    esc = lambda t: (t or "").replace("|", "\\|").replace("\n", " ⏎ ")                   # noqa: E731

    if "exaone" in args.models:
        P("## 9. 재작성 실패 분류 — EXAONE 머리말·여러 줄 402건\n")
        P("첫 줄 = 실제 검색 쿼리. '비질문' = 물음표 없음(self_translate_compare.form) 또는 머리말·레이블로 시작"
          "(self_translate_check.PREAMBLE).\n")
        mixed = [r for r in res["exaone"]["rows"] if r["explanation_mixed"] == "1"]
        nq = [r for r in mixed if first_line_nonquestion(r)]
        okq = [r for r in mixed if not first_line_nonquestion(r)]
        P("| 첫 줄 | hop 수 | finish=length |")
        P("|-|-|-|")
        for name, g in (("정상 질문", okq), ("머리말·레이블·비질문", nq)):
            fl = sum(1 for r in g if str(r.get("finish_reason", "")).lower() in ("length", "max_tokens"))
            P(f"| {name} | {len(g)} | {fl} |")
        P()
        for name, g in (("정상 질문", okq), ("머리말·레이블·비질문", nq)):
            random.seed(args.seed)
            P(f"사례 5개 — 첫 줄 {name} (seed 42)\n")
            P("| qid · hop | finish_reason | 원문 | raw 출력 (앞 220자, 줄바꿈 = ⏎) | 첫 줄 (검색 쿼리) |")
            P("|-|-|-|-|-|")
            for r in random.sample(g, min(5, len(g))):
                P(f"| {r['qid']} · {r['hop']} | {r['finish_reason']} | {esc(r['source_query'])} | "
                  f"{esc(r['raw_output'][:220])} | {esc(r['retrieval_query'])} |")
            P()
        P("민감도 2-b 의 회복률은 §1 의 [탐색] 표.\n")

    if "aya" in args.models:
        P("## 10. 재작성 실패 분류 — Aya 언어 불일치 중 원문 반환이 아닌 hop\n")
        P("`_detect_lang` (pipeline/05_pilot_runner.py:448): 한글 음절(가–힣) 수 / **전체 문자열 길이**(공백·따옴표·구두점·라틴 포함) "
          "> 0.2 이면 ko, 아니면 en. 따옴표는 한글이 아닌 문자로 분모에만 들어가고, 영한 혼용은 한글 비율로만 갈린다.\n")
        P("원문 반환 = 따옴표·끝 구두점·공백 정규화 후 raw 출력 == 원문. 유형은 첫 줄(검색 쿼리)에서 앞뒤 따옴표를 뗀 문자열로 나눈다.\n")
        norm = lambda t: re.sub(r"\s+", " ", (t or "").strip().strip('"\u201c\u201d\'').rstrip(".?？")).lower()  # noqa: E731
        rows_a = res["aya"]["rows"]
        mism = [r for r in rows_a if r["script_ok"] != "1"]
        rest = [r for r in mism if norm(r["raw_output"]) != norm(r["source_query"])]

        def aya_type(r):
            q = (r["retrieval_query"] or "").strip().strip('"\u201c\u201d\'')
            ko, la = bool(re.search(r"[가-힣]", q)), bool(re.search(r"[A-Za-z]", q))
            if ko and la:
                return "한영 혼용"
            if not ko:
                return "한글 없음"
            if "?" in q or "？" in q:
                return "한국어 질문"
            return "한국어 서술문" if form_of(q) == "declarative" else "한국어 비질문 (구·키워드)"
        P(f"언어 불일치 {len(mism)} − 원문 반환 {len(mism) - len(rest)} = **{len(rest)}**\n")
        P("| 유형 | hop 수 | 그중 검색 쿼리가 따옴표로 시작 |")
        P("|-|-|-|")
        tc = Counter(aya_type(r) for r in rest)
        for t, n in tc.most_common():
            qn = sum(1 for r in rest if aya_type(r) == t and (r["retrieval_query"] or "").strip().startswith(('"', "\u201c", "'")))
            P(f"| {t} | {n} | {qn} |")
        P()
        random.seed(args.seed)
        P("사례 10개 (seed 42)\n")
        P("| qid · hop | 유형 | 원문 | raw 출력 | 검색 쿼리 | 한글 비율 (_detect_lang 기준) |")
        P("|-|-|-|-|-|-|")
        for r in sorted(random.sample(rest, min(10, len(rest))), key=lambda r: r["qid"]):
            q = r["retrieval_query"] or ""
            ratio = sum(1 for ch in q if "가" <= ch <= "힣") / max(len(q), 1)
            P(f"| {r['qid']} · {r['hop']} | {aya_type(r)} | {esc(r['source_query'])} | {esc(r['raw_output'][:160])} | "
              f"{esc(q)} | {ratio:.2f} |")
        P()

    P("## 11. self_tr vs norm_dec hop 별 hit@5 불일치 사례 20개 (한쪽만 hit, 사전 명세 (5-c))\n")
    allmm = [r for m in args.models for r in res[m]["mismatch"]]
    P("불일치 hop 수: " + ", ".join(
        f"{m} hop0 {sum(1 for r in res[m]['mismatch'] if r['hop'] == 0)} · hop1 {sum(1 for r in res[m]['mismatch'] if r['hop'] == 1)}"
        for m in args.models) + "\n")
    P("층화: hop0·hop1 각 10개, 6모델 합친 풀에서 seed 42 무작위.\n")
    P("| 모델 | qid · hop | 인덱스 | self_tr hit | norm_dec hit | self_tr 검색 쿼리 | norm_dec 검색 쿼리 |")
    P("|-|-|-|-|-|-|-|")
    random.seed(args.seed)
    for k in (0, 1):
        pool = [r for r in allmm if r["hop"] == k]
        for r in sorted(random.sample(pool, min(10, len(pool))), key=lambda r: (MODELS.index(r["model"]), r["qid"])):
            P(f"| {r['model']} | {r['qid']} · {k} | {r['index_lang']} | {r['self_tr_hit']} | {r['norm_dec_hit']} | "
              f"{esc(r['self_tr_query'])} | {esc(r['norm_dec_query'])} |")
    P()

    out_csv = CSVD / "added_entities.csv"
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(added[0]))
        w.writeheader()
        w.writerows(added)
    print(f"[저장] {out_csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
