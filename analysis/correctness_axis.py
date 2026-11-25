#!/usr/bin/env python3
"""
analysis/correctness_axis.py

리뷰 대응용: 기존 trajectory·annotated 결과로 hop 정답 정확도와 궤적 성공률을 계산한다.
새 생성·채점 없음. 기존 결과 파일은 읽기만 하고, 출력은 reports/rebuttal/ 아래에만 쓴다.

정의
  correct_h   = ¬abstained ∧ 응답(generated_text)에 hop gold 엔티티 또는 alias 포함.
                기권 → 0, ambiguous_answer_marker → None(모든 분모 제외).
                hop1 gold 는 경로의 gold 정답(qa_pairs_core hops[1].gold_entity_qid) — hop0 오답과 무관.
  매처(주)    = 논문 parametric fallback rate 매처 (analysis/16_pdr_analysis.py
                build_alias_surfaces + entity_in_text; 13_bootstrap_ci.build_pdr_data 와 동일 로직)
  매처(민감도)= answer-bearing hit@5 매처 (utils.retrieval_success.check_answer_bearing_hit),
                passages 자리에 [generated_text] 를 넘긴다.
  faithful    = utils.metrics.hop_yield == 1  (¬abstained ∧ f_h ≥ 0.5)
  traj 성공   = correct_0 ∧ correct_1 (한 hop 이라도 None 이면 분모 제외)
  same/cross  = utils.metrics.index_class (gold_lang 기준), 집계 = utils.metrics.lshd

통계: QID-cluster paired bootstrap. analysis/yield_qmc_joint_ci.py 와 같은 재표집
(random.seed(seed) 후 매 draw random.choices(triples, k=N)) — 따라서 yield 값은 그 스크립트 출력과 일치해야 한다.

사용:
  python3 analysis/correctness_axis.py --models llama --conds controlled --tag pilot_llama_ctrl
  python3 analysis/correctness_axis.py --tag all
"""
import argparse
import csv
import importlib.util
import json
import math
import random
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.metrics import hop_faith, hop_yield, index_class, lshd          # noqa: E402
from utils.retrieval_success import check_answer_bearing_hit    # noqa: E402

_spec = importlib.util.spec_from_file_location("pdr16", ROOT / "analysis" / "16_pdr_analysis.py")
pdr16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdr16)

QA = ROOT / "data/dataset/qa_pairs.jsonl"
ALIAS = ROOT / "data/retrieval/alias_cache.json"
MASTER = ROOT / "data/dataset/triples.jsonl"
OUT = ROOT / "reports/rebuttal"

CONDS = ["controlled", "template_matched", "surface_normalized"]
MODELS = ["aya", "exaone", "llama", "qwen", "sonnet", "gemini"]


# scripts/camera_ready/_inputs.sh 의 ann/qm/nm · traj/tqm/tnm 과 같은 정본 입력 (판정, 궤적).
# 경로를 그대로 적는다 — 배포 빌더가 경로 문자열을 배포 이름으로 바꾸므로 조립한 파일명은 배포본에서 깨진다.
_PATHS = {
    ("aya", "controlled"): ("data/annotated/aya_ecological.jsonl",
                            "data/trajectories/aya_ecological.jsonl"),
    ("exaone", "controlled"): ("data/annotated/exaone_ecological.jsonl",
                               "data/trajectories/exaone_ecological.jsonl"),
    ("llama", "controlled"): ("data/annotated/llama_ecological.jsonl",
                              "data/trajectories/llama_ecological.jsonl"),
    ("qwen", "controlled"): ("data/annotated/qwen_ecological.jsonl",
                             "data/trajectories/qwen_ecological.jsonl"),
    ("sonnet", "controlled"): ("data/annotated/sonnet_ecological.jsonl",
                               "data/trajectories/sonnet_ecological.jsonl"),
    ("gemini", "controlled"): ("data/annotated/gemini_ecological.jsonl",
                               "data/trajectories/gemini_ecological.jsonl"),
    ("aya", "template_matched"): ("data/annotated/aya_query_matched.jsonl",
                                  "data/trajectories/aya_query_matched.jsonl"),
    ("exaone", "template_matched"): ("data/annotated/exaone_query_matched.jsonl",
                                     "data/trajectories/exaone_query_matched.jsonl"),
    ("llama", "template_matched"): ("data/annotated/llama_query_matched.jsonl",
                                    "data/trajectories/llama_query_matched.jsonl"),
    ("qwen", "template_matched"): ("data/annotated/qwen_query_matched.jsonl",
                                   "data/trajectories/qwen_query_matched.jsonl"),
    ("sonnet", "template_matched"): ("data/annotated/sonnet_query_matched.jsonl",
                                     "data/trajectories/sonnet_query_matched.jsonl"),
    ("gemini", "template_matched"): ("data/annotated/gemini_query_matched.jsonl",
                                     "data/trajectories/gemini_query_matched.jsonl"),
    ("aya", "surface_normalized"): ("data/annotated/aya_normalized.jsonl",
                                    "data/trajectories/aya_normalized.jsonl"),
    ("exaone", "surface_normalized"): ("data/annotated/exaone_normalized.jsonl",
                                       "data/trajectories/exaone_normalized.jsonl"),
    ("llama", "surface_normalized"): ("data/annotated/llama_normalized.jsonl",
                                      "data/trajectories/llama_normalized.jsonl"),
    ("qwen", "surface_normalized"): ("data/annotated/qwen_normalized.jsonl",
                                     "data/trajectories/qwen_normalized.jsonl"),
    ("sonnet", "surface_normalized"): ("data/annotated/sonnet_normalized.jsonl",
                                       "data/trajectories/sonnet_normalized.jsonl"),
    ("gemini", "surface_normalized"): ("data/annotated/gemini_normalized.jsonl",
                                       "data/trajectories/gemini_normalized.jsonl"),
}


def paths(model: str, cond: str) -> tuple[Path, Path]:
    """정본 입력 (판정, 궤적) 경로."""
    ann, trj = _PATHS[(model, cond)]
    return ROOT / ann, ROOT / trj


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


# ── hop 단위 레코드 ───────────────────────────────────────────────────────────

LIST_SEPS = (",", "、", " 및 ", " and ", "/", " 또는 ", " or ")


def pdr_matched(aliases: set[str], text: str) -> list[str]:
    """16_pdr_analysis.entity_in_text 와 같은 판정에서 어떤 alias 가 걸렸는지만 나열한다."""
    t = pdr16.normalize(text)
    return sorted(a for a in aliases if a and a in t)


def build_hops(model, cond, qa, alias_cache, atype):
    ann_p, trj_p = paths(model, cond)
    ann = {r["qid"]: r for r in load_jsonl(ann_p) if r["qid"] in qa}
    trj = {r["qid"]: r for r in load_jsonl(trj_p) if r["qid"] in qa}
    assert set(ann) == set(qa) == set(trj), (model, cond, len(ann), len(trj))
    records, rows, issues = [], [], Counter()
    for qid in ann:                       # annotated 파일 순서 (bootstrap QID 순서용)
        q, a, t = qa[qid], ann[qid], trj[qid]
        steps = {int(s["hop"]): s for s in t["trajectory"]}
        rec = {"qid": qid, "pattern": q["pattern"], "hops": []}
        for qh in q["hops"]:
            k = qh["hop"]
            ah = next(h for h in a["hops"] if h["hop"] == k)
            st = steps[k]
            if ah["gold_lang"] != qh["gold_lang"]:
                issues["gold_lang_mismatch_ann_vs_qa"] += 1
            if st.get("gold_entity") != qh["gold_entity"]:
                issues["gold_entity_mismatch_traj_vs_qa"] += 1
            text = st.get("generated_text", "") or ""
            aliases = pdr16.build_alias_surfaces(qh["gold_entity_qid"], alias_cache, qh["gold_entity"])
            matched = pdr_matched(aliases, text)
            raw_hit = bool(text) and pdr16.entity_in_text(aliases, text)
            assert raw_hit == bool(matched and text)
            hit5 = check_answer_bearing_hit([text], qh["gold_entity_qid"], alias_cache,
                                            qh["gold_lang"])["answer_bearing_hit"] if text else False
            amb, abst = bool(ah.get("ambiguous_answer_marker")), bool(ah.get("abstained"))
            correct = None if amb else (0 if abst else int(raw_hit))
            correct_alt = None if amb else (0 if abst else int(hit5))
            y = hop_yield(ah)
            query = pdr16.normalize(st.get("retrieval_query", "") or "")
            # 구버전 trajectory(aya dynamic·query_matched, exaone/llama/qwen dynamic)는 bridge_entity/bridge_correct 필드명
            bridge_raw = st.get("prev_answer_raw", st.get("bridge_entity", "")) or ""
            bridge_gold = st.get("prev_answer_is_gold_bridge", st.get("bridge_correct"))
            bridge = pdr16.normalize(bridge_raw)
            flags = {}
            if correct == 1:
                flags = {
                    "flag_digit_alias": any(any(c.isdigit() for c in m) for m in matched),
                    "flag_short_alias_only": all(len(m) <= 3 for m in matched),
                    "flag_single_latin_token_only": all(" " not in m and m.isascii() for m in matched),
                    "flag_list_like_response": sum(text.count(s) for s in LIST_SEPS) >= 1,
                    "flag_alias_in_query": any(m in query for m in matched),
                    "flag_alias_in_bridge": k > 0 and any(m in bridge for m in matched),
                }
            h = {
                "model": model, "cond": cond, "qid": qid, "pattern": q["pattern"], "hop": k,
                "gold_lang": qh["gold_lang"], "index_class": index_class(qh),
                "gold_entity": qh["gold_entity"], "gold_qid": qh["gold_entity_qid"],
                "answer_type_path": atype.get(q["source_path_id"].rsplit("_pat_", 1)[0], ""),
                "abstained": abst, "ambiguous": amb, "hop_faithfulness": ah.get("hop_faithfulness"),
                "faithful": y, "correct": correct, "correct_hit5matcher": correct_alt,
                "gold_in_text_raw": int(raw_hit), "matched_aliases": "|".join(matched),
                "query_contains_gold_answer_flag": st.get("query_contains_gold_answer"),
                "prev_answer_is_gold_bridge": bridge_gold,
                "retrieval_query": st.get("retrieval_query", ""), "prev_answer_raw": bridge_raw,
                "generated_text": text, "aliases": "|".join(sorted(aliases)),
                **flags,
            }
            rows.append(h)
            rec["hops"].append({"hop": k, "gold_lang": qh["gold_lang"], "acc": correct,
                                "acc_alt": correct_alt, "yield": y, "faith": hop_faith(ah)})
        records.append(rec)
    return records, rows, issues


# ── 통계량 ────────────────────────────────────────────────────────────────────

def _gap(records, key):
    g, sm, cr, ns, nc = lshd(records, lambda h: h[key])
    return g, sm, cr


def _traj(rec, key):
    v = [h[key] for h in rec["hops"]]
    return None if any(x is None for x in v) else int(all(v))


def cond_stats(records) -> dict:
    out = {}
    for key, name in (("acc", "acc"), ("acc_alt", "accalt"), ("yield", "yield")):
        g, sm, cr = _gap(records, key)
        out[f"{name}_same"], out[f"{name}_cross"], out[f"{name}_gap"] = sm, cr, g
    for pat in (None, "a", "b", "c", "d"):
        vals = [_traj(r, "acc") for r in records if pat is None or r["pattern"] == pat]
        vals = [v for v in vals if v is not None]
        out[f"traj_{pat or 'all'}"] = sum(vals) / len(vals) if vals else float("nan")
    # 최종 답 정확도 = hop1 정답률(hop0 정오와 무관). hop1 ambiguous 궤적은 분모에서 뺀다.
    for pat in (None, "a", "b", "c", "d"):
        vals = [r["hops"][1]["acc"] for r in records if pat is None or r["pattern"] == pat]
        vals = [v for v in vals if v is not None]
        out[f"final_{pat or 'all'}"] = sum(vals) / len(vals) if vals else float("nan")
    # 탈선했지만 faithful: hop1 faithful(=yield 1) ∧ hop0 correct==0, hop1 faithful 분모
    num = den = 0
    # 하한: 위 조건 + hop1 도 오답 (hop1 ambiguous 궤적은 제외) / 회복: hop1 정답 비율
    low_num = low_den = rec_num = rec_den = 0
    alt_num = alt_den = 0
    for r in records:
        h0, h1 = r["hops"][0], r["hops"][1]
        if h1["yield"] == 1.0 and h0["acc"] is not None:
            den += 1
            num += h0["acc"] == 0
            if h1["acc"] is not None:
                low_den += 1
                low_num += h0["acc"] == 0 and h1["acc"] == 0
                if h0["acc"] == 0:
                    rec_den += 1
                    rec_num += h1["acc"] == 1
            if h1["acc_alt"] is not None and h0["acc_alt"] is not None:
                alt_den += 1
                alt_num += h0["acc_alt"] == 0 and h1["acc_alt"] == 0
    out["derailed_faithful_share"] = num / den if den else float("nan")
    out["derailed_lowbound_share"] = low_num / low_den if low_den else float("nan")
    out["derailed_lowbound_share_hit5"] = alt_num / alt_den if alt_den else float("nan")
    out["derailed_recovery_share"] = rec_num / rec_den if rec_den else float("nan")
    return out


def qmc(c, t, n, name):
    gc, gt, gn = c[f"{name}_gap"], t[f"{name}_gap"], n[f"{name}_gap"]
    return {f"{name}_gap_ctrl": gc, f"{name}_template": gc - gt, f"{name}_bridge": gt - gn, f"{name}_residual": gn}


def all_stats(bundle: list[dict], conds: list[str]) -> dict:
    """bundle: [{cond: record}] (QID 단위). 같은 resample 로 모든 조건·지표를 평가."""
    per = {c: cond_stats([b[c] for b in bundle]) for c in conds}
    out = {f"{c}|{k}": v for c in conds for k, v in per[c].items()}
    if len(conds) == 3:
        c, t, n = (per[x] for x in CONDS)
        for name in ("acc", "accalt", "yield"):
            out.update({f"qmc|{k}": v for k, v in qmc(c, t, n, name).items()})
    return out


def ci(vals, alpha=0.05):
    s = sorted(vals)
    return s[int(alpha / 2 * len(s))], s[int((1 - alpha / 2) * len(s)) - 1]


def bootstrap(bundle, conds, n_boot, seed, clusters=None):
    """clusters 가 주어지면 그 그룹(triple) 단위로 재표집한다 (기본은 QID 단위)."""
    point = all_stats(bundle, conds)
    random.seed(seed)
    boot = defaultdict(list)
    ident_max = 0.0
    for _ in range(n_boot):
        samp = ([b for g in random.choices(clusters, k=len(clusters)) for b in g]
                if clusters is not None else random.choices(bundle, k=len(bundle)))
        m = all_stats(samp, conds)
        for k, v in m.items():
            if not math.isnan(v):
                boot[k].append(v)
        if len(conds) == 3:
            for name in ("acc", "yield"):
                parts = [m[f"qmc|{name}_{x}"] for x in ("template", "bridge", "residual")]
                if not any(math.isnan(p) for p in parts):
                    ident_max = max(ident_max, abs(sum(parts) - m[f"qmc|{name}_gap_ctrl"]))
    return point, {k: ci(v) + (len(v),) for k, v in boot.items()}, ident_max


# ── 표 ────────────────────────────────────────────────────────────────────────

def crosstab(rows):
    """faithful × correct 2×2 (+ 기권 행). ambiguous 제외."""
    t = Counter()
    for h in rows:
        if h["ambiguous"]:
            continue
        if h["abstained"]:
            t[("abstained", h["correct"], h["gold_in_text_raw"])] += 1
        else:
            t[("faithful" if h["faithful"] == 1.0 else "not_faithful", h["correct"], None)] += 1
    return t



def _sq(t: str) -> str:
    """공백·구두점·괄호 제거 + 소문자 (관대 부분 일치용)."""
    import re as _re
    return _re.sub(r"[\s\-·‧・.,'’\"()\[\]]", "", unicodedata.normalize("NFC", t).lower())


def _derail_table(all_rows, stat_rows, args):
    """탈선 faithful 하한 표: 카운트는 원자료에서, CI 는 bootstrap 결과에서 가져온다."""
    ci = {(r["model"], r["stat"]): r for r in stat_rows}

    def fmt(model, stat):
        r = ci.get((model, stat))
        if r is None or r["point"] is None:
            return "N/A"
        p_, lo, hi = float(r["point"]), float(r["ci_lo"]), float(r["ci_hi"])
        tag = " 경계" if (lo == 0.0 or hi == 0.0) else ("*" if (lo > 0 or hi < 0) else "")
        return f"{p_:.3f} [{lo:.3f}, {hi:.3f}]{tag}"

    by = defaultdict(dict)
    for h in all_rows:
        by[(h["model"], h["cond"], h["qid"])][int(h["hop"])] = h
    print(f"탈선 faithful 하한 — triple {250}개 재표집, B={args.n_boot:,}, seed={args.seed}, 주 매처\n")
    hdr = ("모델", "조건", "D", "R_all [95% CI]", "R_low 수", "R_low [95% CI]", "S [95% CI]")
    print(f"{hdr[0]:<8} {hdr[1]:<20} {hdr[2]:>5} {hdr[3]:>24} {hdr[4]:>8} {hdr[5]:>24} {hdr[6]:>24}")
    for model in args.models:
        for cond in args.conds:
            D = [hh for (m, c, q), hh in by.items() if m == model and c == cond
                 and hh[1]["faithful"] == 1.0 and not hh[0]["ambiguous"]]
            amb1 = sum(1 for hh in D if hh[1]["ambiguous"])
            Dl = [hh for hh in D if not hh[1]["ambiguous"]]
            r_all = sum(1 for hh in D if hh[0]["correct"] == 0)
            r_low = sum(1 for hh in Dl if hh[0]["correct"] == 0 and hh[1]["correct"] == 0)
            s_den = sum(1 for hh in Dl if hh[0]["correct"] == 0)
            s_num = sum(1 for hh in Dl if hh[0]["correct"] == 0 and hh[1]["correct"] == 1)
            print(f"{model:<8} {cond:<20} {len(D):>5} {fmt(model, cond + '|derailed_faithful_share'):>24} "
                  f"{r_low:>8} {fmt(model, cond + '|derailed_lowbound_share'):>24} "
                  f"{fmt(model, cond + '|derailed_recovery_share'):>24}"
                  f"   (R_all 수 {r_all}, hop1 ambiguous 제외 {amb1}, S 분모 {s_den}, S 분자 {s_num})")
    print("\nhit@5 매처 기준 R_low")
    print(f"{'모델':<8} {'조건':<20} {'R_low(hit@5) [95% CI]':>28}")
    for model in args.models:
        for cond in args.conds:
            print(f"{model:<8} {cond:<20} {fmt(model, cond + '|derailed_lowbound_share_hit5'):>28}")

    print("\n탈선(hop0 오답) ∧ hop1 정답 궤적에서 hop0 응답에 gold bridge 가 관대 매칭으로 들어있는 비율")
    print(f"{'모델':<8} {'조건':<20} {'해당 궤적':>8} {'관대 일치':>8} {'비율':>8}")
    for model in args.models:
        for cond in args.conds:
            sub = [hh for (m, c, q), hh in by.items() if m == model and c == cond
                   and hh[1]["faithful"] == 1.0 and not hh[0]["ambiguous"]
                   and not hh[1]["ambiguous"]
                   and hh[0]["correct"] == 0 and hh[1]["correct"] == 1]
            hit = sum(1 for hh in sub
                      if any(a and _sq(a) and _sq(a) in _sq(hh[0]["generated_text"])
                             for a in hh[0]["aliases"].split("|")))
            share = f"{hit / len(sub):.3f}" if sub else "N/A"
            print(f"{model:<8} {cond:<20} {len(sub):>8} {hit:>8} {share:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--conds", nargs="+", default=CONDS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--exclude-qids", default=None,
                    help="한 줄에 qid 하나인 파일. 그 qid 를 분석 전체에서 제외한다 (민감도용)")
    ap.add_argument("--resample", choices=["qid", "triple", "path_cluster_id"], default="qid",
                    help="bootstrap 재표집 단위 (triple = 한 triple 의 네 패턴을 함께 뽑는다; "
                         "path_cluster_id = 같은 단위를 analysis/yield_qmc_joint_ci.py --cluster-key path_cluster_id 와 "
                         "같은 순서로 뽑아, 원고 정본 CI 와 같은 resample 을 얻는다)")
    ap.add_argument("--out-dir", default=None,
                    help="출력 디렉터리 (기본 data/correctness/<tag>)")
    ap.add_argument("--derail-table", action="store_true",
                    help="탈선 faithful 하한 표(카운트·CI)와 관대 매칭 점검을 stdout 에 출력")
    ap.add_argument("--sample", action="store_true", help="data/correctness_sample.csv 작성")
    args = ap.parse_args()

    odir = Path(args.out_dir) if args.out_dir else OUT / "correctness" / args.tag
    odir.mkdir(parents=True, exist_ok=True)
    qa = {r["qid"]: r for r in load_jsonl(QA)}
    if args.exclude_qids:
        drop = {l.strip() for l in Path(args.exclude_qids).read_text().split() if l.strip()}
        qa = {q: r for q, r in qa.items() if q not in drop}
        print(f"[제외] {len(drop)} qid → 분석 대상 {len(qa)} qid", file=sys.stderr)
    alias_cache = json.loads(ALIAS.read_text(encoding="utf-8"))
    atype = {r["triple_id"]: r["answer_type"] for r in load_jsonl(MASTER)}

    all_rows, stat_rows, ct_rows, misc_rows, flag_rows = [], [], [], [], []
    for model in args.models:
        per_cond = {}
        for cond in args.conds:
            records, rows, issues = build_hops(model, cond, qa, alias_cache, atype)
            per_cond[cond] = {r["qid"]: r for r in records}
            all_rows += rows
            print(f"[{model}/{cond}] hops={len(rows)} issues={dict(issues)}", file=sys.stderr)
            # 2×2
            for (row, corr, raw), n in sorted(crosstab(rows).items(), key=str):
                ct_rows.append({"model": model, "cond": cond, "row": row, "correct": corr,
                                "abstained_gold_in_text_raw": raw, "n": n})
            # 수 세기 (5·6·ambiguous·매처 일치)
            byq = defaultdict(dict)
            for h in rows:
                byq[h["qid"]][h["hop"]] = h
            derail = Counter()
            for q, hh in byq.items():
                h0, h1 = hh[0], hh[1]
                if h0["ambiguous"] or h1["ambiguous"]:
                    derail["traj_excluded_ambiguous"] += 1
                    continue
                derail["traj_n"] += 1
                if h1["faithful"] == 1.0:
                    derail["hop1_faithful"] += 1
                    if h0["correct"] == 0:
                        derail["derailed_faithful"] += 1
                        derail["derailed_faithful_hop0_abstained" if h0["abstained"]
                               else "derailed_faithful_hop0_wrong_answer"] += 1
                        derail["derailed_faithful_and_hop1_correct"] += h1["correct"] == 1
                if h0["correct"] == 0 and h1["correct"] == 1:
                    derail["hop0_wrong_hop1_right"] += 1
                    derail["hop0_wrong_hop1_right_hop0_abstained" if h0["abstained"]
                           else "hop0_wrong_hop1_right_hop0_answered"] += 1
                    derail["hop0_wrong_hop1_right_bridge_exact_gold"] += str(h1["prev_answer_is_gold_bridge"]) == "True"
            nonamb = [h for h in rows if not h["ambiguous"]]
            misc_rows.append({"model": model, "cond": cond,
                              "n_hops": len(rows), "n_ambiguous": len(rows) - len(nonamb),
                              "n_abstained": sum(h["abstained"] for h in nonamb),
                              "n_correct": sum(h["correct"] for h in nonamb),
                              "n_correct_hit5matcher": sum(h["correct_hit5matcher"] for h in nonamb),
                              "matcher_disagree": sum(h["correct"] != h["correct_hit5matcher"] for h in nonamb),
                              "pdr_only": sum(h["correct"] == 1 and h["correct_hit5matcher"] == 0 for h in nonamb),
                              "hit5_only": sum(h["correct"] == 0 and h["correct_hit5matcher"] == 1 for h in nonamb),
                              "abstained_but_gold_in_text": sum(h["abstained"] and h["gold_in_text_raw"] for h in nonamb),
                              **dict(derail)})
            corr = [h for h in nonamb if h["correct"] == 1]
            fk = [k for k in (corr[0].keys() if corr else []) if k.startswith("flag_")]
            flag_rows.append({"model": model, "cond": cond, "n_correct": len(corr),
                              **{k: sum(bool(h[k]) for h in corr) for k in fk},
                              "any_flag": sum(any(h[k] for k in fk) for h in corr)})
        # QID 순서 = 첫 조건(annotated) 파일 순서 — analysis/yield_qmc_joint_ci.py 의 common 과 같게 해
        # 같은 seed 에서 같은 resample 을 얻는다.
        first = per_cond[args.conds[0]]
        common = [q for q in first if all(q in per_cond[c] for c in args.conds)]
        bundle = [{c: per_cond[c][q] for c in args.conds} for q in common]
        clusters = None
        if args.resample == "triple":
            by_t = defaultdict(list)
            for b in bundle:
                qid = b[args.conds[0]]["qid"]
                by_t[qa[qid]["source_path_id"].rsplit("_pat_", 1)[0]].append(b)
            clusters = [by_t[t] for t in sorted(by_t)]
            print(f"[{model}] triple 재표집: 클러스터 {len(clusters)}개", file=sys.stderr)
        elif args.resample == "path_cluster_id":
            # yield_qmc_joint_ci.py 처럼 common 순서에서 처음 나온 순서로 묶는다 — 같은 seed 에서 같은 resample.
            by_p = {}
            for b in bundle:
                by_p.setdefault(qa[b[args.conds[0]]["qid"]]["path_cluster_id"], []).append(b)
            clusters = list(by_p.values())
            print(f"[{model}] path_cluster_id 재표집: 클러스터 {len(clusters)}개", file=sys.stderr)
        point, cis, ident = bootstrap(bundle, args.conds, args.n_boot, args.seed, clusters)
        print(f"[{model}] bootstrap 완료 (N={len(bundle)}), 항등 최대 오차 {ident:.2e}", file=sys.stderr)
        for k, v in point.items():
            lo, hi, nb = cis.get(k, (float("nan"), float("nan"), 0))
            stat_rows.append({"model": model, "stat": k, "point": v, "ci_lo": lo, "ci_hi": hi,
                              "n_boot_valid": nb, "identity_max_abs_err": ident})

    def dump(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with (odir / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    dump("hops.csv", all_rows)
    dump("stats_bootstrap.csv", stat_rows)
    dump("faithful_x_correct.csv", ct_rows)
    dump("counts.csv", misc_rows)
    dump("match_risk_flags.csv", flag_rows)

    if args.derail_table:
        _derail_table(all_rows, stat_rows, args)

    if args.sample:
        rng = random.Random(args.seed)
        pool = [h for h in all_rows if not h["ambiguous"] and not h["abstained"]]
        pick = rng.sample([h for h in pool if h["correct"] == 1], 30) + \
            rng.sample([h for h in pool if h["correct"] == 0], 30)
        rng.shuffle(pick)
        cols = ["sample_id", "model", "cond", "qid", "hop", "pattern", "index_class", "gold_entity", "gold_qid",
                "aliases", "generated_text", "retrieval_query", "prev_answer_raw", "correct",
                "correct_hit5matcher", "matched_aliases", "faithful", "hop_faithfulness",
                "flag_digit_alias", "flag_short_alias_only", "flag_single_latin_token_only",
                "flag_list_like_response", "flag_alias_in_query", "flag_alias_in_bridge",
                "human_correct", "human_note"]
        with (OUT / "correctness_sample.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for i, h in enumerate(pick, 1):
                w.writerow({**h, "sample_id": i, "human_correct": "", "human_note": ""})
    print(f"저장: {odir}", file=sys.stderr)


if __name__ == "__main__":
    main()
