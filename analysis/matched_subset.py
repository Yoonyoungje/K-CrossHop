#!/usr/bin/env python3
"""
analysis/matched_subset.py

리뷰 대응용 matched-subset 분석. 기존 controlled 조건 결과만 읽는다 (새 생성·채점 없음).
출력은 reports/rebuttal/ 아래에만 쓴다.

단위: master triple (250개). 한 triple 은 패턴 a/b/c/d 네 궤적으로 인스턴스화되어 있다.
  hop0 index:  a,c = KO      b,d = EN
  hop1 index:  a,b = KO      c,d = EN

  0. 사전 확인
     a. Table 3(retrieval-conditioned Δf)·§5.3(hop0 R=1 Δy) 재현 —
        기준은 analysis/12_conditioned_analysis.py 를 그대로 실행한 출력이고,
        이 스크립트도 [0a'] 줄에서 자체 경로로 같은 값을 다시 낸다 (R 맵·faith/yield 추출 검증)
     b. 기존 bootstrap 재표집 단위 확인 — analysis/13_bootstrap_ci.py bootstrap()
     c. 패턴 간 hop0 동일성 (a vs c, b vs d) + hop1 bridge 문자열 동일성
  1. hop0 matched Δf   (양쪽 R=1 · 비기권 triple, KO쪽 − EN쪽)
  2. 같은 R=1 짝 집합의 yield 버전 Δy (기권 = 0)
  3. hop1 matched Δf / Δy (bridge 문자열 고정: a–c, b–d 짝)

R=1 정의·기권 처리는 12/13 과 같다:
  R=1        = retrieval_v3 파일의 answer_bearing_hit == True (해당 qid·hop)
  Δf 분모    = utils.metrics.hop_faith (기권·ambiguous·무효 → None → 제외)
  Δy 분모    = utils.metrics.hop_yield (기권·무claim → 0, ambiguous → None)

사용:
  python3 analysis/matched_subset.py \
      > data/correctness/matched_subset_run.log 2>&1
"""
import argparse
import csv
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.metrics import hop_faith, hop_yield          # noqa: E402

QA = ROOT / "data/dataset/qa_pairs.jsonl"
OUT = ROOT / "reports/rebuttal"

MODELS = ["aya", "exaone", "llama", "qwen", "sonnet", "gemini"]
NAMES = {"aya": "Aya Expanse 8B", "exaone": "EXAONE 3.5 8B", "llama": "Llama 3.1 8B",
         "qwen": "Qwen3 8B", "sonnet": "Claude Sonnet 4.6", "gemini": "Gemini 3.5 Flash"}

# 원고 Table 3 (tab:retrieval-conditioned) hop0 Δf 와 §5.3 hop0 R=1 yield 잔차
TAB3_HOP0 = {"aya": "+0.025 (448/372)", "exaone": "+0.028 (424/354)", "llama": "+0.026 (454/372)",
             "qwen": "+0.004 (452/372)", "sonnet": "-0.003 (446/355)", "gemini": "-0.003 (446/360)"}
TAB3_HOP1 = {"aya": "+0.002 (351/336)", "exaone": "-0.001 (308/294)", "llama": "+0.025 (338/313)",
             "qwen": "+0.012 (329/291)", "sonnet": "-0.007 (312/298)", "gemini": "+0.005 (336/292)"}
YIELD_53 = {"llama": "+0.050 (유의)", "sonnet": "+0.046 (유의)", "gemini": "+0.036 (유의)"}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


# 입력 경로 표 (tools/verify_baseline.sh 의 ann/qm/nm/traj/hits 정본과 같은 파일).
# 배포 빌더는 문자열로 적힌 경로를 배포 경로로 바꾸므로 f-string 으로 조립하지 않는다.
# 8B controlled 판정은 정본 판정 파일에서 QA 1,000 행만 읽는다 — 예전에 읽던 1,000 행 사전 추출 스냅샷과
# 전부 같다 (11/16 확인: "스냅샷 == 정본 제한본: 1000/1000 행 일치", 6모델 모두).
# template_matched·surface_normalized 의 R=1 은 기존 파일이 sonnet 만 있어 10_retrieval_proxy.py 로 재생성했다 (§7 참조).
ANN_PATHS = {
    ("aya", "controlled"):            "data/annotated/aya_ecological.jsonl",
    ("aya", "template_matched"):      "data/annotated/aya_query_matched.jsonl",
    ("aya", "surface_normalized"):    "data/annotated/aya_normalized.jsonl",
    ("exaone", "controlled"):         "data/annotated/exaone_ecological.jsonl",
    ("exaone", "template_matched"):   "data/annotated/exaone_query_matched.jsonl",
    ("exaone", "surface_normalized"): "data/annotated/exaone_normalized.jsonl",
    ("llama", "controlled"):          "data/annotated/llama_ecological.jsonl",
    ("llama", "template_matched"):    "data/annotated/llama_query_matched.jsonl",
    ("llama", "surface_normalized"):  "data/annotated/llama_normalized.jsonl",
    ("qwen", "controlled"):           "data/annotated/qwen_ecological.jsonl",
    ("qwen", "template_matched"):     "data/annotated/qwen_query_matched.jsonl",
    ("qwen", "surface_normalized"):   "data/annotated/qwen_normalized.jsonl",
    ("sonnet", "controlled"):         "data/annotated/sonnet_ecological.jsonl",
    ("sonnet", "template_matched"):   "data/annotated/sonnet_query_matched.jsonl",
    ("sonnet", "surface_normalized"): "data/annotated/sonnet_normalized.jsonl",
    ("gemini", "controlled"):         "data/annotated/gemini_ecological.jsonl",
    ("gemini", "template_matched"):   "data/annotated/gemini_query_matched.jsonl",
    ("gemini", "surface_normalized"): "data/annotated/gemini_normalized.jsonl",
}
HITS_PATHS = {
    ("aya", "controlled"):            "data/retrieval/aya_ecological_hits.jsonl",
    ("aya", "template_matched"):      "data/retrieval/aya_query_matched_hits.jsonl",
    ("aya", "surface_normalized"):    "data/retrieval/aya_normalized_hits.jsonl",
    ("exaone", "controlled"):         "data/retrieval/exaone_ecological_hits.jsonl",
    ("exaone", "template_matched"):   "data/retrieval/exaone_query_matched_hits.jsonl",
    ("exaone", "surface_normalized"): "data/retrieval/exaone_normalized_hits.jsonl",
    ("llama", "controlled"):          "data/retrieval/llama_ecological_hits.jsonl",
    ("llama", "template_matched"):    "data/retrieval/llama_query_matched_hits.jsonl",
    ("llama", "surface_normalized"):  "data/retrieval/llama_normalized_hits.jsonl",
    ("qwen", "controlled"):           "data/retrieval/qwen_ecological_hits.jsonl",
    ("qwen", "template_matched"):     "data/retrieval/qwen_query_matched_hits.jsonl",
    ("qwen", "surface_normalized"):   "data/retrieval/qwen_normalized_hits.jsonl",
    ("sonnet", "controlled"):         "data/retrieval/sonnet_ecological_hits.jsonl",
    ("sonnet", "template_matched"):   "data/retrieval/sonnet_query_matched_hits.jsonl",
    ("sonnet", "surface_normalized"): "data/retrieval/sonnet_normalized_hits.jsonl",
    ("gemini", "controlled"):         "data/retrieval/gemini_ecological_hits.jsonl",
    ("gemini", "template_matched"):   "data/retrieval/gemini_query_matched_hits.jsonl",
    ("gemini", "surface_normalized"): "data/retrieval/gemini_normalized_hits.jsonl",
}
TRAJ_PATHS = {
    ("aya", "controlled"):            "data/trajectories/aya_ecological.jsonl",
    ("aya", "template_matched"):      "data/trajectories/aya_query_matched.jsonl",
    ("aya", "surface_normalized"):    "data/trajectories/aya_normalized.jsonl",
    ("exaone", "controlled"):         "data/trajectories/exaone_ecological.jsonl",
    ("exaone", "template_matched"):   "data/trajectories/exaone_query_matched.jsonl",
    ("exaone", "surface_normalized"): "data/trajectories/exaone_normalized.jsonl",
    ("llama", "controlled"):          "data/trajectories/llama_ecological.jsonl",
    ("llama", "template_matched"):    "data/trajectories/llama_query_matched.jsonl",
    ("llama", "surface_normalized"):  "data/trajectories/llama_normalized.jsonl",
    ("qwen", "controlled"):           "data/trajectories/qwen_ecological.jsonl",
    ("qwen", "template_matched"):     "data/trajectories/qwen_query_matched.jsonl",
    ("qwen", "surface_normalized"):   "data/trajectories/qwen_normalized.jsonl",
    ("sonnet", "controlled"):         "data/trajectories/sonnet_ecological.jsonl",
    ("sonnet", "template_matched"):   "data/trajectories/sonnet_query_matched.jsonl",
    ("sonnet", "surface_normalized"): "data/trajectories/sonnet_normalized.jsonl",
    ("gemini", "controlled"):         "data/trajectories/gemini_ecological.jsonl",
    ("gemini", "template_matched"):   "data/trajectories/gemini_query_matched.jsonl",
    ("gemini", "surface_normalized"): "data/trajectories/gemini_normalized.jsonl",
}


def ann_path(model: str, cond: str = "controlled") -> Path:
    return ROOT / ANN_PATHS[(model, cond)]


def hits_path(model: str, cond: str = "controlled") -> Path:
    return ROOT / HITS_PATHS[(model, cond)]


def traj_path(model: str, cond: str = "controlled") -> Path:
    return ROOT / TRAJ_PATHS[(model, cond)]


def bridge_of(step: dict) -> str:
    """hop1 질의에 삽입된 bridge 문자열. frontier=bridge_insert_text, 구버전 8B=bridge_entity."""
    for k in ("bridge_insert_text", "prev_answer_raw", "bridge_entity"):
        v = step.get(k)
        if v is not None:
            return str(v).strip()
    return ""


# ── 로딩 ──────────────────────────────────────────────────────────────────────

def load_model(model: str, qa: dict, cond: str = "controlled") -> dict:
    """(model, cond) → {(triple, pattern, hop): {...}}. 기권/R/faith/yield/질의·패시지·생성물."""
    ann = {r["qid"]: r for r in load_jsonl(ann_path(model, cond)) if r["qid"] in qa}
    assert len(ann) == len(qa), (model, cond, len(ann), len(qa))
    trj = {r["qid"]: r for r in load_jsonl(traj_path(model, cond)) if r["qid"] in qa}
    assert len(trj) == len(qa), (model, cond, len(trj), len(qa))
    hits = {}
    for r in load_jsonl(hits_path(model, cond)):
        if r["qid"] in qa:
            hits[(r["qid"], int(r.get("hop", 0)))] = r.get("answer_bearing_hit")

    cells = {}
    for qid, q in qa.items():
        a, t = ann[qid], trj[qid]
        steps = {int(s["hop"]): s for s in t["trajectory"]}
        for qh in q["hops"]:
            k = qh["hop"]
            ah = next(h for h in a["hops"] if h["hop"] == k)
            st = steps[k]
            cells[(q["triple"], q["pattern"], k)] = {
                "qid": qid, "gold_lang": qh["gold_lang"],
                "R": hits.get((qid, k)),
                "abstained": bool(ah.get("abstained")),
                "ambiguous": bool(ah.get("ambiguous_answer_marker")),
                "f": hop_faith(ah), "y": hop_yield(ah),
                "query": st.get("retrieval_query", "") or "",
                "passages": tuple(st.get("retrieved_passages", []) or []),
                "gen": st.get("generated_text", "") or "",
                "bridge": bridge_of(st) if k > 0 else "",
                "index_used": st.get("retrieval_index_used"),
            }
    return cells


# ── 0c. 패턴 간 동일성 ────────────────────────────────────────────────────────

def check_pattern_identity(model: str, cells: dict, triples: list[str]) -> dict:
    c = Counter()
    for t in triples:
        for hop, (p1, p2), tag in [(0, ("a", "c"), "hop0_ac"), (0, ("b", "d"), "hop0_bd"),
                                   (1, ("a", "c"), "hop1_ac"), (1, ("b", "d"), "hop1_bd")]:
            x, y = cells[(t, p1, hop)], cells[(t, p2, hop)]
            c[f"{tag}_n"] += 1
            if x["query"] != y["query"]:
                c[f"{tag}_query_diff"] += 1
            if x["passages"] != y["passages"]:
                c[f"{tag}_passages_diff"] += 1
            if x["gen"] != y["gen"]:
                c[f"{tag}_gen_diff"] += 1
            if hop == 1 and x["bridge"] != y["bridge"]:
                c[f"{tag}_bridge_diff"] += 1
            if hop == 0 and x["R"] != y["R"]:
                c[f"{tag}_R_diff"] += 1
    return c


# ── 1~3. matched 지표 ─────────────────────────────────────────────────────────

HOP0_SIDES = {"ko": ("a", "c"), "en": ("b", "d")}
HOP1_SIDES = {"ko": ("a", "b"), "en": ("c", "d")}
HOP1_PAIRS = {"ac": ("a", "c"), "bd": ("b", "d")}          # (KO hop1, EN hop1)


def hop0_rows(cells: dict, triples: list[str]) -> list[dict]:
    """triple 단위 hop0 행. 양쪽(R=1 패턴) f/y 평균."""
    rows = []
    for t in triples:
        row = {"triple": t}
        for side, pats in HOP0_SIDES.items():
            r1 = [p for p in pats if cells[(t, p, 0)]["R"] is True]
            fs = [cells[(t, p, 0)]["f"] for p in r1 if cells[(t, p, 0)]["f"] is not None]
            ys = [cells[(t, p, 0)]["y"] for p in r1 if cells[(t, p, 0)]["y"] is not None]
            row[f"hop0_{side}_R1_pats"] = "".join(r1)
            row[f"hop0_{side}_abstained_pats"] = "".join(p for p in r1 if cells[(t, p, 0)]["abstained"])
            row[f"hop0_{side}_ambiguous_pats"] = "".join(p for p in r1 if cells[(t, p, 0)]["ambiguous"])
            row[f"hop0_{side}_f"] = statistics.mean(fs) if fs else None
            row[f"hop0_{side}_y"] = statistics.mean(ys) if ys else None
        row["hop0_both_R1"] = bool(row["hop0_ko_R1_pats"]) and bool(row["hop0_en_R1_pats"])
        row["hop0_df"] = (row["hop0_ko_f"] - row["hop0_en_f"]
                          if row["hop0_both_R1"] and row["hop0_ko_f"] is not None
                          and row["hop0_en_f"] is not None else None)
        row["hop0_dy"] = (row["hop0_ko_y"] - row["hop0_en_y"]
                          if row["hop0_both_R1"] and row["hop0_ko_y"] is not None
                          and row["hop0_en_y"] is not None else None)
        rows.append(row)
    return rows


def hop1_rows(cells: dict, triples: list[str], rows: list[dict]) -> Counter:
    """짝(a–c, b–d) 단위 hop1. bridge 문자열이 같고 양쪽 R=1 인 짝만."""
    cnt = Counter()
    for row in rows:
        t = row["triple"]
        dfs, dys = [], []
        for tag, (pk, pe) in HOP1_PAIRS.items():
            k, e = cells[(t, pk, 1)], cells[(t, pe, 1)]
            both_r1 = k["R"] is True and e["R"] is True
            same_bridge = k["bridge"] == e["bridge"]
            row[f"hop1_{tag}_both_R1"] = both_r1
            row[f"hop1_{tag}_bridge_equal"] = same_bridge
            row[f"hop1_{tag}_abstained"] = "".join(
                p for p, c in ((pk, k), (pe, e)) if c["abstained"])
            if both_r1:
                cnt[f"pair_{tag}_both_R1"] += 1
                if not same_bridge:
                    cnt[f"pair_{tag}_bridge_diff"] += 1
            ok = both_r1 and same_bridge
            df = (k["f"] - e["f"]) if ok and k["f"] is not None and e["f"] is not None else None
            dy = (k["y"] - e["y"]) if ok and k["y"] is not None and e["y"] is not None else None
            row[f"hop1_{tag}_df"] = df
            row[f"hop1_{tag}_dy"] = dy
            if ok and df is None:
                cnt[f"pair_{tag}_dropped_abstain"] += 1
            if df is not None:
                cnt[f"pair_{tag}_kept_f"] += 1
            if dy is not None:
                cnt[f"pair_{tag}_kept_y"] += 1
        dfs = [row[f"hop1_{g}_df"] for g in HOP1_PAIRS if row[f"hop1_{g}_df"] is not None]
        dys = [row[f"hop1_{g}_dy"] for g in HOP1_PAIRS if row[f"hop1_{g}_dy"] is not None]
        row["hop1_df"] = statistics.mean(dfs) if dfs else None
        row["hop1_dy"] = statistics.mean(dys) if dys else None
        row["hop1_n_pairs_f"] = len(dfs)
        row["hop1_n_pairs_y"] = len(dys)
    return cnt


def hop_level_gap(cells: dict, triples: list[str], hop: int, key: str,
                  only: set[str] | None = None) -> tuple[float, int, int]:
    """12/13 과 같은 hop 단위(비페어링) same − cross. only 로 triple 부분집합 제한."""
    sm, cr = [], []
    for t in (triples if only is None else [t for t in triples if t in only]):
        for side, pats in HOP0_SIDES.items() if hop == 0 else HOP1_SIDES.items():
            for p in pats:
                c = cells[(t, p, hop)]
                if c["R"] is not True or c[key] is None:
                    continue
                (sm if side == "ko" else cr).append(c[key])
    g = (statistics.mean(sm) - statistics.mean(cr)) if sm and cr else float("nan")
    return g, len(sm), len(cr)


def boot_mean(vals: list[float], n_boot: int, seed: int) -> tuple[float, float, float, int]:
    """triple 재표집 paired bootstrap. 통계량별로 독립 RNG(seed 고정)."""
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    point = statistics.mean(vals)
    rng = random.Random(seed)
    b = sorted(statistics.mean(rng.choices(vals, k=len(vals))) for _ in range(n_boot))
    return point, b[int(0.025 * n_boot)], b[int(0.975 * n_boot) - 1], len(vals)


DEC = 3          # 표시 소수점 (--decimals; 경계 CI 확인용으로 4 를 쓸 수 있다)


def fmt(p, lo, hi, n=None) -> str:
    if n == 0:
        return "N/A"
    s = f"{p:+.{DEC}f} [{lo:+.{DEC}f}, {hi:+.{DEC}f}]"
    if n is not None:
        s += f" (n={n})"
    return s + ("*" if (lo > 0 or hi < 0) else "")


def main():
    global DEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--out-dir", default=str(OUT), help="matched_subset_pairs.csv 를 쓸 디렉터리")
    args = ap.parse_args()
    DEC = args.decimals

    qa_raw = load_jsonl(QA)
    qa = {}
    for r in qa_raw:
        r["triple"] = r["source_path_id"].rsplit("_pat_", 1)[0]
        qa[r["qid"]] = r
    by_triple = defaultdict(set)
    for r in qa.values():
        by_triple[r["triple"]].add(r["pattern"])
    triples = sorted(by_triple)
    assert all(v == {"a", "b", "c", "d"} for v in by_triple.values()), "패턴 4종이 아닌 triple 존재"
    print(f"[입력] QA {len(qa)} 궤적 / triple {len(triples)}개 (패턴 a–d 완비)")
    # 패턴 → hop0/hop1 index 언어가 설계대로인지 (gold_lang 기준)
    exp = {"a": ("ko", "ko"), "b": ("en", "ko"), "c": ("ko", "en"), "d": ("en", "en")}
    bad = [r["qid"] for r in qa.values()
           if tuple(h["gold_lang"] for h in r["hops"]) != exp[r["pattern"]]]
    print(f"[입력] 패턴↔gold_lang 불일치: {len(bad)}건")

    all_rows, table = [], []
    for model in args.models:
        cells = load_model(model, qa)
        print(f"\n{'='*78}\n{NAMES[model]}\n{'='*78}")
        print(f"  annotated : {ann_path(model).relative_to(ROOT)}")
        print(f"  hits      : {hits_path(model).relative_to(ROOT)}")
        print(f"  trajectory: {traj_path(model).relative_to(ROOT)}")

        # 0c
        chk = check_pattern_identity(model, cells, triples)
        print("\n[0c] 패턴 간 동일성 (250 triple 당)")
        for tag in ("hop0_ac", "hop0_bd", "hop1_ac", "hop1_bd"):
            extra = (f" bridge≠ {chk[tag+'_bridge_diff']}" if tag.startswith("hop1")
                     else f" R≠ {chk[tag+'_R_diff']}")
            print(f"   {tag}: query≠ {chk[tag+'_query_diff']}  passages≠ {chk[tag+'_passages_diff']}"
                  f"  generated_text≠ {chk[tag+'_gen_diff']}{extra}")

        rows = hop0_rows(cells, triples)
        pair_cnt = hop1_rows(cells, triples, rows)
        for r in rows:
            r["model"] = model
        all_rows += rows

        both = [r for r in rows if r["hop0_both_R1"]]
        ko_r1 = sum(1 for r in rows if r["hop0_ko_R1_pats"])
        en_r1 = sum(1 for r in rows if r["hop0_en_R1_pats"])
        two_pat = Counter(len(r[f"hop0_{s}_R1_pats"]) for r in both for s in HOP0_SIDES)
        ko_out = [r for r in both if r["hop0_ko_f"] is None]
        en_out = [r for r in both if r["hop0_en_f"] is None]
        amb = sum(1 for r in both for s in HOP0_SIDES if r[f"hop0_{s}_ambiguous_pats"])
        dfs = [r["hop0_df"] for r in rows if r["hop0_df"] is not None]
        dys = [r["hop0_dy"] for r in rows if r["hop0_dy"] is not None]
        p_f = boot_mean(dfs, args.n_boot, args.seed)
        p_y = boot_mean(dys, args.n_boot, args.seed)
        # 0a'. 이 스크립트의 R 맵·faith 추출이 12 와 같은 값을 내는지 (hop 단위, 전체 triple)
        for hop in (0, 1):
            g, ns, nc = hop_level_gap(cells, triples, hop, "f")
            print(f"[0a'] hop{hop} hop단위 Δf (R=1, 전체 triple) = {g:+.3f} ({ns}/{nc})"
                  f"   원고 = {(TAB3_HOP0 if hop == 0 else TAB3_HOP1)[model]}")
        g, ns, nc = hop_level_gap(cells, triples, 0, "y")
        print(f"[0a'] hop0 hop단위 Δy (R=1, 전체 triple) = {g:+.3f} ({ns}/{nc})"
              f"   원고 §5.3 = {YIELD_53.get(model, '보고 없음(비유의)')}")

        print(f"\n[1] hop0 R=1 triple: KO쪽 {ko_r1} / EN쪽 {en_r1} / 양쪽 {len(both)}"
              f"  (양쪽 R=1 triple의 side 중 패턴 2개 사용 {two_pat[2]}, 1개 사용 {two_pat[1]})")
        print(f"    기권·ambiguous로 side 전체가 빠진 triple: KO쪽 {len(ko_out)} / EN쪽 {len(en_out)}"
              f" (ambiguous 포함 side {amb}) → n={p_f[3]}")
        print(f"    matched Δf = {fmt(*p_f)}      원고 Table 3 hop0 = {TAB3_HOP0[model]}")
        only = {r["triple"] for r in both}
        for key, lab in (("f", "Δf"), ("y", "Δy")):
            g, ns, nc = hop_level_gap(cells, triples, 0, key, only)
            print(f"    (대조: 같은 양쪽 R=1 triple {len(only)}개의 hop단위 비페어링 {lab}"
                  f" = {g:+.3f}, 홉 {ns}/{nc})")
        print(f"[2] hop0 matched Δy (같은 R=1 짝 집합, 기권=0) = {fmt(*p_y)}"
              f"      원고 §5.3 hop0 R=1 yield 잔차 = {YIELD_53.get(model, 'n.s.')}")

        d1f = [r["hop1_df"] for r in rows if r["hop1_df"] is not None]
        d1y = [r["hop1_dy"] for r in rows if r["hop1_dy"] is not None]
        q_f = boot_mean(d1f, args.n_boot, args.seed)
        q_y = boot_mean(d1y, args.n_boot, args.seed)
        print(f"\n[3] hop1 matched (bridge 고정)")
        for tag in HOP1_PAIRS:
            print(f"    짝 {tag}: 양쪽 R=1 {pair_cnt[f'pair_{tag}_both_R1']}"
                  f" / bridge 불일치 제외 {pair_cnt[f'pair_{tag}_bridge_diff']}"
                  f" / 기권 제외 {pair_cnt[f'pair_{tag}_dropped_abstain']}"
                  f" → Δf 짝 {pair_cnt[f'pair_{tag}_kept_f']}, Δy 짝 {pair_cnt[f'pair_{tag}_kept_y']}")
        print(f"    matched Δf hop1 = {fmt(*q_f)}      원고 Table 3 hop1 = {TAB3_HOP1[model]}")
        print(f"    matched Δy hop1 = {fmt(*q_y)}")

        table.append({
            "model": NAMES[model], "hop0_both_R1": len(both),
            "hop0_abstain_drop": f"{len(ko_out)}/{len(en_out)}", "hop0_n": p_f[3],
            "hop0_df": fmt(*p_f[:3]), "tab3_hop0": TAB3_HOP0[model], "hop0_dy": fmt(*p_y[:3]),
            "hop1_pairs_f": pair_cnt["pair_ac_kept_f"] + pair_cnt["pair_bd_kept_f"],
            "hop1_bridge_diff": pair_cnt["pair_ac_bridge_diff"] + pair_cnt["pair_bd_bridge_diff"],
            "hop1_n": q_f[3], "hop1_df": fmt(*q_f[:3]), "hop1_dy": fmt(*q_y[:3]),
        })

    print(f"\n{'='*78}\n[4] 요약 표\n{'='*78}")
    hdr = ["model", "hop0_both_R1", "hop0_abstain_drop", "hop0_n", "hop0_df", "tab3_hop0",
           "hop0_dy", "hop1_bridge_diff", "hop1_pairs_f", "hop1_n", "hop1_df", "hop1_dy"]
    print(" | ".join(hdr))
    for r in table:
        print(" | ".join(str(r[k]) for k in hdr))

    keys = list(dict.fromkeys(k for r in all_rows for k in r))
    keys = ["model", "triple"] + [k for k in keys if k not in ("model", "triple")]
    out = Path(args.out_dir) / "matched_subset_pairs.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n저장: {out.name} ({len(all_rows)}행)")


if __name__ == "__main__":
    main()
