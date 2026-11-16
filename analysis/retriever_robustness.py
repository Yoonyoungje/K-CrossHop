#!/usr/bin/env python3
"""
analysis/retriever_robustness.py — Retriever Robustness (부록 tab:retriever-robustness, retrieval-only)

camera-ready: work/e5_stage2/stage2_retrieval.py 를 옮겼다. 바뀐 점
  - --matcher: answer_bearing(기본, §4.1 utils.retrieval_success.check_answer_bearing_hit) / diag(이전 판정,
    gold_entity_retrieval_diag.passage_has_entity — 제출본 부록 값 재현용).
  - E5 인덱스 = data/camera_ready/index_e5 (pipeline/04 재생성, 부록 인덱스와 벡터까지 비트 동일).
  - hop_level_hits.jsonl 에 조건별 top-5 passage_id 를 남긴다(검색 없이 다시 판정할 수 있게).

BGE-M3에서 관찰된 query-side 효과가 multilingual-e5-large에서도 방향과 구성비를 보존하는지
확인한다. 지표는 answer-bearing hit@5 하나뿐이며, 생성/judge/parser는 일절 실행하지 않는다.

[A] hop0 — template-language 성분 (model-agnostic)
    A1 controlled       : KO sub_q            → gold-language index
    A2 template-matched : EN-evidence 홉은 sub_q_en, KO-evidence 홉은 sub_q 유지
    (hop0에는 삽입된 bridge가 없으므로 surface-normalized ≡ template-matched. 조건은 둘뿐이다.)

[B] hop1 — gold bridge 고정, EN index 대상 (model-free)
    B1 controlled        : KO 템플릿 + KO 표면형 bridge  (= sub_q)
    B2 template-matched  : EN 템플릿 + KO 표면형 bridge  (sub_q_en의 EN label을 KO label로 치환)
    B3 surface-normalized: EN 템플릿 + EN 표면형 bridge  (= sub_q_en)
    ⚠️ oracle bridge를 쓰므로 논문 §5.2 수치의 재현이 아니라 별개의 추정량이다.

설계상 준수 사항
  - 인덱스 라우팅은 pipeline/05_pilot_runner.resolve_hop_query 를 재사용한다(gold_lang 직접 참조 금지).
  - hop0 sub_q는 패턴 간 공유되므로 고유 (query, index) 쌍으로 dedupe해 검색하고,
    집계는 홉 단위로 되펼친다. bootstrap은 triple(QID) 클러스터 단위.
  - hit 판정은 두 팔 모두 같은 matcher (--matcher).
"""

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))

import gold_entity_retrieval_diag as diag  # noqa: E402  (--matcher diag)
from utils.retrieval_success import check_answer_bearing_hit  # noqa: E402  (--matcher answer_bearing, §4.1)


def _load_pilot_runner():
    """pipeline/05_pilot_runner.py 는 모듈명이 숫자로 시작해 일반 import가 불가능하다."""
    spec = importlib.util.spec_from_file_location(
        "pilot_runner", ROOT / "pipeline" / "05_pilot_runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOP_K = 5
B_BOOT = 10_000
SEED = 42
ARMS = {
    "bge-m3": {"model": "BAAI/bge-m3", "index_dir": "data/faiss_index", "qprefix": ""},
    "e5":     {"model": "intfloat/multilingual-e5-large",
               "index_dir": "data/camera_ready/index_e5", "qprefix": "query: "},
}
EXPECTED_ROUTING = {  # 패턴 정의 (a: KO/KO · b: EN/KO · c: KO/EN · d: EN/EN)
    "a": ("ko", "ko"), "b": ("en", "ko"), "c": ("ko", "en"), "d": ("en", "en"),
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def triple_cluster(path_id: str) -> str:
    return re.sub(r"_pat_[abcd]$", "", path_id)


# ── 홉 레코드 구성 ───────────────────────────────────────────────────────────
def build_hops(qas: list[dict], paths: dict, pr) -> tuple[list[dict], dict]:
    recs, routing_seen = [], defaultdict(set)
    b2_fail = []
    for qa in qas:
        p = paths[qa["source_path_id"]]
        cluster = triple_cluster(qa["source_path_id"])
        for h in (0, 1):
            hop = qa["hops"][h]
            # ── 라우팅: 기존 파이프라인 함수 재사용 (gold_lang 직접 참조 안 함) ──
            a1 = pr.resolve_hop_query("ko_subq", qa, h, hop, "")
            a2 = pr.resolve_hop_query("sub_q",   qa, h, hop, "")
            assert a1["lang"] == a2["lang"], "두 모드의 index 라우팅이 갈라짐"
            lang = a1["lang"]
            routing_seen[(qa["pattern"], h)].add(lang)

            rec = {
                "qid": qa["qid"], "pattern": qa["pattern"], "hop": h,
                "cluster": cluster, "lang": lang,
                "target_qid": hop["gold_entity_qid"], "target_label": hop["gold_entity"],
                "q_A1": a1["query"], "q_A2": a2["query"],
            }

            # ── B 조건: hop1 & EN index 만 ──
            if h == 1 and lang == "en":
                bridge = p["hops"][1]
                ko_surface = hop.get("sub_q_bridge_label") or bridge["label_ko"]
                en_surface = bridge["label_en"]
                sub_q_en = hop.get("sub_q_en")
                rec["q_B1"] = hop["sub_q"]
                rec["q_B3"] = sub_q_en
                if sub_q_en and en_surface and en_surface in sub_q_en:
                    rec["q_B2"] = sub_q_en.replace(en_surface, ko_surface)
                else:
                    rec["q_B2"] = None
                    b2_fail.append(qa["qid"])
                rec["bridge_ko"] = ko_surface
                rec["bridge_en"] = en_surface
            recs.append(rec)

    routing = {f"{pat}-hop{h}": sorted(v) for (pat, h), v in sorted(routing_seen.items())}
    ok = all(routing[f"{pat}-hop{h}"] == [EXPECTED_ROUTING[pat][h]]
             for pat in "abcd" for h in (0, 1))
    return recs, {"routing_observed": routing, "routing_matches_pattern_def": ok,
                  "b2_construction_failures": b2_fail}


# ── 검색 ─────────────────────────────────────────────────────────────────────
def run_arm(name: str, cfg: dict, recs: list[dict]) -> dict:
    """고유 (query, lang) 쌍만 인코딩·검색하고 {(query, lang): [meta rec, ...]} 반환."""
    pairs = set()
    for r in recs:
        for k in ("q_A1", "q_A2", "q_B1", "q_B2", "q_B3"):
            if r.get(k):
                pairs.add((r[k], r["lang"]))
    pairs = sorted(pairs)
    log(f"[{name}] 고유 (query,index) 쌍 {len(pairs)}개 (홉 {len(recs)}건에서 dedupe)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(cfg["model"], device=device)
    log(f"[{name}] {cfg['model']} max_seq_length={model.max_seq_length} "
        f"dtype={next(model.parameters()).dtype} prefix={cfg['qprefix']!r}")

    idx_dir = Path(cfg["index_dir"])
    indexes, metas = {}, {}
    for lang in ("ko", "en"):
        indexes[lang] = faiss.read_index(str(idx_dir / f"{lang}_index.faiss"))
        metas[lang] = json.loads((idx_dir / f"{lang}_meta.json").read_text("utf-8"))
        log(f"[{name}]   [{lang}] {indexes[lang].ntotal} 벡터")

    t0 = time.time()
    vecs = model.encode([cfg["qprefix"] + q for q, _ in pairs], batch_size=64,
                        normalize_embeddings=True, show_progress_bar=False,
                        convert_to_numpy=True).astype("float32")
    log(f"[{name}] 쿼리 인코딩 {time.time()-t0:.1f}s")

    out = {}
    for (q, lang), vec in zip(pairs, vecs):
        _, ids = indexes[lang].search(vec.reshape(1, -1), TOP_K)
        out[(q, lang)] = [metas[lang][i] for i in ids[0] if 0 <= i < len(metas[lang])]

    del model, indexes, metas
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return out


def judge(hits: dict, recs: list[dict], alias_cache: dict, matcher: str) -> None:
    """조건별 hit 판정을 recs에 기록. matcher: answer_bearing(§4.1) 또는 diag(이전 판정)."""
    alias_memo: dict[str, set] = {}

    def aliases(qid: str, label: str) -> set:
        if qid not in alias_memo:
            e = alias_cache.get(qid, {})
            alias_memo[qid] = diag.load_alias_set(e) if e else {diag.normalize(label)}
        return alias_memo[qid]

    for r in recs:
        for cond in ("A1", "A2", "B1", "B2", "B3"):
            q = r.get(f"q_{cond}")
            if not q:
                continue
            recs_top = hits[(q, r["lang"])]
            texts = [x["text"] for x in recs_top]
            if matcher == "diag":
                r[f"hit_{cond}"] = diag.passage_has_entity(texts, aliases(r["target_qid"], r["target_label"]))
            else:
                r[f"hit_{cond}"] = check_answer_bearing_hit(
                    texts, r["target_qid"], alias_cache, r["lang"])["answer_bearing_hit"]
            r[f"pids_{cond}"] = [x.get("passage_id") for x in recs_top]


# ── 집계 + bootstrap ─────────────────────────────────────────────────────────
def cluster_table(recs: list[dict], clusters: list[str]) -> dict[str, np.ndarray]:
    """클러스터별 (분모, 분자) 카운트 → bootstrap용."""
    ci = {c: i for i, c in enumerate(clusters)}
    n = len(clusters)
    z = lambda: np.zeros(n, dtype=np.float64)
    t = {k: z() for k in ("n_same", "k_same", "n_cross", "k_cross", "k_cross_A2",
                          "n_b", "k_B1", "k_B2", "k_B3")}
    for r in recs:
        i = ci[r["cluster"]]
        if r["hop"] == 0:
            if r["lang"] == "ko":
                t["n_same"][i] += 1
                t["k_same"][i] += r["hit_A1"]
            else:
                t["n_cross"][i] += 1
                t["k_cross"][i] += r["hit_A1"]
                t["k_cross_A2"][i] += r["hit_A2"]
        elif r["hop"] == 1 and r["lang"] == "en" and r.get("q_B2"):
            t["n_b"][i] += 1
            t["k_B1"][i] += r["hit_B1"]
            t["k_B2"][i] += r["hit_B2"]
            t["k_B3"][i] += r["hit_B3"]
    return t


def stats_from(t: dict, sel: np.ndarray) -> dict:
    s = lambda k: t[k][sel].sum()
    d = lambda a, b: a / b if b else float("nan")
    same, cross = d(s("k_same"), s("n_same")), d(s("k_cross"), s("n_cross"))
    cross_a2 = d(s("k_cross_A2"), s("n_cross"))
    gap, rec = same - cross, cross_a2 - cross
    b1, b2, b3 = d(s("k_B1"), s("n_b")), d(s("k_B2"), s("n_b")), d(s("k_B3"), s("n_b"))
    return {"same": same, "cross": cross, "gap": gap,
            "cross_A2": cross_a2, "recovery": rec,
            "recovery_ratio": rec / gap if gap else float("nan"),
            "B1": b1, "B2": b2, "B3": b3,
            "template_language": b2 - b1, "bridge_surface": b3 - b2}


def bootstrap(t: dict, clusters: list[str]) -> tuple[dict, dict]:
    n = len(clusters)
    point = stats_from(t, np.arange(n))
    rng = np.random.default_rng(SEED)
    draws = defaultdict(list)
    for _ in range(B_BOOT):
        sel = rng.integers(0, n, n)
        for k, v in stats_from(t, sel).items():
            draws[k].append(v)
    ci = {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
          for k, v in draws.items()}
    return point, ci


def fmt(p: float, ci: tuple) -> str:
    star = "*" if (ci[0] > 0 or ci[1] < 0) else " "
    return f"{p:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]{star}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-pairs",    default="data/dataset/qa_pairs.jsonl")
    ap.add_argument("--paths",       default="data/dataset/paths.jsonl")
    ap.add_argument("--alias-cache", default="data/retrieval/alias_cache.json")
    ap.add_argument("--matcher", default="answer_bearing", choices=["answer_bearing", "diag"],
                    help="answer_bearing = §4.1 check_answer_bearing_hit (기본) / diag = 제출본 부록 판정 재현")
    ap.add_argument("--out-dir",     default="results/camera_ready/retriever_robustness")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pr = _load_pilot_runner()
    qas = [json.loads(l) for l in Path(args.qa_pairs).read_text("utf-8").splitlines() if l.strip()]
    paths = {json.loads(l)["path_id"]: json.loads(l)
             for l in Path(args.paths).read_text("utf-8").splitlines() if l.strip()}
    alias_cache = json.loads(Path(args.alias_cache).read_text("utf-8"))

    recs, meta = build_hops(qas, paths, pr)
    clusters = sorted({r["cluster"] for r in recs})
    log(f"홉 {len(recs)}건 / triple 클러스터 {len(clusters)}개")
    log(f"라우팅 검증: {meta['routing_observed']}  일치={meta['routing_matches_pattern_def']}")
    if not meta["routing_matches_pattern_def"]:
        log("STOP: 라우팅이 패턴 정의와 불일치")
        sys.exit(3)

    arm_hits, results = {}, {}
    for name, cfg in ARMS.items():
        arm_hits[name] = run_arm(name, cfg, recs)

    # 팔별 판정 → 별도 사본에 기록
    per_arm_recs = {}
    for name in ARMS:
        copy = [dict(r) for r in recs]
        judge(arm_hits[name], copy, alias_cache, args.matcher)
        per_arm_recs[name] = copy
        t = cluster_table(copy, clusters)
        point, ci = bootstrap(t, clusters)
        results[name] = {
            "point": point, "ci": ci,
            "n": {"same": int(t["n_same"].sum()), "cross": int(t["n_cross"].sum()),
                  "b": int(t["n_b"].sum())},
        }

    # ── §2-2 두 팔의 hit 판정 동일성 검증 ────────────────────────────────────
    a, b = per_arm_recs["bge-m3"], per_arm_recs["e5"]
    same_top5, verdict_match, mismatches = 0, 0, []
    for ra, rb in zip(a, b):
        for cond in ("A1", "A2", "B1", "B2", "B3"):
            if f"hit_{cond}" not in ra or f"hit_{cond}" not in rb:
                continue
            if ra[f"pids_{cond}"] == rb[f"pids_{cond}"]:
                same_top5 += 1
                if ra[f"hit_{cond}"] == rb[f"hit_{cond}"]:
                    verdict_match += 1
                else:
                    mismatches.append({"qid": ra["qid"], "hop": ra["hop"], "cond": cond})
    meta["identical_top5_cells"] = same_top5
    meta["identical_top5_verdict_match"] = verdict_match
    meta["identical_top5_mismatches"] = mismatches

    meta.update({
        "n_hops": len(recs), "n_clusters": len(clusters),
        "unique_query_index_pairs": {n: len(h) for n, h in arm_hits.items()},
        "bootstrap": {"B": B_BOOT, "seed": SEED, "cluster_unit": "triple (path_id minus _pat_X)"},
        "matcher": ("primary local matcher (analysis/gold_entity_retrieval_diag.py load_alias_set/passage_has_entity)"
                    if args.matcher == "diag" else
                    "§4.1 answer-bearing (utils/retrieval_success.py check_answer_bearing_hit)"),
        "alias_cache": args.alias_cache,
        "routing_source": "pipeline/05_pilot_runner.py resolve_hop_query",
        "top_k": TOP_K,
    })

    (out_dir / "stage2_results.json").write_text(
        json.dumps({"meta": meta, "arms": results}, indent=2, ensure_ascii=False,
                   default=float), "utf-8")
    with open(out_dir / "hop_level_hits.jsonl", "w", encoding="utf-8") as f:
        for ra, rb in zip(a, b):
            row = {k: ra[k] for k in ("qid", "pattern", "hop", "cluster", "lang",
                                      "target_qid", "target_label")}
            for cond in ("A1", "A2", "B1", "B2", "B3"):
                if f"hit_{cond}" in ra:
                    row[f"query_{cond}"] = ra[f"q_{cond}"]
                    row[f"hit_{cond}_bge"] = ra[f"hit_{cond}"]
                    row[f"hit_{cond}_e5"] = rb[f"hit_{cond}"]
                    row[f"same_top5_{cond}"] = ra[f"pids_{cond}"] == rb[f"pids_{cond}"]
                    row[f"pids_{cond}_bge"] = ra[f"pids_{cond}"]
                    row[f"pids_{cond}_e5"] = rb[f"pids_{cond}"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ── 출력 ────────────────────────────────────────────────────────────────
    print("\n[A] hop0 — same/cross gap  (A1 controlled)")
    print(f"{'arm':8} {'same hit@5':>11} {'cross hit@5':>12} {'gap [95% CI]':>30} {'n_same/n_cross':>16}")
    for n in ARMS:
        r = results[n]
        print(f"{n:8} {r['point']['same']:>11.3f} {r['point']['cross']:>12.3f} "
              f"{fmt(r['point']['gap'], r['ci']['gap']):>30} "
              f"{str(r['n']['same'])+'/'+str(r['n']['cross']):>16}")

    print("\n[A] hop0 — template-matched recovery (cross-index 홉만)")
    print(f"{'arm':8} {'cross A1':>9} {'cross A2':>9} {'recovery [95% CI]':>30} {'회수비율':>10}")
    for n in ARMS:
        r = results[n]
        print(f"{n:8} {r['point']['cross']:>9.3f} {r['point']['cross_A2']:>9.3f} "
              f"{fmt(r['point']['recovery'], r['ci']['recovery']):>30} "
              f"{r['point']['recovery_ratio']:>10.3f}")

    print("\n[B] hop1 (gold bridge, EN index)")
    print(f"{'arm':8} {'B1':>7} {'B2':>7} {'B3':>7} {'template-language [CI]':>30} "
          f"{'bridge-surface [CI]':>30} {'n':>5}")
    for n in ARMS:
        r = results[n]
        print(f"{n:8} {r['point']['B1']:>7.3f} {r['point']['B2']:>7.3f} {r['point']['B3']:>7.3f} "
              f"{fmt(r['point']['template_language'], r['ci']['template_language']):>30} "
              f"{fmt(r['point']['bridge_surface'], r['ci']['bridge_surface']):>30} "
              f"{r['n']['b']:>5}")

    # ── 사전등록 기준 ───────────────────────────────────────────────────────
    e5, bg = results["e5"], results["bge-m3"]
    c1 = e5["point"]["gap"] > 0 and e5["ci"]["gap"][0] > 0
    c2 = e5["point"]["recovery_ratio"] >= 0.5 * bg["point"]["recovery_ratio"]
    c3 = e5["point"]["bridge_surface"] > 0
    print("\n사전등록 기준")
    print(f"  ① same/cross gap 양수 + CI가 0 제외      : {'PASS' if c1 else 'FAIL'}")
    print(f"  ② template recovery 비율 ≥ BGE-M3의 1/2  : {'PASS' if c2 else 'FAIL'}"
          f"  (e5 {e5['point']['recovery_ratio']:.3f} vs 기준 {0.5*bg['point']['recovery_ratio']:.3f})")
    print(f"  ③ bridge-surface recovery 부호 양수      : {'PASS' if c3 else 'FAIL'}")
    print(f"\n두 팔 top-5 동일 셀 {same_top5}건 중 hit 판정 일치 {verdict_match}건 "
          f"(불일치 {len(mismatches)}건)")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
