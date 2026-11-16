#!/usr/bin/env python3
"""
analysis/gold_entity_ub_encoder.py — gold-entity retrieval UB, 인코더 교체판 (E5 부록 Table 15 의 UB 행)

camera-ready 때 work/e5_gate/e5_ub.py 에서 옮겼다 (경로 기본값만 바꿈: 정본 core QA, camera-ready E5 인덱스,
results/camera_ready/ 출력). GPU 가 필요하다. 두 인코더 결과의 홉별 일치는 analysis/gold_entity_ub_agreement.py.

  python3 analysis/gold_entity_ub_encoder.py --model BAAI/bge-m3 --query-prefix "" --index-dir data/faiss_index \\
      --label bge --output $D/bge_per_entity.jsonl --summary $D/bge_summary.json
  python3 analysis/gold_entity_ub_encoder.py                       # E5 (기본값)
  (D=results/camera_ready/retriever_robustness/gold_entity_ub)

analysis/gold_entity_retrieval_diag.py의 복사·수정본. 원본은 수정하지 않는다.
원본과의 차이:
  1) --model / --query-prefix 인자화 (e5는 "query: " prefix 필수)
  2) hit 판정을 두 matcher로 병기
       primary   : 원본 로컬 matcher (정본 UB 0.991과 동일 판정) — import 재사용
       secondary : utils.retrieval_success.check_answer_bearing_hit (v3)
  3) per-entity JSONL에 top-5 passage_id를 남겨 사후 조사 가능

BGE-M3 모드(--model BAAI/bge-m3 --query-prefix "")로 실행하면 원본과 동일한
primary UB가 나와야 한다 — 이 스크립트 자체의 sanity check.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = "results/camera_ready/retriever_robustness/gold_entity_ub"
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT))

import gold_entity_retrieval_diag as diag                      # noqa: E402
from utils.retrieval_success import check_answer_bearing_hit   # noqa: E402

TOP_K = diag.TOP_K  # 5


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",        default="intfloat/multilingual-e5-large")
    ap.add_argument("--query-prefix", default="query: ",
                    help='e5는 "query: ", BGE-M3는 빈 문자열')
    ap.add_argument("--qa-pairs",     default="data/dataset/qa_pairs.jsonl")
    ap.add_argument("--alias-cache",  default="data/retrieval/alias_cache.json")
    ap.add_argument("--index-dir",    default="data/camera_ready/index_e5")
    ap.add_argument("--batch-size",   type=int, default=64)
    ap.add_argument("--output",       default=f"{OUT_DIR}/e5_per_entity.jsonl")
    ap.add_argument("--summary",      default=f"{OUT_DIR}/e5_summary.json")
    ap.add_argument("--label",        default="e5")
    args = ap.parse_args()

    qa_items    = [json.loads(l) for l in Path(args.qa_pairs).read_text("utf-8").splitlines() if l.strip()]
    alias_cache = json.loads(Path(args.alias_cache).read_text("utf-8"))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    log(f"{args.model} 로딩 (device={device}) ...")
    model = SentenceTransformer(args.model, device=device)
    dtype = str(next(model.parameters()).dtype)
    log(f"  max_seq_length={model.max_seq_length}  dtype={dtype}  prefix={args.query_prefix!r}")

    index_dir = Path(args.index_dir)
    indexes, metas = {}, {}
    for lang in ("ko", "en"):
        indexes[lang] = faiss.read_index(str(index_dir / f"{lang}_index.faiss"))
        metas[lang]   = json.loads((index_dir / f"{lang}_meta.json").read_text("utf-8"))
        log(f"  [{lang}] {indexes[lang].ntotal}개 벡터")

    # ── 쿼리 일괄 인코딩 (홉 순서 보존) ──────────────────────────────────────
    hops = []
    for it in qa_items:
        for hop in it["hops"]:
            hops.append((it, hop))
    log(f"gold_entity 쿼리 {len(hops)}건 인코딩 ...")
    for s in [args.query_prefix + h["gold_entity"] for _, h in hops][:3]:
        log(f"    QUERY: {s!r}")
    t0 = time.time()
    qvecs = model.encode(
        [args.query_prefix + h["gold_entity"] for _, h in hops],
        batch_size=args.batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    ).astype("float32")
    query_sec = round(time.time() - t0, 1)
    log(f"  {query_sec}s")

    # ── 검색 + 판정 ──────────────────────────────────────────────────────────
    results = []
    for (it, hop), qvec in zip(hops, qvecs):
        lang = hop["gold_lang"]
        _, ids = indexes[lang].search(qvec.reshape(1, -1), TOP_K)
        recs = [metas[lang][i] for i in ids[0] if 0 <= i < len(metas[lang])]
        passages = [r["text"] for r in recs]

        gold_qid = hop["gold_entity_qid"]
        entry = alias_cache.get(gold_qid, {})
        alias_set = diag.load_alias_set(entry) if entry else {diag.normalize(hop["gold_entity"])}
        primary = diag.passage_has_entity(passages, alias_set)
        secondary = check_answer_bearing_hit(passages, gold_qid, alias_cache, lang)

        results.append({
            "qid":             it["qid"],
            "pattern":         it["pattern"],
            "hop":             hop["hop"],
            "gold_lang":       lang,
            "gold_entity":     hop["gold_entity"],
            "gold_entity_qid": gold_qid,
            "retrieval_success_gold_entity_ub": primary,
            "hit_v3":          secondary["answer_bearing_hit"],
            "hit_v3_method":   secondary["match_method"],
            "top5_passage_ids":  [r.get("passage_id") for r in recs],
            "top5_entity_qids":  [r.get("entity_qid") for r in recs],
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"저장: {out} ({len(results)} 레코드)")

    # ── 요약 ─────────────────────────────────────────────────────────────────
    by = defaultdict(lambda: {"primary": 0, "v3": 0, "total": 0})
    for r in results:
        key = f"{r['pattern']}-hop{r['hop']}-{r['gold_lang']}"
        by[key]["total"] += 1
        by[key]["primary"] += int(r["retrieval_success_gold_entity_ub"])
        by[key]["v3"]      += int(r["hit_v3"])

    n = len(results)
    overall_primary = sum(r["retrieval_success_gold_entity_ub"] for r in results) / n
    overall_v3      = sum(r["hit_v3"] for r in results) / n

    print()
    print(f"{'key':14s}  {'UB(primary)':>12}  {'UB(v3)':>8}  {'n':>5}")
    print("-" * 46)
    for key in sorted(by):
        v = by[key]
        print(f"{key:14s}  {v['primary']/v['total']:>12.3f}  {v['v3']/v['total']:>8.3f}  {v['total']:>5}")
    print("-" * 46)
    print(f"{'Overall':14s}  {overall_primary:>12.3f}  {overall_v3:>8.3f}  {n:>5}")

    if overall_primary >= 0.95:
        verdict = "PASS"
    elif overall_primary >= 0.85:
        verdict = "AMBER"
    else:
        verdict = "FAIL"
    print(f"\n게이트 판정 ({args.label}): {verdict}  (Overall UB={overall_primary:.3f})")

    summary = {
        "label": args.label,
        "model": args.model,
        "query_prefix": args.query_prefix,
        "index_dir": str(index_dir),
        "device": device,
        "dtype": dtype,
        "max_seq_length": model.max_seq_length,
        "query_encode_sec": query_sec,
        "gold_entity_ub_overall": round(overall_primary, 4),
        "hit_v3_overall": round(overall_v3, 4),
        "verdict": verdict,
        "per_key": {k: {"ub": round(v["primary"]/v["total"], 4),
                        "ub_v3": round(v["v3"]/v["total"], 4),
                        "n": v["total"]}
                    for k, v in by.items()},
    }
    Path(args.summary).write_text(json.dumps(summary, indent=2, ensure_ascii=False), "utf-8")
    log(f"요약 저장: {args.summary}")


if __name__ == "__main__":
    main()
