#!/usr/bin/env python3
"""
analysis/gold_entity_retrieval_diag.py

gold_entity를 retrieval query로 사용했을 때의 retrieval_success 상한선 측정.
기존 controlled(ko_subq) answer_bearing_hit와 비교해
"retrieval failure 중 얼마가 query 언어 mismatch에서 오는가"를 진단한다.

비교:
  controlled (ko_subq): KO sub_q → gold_lang index   (실험 조건)
  gold_entity UB      : gold_entity → gold_lang index (상한선)

출력:
  data/retrieval/gold_entity_retrieval_ub.jsonl  (hop별 결과)
  results/gold_entity_retrieval_ub_summary.json     (요약)
"""

import argparse
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

BGE_M3_MODEL = "BAAI/bge-m3"
TOP_K        = 5


def normalize(text: str) -> str:
    return re.sub(r"[^\w가-힣]", "", text.lower())


def load_alias_set(entry: dict) -> set[str]:
    aliases: set[str] = set()
    for f in ("label_ko", "label_en", "wiki_title_ko", "wiki_title_en"):
        v = entry.get(f, "")
        if v:
            aliases.add(normalize(v))
    for f in ("aliases_ko", "aliases_en"):
        for a in entry.get(f, []):
            if a:
                aliases.add(normalize(a))
    aliases.discard("")
    return aliases


def passage_has_entity(passages: list[str], alias_set: set[str]) -> bool:
    for p in passages:
        np_ = normalize(p)
        if any(a in np_ for a in alias_set):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-pairs",    default="data/dataset/qa_pairs.jsonl")
    parser.add_argument("--alias-cache", default="data/retrieval/alias_cache.json")
    parser.add_argument("--index-dir",   default="data/faiss_index")
    parser.add_argument("--hits-dir",    default="data/retrieval",
                        help="retrieval_v3_aya_controlled.jsonl 등 answer-bearing hit 파일 디렉터리")
    parser.add_argument("--output",      default="data/retrieval/gold_entity_retrieval_ub.jsonl")
    parser.add_argument("--summary",     default="results/gold_entity_retrieval_ub_summary.json")
    args = parser.parse_args()

    # ── 데이터 로드 ──────────────────────────────────────────────────────────
    qa_items    = [json.loads(l) for l in Path(args.qa_pairs).read_text("utf-8").splitlines() if l.strip()]
    alias_cache = json.loads(Path(args.alias_cache).read_text("utf-8"))

    # ── BGE-M3 + FAISS ───────────────────────────────────────────────────────
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"BGE-M3 로딩 (device={device}) ...", file=sys.stderr)
    model = SentenceTransformer(BGE_M3_MODEL, device=device)

    index_dir = Path(args.index_dir)
    indexes, metas = {}, {}
    for lang in ("ko", "en"):
        idx_p  = index_dir / f"{lang}_index.faiss"
        meta_p = index_dir / f"{lang}_meta.json"
        indexes[lang] = faiss.read_index(str(idx_p))
        metas[lang]   = json.loads(meta_p.read_text("utf-8"))
        print(f"  [{lang}] {indexes[lang].ntotal}개 벡터", file=sys.stderr)

    def retrieve(query: str, lang: str) -> list[str]:
        vec = model.encode([query], normalize_embeddings=True,
                           show_progress_bar=False).astype("float32")
        _, ids = indexes[lang].search(vec, TOP_K)
        return [metas[lang][i]["text"] for i in ids[0] if 0 <= i < len(metas[lang])]

    # ── gold_entity UB 계산 ──────────────────────────────────────────────────
    print(f"hop별 gold_entity retrieval 계산 중 ({len(qa_items)} QA × 2 hops) ...", file=sys.stderr)
    results = []
    for it in qa_items:
        for hop in it["hops"]:
            gold_entity     = hop["gold_entity"]
            gold_entity_qid = hop["gold_entity_qid"]
            gold_lang       = hop["gold_lang"]

            cache_entry = alias_cache.get(gold_entity_qid, {})
            alias_set   = load_alias_set(cache_entry) if cache_entry else {normalize(gold_entity)}

            passages = retrieve(gold_entity, gold_lang)
            success  = passage_has_entity(passages, alias_set)

            results.append({
                "qid":           it["qid"],
                "pattern":       it["pattern"],
                "hop":           hop["hop"],
                "gold_lang":     gold_lang,
                "gold_entity":   gold_entity,
                "retrieval_success_gold_entity_ub": success,
            })

    # 저장
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {out_path} ({len(results)} 레코드)")

    # ── 요약 + controlled v3와 비교 ──────────────────────────────────────────
    from collections import defaultdict

    ub_by: dict[str, dict] = defaultdict(lambda: {"succ": 0, "total": 0})
    for r in results:
        key = f"{r['pattern']}-hop{r['hop']}-{r['gold_lang']}"
        ub_by[key]["total"] += 1
        if r["retrieval_success_gold_entity_ub"]:
            ub_by[key]["succ"] += 1

    ub_overall = {"succ": sum(r["retrieval_success_gold_entity_ub"] for r in results),
                  "total": len(results)}

    # controlled v3 평균 (모델별 평균으로 대표값 계산)
    hit_dir = Path(args.hits_dir)
    models = ["aya", "exaone", "llama", "qwen", "sonnet46"]
    hit_succ: dict[str, list] = defaultdict(list)
    for m in models:
        hits_path = hit_dir / f"retrieval_v3_{m}_controlled.jsonl"
        if not hits_path.exists():
            continue
        for line in hits_path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            key = f"{r['pattern']}-hop{r['hop']}-{r['gold_lang']}"
            hit_succ[key].append(int(r.get("answer_bearing_hit", False)))

    hit_avg = {k: round(sum(v) / len(v), 4) for k, v in hit_succ.items() if v}

    print("\n패턴-hop별 retrieval_success 비교")
    print(f"{'key':25s}  {'controlled(avg)':>16}  {'gold_entity UB':>14}")
    print("-" * 60)
    for key in sorted(ub_by.keys()):
        v = ub_by[key]
        ub_rate  = v["succ"] / v["total"] if v["total"] else 0
        ctrl_avg = hit_avg.get(key, float("nan"))
        print(f"{key:25s}  {ctrl_avg:>16.3f}  {ub_rate:>14.3f}  (n={v['total']})")

    ub_overall_rate = ub_overall["succ"] / ub_overall["total"]
    ctrl_overall    = sum(hit_avg.values()) / len(hit_avg) if hit_avg else float("nan")
    print(f"\n전체: controlled_avg={ctrl_overall:.3f}  gold_entity_UB={ub_overall_rate:.3f}")

    summary = {
        "gold_entity_ub_overall": round(ub_overall_rate, 4),
        "controlled_hit_avg":      round(ctrl_overall, 4),
        "gap":                    round(ub_overall_rate - ctrl_overall, 4),
        "per_key": {k: {"ub": round(v["succ"]/v["total"], 4), "n": v["total"],
                        "controlled_avg": hit_avg.get(k)}
                    for k, v in ub_by.items()},
    }
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"요약 저장: {args.summary}")


if __name__ == "__main__":
    main()
