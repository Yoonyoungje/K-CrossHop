#!/usr/bin/env python3
"""
analysis/gold_entity_ub_table.py

Table 12 (gold-entity 상한): gold entity 를 검색어로 직접 발행했을 때의 answer-bearing hit@5(UB),
controlled 조건의 hit@5, 그 차이(gap). 두 열 모두 §4.1 매처(utils.retrieval_success.check_answer_bearing_hit)
판정을 읽기만 한다 (CPU).
  UB          results/camera_ready/retriever_robustness/gold_entity_ub/bge_per_entity.jsonl 의 hit_v3
              (analysis/gold_entity_ub_encoder.py, BGE-M3 — GPU 산출물)
  controlled  모델별 answer-bearing hit 파일 (hop0·hop1)
행: EN-index hop0 (패턴 b/d), KO-index hop0 (a/c), 양 홉 전체. hop0 검색어는 모델과 무관하다.
hop1 은 모델이 낸 bridge 로 검색하므로 양 홉 전체의 controlled 값은 모델마다 다르다.

  python3 analysis/gold_entity_ub_table.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/dataset/qa_pairs.jsonl"
UB = ROOT / "results/camera_ready/retriever_robustness/gold_entity_ub/bge_per_entity.jsonl"
HITS = {
    "aya":    "data/retrieval/aya_ecological_hits.jsonl",
    "exaone": "data/retrieval/exaone_ecological_hits.jsonl",
    "llama":  "data/retrieval/llama_ecological_hits.jsonl",
    "qwen":   "data/retrieval/qwen_ecological_hits.jsonl",
    "sonnet": "data/retrieval/sonnet_ecological_hits.jsonl",
    "gemini": "data/retrieval/gemini_ecological_hits.jsonl",
}


def load(path: Path, qids: set) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text("utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["qid"] in qids]


def rates(rows: list[dict], field: str) -> dict:
    def rate(rs):
        return sum(bool(r[field]) for r in rs) / len(rs)
    h0 = [r for r in rows if r["hop"] == 0]
    return {"en_hop0": rate([r for r in h0 if r["gold_lang"] == "en"]),
            "ko_hop0": rate([r for r in h0 if r["gold_lang"] == "ko"]),
            "both": rate(rows), "n": len(rows)}


def main() -> None:
    qids = {json.loads(l)["qid"] for l in QA.read_text("utf-8").splitlines() if l.strip()}
    ub = rates(load(UB, qids), "hit_v3")
    ctl = {m: rates(load(ROOT / p, qids), "answer_bearing_hit") for m, p in HITS.items()}
    for key in ("en_hop0", "ko_hop0"):
        assert len({c[key] for c in ctl.values()}) == 1, f"hop0 controlled 값이 모델마다 다르다: {key}"
    any_m = next(iter(ctl.values()))

    per_model = ", ".join(f"{m} {c['n']}홉" for m, c in ctl.items())
    print(f"gold-entity UB {ub['n']}홉, controlled 모델별 {per_model}\n")
    print("| 행 | Gold-entity UB | controlled hit@5 | gap |")
    print("|-|-|-|-|")
    for key, label in (("en_hop0", "EN-index hop0 (b/d)"), ("ko_hop0", "KO-index hop0 (a/c)")):
        print(f"| {label} | {ub[key]:.4f} | {any_m[key]:.4f} | {ub[key] - any_m[key]:.4f} |")
    for m, c in ctl.items():
        print(f"| 양 홉 전체 — {m} | {ub['both']:.4f} | {c['both']:.4f} | {ub['both'] - c['both']:.4f} |")
    lo, hi = min(c["both"] for c in ctl.values()), max(c["both"] for c in ctl.values())
    print(f"\n양 홉 전체 controlled 범위 {lo:.4f}–{hi:.4f}, gap 범위 {ub['both'] - hi:.4f}–{ub['both'] - lo:.4f}")


if __name__ == "__main__":
    main()
