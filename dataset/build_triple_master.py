#!/usr/bin/env python3
"""
dataset/build_triple_master.py

최종 250건 triple master 생성.

cap_applied_v10.json의 kept_triple_ids(250건)를 기준으로
audit 파일 + answer_type_labels + Wikidata 레이블을 합쳐 master row를 만든다.

Output: data/dataset/triples.jsonl
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_gate_lib as gate
from entity_cache import fetch_entity_cache

DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = DATA_DIR / "build" / "audit"
ORACLE_DIR = DATA_DIR / "build" / "oracle"

CAP_PATH = AUDIT_DIR / "cap_applied_v10.json"
ANSWER_TYPE_PATH = AUDIT_DIR / "answer_type_labels_v10.json"
AUDIT_PATHS = [
    AUDIT_DIR / "base_reuse_a644_triple_audit_refetched.jsonl",
    AUDIT_DIR / "phase2_candidates_triple_audit_refetched.jsonl",
    AUDIT_DIR / "phase3_candidates_triple_audit_refetched.jsonl",
    AUDIT_DIR / "phase4_candidates_triple_audit_refetched.jsonl",
]
OUT_PATH = DATA_DIR / "build" / "triples.jsonl"


def phase_from_id(triple_id: str) -> str:
    if triple_id.startswith("p1_"):
        return "phase1"
    if triple_id.startswith("p2_"):
        return "phase2"
    if triple_id.startswith("p3_"):
        return "phase3"
    if triple_id.startswith("p4_"):
        return "phase4"
    return "unknown"


def main():
    cap = json.loads(CAP_PATH.read_text())
    kept_ids = cap["kept_triple_ids"]
    kept_set = set(kept_ids)
    print(f"kept_triple_ids: {len(kept_ids)}건")

    # audit 행 로드
    audit_rows: dict[str, dict] = {}
    for path in AUDIT_PATHS:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["triple_id"] in kept_set:
                    audit_rows[d["triple_id"]] = d
    print(f"audit 행 로드: {len(audit_rows)}건")

    answer_types = json.loads(ANSWER_TYPE_PATH.read_text())

    # Wikidata 레이블 수집
    all_qids: set[str] = set()
    for d in audit_rows.values():
        all_qids.update([d["subj_qid"], d["bridge_qid"], d["answer_qid"]])
    print(f"Wikidata 레이블 조회: {len(all_qids)}개 QID")
    cache = fetch_entity_cache(sorted(all_qids), WIKIDATA_CACHE_PATH)

    def label(qid: str, lang: str) -> str:
        entry = cache.get(qid, {})
        return entry.get(f"label_{lang}") or entry.get("label_en") or qid

    # master 행 조립
    master_rows = []
    for triple_id in kept_ids:
        d = audit_rows.get(triple_id)
        if d is None:
            print(f"  WARNING: audit 행 없음 — {triple_id}")
            continue

        s, b, a = d["subj_qid"], d["bridge_qid"], d["answer_qid"]
        schema_id = d.get("schema_id") or gate.extract_schema_id(d.get("query_type", ""))
        at_entry = answer_types.get(a, {})

        row = {
            "triple_id": triple_id,
            "phase": phase_from_id(triple_id),
            "subj_qid": s,
            "bridge_qid": b,
            "answer_qid": a,
            "subj_label_ko": label(s, "ko"),
            "subj_label_en": label(s, "en"),
            "bridge_label_ko": label(b, "ko"),
            "bridge_label_en": label(b, "en"),
            "answer_label_ko": label(a, "ko"),
            "answer_label_en": label(a, "en"),
            "query_type": d.get("query_type"),
            "schema_id": schema_id,
            "relations": d.get("relations"),
            "source_patterns": d.get("source_patterns"),
            "path_ids": d.get("path_ids"),
            "answer_type": at_entry.get("answer_type", "other"),
            "answer_p31_qids": at_entry.get("p31_qids", []),
            "s_ko_page_exists": d.get("s_ko_page_exists"),
            "s_en_page_exists": d.get("s_en_page_exists"),
            "b_ko_page_exists": d.get("b_ko_page_exists"),
            "b_en_page_exists": d.get("b_en_page_exists"),
            "lexical_hop0_ko": d.get("lexical_hop0_ko"),
            "lexical_hop0_en": d.get("lexical_hop0_en"),
            "lexical_hop1_ko": d.get("lexical_hop1_ko"),
            "lexical_hop1_en": d.get("lexical_hop1_en"),
            "lexical_hop0_ko_matched_alias": d.get("lexical_hop0_ko_matched_alias"),
            "lexical_hop0_en_matched_alias": d.get("lexical_hop0_en_matched_alias"),
            "lexical_hop1_ko_matched_alias": d.get("lexical_hop1_ko_matched_alias"),
            "lexical_hop1_en_matched_alias": d.get("lexical_hop1_en_matched_alias"),
        }
        master_rows.append(row)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in master_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n저장 -> {OUT_PATH}")
    print(f"총 {len(master_rows)}행")

    # 간단 검증
    from collections import Counter
    phase_dist = Counter(r["phase"] for r in master_rows)
    at_dist = Counter(r["answer_type"] for r in master_rows)
    print("\nphase 분포:")
    for k, v in sorted(phase_dist.items()):
        print(f"  {k}: {v}")
    print("\nanswer_type 분포:")
    for k, v in at_dist.most_common():
        print(f"  {k:<15} {v} ({v/len(master_rows)*100:.1f}%)")


if __name__ == "__main__":
    main()
