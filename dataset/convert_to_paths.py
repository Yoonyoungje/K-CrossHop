#!/usr/bin/env python3
"""
dataset/convert_to_paths.py

oracle_pass.jsonl + triples.jsonl → pipeline/03_generate_qa.py 입력 포맷 변환.

cap_applied_v10.json의 kept_triple_ids(250건) × 4 conditions = 1000 path rows 생성.
pipeline/03_generate_qa.py --paths 로 넘길 wikidata_paths 포맷.

Output: data/dataset/paths.jsonl
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "build"

CAP_PATH = DATA_DIR / "audit" / "cap_applied_v10.json"
ORACLE_PASS_PATH = DATA_DIR / "oracle" / "oracle_pass.jsonl"
TRIPLE_MASTER_PATH = DATA_DIR / "triples.jsonl"
OUT_PATH = DATA_DIR / "paths_v10.jsonl"

COND_TO_PATTERN = {
    "a_ko_ko": "a",
    "b_en_ko": "b",
    "c_ko_en": "c",
    "d_en_en": "d",
}

LANG_SEQ = {
    "a": ["ko", "ko", "ko"],
    "b": ["ko", "en", "ko"],
    "c": ["ko", "ko", "en"],
    "d": ["ko", "en", "en"],
}

# hop_index: gold_lang per pattern
# hop0(subject): always "ko" (starting entity, Korean Wikipedia)
# hop1(bridge): "en" for b/d, "ko" for a/c
# hop2(answer): "en" for c/d, "ko" for a/b
BRIDGE_GOLD_LANG = {"a": "ko", "b": "en", "c": "ko", "d": "en"}
ANSWER_GOLD_LANG = {"a": "ko", "b": "ko", "c": "en", "d": "en"}


def main():
    cap = json.loads(CAP_PATH.read_text())
    kept_set = set(cap["kept_triple_ids"])
    print(f"kept_triple_ids: {len(kept_set)}건")

    # triple_master → label 조회용
    master: dict[str, dict] = {}
    for line in TRIPLE_MASTER_PATH.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        master[d["triple_id"]] = d

    # oracle_pass → kept 트리플만 필터
    oracle_rows: dict[tuple, dict] = {}  # (triple_id, condition) → row
    for line in ORACLE_PASS_PATH.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["triple_id"] in kept_set:
            oracle_rows[(d["triple_id"], d["condition"])] = d

    print(f"oracle_pass 로드: {len(oracle_rows)}건 (예상 1000)")

    path_rows = []
    missing = []

    for triple_id in cap["kept_triple_ids"]:
        m = master.get(triple_id)
        if m is None:
            print(f"  WARNING: triple_master에 없음 — {triple_id}")
            continue

        for cond, pattern in COND_TO_PATTERN.items():
            oracle = oracle_rows.get((triple_id, cond))
            if oracle is None:
                missing.append((triple_id, cond))
                continue

            s_ko = m["subj_label_ko"] or m["subj_label_en"]
            s_en = m["subj_label_en"] or m["subj_label_ko"]
            b_ko = m["bridge_label_ko"] or m["bridge_label_en"]
            b_en = m["bridge_label_en"] or m["bridge_label_ko"]
            a_ko = m["answer_label_ko"] or m["answer_label_en"]
            a_en = m["answer_label_en"] or m["answer_label_ko"]

            bridge_gl = BRIDGE_GOLD_LANG[pattern]
            answer_gl = ANSWER_GOLD_LANG[pattern]

            row = {
                "path_id": f"{triple_id}_pat_{pattern}",
                "pattern": pattern,
                "lang_sequence": LANG_SEQ[pattern],
                "hops": [
                    {
                        "hop": 0,
                        "qid": m["subj_qid"],
                        "label_ko": s_ko,
                        "label_en": s_en,
                        "gold_lang": "ko",
                        "primary_lang": "ko",
                    },
                    {
                        "hop": 1,
                        "qid": m["bridge_qid"],
                        "label_ko": b_ko,
                        "label_en": b_en,
                        "gold_lang": bridge_gl,
                        "primary_lang": bridge_gl,
                    },
                    {
                        "hop": 2,
                        "qid": m["answer_qid"],
                        "label_ko": a_ko,
                        "label_en": a_en,
                        "gold_lang": answer_gl,
                        "primary_lang": answer_gl,
                    },
                ],
                "query_type": oracle["query_type"].rsplit("_", 1)[0] + "_" + pattern,
                "relations": oracle.get("relations", m.get("relations", [])),
                "oracle_question": oracle.get("oracle_question", ""),
                "oracle_hop0_bridge_support": oracle.get("oracle_hop0_bridge_supported", ""),
                "oracle_hop1_answer_support": oracle.get("oracle_hop1_answer_supported", ""),
            }
            path_rows.append(row)

    if missing:
        print(f"\n누락 (oracle_pass 없음): {len(missing)}건")
        for tid, cond in missing[:10]:
            print(f"  {tid} / {cond}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in path_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n저장 -> {OUT_PATH}")
    print(f"총 {len(path_rows)}행")

    from collections import Counter
    pat_dist = Counter(r["pattern"] for r in path_rows)
    print("pattern 분포:")
    for p in "abcd":
        print(f"  {p}: {pat_dist.get(p, 0)}")


if __name__ == "__main__":
    main()
