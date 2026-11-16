#!/usr/bin/env python3
"""
dataset/repair_labels.py

triples.jsonl의 QID 레이블 문제 일괄 수정 + 대체 triple 보충.

수정 내용:
  - answer_label_en == QID (Q82595 → "Bundesliga")
  - bridge_label_en == QID (Q129591 → "Hugh Jackman")
  - subj_label_en == QID (8개 → enwiki title)
  - source_patterns null → [], 중복 제거
  - 차두리 bridge 중복 triple 제거 (p3_Q310971_Q104761_Q82595)
  - 대체 triple 추가 (p1_Q1849822_Q294587_Q282263: 버추어 레이싱 → Sega AM2 → Yu Suzuki)
  - cap_applied_v10.json / answer_type_labels_v10.json 동기화

실행 후:
  python dataset/convert_to_paths.py   # paths 재생성
  python pipeline/03_generate_qa.py --paths data/dataset/paths.jsonl --output data/dataset/qa_pairs.jsonl
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "build"
MASTER_PATH = DATA_DIR / "triples.jsonl"
CAP_PATH = DATA_DIR / "audit" / "cap_applied_v10.json"
ANSWER_TYPE_PATH = DATA_DIR / "audit" / "answer_type_labels_v10.json"
AUDIT_PATH = DATA_DIR / "audit" / "base_reuse_a644_triple_audit_refetched.jsonl"

LABEL_FIXES: dict[tuple[str, str], str] = {
    # answer_label_en == Q82595
    ("p3_Q310971_Q38245_Q82595",  "answer_label_en"): "Bundesliga",
    ("p3_Q486787_Q38245_Q82595",  "answer_label_en"): "Bundesliga",
    ("p3_Q310971_Q104761_Q82595", "answer_label_en"): "Bundesliga",
    # bridge_label_en == Q129591
    ("p1_Q102180106_Q129591_Q3130", "bridge_label_en"): "Hugh Jackman",
    # subj_label_en == QID (8건)
    ("p1_Q136148940_Q130744_Q21",   "subj_label_en"): "Forza Horizon 6",
    ("p1_Q112435027_Q4806100_Q270",  "subj_label_en"): "The Alters",
    ("p1_Q109285421_Q13462819_Q62",  "subj_label_en"): "Pikmin Bloom",
    ("p3_Q1060929_Q35886_Q311999",   "subj_label_en"): "Lamborghini Silhouette",
    ("p1_Q239288_Q178824_Q52994",   "subj_label_en"): "StarCraft II: Wings of Liberty",
    ("p1_Q131436227_Q720871_Q47164", "subj_label_en"): "Intergalactic: The Heretic Prophet",
    ("p1_Q163628_Q193559_Q347418",  "subj_label_en"): "Counter-Strike",
    ("p3_Q107431877_Q8093_Q456052",  "subj_label_en"): "Nintendo Switch OLED",
}

REMOVE_TRIPLE_ID = "p3_Q310971_Q104761_Q82595"
REMOVE_REASON = (
    "bridge_ambiguity_duplicate_SA_pair: same subject(차두리) and answer(Bundesliga) "
    "with different bridge causes 8-repeat outer question"
)

# 대체 triple: oracle_pass 4조건 통과, entity cap 여유 (s=0,b=0,a=0)
ADD_TRIPLE_ID = "p1_Q1849822_Q294587_Q282263"
ADD_LABELS = {
    "subj_qid": "Q1849822", "subj_label_ko": "버추어 레이싱", "subj_label_en": "Virtua Racing",
    "bridge_qid": "Q294587", "bridge_label_ko": "세가 AM2",    "bridge_label_en": "Sega AM2",
    "answer_qid": "Q282263", "answer_label_ko": "스즈키 유",   "answer_label_en": "Yu Suzuki",
}
ADD_ANSWER_TYPE = {"answer_type": "person", "p31_qids": ["Q5"]}


def load_audit_row(triple_id: str) -> dict | None:
    for line in AUDIT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        if d["triple_id"] == triple_id:
            return d
    return None


def build_master_row(audit: dict) -> dict:
    tid = audit["triple_id"]
    qt = audit.get("query_type", "")
    schema_id = qt.rsplit("_", 1)[0] if qt else ""
    return {
        "triple_id": tid,
        "phase": "phase1",
        "subj_qid":        ADD_LABELS["subj_qid"],
        "bridge_qid":      ADD_LABELS["bridge_qid"],
        "answer_qid":      ADD_LABELS["answer_qid"],
        "subj_label_ko":   ADD_LABELS["subj_label_ko"],
        "subj_label_en":   ADD_LABELS["subj_label_en"],
        "bridge_label_ko": ADD_LABELS["bridge_label_ko"],
        "bridge_label_en": ADD_LABELS["bridge_label_en"],
        "answer_label_ko": ADD_LABELS["answer_label_ko"],
        "answer_label_en": ADD_LABELS["answer_label_en"],
        "query_type":  qt,
        "schema_id":   schema_id,
        "relations":   audit.get("relations", []),
        "source_patterns": [],
        "path_ids":    audit.get("path_ids"),
        "answer_type": ADD_ANSWER_TYPE["answer_type"],
        "answer_p31_qids": ADD_ANSWER_TYPE["p31_qids"],
        "s_ko_page_exists": audit.get("s_ko_page_exists"),
        "s_en_page_exists": audit.get("s_en_page_exists"),
        "b_ko_page_exists": audit.get("b_ko_page_exists"),
        "b_en_page_exists": audit.get("b_en_page_exists"),
        "lexical_hop0_ko": audit.get("lexical_hop0_ko"),
        "lexical_hop0_en": audit.get("lexical_hop0_en"),
        "lexical_hop1_ko": audit.get("lexical_hop1_ko"),
        "lexical_hop1_en": audit.get("lexical_hop1_en"),
        "lexical_hop0_ko_matched_alias": audit.get("lexical_hop0_ko_matched_alias"),
        "lexical_hop0_en_matched_alias": audit.get("lexical_hop0_en_matched_alias"),
        "lexical_hop1_ko_matched_alias": audit.get("lexical_hop1_ko_matched_alias"),
        "lexical_hop1_en_matched_alias": audit.get("lexical_hop1_en_matched_alias"),
    }


def main():
    rows = [json.loads(l) for l in MASTER_PATH.read_text().splitlines() if l.strip()]
    print(f"triple_master 로드: {len(rows)}건")

    # 이미 추가된 경우 스킵
    existing_ids = {r["triple_id"] for r in rows}
    if ADD_TRIPLE_ID in existing_ids:
        print(f"대체 triple 이미 존재, 추가 스킵: {ADD_TRIPLE_ID}")
        add_new = False
    else:
        add_new = True

    label_fixed = 0
    source_fixed = 0
    removed = False

    patched = []
    for row in rows:
        tid = row["triple_id"]

        if tid == REMOVE_TRIPLE_ID:
            removed = True
            print(f"  제거: {tid}")
            continue

        for (fix_tid, field), new_val in LABEL_FIXES.items():
            if tid == fix_tid and row.get(field) != new_val:
                print(f"  [{tid}] {field}: {row[field]!r} → {new_val!r}")
                row[field] = new_val
                label_fixed += 1

        sp = row.get("source_patterns")
        if sp is None:
            row["source_patterns"] = []
            source_fixed += 1
        elif isinstance(sp, list) and len(sp) != len(set(sp)):
            row["source_patterns"] = sorted(set(sp))
            source_fixed += 1

        patched.append(row)

    # 대체 triple 추가
    if add_new:
        audit = load_audit_row(ADD_TRIPLE_ID)
        if audit is None:
            print(f"  ERROR: audit 행 없음 — {ADD_TRIPLE_ID}")
        else:
            new_row = build_master_row(audit)
            patched.append(new_row)
            print(f"\n  추가: {ADD_TRIPLE_ID} ({ADD_LABELS['subj_label_ko']} → {ADD_LABELS['bridge_label_ko']} → {ADD_LABELS['answer_label_ko']})")

    print(f"\n레이블 수정: {label_fixed}건")
    print(f"source_patterns 정리: {source_fixed}건")
    print(f"triple 제거: {'1건' if removed else '0건'}, 추가: {'1건' if add_new else '0건'}")
    print(f"최종 triple 수: {len(patched)}건 (기존 {len(rows)}건)")

    MASTER_PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in patched) + "\n",
        encoding="utf-8",
    )
    print(f"저장 → {MASTER_PATH}")

    # cap_applied_v10.json 동기화
    cap = json.loads(CAP_PATH.read_text())
    kept = cap["kept_triple_ids"]
    changed = False

    if REMOVE_TRIPLE_ID in kept:
        kept.remove(REMOVE_TRIPLE_ID)
        removed_list = cap.setdefault("manual_audit_removed", [])
        if REMOVE_TRIPLE_ID not in removed_list:
            removed_list.append(REMOVE_TRIPLE_ID)
        changed = True

    if ADD_TRIPLE_ID not in kept:
        kept.append(ADD_TRIPLE_ID)
        cap.setdefault("replacement_added", []).append({
            "triple_id": ADD_TRIPLE_ID,
            "reason": "replacement for bridge_ambiguity removal of p3_Q310971_Q104761_Q82595",
        })
        changed = True

    if changed:
        cap["kept_triple_ids"] = kept
        cap["n_after_entity_cap"] = len(kept)
        CAP_PATH.write_text(json.dumps(cap, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"cap_applied_v10.json 업데이트 → kept={len(kept)}건")

    # answer_type_labels_v10.json 동기화
    if add_new:
        at = json.loads(ANSWER_TYPE_PATH.read_text())
        aqid = ADD_LABELS["answer_qid"]
        if aqid not in at:
            at[aqid] = ADD_ANSWER_TYPE
            ANSWER_TYPE_PATH.write_text(json.dumps(at, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"answer_type_labels_v10.json 추가 → {aqid} = person")

    print("\n다음 단계:")
    print("  python dataset/convert_to_paths.py")
    print("  python pipeline/03_generate_qa.py --paths data/dataset/paths.jsonl --output data/dataset/qa_pairs.jsonl")


if __name__ == "__main__":
    main()
