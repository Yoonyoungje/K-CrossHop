#!/usr/bin/env python3
"""
dataset/apply_caps.py

oracle gate를 통과한 410 triple(full 2×2 pass)에 개체 cap과 그룹 cap을 greedy로 적용한다
(데이터셋 구축 funnel의 410 → 263 단계).

Cap 규칙:
  answer_qid  <= 3 triples
  subject_qid <= 2
  bridge_qid  <= 2
  그룹 cap(greedy 단계에서 강제): P178(개발사) 계열 <= 20%, location 정답 <= 35%
  query_type 25% · answer_type 35% 기준은 cap 후 분포를 보고만 한다(강제 절단 없음)

순서 독립: 후보를 triple_id 정렬 순서로 읽은 뒤 seed 42로 섞는다. 기록된 263은 집합 순회
순서(PYTHONHASHSEED)에 의존한 원래 실행에서 나왔으므로 이 스크립트로 다시 나오지 않는다 —
정본은 배포하는 410 / 263 / 250 ID 목록이고, 그 목록이 위 규칙을 지키는지는 별도로 검증한다.

  python3 dataset/apply_caps.py [--output data/audit/cap_applied_sorted.json]

triple 필드(subject·bridge·answer QID, relation, query_type)는 게이트 판정 파일에서 읽는다 — 공개 데이터만으로
돌게 하려는 것이다. 구축 감사 파일(*_triple_audit_refetched.jsonl)에서 읽던 이전 판과 출력이 byte 동일하다(11/16 확인).
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "data" / "build" / "audit"

# 배포 빌더가 경로 문자열 단위로 공개 경로(data/construction/)로 바꾼다 — 조립하지 않고 전체 경로로 쓴다.
FULL_IDS_PATH = REPO_ROOT / "data/construction/ids_410.json"
GATE_PATH = REPO_ROOT / "data/construction/gate_decisions.jsonl"
ANSWER_TYPE_PATH = REPO_ROOT / "data/construction/answer_types.json"
OUT_PATH = AUDIT_DIR / "cap_applied_sorted.json"   # 기록본(cap_applied_v10.json)은 덮지 않는다

ANSWER_QID_CAP = 3
SUBJECT_QID_CAP = 2  # 권장 1, 불가피 시 2 — 여기서는 2로 적용(보수적이지 않은 쪽)
BRIDGE_QID_CAP = 2
QUERY_TYPE_PCT_CAP = 0.25
ANSWER_TYPE_PCT_CAP = 0.35

# 사용자 결정: entity cap만으로는 P178(개발사) 군과 location-answer 군이 과집중
# (각각 35.6%, 41.7%)되는 게 확인됨. query_type/answer_type 개별 cap과 별도로,
# "스키마 군" 단위 group cap을 greedy 단계에서 직접 강제한다.
GROUP_PCT_CAPS = {
    "p178_group": 0.20,          # relations[0] == "P178" (game/software developer 계열)
    "location_answer_group": 0.35,  # answer_type == "location"
}

SEED = 42


def triple_groups(t: dict, answer_types: dict) -> list[str]:
    groups = []
    if t["relations"] and t["relations"][0] == "P178":
        groups.append("p178_group")
    if answer_types.get(t["answer_qid"], {}).get("answer_type") == "location":
        groups.append("location_answer_group")
    return groups


def load_triples() -> list[dict]:
    full_ids = set(json.load(FULL_IDS_PATH.open(encoding="utf-8")))
    rows = {}
    with GATE_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["condition"] == "a_ko_ko" and d["triple_id"] in full_ids:   # triple 필드는 네 패턴 행에서 같다
                rows[d["triple_id"]] = d
    return [rows[tid] for tid in sorted(full_ids)]   # 집합 순회 순서 의존 제거


def apply_entity_caps(triples: list[dict], answer_types: dict) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    order = list(triples)
    rng.shuffle(order)

    answer_count: Counter = Counter()
    subject_count: Counter = Counter()
    bridge_count: Counter = Counter()
    group_count: Counter = Counter()

    kept, dropped = [], []
    for t in order:
        a, s, b = t["answer_qid"], t["subj_qid"], t["bridge_qid"]
        if (answer_count[a] >= ANSWER_QID_CAP or
                subject_count[s] >= SUBJECT_QID_CAP or
                bridge_count[b] >= BRIDGE_QID_CAP):
            dropped.append(t)
            continue

        # group cap: (현재 group 건수 + 1) / (현재 kept 건수 + 1) <= 목표 비율
        groups = triple_groups(t, answer_types)
        if any((group_count[g] + 1) / (len(kept) + 1) > GROUP_PCT_CAPS[g] for g in groups):
            dropped.append(t)
            continue

        answer_count[a] += 1
        subject_count[s] += 1
        bridge_count[b] += 1
        for g in groups:
            group_count[g] += 1
        kept.append(t)
    return kept, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(OUT_PATH))
    args = ap.parse_args()
    out_path = Path(args.output)

    triples = load_triples()
    answer_types = json.loads(ANSWER_TYPE_PATH.read_text(encoding="utf-8"))

    print(f"cap 적용 전: {len(triples)}개")

    kept, dropped = apply_entity_caps(triples, answer_types)
    print(f"entity cap(answer<=3, subject<=2, bridge<=2) + group cap"
          f"(P178<={GROUP_PCT_CAPS['p178_group']*100:.0f}%, "
          f"location<={GROUP_PCT_CAPS['location_answer_group']*100:.0f}%) 적용 후: "
          f"{len(kept)}개 (탈락 {len(dropped)}개)")

    qt_dist = Counter(t["query_type"] for t in kept)
    at_dist = Counter(answer_types.get(t["answer_qid"], {}).get("answer_type", "other") for t in kept)
    rel_dist = Counter(tuple(t["relations"]) if t["relations"] else None for t in kept)

    print("\nquery_type 분포 (cap 후, 상위 10, 25% 기준 점검):")
    for k, v in qt_dist.most_common(10):
        share = v / len(kept)
        flag = "  OVER 25%" if share > QUERY_TYPE_PCT_CAP else ""
        print(f"  {k:<35} {v:>3} ({share*100:.1f}%){flag}")

    print("\nanswer_type 분포 (cap 후, 35% 기준 점검):")
    for k, v in at_dist.most_common():
        share = v / len(kept)
        flag = "  OVER 35%" if share > ANSWER_TYPE_PCT_CAP else ""
        print(f"  {k:<15} {v:>3} ({share*100:.1f}%){flag}")

    # P178 계열(developer) 비중 — 사용자가 지적한 과집중 확인
    p178_count = sum(1 for t in kept if t["relations"] and t["relations"][0] == "P178")
    print(f"\nP178(개발사) 계열 비중: {p178_count}/{len(kept)} ({p178_count/len(kept)*100:.1f}%)")

    out = {
        "n_before_cap": len(triples),
        "n_after_entity_cap": len(kept),
        "n_dropped_by_entity_cap": len(dropped),
        "query_type_distribution": {str(k): v for k, v in qt_dist.items()},
        "answer_type_distribution": dict(at_dist),
        "relation_pair_distribution": {str(k): v for k, v in rel_dist.items()},
        "p178_share": round(p178_count / len(kept), 4) if kept else None,
        "kept_triple_ids": [t["triple_id"] for t in kept],
        "dropped_triple_ids": [t["triple_id"] for t in dropped],
        "cap_config": {
            "answer_qid_cap": ANSWER_QID_CAP, "subject_qid_cap": SUBJECT_QID_CAP,
            "bridge_qid_cap": BRIDGE_QID_CAP, "query_type_pct_cap": QUERY_TYPE_PCT_CAP,
            "answer_type_pct_cap": ANSWER_TYPE_PCT_CAP, "seed": SEED,
        },
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장 -> {out_path}")


if __name__ == "__main__":
    main()
