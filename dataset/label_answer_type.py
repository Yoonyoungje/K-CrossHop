#!/usr/bin/env python3
"""
dataset/label_answer_type.py

answer_type 라벨링. Wikidata P31(instance of) claims를 조회해 answer_qid를
분류한다. LLM 호출 없음.

분류 카테고리:
  location, organization, country, award, person, creative_work,
  software, taxonomic, date_event, other

사용법:
  python dataset/label_answer_type.py                    # triple master 기준
"""

import os
import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
AUDIT_DIR = DATA_DIR / "build" / "audit"

REFETCHED_AUDIT_PATH = AUDIT_DIR / "base_reuse_a644_triple_audit_refetched.jsonl"
LABELS_PATH = AUDIT_DIR / "answer_type_labels_v10.json"
STATS_PATH = AUDIT_DIR / "base_reuse_a644_answer_type_stats.json"

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = os.getenv("WIKI_USER_AGENT", "K-CrossHop-Research/1.0 (research use)")
BATCH_SIZE = 50
RATE_DELAY = 1.0

# P31(instance of) 값(QID) -> answer_type 카테고리.
# 흔히 등장하는 클래스만 우선 다루고, 매칭 안 되면 "other"로 남겨 수동 보강 대상으로 표시한다.
P31_TO_ANSWER_TYPE: dict[str, str] = {
    # person
    "Q5": "person",
    # location (행정구역/지명/지형)
    "Q486972": "location",       # human settlement
    "Q515": "location",          # city
    "Q3957": "location",         # town
    "Q5119": "location",         # capital city
    "Q1549591": "location",      # big city
    "Q15284": "location",        # municipality
    "Q10864048": "location",     # second-order administrative country subdivision
    "Q1496967": "location",      # territorial entity
    "Q56061": "location",        # administrative territorial entity
    "Q35657": "location",        # U.S. state
    "Q6256": "country",
    "Q3624078": "country",       # sovereign state
    "Q4830453": "organization",  # business
    "Q43229": "organization",
    "Q891723": "organization",   # public company
    "Q484652": "organization",   # international organization
    "Q31855": "organization",    # research institute
    "Q3918": "organization",     # university
    "Q875538": "organization",   # public university
    "Q23397": "location",        # lake (안전망, 드물게 등장)
    "Q618123": "location",       # geographical feature
    "Q618779": "award",
    "Q230788": "award",          # grant/award (기타 표기)
    "Q11448906": "award",        # award ceremony 관련(드묾)
    "Q11424": "creative_work",   # film
    "Q5398426": "creative_work", # television series
    "Q571": "creative_work",     # book
    "Q482994": "creative_work",  # album
    "Q7725634": "creative_work", # literary work
    "Q7889": "software",         # video game
    "Q341": "software",          # free software (드묾)
    "Q9135": "software",         # operating system
    "Q1639825": "person",        # athlete가 보통 person으로 분류되지만 명시

    # location 추가 (도시/행정구역 세부 클래스)
    "Q1093829": "location",      # city in the United States
    "Q13218357": "location",     # city of California
    "Q62049": "location",        # county seat
    "Q13218391": "location",     # charter city
    "Q1187811": "location",      # college town
    "Q5327704": "location",      # special ward of Japan
    "Q1776764": "location",      # county of Romania
    "Q1749269": "location",      # city designated by government ordinance
    "Q1801244": "location",      # ward (일본 행정구역)

    # organization 추가 (대학/출판/연구기관 세부 클래스)
    "Q902104": "organization",   # private university
    "Q23002054": "organization", # private not-for-profit educational institution
    "Q96888669": "organization", # academic publisher
    "Q15936437": "organization", # research university
    "Q62078547": "organization", # public research university
    "Q23002039": "organization", # public educational institution of the United States
    "Q1188663": "organization",  # Colonial Colleges

    # award 추가
    "Q19020": "award",           # Academy Awards
    "Q378427": "award",          # literary award
    "Q4220917": "award",         # film award
    "Q107655869": "award",       # group of awards
    "Q1364556": "award",         # music award
}


def _fetch_batch(qids: list[str]) -> dict:
    params = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "claims|labels",
        "languages": "ko|en",
        "format": "json",
    })
    req = urllib.request.Request(f"{WIKIDATA_API}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())["entities"]


def _p31_qids(ent: dict) -> list[str]:
    claims = ent.get("claims", {}).get("P31", [])
    out = []
    for c in claims:
        try:
            out.append(c["mainsnak"]["datavalue"]["value"]["id"])
        except (KeyError, TypeError):
            continue
    return out


def classify(p31_qids: list[str]) -> str:
    for qid in p31_qids:
        if qid in P31_TO_ANSWER_TYPE:
            return P31_TO_ANSWER_TYPE[qid]
    return "other"


PHASE2_REFETCHED_AUDIT_PATH = AUDIT_DIR / "phase2_candidates_triple_audit_refetched.jsonl"
PHASE3_REFETCHED_AUDIT_PATH = AUDIT_DIR / "phase3_candidates_triple_audit_refetched.jsonl"
PHASE4_REFETCHED_AUDIT_PATH = AUDIT_DIR / "phase4_candidates_triple_audit_refetched.jsonl"


def load_eligible_answer_qids() -> tuple[list[dict], list[str]]:
    rows = []
    paths = [REFETCHED_AUDIT_PATH]
    if PHASE2_REFETCHED_AUDIT_PATH.exists():
        paths.append(PHASE2_REFETCHED_AUDIT_PATH)
    if PHASE3_REFETCHED_AUDIT_PATH.exists():
        paths.append(PHASE3_REFETCHED_AUDIT_PATH)
    if PHASE4_REFETCHED_AUDIT_PATH.exists():
        paths.append(PHASE4_REFETCHED_AUDIT_PATH)
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("eligible_full_2x2_lexical"):
                    rows.append(d)
    qids = sorted({r["answer_qid"] for r in rows})
    return rows, qids


def fetch_p31_labels(qids: list[str]) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for i in range(0, len(qids), BATCH_SIZE):
        batch = qids[i:i + BATCH_SIZE]
        print(f"  배치 {i // BATCH_SIZE + 1}: {len(batch)}개 ... ", end="", flush=True)
        try:
            entities = _fetch_batch(batch)
            for qid in batch:
                ent = entities.get(qid, {})
                if "missing" in ent:
                    labels[qid] = {"answer_type": "other", "p31_qids": [], "p31_labels": []}
                    continue
                p31 = _p31_qids(ent)
                answer_type = classify(p31)
                p31_labels = [ent.get("labels", {}).get("en", {}).get("value", "")
                              for _ in p31]  # 자리표시, 아래에서 실제로 보강
                labels[qid] = {"answer_type": answer_type, "p31_qids": p31, "p31_labels": p31}
            print("완료")
        except Exception as e:
            print(f"실패: {e}")
        if i + BATCH_SIZE < len(qids):
            time.sleep(RATE_DELAY)
    return labels


def _run_labeling(rows: list[dict], qids: list[str], labels_path: Path, stats_path: Path) -> None:
    """labels_path에 P31 라벨을 저장하고 stats_path에 분포 통계를 저장한다."""
    labels: dict[str, dict] = {}

    # 기존 캐시 로드 (이 데이터셋용)
    if labels_path.exists():
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        print(f"기존 라벨 로드: {len(labels)}개")

    missing = [q for q in qids if q not in labels]
    if missing:
        print(f"Wikidata P31 조회: {len(missing)}개")
        new_labels = fetch_p31_labels(missing)
        labels.update(new_labels)

    # 매핑 테이블이 갱신됐을 수 있으므로 캐시된 p31_qids로 항상 재분류
    for qid, entry in labels.items():
        entry["answer_type"] = classify(entry.get("p31_qids", []))

    labels_path.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"라벨 저장 -> {labels_path}")

    dist = Counter(labels.get(r["answer_qid"], {}).get("answer_type", "other") for r in rows)
    other_qids = sorted({r["answer_qid"] for r in rows
                         if labels.get(r["answer_qid"], {}).get("answer_type") == "other"})

    stats = {
        "n_items": len(rows),
        "n_unique_answer_qid": len(qids),
        "answer_type_distribution": dict(dist),
        "answer_type_share": {k: round(v / len(rows), 4) for k, v in dist.items()},
        "other_qids_needing_manual_review": other_qids,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nanswer_type 분포 ({len(rows)}건 기준):")
    for k, v in dist.most_common():
        print(f"  {k:<15} {v:>4} ({v / len(rows) * 100:.1f}%)")
    print(f"\n'other' unique qid: {len(other_qids)}개")
    print(f"통계 저장 -> {stats_path}")

    return labels


def main_v10() -> None:
    rows, qids = load_eligible_answer_qids()
    print(f"eligible triple {len(rows)}개, unique answer_qid {len(qids)}개")
    _run_labeling(rows, qids, LABELS_PATH, STATS_PATH)


def main() -> None:
    argparse.ArgumentParser(description="triple master 의 answer_qid 에 answer_type 을 붙인다").parse_args()
    main_v10()


if __name__ == "__main__":
    main()
