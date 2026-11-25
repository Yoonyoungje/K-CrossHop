#!/usr/bin/env python3
"""
pipeline/01_extract_paths.py
====================
Wikidata SPARQL로 KO-EN 멀티홉 경로를 추출한다. (schema-factory 기반)

출력: data/raw/wikidata_paths.jsonl (config.yaml의 output.raw_paths)

실행:
  python pipeline/01_extract_paths.py [--config config.yaml] [--dry-run] [--schema SCHEMA_ID]
"""

import json
import argparse
import logging
from collections import Counter
from pathlib import Path
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.sparql import WikidataSPARQL, extract_wiki_title
from utils.schema_registry import SCHEMAS
from utils.sparql_factory import build_query

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def determine_primary_lang(wiki_ko: bool, wiki_en: bool) -> str:
    if wiki_ko and not wiki_en:
        return "ko"
    if wiki_en and not wiki_ko:
        return "en"
    return "both"


def rows_to_path(rows: list[dict], pattern: str,
                 query_type: str, relations: list[str]) -> list[dict]:
    """
    SPARQL 결과 rows → 표준 path 형식으로 변환.
    rows: [{subj, wiki_subj, subjLabelKo, subjLabelEn,
             mid,  wiki_mid,  midLabelKo,  midLabelEn,
             obj,  wiki_obj,  objLabelKo,  objLabelEn}, ...]
    """
    lang_map = {
        "a": ["ko", "ko", "ko"],
        "b": ["ko", "en", "ko"],
        "c": ["ko", "ko", "en"],
        "d": ["ko", "en", "en"],
    }

    paths = []
    seen = set()

    for row in rows:
        key = (row["subj"], row["mid"])
        if key in seen:
            continue
        seen.add(key)

        gold_langs = lang_map[pattern]

        subj_ko, subj_en = True, True
        mid_ko, mid_en   = True, True
        obj_ko, obj_en   = False, True

        if pattern == "a":
            subj_en, mid_en, obj_en = False, False, False
            obj_ko = True
        elif pattern == "b":
            mid_ko = False
            obj_ko, obj_en = True, False
        elif pattern == "d":
            mid_ko = False

        subj_qid = _qid(row["subj"])
        mid_qid  = _qid(row["mid"])
        obj_qid  = _qid(row["obj"])

        hops = [
            {
                "hop": 0,
                "qid": subj_qid,
                "label_ko": row["subjLabelKo"],
                "label_en": row["subjLabelEn"],
                "wiki_title": extract_wiki_title(row.get("wiki_subj", "")),
                "gold_lang": gold_langs[0],
                "wiki_ko": subj_ko,
                "wiki_en": subj_en,
                "primary_lang": determine_primary_lang(subj_ko, subj_en),
            },
            {
                "hop": 1,
                "qid": mid_qid,
                "label_ko": row["midLabelKo"],
                "label_en": row["midLabelEn"],
                "wiki_title": extract_wiki_title(row.get("wiki_mid", "")),
                "gold_lang": gold_langs[1],
                "wiki_ko": mid_ko,
                "wiki_en": mid_en,
                "primary_lang": determine_primary_lang(mid_ko, mid_en),
            },
            {
                "hop": 2,
                "qid": obj_qid,
                "label_ko": row["objLabelKo"],
                "label_en": row["objLabelEn"],
                "wiki_title": extract_wiki_title(row.get("wiki_obj", "")),
                "gold_lang": gold_langs[2],
                "wiki_ko": obj_ko,
                "wiki_en": obj_en,
                "primary_lang": determine_primary_lang(obj_ko, obj_en),
            },
        ]

        lang_seq = [h["primary_lang"] for h in hops]

        paths.append({
            "path_id": f"{query_type}_{subj_qid}_{mid_qid}_{obj_qid}",
            "pattern": pattern,
            "lang_sequence": lang_seq,
            "hops": hops,
            "relations": relations,
            "query_type": query_type,
        })

    return paths


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _bind_to_dict(r: dict, keys: list[str]) -> dict:
    return {k: r.get(k, {}).get("value", "") for k in keys}

ROW_KEYS = [
    "subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
    "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
    "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn",
]


def build_query_jobs(schema_filter: str | None = None) -> list[tuple]:
    """
    SCHEMAS registry에서 (schema_id, schema, pattern, limit) job 목록 생성.
    --schema 인자로 특정 schema만 필터링 가능.
    """
    jobs = []
    for schema_id, schema in SCHEMAS.items():
        if schema_filter and schema_id != schema_filter:
            continue
        for pattern in schema["patterns"]:
            jobs.append((schema_id, schema, pattern, schema["limit"]))
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   default="config.yaml")
    parser.add_argument("--dry-run",  action="store_true",
                        help="각 쿼리 결과 첫 3개만 출력하고 종료")
    parser.add_argument("--schema",   default=None,
                        help="특정 schema만 실행 (예: film_director_country)")
    parser.add_argument("--append",   action="store_true",
                        help="기존 출력 파일에 있는 query_type은 건너뛰고 새 결과만 추가")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    wd_cfg = cfg["wikidata"]

    sparql = WikidataSPARQL(
        endpoint=wd_cfg["endpoint"],
        user_agent=wd_cfg["user_agent"],
        timeout=wd_cfg["timeout"],
        sleep=wd_cfg["sleep_between_requests"],
        max_retries=wd_cfg["max_retries"],
    )

    output_path = Path(cfg["output"]["raw_paths"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # --append: 기존 파일의 query_type 읽어서 중복 제거
    existing_query_types: set[str] = set()
    if args.append and output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                qt = json.loads(line).get("query_type", "")
                if qt:
                    existing_query_types.add(qt)
        logger.info(f"기존 파일에서 {len(existing_query_types)}개 query_type 감지 → 건너뜀")

    query_jobs = build_query_jobs(args.schema)
    if existing_query_types:
        before = len(query_jobs)
        query_jobs = [
            j for j in query_jobs
            if f"{j[0]}_{j[2]}" not in existing_query_types
        ]
        logger.info(f"job 필터링: {before}개 → {len(query_jobs)}개 (신규만 실행)")

    logger.info(f"총 {len(query_jobs)}개 쿼리 job 실행")

    all_paths = []

    for schema_id, schema, pattern, limit in tqdm(query_jobs, desc="SPARQL 쿼리"):
        query_type = f"{schema_id}_{pattern}"
        logger.info(f"쿼리: {query_type} (domain={schema['domain']}, limit={limit})")

        sparql_str = build_query(schema, pattern, limit)
        raw_rows   = sparql.query(sparql_str)
        rows       = [_bind_to_dict(r, ROW_KEYS) for r in raw_rows]
        logger.info(f"  → {len(rows)}개 결과")

        if args.dry_run:
            print(f"\n=== {query_type} (처음 3개) ===")
            for r in rows[:3]:
                print(json.dumps(r, ensure_ascii=False, indent=2))
            continue

        paths = rows_to_path(rows, pattern, query_type, schema["path"])
        all_paths.extend(paths)
        logger.info(f"  → {len(paths)}개 경로 변환")

    if args.dry_run:
        print(f"\n[dry-run] 총 job: {len(query_jobs)}개")
        return

    # ── 저장 ─────────────────────────────────────────────────────────────────
    write_mode = "a" if args.append else "w"
    with open(output_path, write_mode, encoding="utf-8") as f:
        for p in all_paths:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    # ── 통계 ─────────────────────────────────────────────────────────────────
    pat_labels = {"a": "KO→KO→KO", "b": "KO→EN→KO", "c": "KO→KO→EN", "d": "KO→EN→EN"}
    pat_counts = Counter(p["pattern"] for p in all_paths)
    domain_counts = Counter(
        SCHEMAS[p["query_type"].rsplit("_", 1)[0]]["domain"]
        for p in all_paths
        if p["query_type"].rsplit("_", 1)[0] in SCHEMAS
    )

    print("\n" + "=" * 60)
    print(f"총 경로 수: {len(all_paths):,}")
    print("\n패턴별:")
    for pat, cnt in sorted(pat_counts.items()):
        print(f"  패턴 {pat} ({pat_labels.get(pat, '?')}): {cnt:,}개")
    print("\n도메인별:")
    for dom, cnt in domain_counts.most_common():
        print(f"  {dom}: {cnt:,}개")
    print(f"\n저장 완료: {output_path}")


if __name__ == "__main__":
    main()
