#!/usr/bin/env python3
"""
dataset/entity_cache.py

triple 구축 단계에서 쓰는 두 가지 조회 헬퍼.

  fetch_entity_cache() : QID 목록을 Wikidata API로 조회해 label/alias/sitelink 캐시를 만든다.
                         entity_index 는 일부 QID만 커버해 false negative 가 많으므로
                         필요한 QID 전체를 직접 재조회한다.
  build_passage_index(): passages.jsonl 을 스트리밍하며 필요한 QID 의 본문만 적재한다.

User-Agent 는 WIKI_USER_AGENT 환경변수로 지정한다 (Wikimedia 정책상 연락처 포함 권장).
"""

import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = os.getenv("WIKI_USER_AGENT", "K-CrossHop-Research/1.0 (research use)")
BATCH_SIZE = 50
RATE_DELAY = 1.0


def _fetch_batch(qids: list[str], props: str = "labels|aliases|sitelinks") -> dict:
    params = urllib.parse.urlencode({
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": props,
        "languages": "ko|en",
        "sitefilter": "kowiki|enwiki",
        "format": "json",
    })
    req = urllib.request.Request(f"{WIKIDATA_API}?{params}", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())["entities"]


def _parse_entity(ent: dict) -> dict:
    def label(lang):
        return ent.get("labels", {}).get(lang, {}).get("value", "")

    def aliases(lang):
        return [a["value"] for a in ent.get("aliases", {}).get(lang, [])]

    def wiki_title(site):
        return ent.get("sitelinks", {}).get(site, {}).get("title", "")

    return {
        "label_ko": label("ko"),
        "label_en": label("en"),
        "wiki_title_ko": wiki_title("kowiki"),
        "wiki_title_en": wiki_title("enwiki"),
        "aliases_ko": aliases("ko"),
        "aliases_en": aliases("en"),
    }


_EMPTY = {"label_ko": "", "label_en": "", "wiki_title_ko": "",
          "wiki_title_en": "", "aliases_ko": [], "aliases_en": []}


def fetch_entity_cache(qids: list[str], cache_path: Path) -> dict:
    """cache_path 에 없는 QID만 조회해 채우고 저장한 뒤 전체 캐시를 반환한다."""
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"기존 entity cache 로드: {len(cache)}개")

    missing = [q for q in qids if q not in cache]
    if missing:
        print(f"Wikidata API 조회: {len(missing)}개 QID ({(len(missing) - 1) // BATCH_SIZE + 1}배치)")
        for i in range(0, len(missing), BATCH_SIZE):
            batch = missing[i:i + BATCH_SIZE]
            print(f"  배치 {i // BATCH_SIZE + 1}: {len(batch)}개 ... ", end="", flush=True)
            try:
                entities = _fetch_batch(batch)
                for qid in batch:
                    ent = entities.get(qid, {})
                    cache[qid] = dict(_EMPTY) if "missing" in ent else _parse_entity(ent)
                print("완료")
            except Exception as e:
                print(f"실패: {e}")
            if i + BATCH_SIZE < len(missing):
                time.sleep(RATE_DELAY)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"entity cache 저장: {cache_path} ({len(cache)}개)")
    return cache


def fetch_instance_of(qids: list[str], cache_path: Path) -> dict[str, list[str]]:
    """QID → P31(instance of) 값 QID 목록. cache_path 에 없는 QID만 조회해 채우고 저장한다.
    조회 실패는 그대로 올린다 — 빈 목록으로 채우면 유형 판정이 조용히 틀린다."""
    cache: dict = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    missing = sorted(set(qids) - set(cache))
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        entities = _fetch_batch(batch, props="claims")
        for qid in batch:
            claims = entities[qid].get("claims", {}).get("P31", [])
            cache[qid] = sorted({c["mainsnak"]["datavalue"]["value"]["id"]
                                 for c in claims if c["mainsnak"].get("snaktype") == "value"})
        if i + BATCH_SIZE < len(missing):
            time.sleep(RATE_DELAY)
    if missing:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
        print(f"P31 조회 {len(missing)}개 → {cache_path} ({len(cache)}개)")
    return cache


def build_passage_index(needed_qids: set[str], passages_path: Path) -> dict[tuple, list[str]]:
    """(entity_qid, lang) -> [passage_text, ...]. 필요한 qid만 골라서 메모리에 적재."""
    index: dict[tuple, list[str]] = defaultdict(list)
    with passages_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            qid = row.get("entity_qid")
            if qid not in needed_qids:
                continue
            index[(qid, row.get("lang"))].append(row.get("text", ""))
    return index
