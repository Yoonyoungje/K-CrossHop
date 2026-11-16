#!/usr/bin/env python3
"""
pipeline/02_fetch_passages.py
=====================
wikidata_paths.jsonl의 각 엔티티에 대해
KO/EN Wikipedia 패시지를 수집한다.

입력:  data/raw/wikidata_paths.jsonl
출력:  data/raw/passages.jsonl  (엔티티별 패시지)
       data/raw/entity_index.json (QID → {ko_title, en_title, ...})

실행:
  python pipeline/02_fetch_passages.py [--config config.yaml] [--max-entities 50]
"""

import os
import json
import argparse
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional
import requests
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.validate import validate_text_coherence
from utils.wiki_api import WikiPassageFetcher
from wikipediaapi.exceptions import WikiRateLimitError

# 스레드별 fetcher 인스턴스 (Wikipedia API 세션은 스레드 비안전)
_thread_local = threading.local()


def _get_fetcher(wiki_cfg: dict) -> WikiPassageFetcher:
    if not hasattr(_thread_local, "fetcher"):
        _thread_local.fetcher = WikiPassageFetcher(
            chunk_size=wiki_cfg["chunk_size"],
            overlap=wiki_cfg["chunk_overlap"],
            min_len=wiki_cfg["min_passage_len"],
        )
    return _thread_local.fetcher



def _fetch_entity(args: tuple) -> tuple:
    """단일 엔티티의 KO/EN 패시지를 수집. ThreadPoolExecutor worker.

    반환: (qid, ko_passages, en_passages, failures)
      failures: 수집 실패 항목 목록 [{qid, lang, title, reason, error?}]
        reason: "not_found" | "rate_limited" | "error"
    """
    qid, meta, wiki_cfg = args
    fetcher = _get_fetcher(wiki_cfg)
    ko_passages, en_passages = [], []
    failures = []

    for lang, use_wiki, label, title_key in [
        ("ko", meta["wiki_ko"], meta["label_ko"], "title_ko"),
        ("en", meta["wiki_en"], meta["label_en"], "title_en"),
    ]:
        if not use_wiki:
            continue
        title = meta.get(title_key) or ""
        if not title:
            continue
        try:
            passages, status = fetcher.fetch_passages(qid, label, title, lang)
            if lang == "ko":
                ko_passages = passages
            else:
                en_passages = passages
            if status == "not_found":
                failures.append({"qid": qid, "lang": lang, "title": title, "reason": "not_found"})
        except WikiRateLimitError as e:
            logger.warning(f"Rate limit 소진 ({qid}/{lang}): {e}")
            failures.append({"qid": qid, "lang": lang, "title": title,
                             "reason": "rate_limited", "error": str(e)[:300]})
        except Exception as e:
            logger.warning(f"패시지 수집 실패 ({qid}/{lang}): {e}")
            failures.append({"qid": qid, "lang": lang, "title": title,
                             "reason": "error", "error": str(e)[:300]})

    return qid, ko_passages, en_passages, failures

def _setup_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"02_fetch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file, encoding="utf-8"),
                        ])
    return log_file

logger = logging.getLogger(__name__)


def load_unique_entities(paths_file: Path) -> dict[str, dict]:
    """
    paths.jsonl에서 유일한 QID 목록과 레이블/타이틀 정보를 추출한다.
    wiki_title(SPARQL sitelink URL에서 추출)을 gold_lang에 따라 title_ko/title_en에 저장.
    반환: {qid: {label_ko, label_en, wiki_ko, wiki_en, primary_lang, title_ko, title_en}}
    """
    entities: dict[str, dict] = {}
    with open(paths_file, encoding="utf-8") as f:
        for line in f:
            path = json.loads(line)
            for hop in path["hops"]:
                qid = hop["qid"]
                if qid not in entities:
                    entities[qid] = {
                        "label_ko": hop["label_ko"],
                        "label_en": hop["label_en"],
                        "wiki_ko":  hop["wiki_ko"],
                        "wiki_en":  hop["wiki_en"],
                        "primary_lang": hop["primary_lang"],
                        "title_ko": None,
                        "title_en": None,
                    }
                else:
                    # 패턴마다 wiki 플래그가 다르게 설정되므로 OR로 병합
                    entities[qid]["wiki_ko"] = entities[qid]["wiki_ko"] or hop["wiki_ko"]
                    entities[qid]["wiki_en"] = entities[qid]["wiki_en"] or hop["wiki_en"]
                # wiki_title을 gold_lang에 따라 해당 언어 title에 저장 (최초 등장 기준)
                wiki_title = hop.get("wiki_title", "")
                if wiki_title:
                    gold_lang = hop.get("gold_lang", "ko")
                    meta = entities[qid]
                    if gold_lang == "ko" and not meta["title_ko"]:
                        meta["title_ko"] = wiki_title
                    elif gold_lang == "en" and not meta["title_en"]:
                        meta["title_en"] = wiki_title
    return entities


def resolve_missing_titles(entities: dict[str, dict], batch_size: int = 50) -> int:
    """
    wiki_{lang}=True이지만 title_{lang}이 없는 엔티티의 Wikipedia 제목을
    Wikidata API(wbgetentities sitelinks)로 조회해 채운다. ko/en 모두 처리.

    반환: 해결된 (엔티티, 언어) 쌍 수
    """
    resolved_total = 0

    for lang, site in [("ko", "kowiki"), ("en", "enwiki")]:
        title_key = f"title_{lang}"
        need = [qid for qid, m in entities.items()
                if m.get(f"wiki_{lang}") and not m.get(title_key)]
        if not need:
            continue

        logger.info(f"Wikidata API로 {title_key} 보완: {len(need)}개 엔티티")
        resolved = 0

        for i in range(0, len(need), batch_size):
            batch = need[i:i + batch_size]
            try:
                resp = requests.get(
                    "https://www.wikidata.org/w/api.php",
                    params={
                        "action": "wbgetentities",
                        "ids": "|".join(batch),
                        "props": "sitelinks",
                        "sitefilter": site,
                        "format": "json",
                    },
                    headers={"User-Agent": os.getenv("WIKI_USER_AGENT", "K-CrossHop-Research/1.0 (research use)")},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                logger.warning(f"Wikidata API 배치 실패 ({batch[0]}…): {e}")
                time.sleep(2)
                continue

            for qid in batch:
                entity_data = data.get("entities", {}).get(qid, {})
                title = entity_data.get("sitelinks", {}).get(site, {}).get("title")
                if title:
                    entities[qid][title_key] = title
                    resolved += 1

            time.sleep(0.5)  # Wikidata API rate limit 준수

        logger.info(f"  → {title_key} 보완 완료: {resolved}/{len(need)}개")
        resolved_total += resolved

    return resolved_total


def validate_and_filter_paths(
    paths_file: Path,
    entity_texts: dict[str, str],
    entity_passage_counts: dict[str, int],
    output_file: Path,
    validate_coherence: bool = False,
) -> dict:
    """
    수집된 패시지 기반으로 경로를 검증하고 유효한 경로만 저장한다.

    조건 A: 각 홉 엔티티가 패시지를 1개 이상 보유
    조건 B: 각 홉의 다음 홉 엔티티 레이블이 현재 홉 패시지에 등장 (validate_coherence=True일 때)

    반환: {total, passed, failed, fail_reasons}
    """
    paths = [json.loads(l) for l in paths_file.read_text().splitlines() if l.strip()]
    passed, failed = [], []
    fail_reasons: dict[str, list[str]] = {}

    for path in paths:
        pid = path["path_id"]
        reasons = []

        for i, hop in enumerate(path["hops"]):
            qid  = hop["qid"]
            lang = "ko" if hop["primary_lang"] in ("ko", "both") else "en"
            key  = f"{qid}_{lang}"

            # 조건 A: 패시지 존재
            if entity_passage_counts.get(key, 0) == 0:
                reasons.append(f"hop{i}({qid}): no passages [{lang}]")

        # 조건 B: 텍스트 정합성 (조건 A 통과한 경로에만 적용)
        if validate_coherence and not reasons:
            result = validate_text_coherence(path, entity_texts)
            if not result["pass"]:
                reasons.append(
                    f"condition_b: hop0={result['hop0_pass']}, hop1={result['hop1_pass']}"
                )

        if reasons:
            failed.append(path)
            fail_reasons[pid] = reasons
        else:
            passed.append(path)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for p in passed:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    return {
        "total":   len(paths),
        "passed":  len(passed),
        "failed":  len(failed),
        "fail_reasons": fail_reasons,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--max-entities", type=int, default=None,
                        help="테스트용: 처음 N개 엔티티만 처리")
    parser.add_argument("--validate-coverage", action="store_true",
                        help="패시지 수집 후 경로 커버리지 검증 및 필터링")
    parser.add_argument("--validate-coherence", action="store_true",
                        help="조건 B: 텍스트 정합성 검증 (--validate-coverage와 함께 사용)")
    parser.add_argument("--paths",  default=None,
                        help="입력 경로 파일 (미지정 시 config output.raw_paths 사용)")
    parser.add_argument("--output", default=None,
                        help="출력 패시지 파일 (미지정 시 config output.passages 사용)")
    parser.add_argument("--workers", type=int, default=4,
                        help="병렬 수집 스레드 수 (기본: 4, 429 방지)")
    parser.add_argument("--append", action="store_true",
                        help="기존 passages.jsonl에 없는 신규 엔티티만 수집 후 추가")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    wiki_cfg = cfg["wikipedia"]

    log_file = _setup_logging(Path("logs"))
    logger.info(f"로그 파일: {log_file}")

    paths_file = Path(args.paths or cfg["output"]["raw_paths"])
    if not paths_file.exists():
        logger.error(f"경로 파일 없음: {paths_file} → pipeline/01_extract_paths.py 먼저 실행하세요.")
        return

    passages_file = Path(args.output or cfg["output"]["passages"])
    passages_file.parent.mkdir(parents=True, exist_ok=True)
    entity_index_file = passages_file.parent / "entity_index.json"

    # ── 유일 엔티티 목록 로드 ─────────────────────────────────────
    entities = load_unique_entities(paths_file)
    logger.info(f"유일 엔티티: {len(entities)}개")

    # --append: 이미 수집된 엔티티 제외
    if args.append and passages_file.exists():
        existing_qids: set[str] = set()
        with open(passages_file, encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                existing_qids.add(p["entity_qid"])
        before = len(entities)
        entities = {qid: meta for qid, meta in entities.items() if qid not in existing_qids}
        logger.info(f"--append: 기존 {len(existing_qids)}개 QID 제외 → 신규 {len(entities)}개 / 전체 {before}개")

    if args.max_entities:
        entities = dict(list(entities.items())[:args.max_entities])
        logger.info(f"  → 처음 {args.max_entities}개만 처리")

    # title_en 누락 엔티티를 Wikidata API로 보완
    resolve_missing_titles(entities)

    # entity_index 저장
    with open(entity_index_file, "w", encoding="utf-8") as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    logger.info(f"엔티티 인덱스 저장: {entity_index_file}")

    # ── 패시지 수집 (병렬) ────────────────────────────────────────
    total_passages = 0
    entity_texts:          dict[str, str] = {}   # "{qid}_{lang}" → 전체 텍스트 (커버리지 검증용)
    entity_passage_counts: dict[str, int] = {}   # "{qid}_{lang}" → 패시지 수

    entity_args = [(qid, meta, wiki_cfg) for qid, meta in entities.items()]
    fetch_results: dict[str, tuple] = {}  # qid → (ko_passages, en_passages)
    all_failures: list[dict] = []

    logger.info(f"패시지 수집 시작 (workers={args.workers})")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_entity, arg): arg[0] for arg in entity_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc="패시지 수집"):
            qid, ko_passages, en_passages, failures = future.result()
            fetch_results[qid] = (ko_passages, en_passages)
            all_failures.extend(failures)

    # 원래 엔티티 순서대로 파일에 기록 + 통계 집계
    write_mode = "a" if args.append else "w"
    with open(passages_file, write_mode, encoding="utf-8") as out_f:
        for qid, meta in entities.items():
            ko_passages, en_passages = fetch_results.get(qid, ([], []))

            if meta["wiki_ko"]:
                for p in ko_passages:
                    out_f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
                total_passages += len(ko_passages)
                key = f"{qid}_ko"
                entity_passage_counts[key] = len(ko_passages)
                entity_texts[key] = " ".join(p.text for p in ko_passages)

            if meta["wiki_en"]:
                for p in en_passages:
                    out_f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
                total_passages += len(en_passages)
                key = f"{qid}_en"
                entity_passage_counts[key] = len(en_passages)
                entity_texts[key] = " ".join(p.text for p in en_passages)

    # passages_missing.jsonl 저장 (not_found / rate_limited / error 분리)
    missing_file = passages_file.parent / "passages_missing.jsonl"
    with open(missing_file, "w", encoding="utf-8") as f:
        for item in all_failures:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    not_found    = sum(1 for x in all_failures if x["reason"] == "not_found")
    rate_limited = sum(1 for x in all_failures if x["reason"] == "rate_limited")
    other_error  = sum(1 for x in all_failures if x["reason"] == "error")

    print("\n" + "="*50)
    print(f"수집된 총 패시지: {total_passages}개")
    print(f"\n수집 실패 내역 ({len(all_failures)}개 엔티티-언어 쌍):")
    print(f"  페이지 없음(not_found):  {not_found}개")
    print(f"  Rate limit(rate_limited): {rate_limited}개")
    print(f"  기타 오류(error):         {other_error}개")
    print(f"\n실패 목록 저장: {missing_file}")
    print(f"저장 완료: {passages_file}")

    # ── 커버리지 검증 ─────────────────────────────────────────────
    if args.validate_coverage:
        validated_file = passages_file.parent / "wikidata_paths_validated.jsonl"
        logger.info("경로 커버리지 검증 중...")
        stats = validate_and_filter_paths(
            paths_file,
            entity_texts,
            entity_passage_counts,
            validated_file,
            validate_coherence=args.validate_coherence,
        )
        print("\n" + "="*50)
        print("커버리지 검증 결과")
        print("="*50)
        print(f"  전체 경로: {stats['total']}개")
        print(f"  통과:      {stats['passed']}개 ({stats['passed']/stats['total']:.1%})")
        print(f"  제외:      {stats['failed']}개 ({stats['failed']/stats['total']:.1%})")
        print(f"\n  제외 이유 샘플 (최대 5개):")
        for pid, reasons in list(stats["fail_reasons"].items())[:5]:
            print(f"    {pid}: {reasons[0]}")
        print(f"\n검증된 경로 저장: {validated_file}")
        print("→ 이후 단계에서 wikidata_paths_validated.jsonl을 사용하세요.")


if __name__ == "__main__":
    main()
