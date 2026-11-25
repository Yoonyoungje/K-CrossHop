"""
utils/sparql.py
Wikidata SPARQL 쿼리 유틸리티
- 멀티홉 KO-EN 경로 추출용 쿼리
- Rate-limit-aware wrapper
"""

import time
import logging
from urllib.parse import unquote
from SPARQLWrapper import SPARQLWrapper, JSON

logger = logging.getLogger(__name__)


def extract_wiki_title(url_str: str) -> str:
    """Wikidata Wikipedia URL에서 정확한 문서 타이틀 추출. URL 인코딩 디코딩 포함."""
    if not url_str:
        return ""
    return unquote(url_str.split("/wiki/")[-1])


class WikidataSPARQL:
    """Wikidata SPARQL 엔드포인트 wrapper."""

    def __init__(self, endpoint: str, user_agent: str, timeout: int = 60,
                 sleep: float = 1.0, max_retries: int = 3):
        self.endpoint = endpoint
        self.sleep = sleep
        self.max_retries = max_retries

        self.sparql = SPARQLWrapper(endpoint, agent=user_agent)
        self.sparql.setTimeout(timeout)
        self.sparql.setReturnFormat(JSON)

    def query(self, sparql_str: str) -> list[dict]:
        """SPARQL 쿼리를 실행하고 결과를 반환. 재시도 포함."""
        for attempt in range(self.max_retries):
            try:
                self.sparql.setQuery(sparql_str)
                results = self.sparql.query().convert()
                time.sleep(self.sleep)
                return results["results"]["bindings"]
            except Exception as e:
                error_msg = str(e)
                logger.error(f"SPARQL Error Details: {error_msg[:200]}")

                # 429 에러가 포함되어 있거나 HTTP Error 429인 경우
                if "429" in error_msg:
                    wait = 65  # 안전하게 65초 대기
                    logger.warning(f"서버 Rate Limit (429) 감지. {wait}초 동안 대기합니다...")
                else:
                    wait = 2 ** attempt * 3

                time.sleep(wait)
        logger.error("최대 재시도 횟수 초과. 빈 결과 반환.")
        return []


# ─── 쿼리 템플릿 ────────────────────────────────────────────────────────────


def query_ko_person_employer_region(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (a) 기준선: 한국인(KO) →[P108 고용주]→ 한국 기관(KO) →[P131]→ 행정구역(KO)

    gold_lang 할당: hop-0 "ko", hop-1 "ko"

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  고용주(P108), KO Wikipedia
      obj:  행정구역(P131), KO Wikipedia

    QA 예시: "홍길동이 근무한 기관이 속한 행정구역은?"
    수율: distinct subj 2,828개 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 고용주 (P108, KO Wikipedia)
  ?subj wdt:P108 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 3. 행정구역 (P131, KO Wikipedia)
  ?mid wdt:P131 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}
  OPTIONAL {{ ?mid  rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid  rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_sister_city_region(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (c): 한국 도시(KO) →[P190 자매결연]→ 해외 도시(EN) →[P131]→ 상위 행정구역(EN)
    FILTER→MINUS 수정 완료 (QLever IRI 비교 버그 — MINUS { FILTER(?x = wd:QID) } 원칙 적용)

    gold_lang 할당: hop-0 "ko", hop-1 "en"

    설계 변경 이유:
      기존에는 mid(해외 도시)에 KO+EN Wikipedia 동시 요구 → 수율 극도로 제한.
      gold_lang을 파이프라인에서 명시 할당하므로 sitelink 조건은 패시지 수집
      가능 여부(EN Wikipedia 존재)만 확인. KO Wikipedia 조건 제거.

    엔티티 조건:
      subj: 한국 소재(P17=Q884), KO Wikipedia
      mid:  해외 자매결연 도시(P190, MINUS 한국), EN Wikipedia
      obj:  상위 행정구역(P131), EN Wikipedia

    QA 예시: "서울의 자매결연 해외 도시가 속한 행정구역은?"
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국에 위치함 (P17: South Korea), KO Wikipedia
  ?subj wdt:P17 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 자매결연 도시 (P190)
  ?subj wdt:P190 ?mid .

  # 3. 해외 자매결연 도시 — MINUS로 한국 제외
  MINUS {{ ?mid wdt:P17 wd:Q884 . }}

  # 4. 해외 도시 EN Wikipedia (KO 조건 제거 — gold_lang "ko"는 파이프라인 할당)
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 5. 상위 행정구역 (P131), EN Wikipedia
  ?mid wdt:P131 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_scholar_foreign_univ(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (d): 한국인(KO) → 해외 대학(EN) → 대학 소재지(EN)
    FILTER→MINUS 수정 완료 (QLever IRI 비교 버그 — MINUS { FILTER(?x = wd:QID) } 원칙 적용)
    VALUES로 영어권 국가(US/UK/CA/AU)를 고정 → 확실한 KO→EN→EN 경로.

    gold_lang 할당: hop-0 "en", hop-1 "en"

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  영어권 대학(P69, VALUES로 4개국), EN Wikipedia
      obj:  대학 소재 행정구역(P131), EN Wikipedia

    QA 예시: "김OO이 졸업한 해외 대학이 위치한 행정구역은?"
    수율: ~151개 (limit=300)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (P27=Q884, KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 해외 대학 (P69, 영어권 국가, EN Wikipedia)
  ?subj wdt:P69 ?mid .
  ?mid wdt:P17 ?country .
  VALUES ?country {{ wd:Q30 wd:Q145 wd:Q16 wd:Q408 }}
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 대학 소재 행정구역 (P131, EN Wikipedia)
  ?mid wdt:P131 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_athlete_global_event_kr(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (b): 한국인(KO) → 국제 대회(EN-dominant) → 한국 개최 도시(KO)
    mid의 KO Wikipedia 조건 없음 → rows_to_path의 mid_ko=False 처리와 일관성 유지.
    gold_lang="en"이 hop-1에 올바르게 적용되기 위한 필수 조건.

    gold_lang 할당: hop-0 "en", hop-1 "ko"

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  국제 대회(P1344), EN Wikipedia
      obj:  한국 내 개최 도시(P276, P17=Q884), KO Wikipedia

    QA 예시: "김OO이 참가한 국제 대회의 개최 도시는?"
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 국제 대회 참가 (P1344, EN Wikipedia만 요구)
  ?subj wdt:P1344 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 개최 장소가 한국 내 도시 (P276, P17=Q884, KO Wikipedia)
  ?mid wdt:P276 ?obj .
  ?obj wdt:P17 wd:Q884 .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_person_foreign_team(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (b): 한국인(KO) →[P54 소속팀]→ 해외 스포츠팀(EN) →[P17 국가]→ 국가(KO)

    gold_lang 할당: hop-0 "en", hop-1 "ko"

    설계 변경 이유:
      기존 athlete_event_kr(P1344 대회 참가)을 스포츠팀 소속 경로로 보완.
      mid에 KO Wikipedia MINUS 조건 없음 — gold_lang "en"은 파이프라인에서 할당.
      FILTER(?obj != wd:Q884)로 한국 국가대표 경로(자기 참조) 제외.

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  해외 스포츠팀(P54), EN Wikipedia
      obj:  국가(P17), KO Wikipedia, obj != Q884(한국 제외)

    QA 예시: "박지성이 소속했던 해외 팀의 국가는?"
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 소속 스포츠팀 (P54, EN Wikipedia — KO MINUS 없음)
  ?subj wdt:P54 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 팀 소속 국가 (P17, KO Wikipedia, 한국 제외 — FILTER != IRI QLever 버그 우회)
  ?mid wdt:P17 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .
  MINUS {{ ?mid wdt:P17 wd:Q884 . }}

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_person_foreign_univ_city(sparql: WikidataSPARQL, limit: int = 200) -> list[dict]:
    """
    패턴 (b): 한국인(KO) →[P69 졸업 대학]→ 영어권 대학(EN) →[P131]→ 소재 행정구역(KO)

    gold_lang 할당: hop-0 "en", hop-1 "ko"

    설계 변경 이유:
      기존 query_scholar_foreign_univ(패턴 d, obj=EN)의 패턴 b 변형.
      mid(대학) EN Wikipedia만 요구, KO MINUS 없음 — gold_lang은 파이프라인 할당.
      obj(행정구역)는 KO Wikipedia 보유로 KO→EN→KO 경로를 구성.
      VALUES로 영어권 4개국(US/UK/CA/AU)을 고정하여 수율 확보 (최대 ~151개).

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  영어권 대학(P69, P17 IN {US,UK,CA,AU}), EN Wikipedia
      obj:  소재 행정구역(P131), KO Wikipedia
      FILTER(?mid != ?obj)

    QA 예시: "김연아가 훈련받은 해외 대학이 위치한 행정구역은?"
    수율: limit=200 (QLever 기준 ~151개 확인)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 영어권 대학 (P69, VALUES로 4개국 고정, EN Wikipedia)
  ?subj wdt:P69 ?mid .
  ?mid wdt:P17 ?country .
  VALUES ?country {{ wd:Q30 wd:Q145 wd:Q16 wd:Q408 }}
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 소재 행정구역 (P131, KO Wikipedia)
  ?mid wdt:P131 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .
  FILTER(?mid != ?obj)

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_person_award_org(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (c): 한국인(KO) →[P166 수상]→ 수상(EN) →[P1027 수여기관]→ 수여 기관(EN)
    FILTER→MINUS 수정 완료 (QLever IRI 비교 버그 — MINUS { FILTER(?x = wd:QID) } 원칙 적용)

    gold_lang 할당: hop-0 "ko", hop-1 "en"

    설계 변경 이유:
      기존에는 mid(수상)에 KO+EN Wikipedia 동시 요구 → 수율 제한(132개).
      gold_lang을 파이프라인에서 명시 할당하므로 mid는 EN Wikipedia만 요구.
      KO Wikipedia 조건 제거로 수율 증가 기대. midLabel도 "en"으로 변경.

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  수상(P166), EN Wikipedia (KO 조건 제거)
      obj:  수여 기관(P1027), EN Wikipedia

    QA 예시: "봉준호 감독이 받은 상의 수여 기관은?"
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 수상 이력 (P166, EN Wikipedia — KO 조건 제거, gold_lang은 파이프라인 할당)
  ?subj wdt:P166 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 수여 기관 (P1027, EN Wikipedia)
  ?mid wdt:P1027 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_person_intl_award(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (d): 한국인(KO) → 수상 이력(EN-only) → 수여 기관(EN)
    FILTER→MINUS 수정 완료 (QLever IRI 비교 버그 — MINUS { FILTER(?x = wd:QID) } 원칙 적용)
    query_person_award_org에서 mid의 KO Wikipedia 조건 제거 → hop-1 EN-dominant.

    gold_lang 할당: hop-0 "en", hop-1 "en"

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  수상(P166), EN Wikipedia (bridge 아님)
      obj:  수여 기관(P1027), EN Wikipedia

    QA 예시: "한국인이 받은 국제 시상의 수여 기관은?"
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 수상 이력 (P166, EN Wikipedia만 요구 — bridge 아님)
  ?subj wdt:P166 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 수여 기관 (P1027, EN Wikipedia)
  ?mid wdt:P1027 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


# ─── 신규 쿼리 (패턴 b) ─────────────────────────────────────────────────────


def query_ko_film_director_nationality(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (b): 한국 영화(KO) →[P57 감독]→ 감독(EN) →[P27 국적]→ 국가(KO, 비한국)

    gold_lang 할당: hop-0 "en", hop-1 "ko"

    엔티티 조건:
      subj: 한국 영화(P31=Q11424, P495=Q884), KO Wikipedia
      mid:  감독(P57), EN Wikipedia
      obj:  국적(P27), KO Wikipedia, MINUS { ?mid wdt:P27 wd:Q884 . }

    QA 예시: "이 한국 영화의 감독은 어느 나라 출신인가?"
    수율: ~86 (QLever 기준, MINUS 트리플 패턴 적용 후)
    주의: MINUS { FILTER(?x = wd:QID) }는 QLever에서 필터링 안 됨 — 트리플 패턴으로 대체
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국 영화 (KO Wikipedia)
  ?subj wdt:P31 wd:Q11424 ;
        wdt:P495 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 감독 (P57, EN Wikipedia)
  ?subj wdt:P57 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 국적 (P27, KO Wikipedia, 한국 제외)
  ?mid wdt:P27 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .
  MINUS {{ ?mid wdt:P27 wd:Q884 . }}

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_person_foreign_film_country(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (b): 한국인(KO) ←[P161]← 외국 영화(EN, 비한국) →[P495]→ 제작국(KO)

    gold_lang 할당: hop-0 "en", hop-1 "ko"

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  출연 외국 영화(P161 역방향, P31=Q11424), EN Wikipedia, MINUS { ?mid wdt:P495 wd:Q884 }
      obj:  제작국(P495), KO Wikipedia

    QA 예시: "이 한국인 배우가 출연한 외국 영화는 어느 나라에서 제작됐나?"
    수율: 389 dedup (limit=500 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 출연 외국 영화 (P161 역방향, EN Wikipedia, 한국 제외)
  ?mid wdt:P161 ?subj ;
       wdt:P31 wd:Q11424 .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .
  MINUS {{ ?mid wdt:P495 wd:Q884 . }}

  # 3. 제작국 (P495, KO Wikipedia)
  ?mid wdt:P495 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_person_intl_org_country(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (b): 한국인(KO) →[P463 소속기구]→ 국제기구(EN, 비한국 소재) →[P17]→ 기구 소재국(KO)

    gold_lang 할당: hop-0 "en", hop-1 "ko"

    엔티티 조건:
      subj: 한국인(P27=Q884, P31=Q5), KO Wikipedia
      mid:  국제기구(P463), EN Wikipedia, MINUS { ?mid wdt:P17 wd:Q884 }
      obj:  기구 소재국(P17), KO Wikipedia

    QA 예시: "이 한국인이 소속된 국제기구는 어느 나라에 있나?"
    수율: 122 dedup (limit=500 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국인 (KO Wikipedia)
  ?subj wdt:P27 wd:Q884 ;
        wdt:P31 wd:Q5 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 국제기구 (P463, EN Wikipedia, 한국 소재 제외)
  ?subj wdt:P463 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .
  MINUS {{ ?mid wdt:P17 wd:Q884 . }}

  # 3. 기구 소재국 (P17, KO Wikipedia)
  ?mid wdt:P17 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://ko.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj  rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj  rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


# ─── 신규 쿼리 (패턴 c) ─────────────────────────────────────────────────────


def query_ko_film_director_birthplace(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (c): 한국 영화(KO) →[P57 감독]→ 감독(KO) →[P19 출생지]→ 출생지(EN)

    gold_lang 할당: hop-0 "ko", hop-1 "en"

    엔티티 조건:
      subj: 한국 영화(P31=Q11424, P495=Q884), KO Wikipedia
      mid:  감독(P57), KO Wikipedia
      obj:  출생지(P19), EN Wikipedia

    QA 예시: "이 한국 영화의 감독이 태어난 곳은?"
    수율: 2,884 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국 영화 (KO Wikipedia)
  ?subj wdt:P31 wd:Q11424 ;
        wdt:P495 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 감독 (P57, KO Wikipedia)
  ?subj wdt:P57 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 3. 출생지 (P19, EN Wikipedia)
  ?mid wdt:P19 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}
  OPTIONAL {{ ?mid  rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid  rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_film_actor_birthplace(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (c): 한국 영화(KO) →[P161 출연 배우]→ 배우(KO) →[P19 출생지]→ 출생지(EN)

    gold_lang 할당: hop-0 "ko", hop-1 "en"

    엔티티 조건:
      subj: 한국 영화(P31=Q11424, P495=Q884), KO Wikipedia
      mid:  출연 배우(P161), KO Wikipedia
      obj:  출생지(P19), EN Wikipedia

    QA 예시: "이 한국 영화의 주연 배우가 태어난 곳은?"
    수율: 3,075 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국 영화 (KO Wikipedia)
  ?subj wdt:P31 wd:Q11424 ;
        wdt:P495 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 출연 배우 (P161, KO Wikipedia)
  ?subj wdt:P161 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 3. 출생지 (P19, EN Wikipedia)
  ?mid wdt:P19 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}
  OPTIONAL {{ ?mid  rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid  rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_game_developer_hq(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (c): 한국 게임(KO) →[P178 개발사]→ 개발사(KO) →[P159 본사 소재지]→ 소재지(EN)

    gold_lang 할당: hop-0 "ko", hop-1 "en"

    엔티티 조건:
      subj: 비디오 게임(P31 VALUES), KO Wikipedia
            VALUES: Q7889/Q166080/Q175173/Q805166/Q20650540
            (P31/P279* 경로 대신 직접 타입 열거 — QLever 성능 개선)
      mid:  개발사(P178), KO Wikipedia
      obj:  본사 소재지(P159), EN Wikipedia

    라벨: subj/mid는 COALESCE(ko, en), obj는 COALESCE(en, ko)
    QA 예시: "이 한국 게임을 개발한 회사의 본사는 어디인가?"
    수율: 3,075 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 비디오 게임 (P31 VALUES, KO Wikipedia)
  ?subj wdt:P31 ?gameType .
  VALUES ?gameType {{ wd:Q7889 wd:Q166080 wd:Q175173 wd:Q805166 wd:Q20650540 }}
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 개발사 (P178, KO Wikipedia)
  ?subj wdt:P178 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 3. 본사 소재지 (P159, EN Wikipedia)
  ?mid wdt:P159 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


# ─── 신규 쿼리 (패턴 d) ─────────────────────────────────────────────────────


def query_ko_film_actor_birthplace_en(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (d): 한국 영화(KO) →[P161 출연 배우]→ 배우(EN) →[P19 출생지]→ 출생지(EN)

    gold_lang 할당: hop-0 "en", hop-1 "en"

    엔티티 조건:
      subj: 한국 영화(P31/P279* Q11424, P495=Q884), KO Wikipedia
      mid:  출연 배우(P161), EN Wikipedia (KO MINUS 없음)
      obj:  출생지(P19), EN Wikipedia

    QA 예시: "이 한국 영화의 외국인 배우가 태어난 곳은?"
    수율: 1,885 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국 영화 (KO Wikipedia)
  ?subj wdt:P31/wdt:P279* wd:Q11424 ;
        wdt:P495 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 출연 배우 (P161, EN Wikipedia — KO MINUS 없음)
  ?subj wdt:P161 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 출생지 (P19, EN Wikipedia)
  ?mid wdt:P19 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_film_director_nationality_en(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (d): 한국 영화(KO) →[P57 감독]→ 감독(EN) →[P27 국적]→ 국가(EN)

    gold_lang 할당: hop-0 "en", hop-1 "en"

    엔티티 조건:
      subj: 한국 영화(P31=Q11424, P495=Q884), KO Wikipedia
      mid:  감독(P57), EN Wikipedia
      obj:  국적(P27), EN Wikipedia (한국 제외 조건 없음 — Python 필터링으로 처리)

    라벨: subj는 COALESCE(ko, en), mid/obj는 COALESCE(en, ko)
    QA 예시: "이 한국 영화의 감독은 어느 나라 출신인가?"
    수율: 2,242 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국 영화 (KO Wikipedia)
  ?subj wdt:P31 wd:Q11424 ;
        wdt:P495 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 감독 (P57, EN Wikipedia)
  ?subj wdt:P57 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 국적 (P27, EN Wikipedia)
  ?mid wdt:P27 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


def query_ko_software_language_designer(sparql: WikidataSPARQL, limit: int = 300) -> list[dict]:
    """
    패턴 (d): 한국 소프트웨어(KO) →[P277 프로그래밍 언어]→ 언어(EN) →[P287 설계자]→ 설계자(EN)

    gold_lang 할당: hop-0 "en", hop-1 "en"

    엔티티 조건:
      subj: 소프트웨어(P31/P279* Q7397), KO Wikipedia
      mid:  프로그래밍 언어(P277), EN Wikipedia
      obj:  설계자(P287), EN Wikipedia

    QA 예시: "이 소프트웨어에 사용된 프로그래밍 언어의 설계자는?"
    수율: 2,081 (QLever 기준)
    """
    q = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?subj ?wiki_subj ?subjLabelKo ?subjLabelEn
                ?mid  ?wiki_mid  ?midLabelKo  ?midLabelEn
                ?obj  ?wiki_obj  ?objLabelKo  ?objLabelEn WHERE {{
  # 1. 한국 소프트웨어 (P31/P279* Q7397, 한국 원산지 P495=Q884, KO Wikipedia)
  ?subj wdt:P31/wdt:P279* wd:Q7397 ;
        wdt:P495 wd:Q884 .
  ?wiki_subj schema:about ?subj ; schema:isPartOf <https://ko.wikipedia.org/> .

  # 2. 사용 프로그래밍 언어 (P277, EN Wikipedia)
  ?subj wdt:P277 ?mid .
  ?wiki_mid schema:about ?mid ; schema:isPartOf <https://en.wikipedia.org/> .

  # 3. 설계자 (P287, EN Wikipedia)
  ?mid wdt:P287 ?obj .
  ?wiki_obj schema:about ?obj ; schema:isPartOf <https://en.wikipedia.org/> .

  OPTIONAL {{ ?subj rdfs:label ?subjLabelKo . FILTER(LANG(?subjLabelKo) = "ko") }}
  OPTIONAL {{ ?subj rdfs:label ?subjLabelEn . FILTER(LANG(?subjLabelEn) = "en") }}

  OPTIONAL {{ ?mid rdfs:label ?midLabelKo . FILTER(LANG(?midLabelKo) = "ko") }}
  OPTIONAL {{ ?mid rdfs:label ?midLabelEn . FILTER(LANG(?midLabelEn) = "en") }}

  OPTIONAL {{ ?obj rdfs:label ?objLabelKo . FILTER(LANG(?objLabelKo) = "ko") }}
  OPTIONAL {{ ?obj rdfs:label ?objLabelEn . FILTER(LANG(?objLabelEn) = "en") }}
}}
LIMIT {limit}
"""
    rows = sparql.query(q)
    return [_bind_to_dict(r, ["subj", "wiki_subj", "subjLabelKo", "subjLabelEn",
                               "mid",  "wiki_mid",  "midLabelKo",  "midLabelEn",
                               "obj",  "wiki_obj",  "objLabelKo",  "objLabelEn"]) for r in rows]


# ─── 패턴별 쿼리 함수 목록 ───────────────────────────────────────────────────
# 패턴 (a): query_ko_person_employer_region
# 패턴 (b): query_ko_film_director_nationality * (MINUS 수정, 실수율 ~86)
#            query_ko_person_foreign_film_country *
#            query_ko_person_intl_org_country *
#            query_athlete_global_event_kr
#            query_ko_person_foreign_team
#            query_ko_person_foreign_univ_city
# 패턴 (c): query_ko_film_director_birthplace *
#            query_ko_film_actor_birthplace *
#            query_ko_game_developer_hq *
#            query_sister_city_region (수정)
#            query_person_award_org (수정)
# 패턴 (d): query_ko_film_actor_birthplace_en *
#            query_ko_film_director_nationality_en *
#            query_ko_software_language_designer *
#            query_scholar_foreign_univ (수정)
#            query_person_intl_award (수정)
# * 신규

# ─── helpers ────────────────────────────────────────────────────────────────

def _bind_to_dict(binding: dict, keys: list[str]) -> dict:
    """SPARQL 바인딩 row를 평탄한 dict로 변환."""
    return {k: binding[k]["value"] if k in binding else "" for k in keys}
