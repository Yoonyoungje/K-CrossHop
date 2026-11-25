#!/usr/bin/env python3
"""
pipeline/03_generate_qa.py
==================
Wikidata 경로 + Wikipedia 패시지 → 멀티홉 QA 쌍 생성

전략:
  1단계(파일럿): 템플릿 기반 질문 생성 (빠르고 저비용)
  2단계(본실험): Anthropic API로 자연스러운 한국어 질문 생성

출력: data/processed/qa_pairs.jsonl
  각 줄: {
    "qid": "q00042",
    "question": "...",          # 한국어 질문
    "answer": "...",            # 최종 정답 레이블
    "hops": [
      {"hop": 0, "sub_q": "...", "gold_entity": "...", "gold_lang": "ko"},
      {"hop": 1, "sub_q": "...", "gold_entity": "...", "gold_lang": "en"},
      ...
    ],
    "pattern": "d",             # a/b/c/d
    "lang_sequence": ["ko","both","en"],
    "switch_at_hop": [1],
    "source_path_id": "film_00042",
    "gen_method": "template"    # "template" | "llm"
  }

실행:
  python pipeline/03_generate_qa.py [--config config.yaml] [--method template|llm]
"""

import json
import random
import argparse
import logging
import re
import requests
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


_SWITCH_AT_HOP = {"a": [], "b": [1, 2], "c": [2], "d": [1]}

# ─── 한국어 조사 자동 선택 ────────────────────────────────────────────────────

_PARTICLE_MAP = {
    "과(와)": ("과", "와"),
    "이(가)": ("이", "가"),
    "을(를)": ("을", "를"),
    "은(는)": ("은", "는"),
}


def _has_jongseong(char: str) -> bool:
    code = ord(char)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    return False


def resolve_particles(text: str) -> str:
    for dual, (with_j, without_j) in _PARTICLE_MAP.items():
        idx = 0
        result = []
        while idx < len(text):
            pos = text.find(dual, idx)
            if pos == -1:
                result.append(text[idx:])
                break
            result.append(text[idx:pos])
            prev = text[pos - 1] if pos > 0 else ""
            result.append(with_j if (prev and _has_jongseong(prev)) else without_j)
            idx = pos + len(dual)
        text = "".join(result)
    return text


# ─── 템플릿 기반 질문 생성 ────────────────────────────────────────────────────

# (query_type, relation_pair) → (질문 템플릿, 서브질문 0, 서브질문 1, 서브질문 2)
# {0}=hop0 레이블, {1}=hop1 레이블, {2}=hop2 레이블
QA_TEMPLATES: dict[str, dict] = {
    # 패턴 a: KO→KO→KO
    "person_employer_ko": {
        "question":  "{0}이(가) 근무한 기관이 속한 행정구역은 어디인가?",
        "sub_q": [
            "{0}이(가) 근무한 기관은 어디인가?",
            "{1}이(가) 속한 행정구역은 어디인가?",
        ],
        "answer_hop": 2,
    },
    # 패턴 b: KO→EN→KO
    "athlete_event_kr": {
        "question":  "{0}이(가) 참가한 국제 대회의 개최 도시는 어디인가?",
        "sub_q": [
            "{0}이(가) 참가한 국제 대회는 무엇인가?",
            "{1}의 개최 도시는 어디인가?",
        ],
        "sub_q_en": [
            "What international event did {0} participate in?",
            "Where is the host city of {1}?",
        ],
        "answer_hop": 2,
    },
    # 패턴 c: KO→KO→EN
    "sister_city": {
        "question":  "{0}의 자매결연 해외 도시가 속한 행정구역은 어디인가?",
        "sub_q": [
            "{0}의 자매결연 해외 도시는 어디인가?",
            "{1}이(가) 속한 상위 행정구역은 어디인가?",
        ],
        "sub_q_en": [
            "What is the overseas sister city of {0}?",
            "What administrative region is {1} located in?",
        ],
        "answer_hop": 2,
    },
    # 패턴 d: KO→EN→EN
    "scholar_univ": {
        "question":  "{0}이(가) 졸업한 해외 대학이 위치한 행정구역은 어디인가?",
        "sub_q": [
            "{0}이(가) 졸업한 해외 대학은 어디인가?",
            "{1}이(가) 위치한 행정구역은 어디인가?",
        ],
        "sub_q_en": [
            "What foreign university did {0} graduate from?",
            "What region is {1} located in?",
        ],
        "answer_hop": 2,
    },
    "person_intl_award": {
        "question":  "{0}이(가) 받은 상의 수여 기관은 어디인가?",
        "sub_q": [
            "{0}이(가) 받은 국제 수상 이력은 무엇인가?",
            "{1}을(를) 수여하는 기관은 어디인가?",
        ],
        "sub_q_en": [
            "What international award did {0} receive?",
            "What organization awards {1}?",
        ],
        "answer_hop": 2,
    },
}

DEFAULT_TEMPLATE = {
    "question":  "{0}과(와) 관련된 {1}의 세부 정보는 무엇인가?",
    "sub_q": [
        "{0}과(와) 관련된 항목은 무엇인가?",
        "{1}에 대한 추가 정보는 무엇인가?",
    ],
    "answer_hop": 2,
}

# Y2: 단계별 지시 형태 템플릿 ({0}=subj 레이블)
QA_TEMPLATES_Y2: dict[str, str] = {
    "person_employer_ko": "{0}에 대해 검색하라. 그런 다음 {0}이(가) 근무한 기관을 찾아라. 마지막으로 그 기관이 속한 행정구역을 답하라.",
    "athlete_event_kr":   "{0}에 대해 검색하라. 그런 다음 {0}이(가) 참가한 국제 대회를 찾아라. 마지막으로 그 대회의 개최 도시를 답하라.",
    "foreign_team":       "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 해외 팀을 찾아라. 마지막으로 그 팀이 속한 국가를 답하라.",
    "foreign_univ_city":  "{0}에 대해 검색하라. 그런 다음 {0}이(가) 졸업한 해외 대학을 찾아라. 마지막으로 그 대학이 위치한 행정구역을 답하라.",
    "sister_city":        "{0}에 대해 검색하라. 그런 다음 {0}의 자매결연 해외 도시를 찾아라. 마지막으로 그 도시가 속한 행정구역을 답하라.",
    "person_award_org":   "{0}에 대해 검색하라. 그런 다음 {0}이(가) 받은 수상을 찾아라. 마지막으로 그 상의 수여 기관을 답하라.",
    "scholar_univ":       "{0}에 대해 검색하라. 그런 다음 {0}이(가) 졸업한 해외 대학을 찾아라. 마지막으로 그 대학이 위치한 행정구역을 답하라.",
    "person_intl_award":  "{0}에 대해 검색하라. 그런 다음 {0}이(가) 받은 국제 수상을 찾아라. 마지막으로 그 상의 수여 기관을 답하라.",
}
DEFAULT_TEMPLATE_Y2 = "{0}에 대해 검색하라. 그런 다음 {0}과(와) 관련된 항목을 찾아라. 마지막으로 그것의 세부 정보를 답하라."

# ── schema-factory 기반 템플릿 (schema_id 단위로 정의, pattern 자동 확장) ──────

_SCHEMA_BASE_TEMPLATES: dict[str, dict] = {
    # Biography / Person
    "person_employer_admin": {
        "question": "{0}이(가) 근무한 기관 {1}이(가) 속한 행정구역은 어디인가?",
        "sub_q": ["{0}이(가) 근무한 기관은 어디인가?", "{1}이(가) 속한 행정구역은 어디인가?"],
        "sub_q_en": ["Where did {0} work?", "What administrative region is {1} in?"],
        "answer_hop": 2,
    },
    "person_edu_admin": {
        "question": "{0}이(가) 졸업한 대학 {1}이(가) 위치한 행정구역은 어디인가?",
        "sub_q": ["{0}이(가) 졸업한 대학은 어디인가?", "{1}이(가) 위치한 행정구역은 어디인가?"],
        "sub_q_en": ["Which university did {0} graduate from?", "What administrative region is {1} in?"],
        "answer_hop": 2,
    },
    "person_affiliation_country": {
        "question": "{0}이(가) 소속된 단체 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}이(가) 소속된 단체는 무엇인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["What organization is {0} a member of?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "person_award_org": {
        "question": "{0}이(가) 받은 상 {1}의 시상기관은 어디인가?",
        "sub_q": ["{0}이(가) 받은 상은 무엇인가?", "{1}을(를) 수여하는 기관은 어디인가?"],
        "sub_q_en": ["What award did {0} receive?", "What organization awards {1}?"],
        "answer_hop": 2,
    },
    # Film
    "film_director_country": {
        "question": "{0}을(를) 감독한 {1}의 국적은 어디인가?",
        "sub_q": ["{0}을(를) 감독한 사람은 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who directed {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    "film_director_birthplace": {
        "question": "{0}을(를) 감독한 {1}의 출생지는 어디인가?",
        "sub_q": ["{0}을(를) 감독한 사람은 누구인가?", "{1}의 출생지는 어디인가?"],
        "sub_q_en": ["Who directed {0}?", "Where was {1} born?"],
        "answer_hop": 2,
    },
    "film_actor_country": {
        "question": "{0}에 출연한 배우 {1}의 국적은 어디인가?",
        "sub_q": ["{0}에 출연한 배우는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who appeared in {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    "film_actor_birthplace": {
        "question": "{0}에 출연한 배우 {1}의 출생지는 어디인가?",
        "sub_q": ["{0}에 출연한 배우는 누구인가?", "{1}의 출생지는 어디인가?"],
        "sub_q_en": ["Who appeared in {0}?", "Where was {1} born?"],
        "answer_hop": 2,
    },
    "film_production_company_country": {
        "question": "{0}의 제작사 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 제작사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which company produced {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "film_distributor_country": {
        "question": "{0}의 배급사 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 배급사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which distributor distributed {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    # Sports
    "athlete_team_country": {
        "question": "{0}이(가) 소속된 팀 {1}의 국가는 어디인가?",
        "sub_q": ["{0}이(가) 소속된 팀은 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["What team does {0} play for?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "team_league_country": {
        "question": "{0}이(가) 속한 리그 {1}의 국가는 어디인가?",
        "sub_q": ["{0}이(가) 속한 리그는 무엇인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["What league is {0} in?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "athlete_event_venue": {
        "question": "{0}이(가) 참가한 대회 {1}의 개최지는 어디인가?",
        "sub_q": ["{0}이(가) 참가한 대회는 무엇인가?", "{1}의 개최지는 어디인가?"],
        "sub_q_en": ["What event did {0} participate in?", "Where was {1} held?"],
        "answer_hop": 2,
    },
    # Organization / Company
    "org_hq_country": {
        "question": "{0}의 본사 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 본사는 어디에 있는가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Where is {0}'s headquarters located?", "What country is {1} in?"],
        "answer_hop": 2,
    },
    "company_founder_country": {
        "question": "{0}을(를) 설립한 {1}의 국적은 어디인가?",
        "sub_q": ["{0}의 설립자는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who founded {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    "company_parent_hq": {
        "question": "{0}의 모기업 {1}의 본사는 어디에 위치하는가?",
        "sub_q": ["{0}의 모기업은 어디인가?", "{1}의 본사는 어디인가?"],
        "sub_q_en": ["What is the parent company of {0}?", "Where is {1}'s headquarters located?"],
        "answer_hop": 2,
    },
    # Tech / Software
    "software_developer_hq": {
        "question": "{0}을(를) 개발한 {1}의 본사는 어디에 위치하는가?",
        "sub_q": ["{0}의 개발사는 어디인가?", "{1}의 본사는 어디인가?"],
        "sub_q_en": ["Which company developed {0}?", "Where is {1}'s headquarters located?"],
        "answer_hop": 2,
    },
    "game_developer_hq": {
        "question": "{0}을(를) 개발한 {1}의 본사는 어디에 위치하는가?",
        "sub_q": ["{0}의 개발사는 어디인가?", "{1}의 본사는 어디인가?"],
        "sub_q_en": ["Which company developed {0}?", "Where is {1}'s headquarters located?"],
        "answer_hop": 2,
    },
    "software_lang_designer": {
        "question": "{0}에 사용된 프로그래밍 언어 {1}의 설계자는 누구인가?",
        "sub_q": ["{0}에 사용된 프로그래밍 언어는 무엇인가?", "{1}의 설계자는 누구인가?"],
        "sub_q_en": ["What programming language is used in {0}?", "Who designed {1}?"],
        "answer_hop": 2,
    },
    "software_developer_country": {
        "question": "{0}을(를) 개발한 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 개발사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which company developed {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    # Literature
    "book_author_country": {
        "question": "{0}의 저자 {1}의 국적은 어디인가?",
        "sub_q": ["{0}의 저자는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who wrote {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    "book_publisher_country": {
        "question": "{0}의 출판사 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 출판사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which publisher published {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    # Music
    "album_record_label_country": {
        "question": "{0}의 레이블 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 레이블은 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which record label released {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "song_performer_country": {
        "question": "{0}을(를) 부른 아티스트 {1}의 국적은 어디인가?",
        "sub_q": ["{0}을(를) 부른 아티스트는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who performed {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    # Award
    "award_org_hq": {
        "question": "{0}의 시상기관 {1}의 본사는 어디에 위치하는가?",
        "sub_q": ["{0}의 시상기관은 어디인가?", "{1}의 본사는 어디인가?"],
        "sub_q_en": ["What organization awards {0}?", "Where is {1}'s headquarters located?"],
        "answer_hop": 2,
    },
    # Architecture / Infrastructure
    "building_architect_country": {
        "question": "{0}을(를) 설계한 건축가 {1}의 국적은 어디인가?",
        "sub_q": ["{0}을(를) 설계한 건축가는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who designed {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    "airport_operator_country": {
        "question": "{0}을(를) 운영하는 기관 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 운영기관은 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which organization operates {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    # Education
    "university_country_continent": {
        "question": "{0}이(가) 위치한 국가 {1}의 대륙은 어디인가?",
        "sub_q": ["{0}이(가) 위치한 국가는 어디인가?", "{1}이(가) 속한 대륙은 어디인가?"],
        "sub_q_en": ["What country is {0} located in?", "What continent is {1} on?"],
        "answer_hop": 2,
    },
    # Biography (Core additions)
    "person_employer_country": {
        "question": "{0}이(가) 근무한 기관 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}이(가) 근무한 기관은 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Where did {0} work?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "person_educated_country": {
        "question": "{0}이(가) 졸업한 대학 {1}이(가) 위치한 국가는 어디인가?",
        "sub_q": ["{0}이(가) 졸업한 대학은 어디인가?", "{1}이(가) 위치한 국가는 어디인가?"],
        "sub_q_en": ["Which university did {0} graduate from?", "What country is {1} in?"],
        "answer_hop": 2,
    },
    "person_position_jurisdiction": {
        "question": "{0}이(가) 맡은 직위 {1}의 관할 지역은 어디인가?",
        "sub_q": ["{0}이(가) 맡은 직위는 무엇인가?", "{1}의 관할 지역은 어디인가?"],
        "sub_q_en": ["What position did {0} hold?", "What jurisdiction does {1} cover?"],
        "answer_hop": 2,
    },
    # Film (Diversity additions)
    "film_original_country": {
        "question": "{0}의 제작국 {1}이(가) 속한 대륙은 어디인가?",
        "sub_q": ["{0}의 제작국은 어디인가?", "{1}이(가) 속한 대륙은 어디인가?"],
        "sub_q_en": ["What country produced {0}?", "What continent is {1} on?"],
        "answer_hop": 2,
    },
    # Sports (Core/Diversity additions)
    "athlete_event_country": {
        "question": "{0}이(가) 참가한 대회 {1}이(가) 개최된 국가는 어디인가?",
        "sub_q": ["{0}이(가) 참가한 대회는 무엇인가?", "{1}이(가) 개최된 국가는 어디인가?"],
        "sub_q_en": ["What event did {0} participate in?", "What country hosted {1}?"],
        "answer_hop": 2,
    },
    "athlete_team_league": {
        "question": "{0}이(가) 소속된 팀 {1}이(가) 참가하는 리그는 무엇인가?",
        "sub_q": ["{0}이(가) 소속된 팀은 어디인가?", "{1}이(가) 속한 리그는 무엇인가?"],
        "sub_q_en": ["What team does {0} play for?", "What league is {1} in?"],
        "answer_hop": 2,
    },
    "team_home_venue_city": {
        "question": "{0}의 홈 경기장 {1}이(가) 위치한 행정구역은 어디인가?",
        "sub_q": ["{0}의 홈 경기장은 어디인가?", "{1}이(가) 위치한 행정구역은 어디인가?"],
        "sub_q_en": ["What is {0}'s home venue?", "What administrative region is {1} in?"],
        "answer_hop": 2,
    },
    "event_venue_country": {
        "question": "{0}의 개최지 {1}이(가) 위치한 국가는 어디인가?",
        "sub_q": ["{0}의 개최지는 어디인가?", "{1}이(가) 위치한 국가는 어디인가?"],
        "sub_q_en": ["Where was {0} held?", "What country is {1} in?"],
        "answer_hop": 2,
    },
    # Organization (Core/Diversity additions)
    "org_hq_admin": {
        "question": "{0}의 본사 {1}이(가) 위치한 행정구역은 어디인가?",
        "sub_q": ["{0}의 본사는 어디에 있는가?", "{1}이(가) 위치한 행정구역은 어디인가?"],
        "sub_q_en": ["Where is {0}'s headquarters?", "What administrative region is {1} in?"],
        "answer_hop": 2,
    },
    "org_member_country": {
        "question": "{0}이(가) 소속된 단체 {1}이(가) 위치한 국가는 어디인가?",
        "sub_q": ["{0}이(가) 소속된 상위 단체는 무엇인가?", "{1}이(가) 위치한 국가는 어디인가?"],
        "sub_q_en": ["What organization is {0} a member of?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "company_founder_birthplace": {
        "question": "{0}의 설립자 {1}이(가) 태어난 곳은 어디인가?",
        "sub_q": ["{0}의 설립자는 누구인가?", "{1}은(는) 어디서 태어났는가?"],
        "sub_q_en": ["Who founded {0}?", "Where was {1} born?"],
        "answer_hop": 2,
    },
    "newspaper_publisher_country": {
        "question": "{0}을(를) 발행하는 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 발행사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Who publishes {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    # Tech (Diversity additions)
    "game_publisher_country": {
        "question": "{0}을(를) 배급한 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 배급사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which company published {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "programming_language_designer_birthplace": {
        "question": "{0}을(를) 설계한 {1}이(가) 태어난 곳은 어디인가?",
        "sub_q": ["{0}을(를) 설계한 사람은 누구인가?", "{1}은(는) 어디서 태어났는가?"],
        "sub_q_en": ["Who designed {0}?", "Where was {1} born?"],
        "answer_hop": 2,
    },
    "programming_language_designer_country": {
        "question": "{0}을(를) 설계한 {1}의 국적은 어디인가?",
        "sub_q": ["{0}을(를) 설계한 사람은 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who designed {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    "product_manufacturer_country": {
        "question": "{0}을(를) 제조한 {1}이(가) 속한 국가는 어디인가?",
        "sub_q": ["{0}의 제조사는 어디인가?", "{1}이(가) 속한 국가는 어디인가?"],
        "sub_q_en": ["Which company manufactured {0}?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "invention_inventor_country": {
        "question": "{0}을(를) 발명한 {1}의 국적은 어디인가?",
        "sub_q": ["{0}을(를) 발명한 사람은 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who invented {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    # Literature (Diversity addition)
    "book_author_birthplace": {
        "question": "{0}의 저자 {1}이(가) 태어난 곳은 어디인가?",
        "sub_q": ["{0}의 저자는 누구인가?", "{1}은(는) 어디서 태어났는가?"],
        "sub_q_en": ["Who wrote {0}?", "Where was {1} born?"],
        "answer_hop": 2,
    },
    # Music (Diversity addition)
    "album_performer_country": {
        "question": "{0}을(를) 발매한 아티스트 {1}의 국적은 어디인가?",
        "sub_q": ["{0}을(를) 발매한 아티스트는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who released {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    # Politics (Diversity additions)
    "politician_party_country": {
        "question": "{0}이(가) 소속된 정당 {1}이(가) 위치한 국가는 어디인가?",
        "sub_q": ["{0}이(가) 소속된 정당은 무엇인가?", "{1}이(가) 위치한 국가는 어디인가?"],
        "sub_q_en": ["What party does {0} belong to?", "What country is {1} based in?"],
        "answer_hop": 2,
    },
    "party_hq_country": {
        "question": "{0}의 본부 {1}이(가) 위치한 국가는 어디인가?",
        "sub_q": ["{0}의 본부는 어디에 있는가?", "{1}이(가) 위치한 국가는 어디인가?"],
        "sub_q_en": ["Where is {0}'s headquarters?", "What country is {1} in?"],
        "answer_hop": 2,
    },
    # Education (Diversity addition)
    "university_founder_country": {
        "question": "{0}을(를) 설립한 {1}의 국적은 어디인가?",
        "sub_q": ["{0}의 설립자는 누구인가?", "{1}의 국적은 어디인가?"],
        "sub_q_en": ["Who founded {0}?", "What is the nationality of {1}?"],
        "answer_hop": 2,
    },
    # ── 추가 스키마 (schema_registry 등록분) ──────────────────────────
    "film_director_award": {
        "question": "{0}을(를) 감독한 {1}이(가) 받은 상은 무엇인가?",
        "sub_q": ["{0}을(를) 감독한 사람은 누구인가?", "{1}이(가) 받은 상은 무엇인가?"],
        "sub_q_en": ["Who directed {0}?", "What award did {1} receive?"],
        "answer_hop": 2,
    },
    "film_director_educated_at": {
        "question": "{0}을(를) 감독한 {1}이(가) 졸업한 학교는 어디인가?",
        "sub_q": ["{0}을(를) 감독한 사람은 누구인가?", "{1}이(가) 졸업한 학교는 어디인가?"],
        "sub_q_en": ["Who directed {0}?", "Which university did {1} graduate from?"],
        "answer_hop": 2,
    },
    "film_actor_award": {
        "question": "{0}에 출연한 배우 {1}이(가) 받은 상은 무엇인가?",
        "sub_q": ["{0}에 출연한 배우는 누구인가?", "{1}이(가) 받은 상은 무엇인가?"],
        "sub_q_en": ["Who appeared in {0}?", "What award did {1} receive?"],
        "answer_hop": 2,
    },
    "athlete_team_league_domestic": {
        "question": "{0}이(가) 소속된 팀 {1}이(가) 참가하는 리그는 무엇인가?",
        "sub_q": ["{0}이(가) 소속된 팀은 어디인가?", "{1}이(가) 속한 리그는 무엇인가?"],
        "sub_q_en": ["What team does {0} play for?", "What league is {1} in?"],
        "answer_hop": 2,
    },
    "company_founder_award": {
        "question": "{0}의 설립자 {1}이(가) 받은 상은 무엇인가?",
        "sub_q": ["{0}의 설립자는 누구인가?", "{1}이(가) 받은 상은 무엇인가?"],
        "sub_q_en": ["Who founded {0}?", "What award did {1} receive?"],
        "answer_hop": 2,
    },
    "company_founder_educated_at": {
        "question": "{0}의 설립자 {1}이(가) 졸업한 학교는 어디인가?",
        "sub_q": ["{0}의 설립자는 누구인가?", "{1}이(가) 졸업한 학교는 어디인가?"],
        "sub_q_en": ["Who founded {0}?", "Which university did {1} graduate from?"],
        "answer_hop": 2,
    },
    "book_author_award": {
        "question": "{0}의 저자 {1}이(가) 받은 상은 무엇인가?",
        "sub_q": ["{0}의 저자는 누구인가?", "{1}이(가) 받은 상은 무엇인가?"],
        "sub_q_en": ["Who wrote {0}?", "What award did {1} receive?"],
        "answer_hop": 2,
    },
    "album_performer_award": {
        "question": "{0}을(를) 발매한 아티스트 {1}이(가) 받은 상은 무엇인가?",
        "sub_q": ["{0}을(를) 발매한 아티스트는 누구인가?", "{1}이(가) 받은 상은 무엇인가?"],
        "sub_q_en": ["Who released {0}?", "What award did {1} receive?"],
        "answer_hop": 2,
    },
    "building_architect_award": {
        "question": "{0}을(를) 설계한 건축가 {1}이(가) 받은 상은 무엇인가?",
        "sub_q": ["{0}을(를) 설계한 건축가는 누구인가?", "{1}이(가) 받은 상은 무엇인가?"],
        "sub_q_en": ["Who designed {0}?", "What award did {1} receive?"],
        "answer_hop": 2,
    },
    "university_founder_educated_at": {
        "question": "{0}의 설립자 {1}이(가) 졸업한 학교는 어디인가?",
        "sub_q": ["{0}의 설립자는 누구인가?", "{1}이(가) 졸업한 학교는 어디인가?"],
        "sub_q_en": ["Who founded {0}?", "Which university did {1} graduate from?"],
        "answer_hop": 2,
    },
    "software_developer_founder": {
        "question": "{0}을(를) 개발한 {1}의 설립자는 누구인가?",
        "sub_q": ["{0}의 개발사는 어디인가?", "{1}의 설립자는 누구인가?"],
        "sub_q_en": ["Which company developed {0}?", "Who founded {1}?"],
        "answer_hop": 2,
    },
    "game_developer_founder": {
        "question": "{0}을(를) 개발한 {1}의 설립자는 누구인가?",
        "sub_q": ["{0}의 개발사는 어디인가?", "{1}의 설립자는 누구인가?"],
        "sub_q_en": ["Which company developed {0}?", "Who founded {1}?"],
        "answer_hop": 2,
    },
    "game_publisher_founder": {
        "question": "{0}을(를) 배급한 {1}의 설립자는 누구인가?",
        "sub_q": ["{0}의 배급사는 어디인가?", "{1}의 설립자는 누구인가?"],
        "sub_q_en": ["Which company published {0}?", "Who founded {1}?"],
        "answer_hop": 2,
    },
    "newspaper_publisher_founder": {
        "question": "{0}을(를) 발행하는 {1}의 설립자는 누구인가?",
        "sub_q": ["{0}의 발행사는 어디인가?", "{1}의 설립자는 누구인가?"],
        "sub_q_en": ["Who publishes {0}?", "Who founded {1}?"],
        "answer_hop": 2,
    },
    "product_manufacturer_founder": {
        "question": "{0}을(를) 제조한 {1}의 설립자는 누구인가?",
        "sub_q": ["{0}의 제조사는 어디인가?", "{1}의 설립자는 누구인가?"],
        "sub_q_en": ["Which company manufactured {0}?", "Who founded {1}?"],
        "answer_hop": 2,
    },
    "programming_language_designer_educated_at": {
        "question": "{0}을(를) 설계한 {1}이(가) 졸업한 학교는 어디인가?",
        "sub_q": ["{0}을(를) 설계한 사람은 누구인가?", "{1}이(가) 졸업한 학교는 어디인가?"],
        "sub_q_en": ["Who designed {0}?", "Which university did {1} graduate from?"],
        "answer_hop": 2,
    },
}

_SCHEMA_BASE_Y2: dict[str, str] = {
    "person_employer_admin":           "{0}에 대해 검색하라. 그런 다음 {0}이(가) 근무한 기관을 찾아라. 마지막으로 그 기관이 속한 행정구역을 답하라.",
    "person_edu_admin":                "{0}에 대해 검색하라. 그런 다음 {0}이(가) 졸업한 대학을 찾아라. 마지막으로 그 대학이 위치한 행정구역을 답하라.",
    "person_affiliation_country":      "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 단체를 찾아라. 마지막으로 그 단체가 속한 국가를 답하라.",
    "person_award_org":                "{0}에 대해 검색하라. 그런 다음 {0}이(가) 받은 상을 찾아라. 마지막으로 그 상의 시상기관을 답하라.",
    "film_director_country":           "{0}에 대해 검색하라. 그런 다음 {0}을(를) 감독한 사람을 찾아라. 마지막으로 그의 국적을 답하라.",
    "film_director_birthplace":        "{0}에 대해 검색하라. 그런 다음 {0}을(를) 감독한 사람을 찾아라. 마지막으로 그의 출생지를 답하라.",
    "film_actor_country":              "{0}에 대해 검색하라. 그런 다음 {0}에 출연한 배우를 찾아라. 마지막으로 그의 국적을 답하라.",
    "film_actor_birthplace":           "{0}에 대해 검색하라. 그런 다음 {0}에 출연한 배우를 찾아라. 마지막으로 그의 출생지를 답하라.",
    "film_production_company_country": "{0}에 대해 검색하라. 그런 다음 {0}의 제작사를 찾아라. 마지막으로 그 제작사가 속한 국가를 답하라.",
    "film_distributor_country":        "{0}에 대해 검색하라. 그런 다음 {0}의 배급사를 찾아라. 마지막으로 그 배급사가 속한 국가를 답하라.",
    "athlete_team_country":            "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 팀을 찾아라. 마지막으로 그 팀의 국가를 답하라.",
    "team_league_country":             "{0}에 대해 검색하라. 그런 다음 {0}이(가) 속한 리그를 찾아라. 마지막으로 그 리그의 국가를 답하라.",
    "athlete_event_venue":             "{0}에 대해 검색하라. 그런 다음 {0}이(가) 참가한 대회를 찾아라. 마지막으로 그 대회의 개최지를 답하라.",
    "org_hq_country":                  "{0}에 대해 검색하라. 그런 다음 {0}의 본사를 찾아라. 마지막으로 그 본사가 속한 국가를 답하라.",
    "company_founder_country":         "{0}에 대해 검색하라. 그런 다음 {0}의 설립자를 찾아라. 마지막으로 그의 국적을 답하라.",
    "company_parent_hq":               "{0}에 대해 검색하라. 그런 다음 {0}의 모기업을 찾아라. 마지막으로 그 모기업의 본사를 답하라.",
    "software_developer_hq":           "{0}에 대해 검색하라. 그런 다음 {0}의 개발사를 찾아라. 마지막으로 그 개발사의 본사를 답하라.",
    "game_developer_hq":               "{0}에 대해 검색하라. 그런 다음 {0}의 개발사를 찾아라. 마지막으로 그 개발사의 본사를 답하라.",
    "software_lang_designer":          "{0}에 대해 검색하라. 그런 다음 {0}에 사용된 프로그래밍 언어를 찾아라. 마지막으로 그 언어의 설계자를 답하라.",
    "software_developer_country":      "{0}에 대해 검색하라. 그런 다음 {0}의 개발사를 찾아라. 마지막으로 그 개발사가 속한 국가를 답하라.",
    "book_author_country":             "{0}에 대해 검색하라. 그런 다음 {0}의 저자를 찾아라. 마지막으로 그의 국적을 답하라.",
    "book_publisher_country":          "{0}에 대해 검색하라. 그런 다음 {0}의 출판사를 찾아라. 마지막으로 그 출판사가 속한 국가를 답하라.",
    "album_record_label_country":      "{0}에 대해 검색하라. 그런 다음 {0}의 레이블을 찾아라. 마지막으로 그 레이블이 속한 국가를 답하라.",
    "song_performer_country":          "{0}에 대해 검색하라. 그런 다음 {0}을(를) 부른 아티스트를 찾아라. 마지막으로 그의 국적을 답하라.",
    "award_org_hq":                    "{0}에 대해 검색하라. 그런 다음 {0}의 시상기관을 찾아라. 마지막으로 그 기관의 본사를 답하라.",
    "building_architect_country":      "{0}에 대해 검색하라. 그런 다음 {0}을(를) 설계한 건축가를 찾아라. 마지막으로 그의 국적을 답하라.",
    "airport_operator_country":        "{0}에 대해 검색하라. 그런 다음 {0}의 운영기관을 찾아라. 마지막으로 그 기관이 속한 국가를 답하라.",
    "university_country_continent":    "{0}에 대해 검색하라. 그런 다음 {0}이(가) 위치한 국가를 찾아라. 마지막으로 그 국가의 대륙을 답하라.",
    # Biography (Core additions)
    "person_employer_country":         "{0}에 대해 검색하라. 그런 다음 {0}이(가) 근무한 기관을 찾아라. 마지막으로 그 기관이 속한 국가를 답하라.",
    "person_educated_country":         "{0}에 대해 검색하라. 그런 다음 {0}이(가) 졸업한 대학을 찾아라. 마지막으로 그 대학이 위치한 국가를 답하라.",
    "person_position_jurisdiction":    "{0}에 대해 검색하라. 그런 다음 {0}이(가) 맡은 직위를 찾아라. 마지막으로 그 직위의 관할 지역을 답하라.",
    # Film (Diversity)
    "film_original_country":           "{0}에 대해 검색하라. 그런 다음 {0}의 제작국을 찾아라. 마지막으로 그 국가가 속한 대륙을 답하라.",
    # Sports (Core/Diversity)
    "athlete_event_country":           "{0}에 대해 검색하라. 그런 다음 {0}이(가) 참가한 대회를 찾아라. 마지막으로 그 대회가 개최된 국가를 답하라.",
    "athlete_team_league":             "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 팀을 찾아라. 마지막으로 그 팀이 속한 리그를 답하라.",
    "team_home_venue_city":            "{0}에 대해 검색하라. 그런 다음 {0}의 홈 경기장을 찾아라. 마지막으로 그 경기장이 위치한 행정구역을 답하라.",
    "event_venue_country":             "{0}에 대해 검색하라. 그런 다음 {0}의 개최지를 찾아라. 마지막으로 그 개최지가 위치한 국가를 답하라.",
    # Organization (Core/Diversity)
    "org_hq_admin":                    "{0}에 대해 검색하라. 그런 다음 {0}의 본사를 찾아라. 마지막으로 그 본사가 위치한 행정구역을 답하라.",
    "org_member_country":              "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 상위 단체를 찾아라. 마지막으로 그 단체가 위치한 국가를 답하라.",
    "company_founder_birthplace":      "{0}에 대해 검색하라. 그런 다음 {0}의 설립자를 찾아라. 마지막으로 그의 출생지를 답하라.",
    "newspaper_publisher_country":     "{0}에 대해 검색하라. 그런 다음 {0}의 발행사를 찾아라. 마지막으로 그 발행사가 속한 국가를 답하라.",
    # Tech (Diversity)
    "game_publisher_country":          "{0}에 대해 검색하라. 그런 다음 {0}의 배급사를 찾아라. 마지막으로 그 배급사가 속한 국가를 답하라.",
    "programming_language_designer_birthplace": "{0}에 대해 검색하라. 그런 다음 {0}을(를) 설계한 사람을 찾아라. 마지막으로 그의 출생지를 답하라.",
    "programming_language_designer_country":    "{0}에 대해 검색하라. 그런 다음 {0}을(를) 설계한 사람을 찾아라. 마지막으로 그의 국적을 답하라.",
    "product_manufacturer_country":    "{0}에 대해 검색하라. 그런 다음 {0}의 제조사를 찾아라. 마지막으로 그 제조사가 속한 국가를 답하라.",
    "invention_inventor_country":      "{0}에 대해 검색하라. 그런 다음 {0}을(를) 발명한 사람을 찾아라. 마지막으로 그의 국적을 답하라.",
    # Literature (Diversity)
    "book_author_birthplace":          "{0}에 대해 검색하라. 그런 다음 {0}의 저자를 찾아라. 마지막으로 그의 출생지를 답하라.",
    # Music (Diversity)
    "album_performer_country":         "{0}에 대해 검색하라. 그런 다음 {0}을(를) 발매한 아티스트를 찾아라. 마지막으로 그의 국적을 답하라.",
    # Politics (Diversity)
    "politician_party_country":        "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 정당을 찾아라. 마지막으로 그 정당이 위치한 국가를 답하라.",
    "party_hq_country":                "{0}에 대해 검색하라. 그런 다음 {0}의 본부를 찾아라. 마지막으로 그 본부가 위치한 국가를 답하라.",
    # Education (Diversity)
    "university_founder_country":      "{0}에 대해 검색하라. 그런 다음 {0}의 설립자를 찾아라. 마지막으로 그의 국적을 답하라.",
    # ── 추가 스키마 ────────────────────────────────────────────────────────
    "film_director_award":                       "{0}에 대해 검색하라. 그런 다음 {0}을(를) 감독한 사람을 찾아라. 마지막으로 그가 받은 상을 답하라.",
    "film_director_educated_at":                 "{0}에 대해 검색하라. 그런 다음 {0}을(를) 감독한 사람을 찾아라. 마지막으로 그가 졸업한 학교를 답하라.",
    "film_actor_award":                          "{0}에 대해 검색하라. 그런 다음 {0}에 출연한 배우를 찾아라. 마지막으로 그가 받은 상을 답하라.",
    "athlete_team_league_domestic":              "{0}에 대해 검색하라. 그런 다음 {0}이(가) 소속된 팀을 찾아라. 마지막으로 그 팀이 속한 리그를 답하라.",
    "company_founder_award":                     "{0}에 대해 검색하라. 그런 다음 {0}의 설립자를 찾아라. 마지막으로 그가 받은 상을 답하라.",
    "company_founder_educated_at":               "{0}에 대해 검색하라. 그런 다음 {0}의 설립자를 찾아라. 마지막으로 그가 졸업한 학교를 답하라.",
    "book_author_award":                         "{0}에 대해 검색하라. 그런 다음 {0}의 저자를 찾아라. 마지막으로 그가 받은 상을 답하라.",
    "album_performer_award":                     "{0}에 대해 검색하라. 그런 다음 {0}을(를) 발매한 아티스트를 찾아라. 마지막으로 그가 받은 상을 답하라.",
    "building_architect_award":                  "{0}에 대해 검색하라. 그런 다음 {0}을(를) 설계한 건축가를 찾아라. 마지막으로 그가 받은 상을 답하라.",
    "university_founder_educated_at":            "{0}에 대해 검색하라. 그런 다음 {0}의 설립자를 찾아라. 마지막으로 그가 졸업한 학교를 답하라.",
    "software_developer_founder":                "{0}에 대해 검색하라. 그런 다음 {0}의 개발사를 찾아라. 마지막으로 그 개발사의 설립자를 답하라.",
    "game_developer_founder":                    "{0}에 대해 검색하라. 그런 다음 {0}의 개발사를 찾아라. 마지막으로 그 개발사의 설립자를 답하라.",
    "game_publisher_founder":                    "{0}에 대해 검색하라. 그런 다음 {0}의 배급사를 찾아라. 마지막으로 그 배급사의 설립자를 답하라.",
    "newspaper_publisher_founder":               "{0}에 대해 검색하라. 그런 다음 {0}의 발행사를 찾아라. 마지막으로 그 발행사의 설립자를 답하라.",
    "product_manufacturer_founder":              "{0}에 대해 검색하라. 그런 다음 {0}의 제조사를 찾아라. 마지막으로 그 제조사의 설립자를 답하라.",
    "programming_language_designer_educated_at": "{0}에 대해 검색하라. 그런 다음 {0}을(를) 설계한 사람을 찾아라. 마지막으로 그가 졸업한 학교를 답하라.",
}

# schema_id × pattern → query_type 키로 자동 확장
from utils.schema_registry import SCHEMAS as _SCHEMAS
for _sid, _s in _SCHEMAS.items():
    for _pat in _s["patterns"]:
        _key = f"{_sid}_{_pat}"
        if _sid in _SCHEMA_BASE_TEMPLATES:
            QA_TEMPLATES[_key] = _SCHEMA_BASE_TEMPLATES[_sid]
        if _sid in _SCHEMA_BASE_Y2:
            QA_TEMPLATES_Y2[_key] = _SCHEMA_BASE_Y2[_sid]


def _apply_template(template: dict, labels: list[str], labels_en: list[str] | None = None) -> dict:
    """템플릿에 레이블을 채워 질문/서브질문을 생성."""
    question = resolve_particles(template["question"].format(*labels))
    sub_qs = [resolve_particles(sq.format(*labels)) for sq in template["sub_q"]]
    _en = labels_en if labels_en is not None else labels
    sub_qs_en = [sq.format(*_en) for sq in template.get("sub_q_en", [])]
    return {"question": question, "sub_qs": sub_qs, "sub_qs_en": sub_qs_en,
            "answer_hop": template["answer_hop"]}


def generate_template_qa(path: dict) -> Optional[dict]:
    """단일 경로 → 템플릿 기반 QA 쌍."""
    hops = path["hops"]
    if len(hops) < 3:
        return None

    labels = [h["label_ko"] or h["label_en"] for h in hops]
    labels_en = [h["label_en"] or h["label_ko"] for h in hops]
    qtype = path.get("query_type", "")
    tmpl = QA_TEMPLATES.get(qtype, DEFAULT_TEMPLATE)
    filled = _apply_template(tmpl, labels, labels_en)

    answer_hop = filled["answer_hop"]
    answer_hop_data = hops[answer_hop]
    final_gold_lang = (answer_hop_data.get("gold_lang")
                       or ("en" if answer_hop_data["primary_lang"] == "en" else "ko"))
    answer_ko = answer_hop_data.get("label_ko", "") or ""
    answer_en = answer_hop_data.get("label_en", "") or ""
    answer_entity = (answer_en or answer_ko) if final_gold_lang == "en" else (answer_ko or answer_en)

    # 홉별 상세 정보
    hop_details = []
    for i, (hop, sub_q) in enumerate(zip(hops[:-1], filled["sub_qs"])):
        next_hop = hops[i + 1]
        gold_lang = (next_hop.get("gold_lang")
                     or ("en" if next_hop["primary_lang"] == "en" else "ko"))
        # gold_entity는 gold_lang에 맞는 레이블을 우선 사용
        if gold_lang == "en":
            gold_entity = next_hop["label_en"] or next_hop["label_ko"]
        else:
            gold_entity = next_hop["label_ko"] or next_hop["label_en"]

        hop_detail = {
            "hop": i,
            "sub_q": sub_q,
            "gold_entity": gold_entity,
            "gold_entity_qid": next_hop["qid"],
            "gold_lang": gold_lang,
        }
        if gold_lang == "en" and i < len(filled["sub_qs_en"]):
            hop_detail["sub_q_en"] = filled["sub_qs_en"][i]
        # dynamic_subq용: sub_q에 실제 치환된 bridge entity의 KO 형태
        # hop_i의 sub_q는 labels[i]를 {i} 위치에 사용하므로,
        # hop_idx > 0일 때 labels[i] = 이전 홉 gold_entity의 KO 형태 = bridge entity
        if i > 0:
            hop_detail["sub_q_bridge_label"] = labels[i]
        hop_details.append(hop_detail)

    y2_tmpl = QA_TEMPLATES_Y2.get(qtype, DEFAULT_TEMPLATE_Y2)
    question_y2 = resolve_particles(y2_tmpl.format(*labels))

    return {
        "question":    question_y2,
        "answer": answer_entity,
        "answer_ko": answer_ko,
        "answer_en": answer_en,
        "answer_qid": hops[answer_hop]["qid"],
        "pattern": path["pattern"],
        "query_type": path.get("query_type", ""),
        "source_path_id": path["path_id"],
        "relations": path.get("relations", []),
        "hops": hop_details,
        "lang_sequence": path["lang_sequence"],
        "switch_at_hop": path.get("switch_at_hop", _SWITCH_AT_HOP[path["pattern"]]),
        "gen_method": "template",
        "oracle_question": path.get("oracle_question", ""),
        "oracle_hop0_bridge_support": path.get("oracle_hop0_bridge_support", ""),
        "oracle_hop1_answer_support": path.get("oracle_hop1_answer_support", ""),
    }


# ─── LLM 기반 질문 생성 (2단계, 선택) ────────────────────────────────────────

LLM_PROMPT = """다음 멀티홉 경로를 바탕으로 자연스러운 한국어 질문과 단계별 서브질문을 생성하세요.

경로 정보:
- 엔티티 0 (한국어 출처): {ent0}
- 관계 0→1: {rel0}
- 엔티티 1 (한국어+영어 출처): {ent1}
- 관계 1→2: {rel1}
- 엔티티 2 (영어 출처): {ent2}

요구사항:
1. 전체 질문은 ent0에 대해 묻는 한국어 질문 1문장
2. 서브질문 2개: 각 홉에서 중간 엔티티를 찾는 질문
3. 정답: ent2의 한국어 이름 또는 설명
4. 자연스러운 한국어, 어색한 번역투 금지

JSON 형식으로만 응답하세요:
{{
  "question": "...",
  "sub_q_0": "...",
  "sub_q_1": "...",
  "answer": "..."
}}"""


def generate_llm_qa(path: dict, api_key: str, model: str) -> Optional[dict]:
    """Anthropic API로 자연스러운 QA 생성."""

    hops = path["hops"]
    if len(hops) < 3:
        return None

    rels = path.get("relations", ["?", "?"])
    prompt = LLM_PROMPT.format(
        ent0=hops[0]["label_ko"],
        rel0=rels[0] if rels else "?",
        ent1=hops[1]["label_ko"],
        rel1=rels[1] if len(rels) > 1 else "?",
        ent2=hops[2]["label_en"],
    )

    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": model,
                "max_tokens": 512,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        raw = resp.json()["content"][0]["text"]
        # JSON 파싱 (마크다운 펜스 제거)
        raw = re.sub(r"```json|```", "", raw).strip()
        parsed = json.loads(raw)
    except Exception as e:
        logger.warning(f"LLM QA 생성 실패 ({path['path_id']}): {e}")
        return None

    hop_details = []
    for i, hop in enumerate(hops[:-1]):
        next_hop = hops[i + 1]
        gold_lang = (next_hop.get("gold_lang")
                     or ("en" if next_hop["primary_lang"] == "en" else "ko"))
        if gold_lang == "en":
            gold_entity = next_hop["label_en"] or next_hop["label_ko"]
        else:
            gold_entity = next_hop["label_ko"] or next_hop["label_en"]
        hop_details.append({
            "hop": i,
            "sub_q": parsed.get(f"sub_q_{i}", ""),
            "gold_entity": gold_entity,
            "gold_entity_qid": next_hop["qid"],
            "gold_lang": gold_lang,
        })

    last = hops[-1]
    final_gold_lang = (last.get("gold_lang") or ("en" if last["primary_lang"] == "en" else "ko"))
    llm_answer_ko = last.get("label_ko", "") or ""
    llm_answer_en = last.get("label_en", "") or ""
    llm_answer = (llm_answer_en or llm_answer_ko) if final_gold_lang == "en" else (llm_answer_ko or llm_answer_en)

    return {
        "question": parsed.get("question", ""),
        "answer": parsed.get("answer", llm_answer),
        "answer_ko": llm_answer_ko,
        "answer_en": llm_answer_en,
        "answer_qid": hops[-1]["qid"],
        "pattern": path["pattern"],
        "query_type": path.get("query_type", ""),
        "source_path_id": path["path_id"],
        "relations": path.get("relations", []),
        "hops": hop_details,
        "lang_sequence": path["lang_sequence"],
        "switch_at_hop": path.get("switch_at_hop", _SWITCH_AT_HOP[path["pattern"]]),
        "gen_method": "llm",
        "oracle_question": path.get("oracle_question", ""),
        "oracle_hop0_bridge_support": path.get("oracle_hop0_bridge_support", ""),
        "oracle_hop1_answer_support": path.get("oracle_hop1_answer_support", ""),
    }


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--method", choices=["template", "llm"], default="template")
    parser.add_argument("--api-key", default=None, help="Anthropic API key (llm 모드)")
    parser.add_argument("--paths",  default=None, help="입력 경로 파일 (미지정 시 config 사용)")
    parser.add_argument("--output", default=None, help="출력 QA 파일 (미지정 시 config 사용)")
    parser.add_argument("--no-cap", action="store_true", help="패턴별 per_pattern 캡 무시 (사전 샘플링된 입력 사용 시)")
    parser.add_argument("--trim-per-pattern", type=int, default=None,
                        help="dedup 후 패턴별 최대 건수로 트리밍 (버퍼 샘플 사용 시)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    paths_file = Path(args.paths or cfg["output"].get("validated_paths", cfg["output"]["raw_paths"]))

    if not paths_file.exists():
        logger.error(f"경로 파일 없음: {paths_file} → pipeline/01_extract_paths.py 먼저 실행하세요.")
        return

    output_file = Path(args.output or cfg["output"]["qa_pairs"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model = cfg["qa"]["llm_model"]
    api_key = args.api_key

    # 경로 로드 + 시드 고정 셔플 (query_type 독식 방지)
    paths = [json.loads(l) for l in paths_file.read_text().splitlines() if l.strip()]
    logger.info(f"경로 {len(paths)}개 로드")
    seed = cfg.get("seed", 42)
    random.seed(seed)
    random.shuffle(paths)

    # 패턴별 목표 수
    per_pattern = cfg["qa"]["per_pattern"]
    pattern_counts: Counter = Counter()

    qa_pairs = []
    skipped = 0

    for path in tqdm(paths, desc="QA 생성"):
        pat = path["pattern"]
        if not args.no_cap and pattern_counts[pat] >= per_pattern:
            continue

        if args.method == "template":
            qa = generate_template_qa(path)
        else:
            if not api_key:
                logger.error("--api-key가 필요합니다 (llm 모드)")
                return
            qa = generate_llm_qa(path, api_key, model)

        if qa is None:
            skipped += 1
            continue

        qa["qid"] = f"kch_{len(qa_pairs)+1:06d}"
        qa_pairs.append(qa)
        pattern_counts[pat] += 1

    # oracle_question 기준 multi-answer 중복 제거
    q_to_answer_qids: dict[str, set] = defaultdict(set)
    for qa in qa_pairs:
        q_to_answer_qids[qa.get("oracle_question", qa["question"])].add(qa["answer_qid"])
    ambiguous_qs = {q for q, aids in q_to_answer_qids.items() if len(aids) > 1}
    if ambiguous_qs:
        seen_ambig: set[str] = set()
        deduped = []
        removed = 0
        for qa in qa_pairs:
            q = qa.get("oracle_question", qa["question"])
            if q in ambiguous_qs:
                if q in seen_ambig:
                    removed += 1
                    continue
                seen_ambig.add(q)
            deduped.append(qa)
        logger.info(f"multi-answer 중복 제거: {removed}건 제거 → {len(deduped)}건")
        qa_pairs = deduped

    # 패턴별 트리밍 (버퍼 샘플 사용 시)
    if args.trim_per_pattern:
        trim_cnt: Counter = Counter()
        trimmed = []
        for qa in qa_pairs:
            if trim_cnt[qa["pattern"]] < args.trim_per_pattern:
                trimmed.append(qa)
                trim_cnt[qa["pattern"]] += 1
        before = len(qa_pairs)
        qa_pairs = trimmed
        logger.info(f"패턴별 트리밍 (≤{args.trim_per_pattern}): {before}건 → {len(qa_pairs)}건")

    # qid 재부여 (dedup 후 연속성 보장)
    for i, qa in enumerate(qa_pairs):
        qa["qid"] = f"kch_{i+1:06d}"

    # 저장
    with open(output_file, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print("\n" + "="*50)
    print(f"생성된 QA 쌍: {len(qa_pairs)}개  (스킵: {skipped})")
    print("패턴별 분포:")
    labels = {"a": "KO→KO→KO", "b": "KO→EN→KO",
              "c": "KO→KO→EN", "d": "KO→EN→EN"}
    for pat, cnt in sorted(pattern_counts.items()):
        print(f"  {pat} ({labels.get(pat,'?')}): {cnt}개")
    print(f"저장 완료: {output_file}")


if __name__ == "__main__":
    main()
