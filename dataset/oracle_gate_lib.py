#!/usr/bin/env python3
"""
dataset/oracle_gate_lib.py

Oracle Solvability Gate 공용 라이브러리 — 질문 템플릿 · 프롬프트 구성 · 응답 판정.

dataset/oracle_gate.py 와 dataset/build_triple_master.py 가 import 해서 쓴다.
독립 실행 진입점은 없다(게이트 실행은 dataset/oracle_gate.py).

  질문 템플릿 : QUESTION_MAP / _OVERRIDES / _SCHEMA_OVERRIDES → get_question()
  프롬프트    : extract_evidence_window() → make_prompt()
  판정        : parse_response() → determine_verdict()
"""

import asyncio
import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

PROMPT_VERSION = "1.0"

import yaml

# 어느 디렉터리에서 실행해도 utils/ 를 찾을 수 있도록
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Property Korean labels ────────────────────────────────────────────────────

_PROP_KO: dict[str, str] = {
    "P50":   "저자",
    "P54":   "소속 팀",
    "P57":   "감독",
    "P61":   "발명자",
    "P69":   "졸업 학교",
    "P84":   "건축가",
    "P102":  "소속 정당",
    "P108":  "소속 기관",
    "P112":  "설립자",
    "P115":  "홈 경기장",
    "P117":  "리그",
    "P118":  "리그",
    "P123":  "출판사",
    "P137":  "운영 기관",
    "P159":  "본사",
    "P161":  "출연 배우",
    "P166":  "수상한 상",
    "P175":  "공연자",
    "P176":  "제조사",
    "P178":  "개발사",
    "P264":  "음반사",
    "P272":  "제작사",
    "P277":  "사용 언어",
    "P287":  "설계자",
    "P463":  "소속 단체",
    "P495":  "제작 국가",
    "P749":  "모기업",
    "P750":  "배급사",
    "P1027": "시상 기관",
    "P1344": "참가한 대회",
    "P17":   "국가",
    "P19":   "출생지",
    "P27":   "국적",
    "P30":   "대륙",
    "P39":   "역임 직위",
    "P131":  "소재 행정 구역",
    "P276":  "개최지",
    "P1001": "관할 지역",
}


def extract_schema_id(query_type: str) -> str:
    """마지막 토큰이 패턴 suffix(a/b/c/d)일 때만 제거한다."""
    parts = query_type.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in ("a", "b", "c", "d"):
        return parts[0]
    return query_type


def _auto_question(p1: str, p2: str) -> str:
    p1_ko = _PROP_KO.get(p1, p1)
    p2_ko = _PROP_KO.get(p2, p2)
    return f"{{subj}}의 {p1_ko}의 {p2_ko}은(는) 무엇입니까?"


def _build_question_map() -> dict[str, str]:
    try:
        from utils.schema_registry import SCHEMAS
        qmap: dict[str, str] = {}
        for s in SCHEMAS.values():
            p1, p2 = s["path"]
            key = f"{p1}→{p2}"
            if key not in qmap:
                qmap[key] = _auto_question(p1, p2)
        return qmap
    except ImportError:
        return {}


# relation_pair 수준 재정의 (P1→P2 키)
_OVERRIDES: dict[str, str] = {
    # Biography / Person
    "P108→P131": "{subj}의 소속 기관이 위치한 행정구역은 어디입니까?",
    "P108→P17":  "{subj}의 소속 기관이 위치한 국가는 어디입니까?",
    "P69→P131":  "{subj}가 졸업한 학교가 위치한 행정구역은 어디입니까?",
    "P69→P17":   "{subj}가 졸업한 학교가 위치한 국가는 어디입니까?",
    "P463→P17":  "{subj}가 소속된 단체가 위치한 국가는 어디입니까?",
    "P166→P1027":"{subj}가 수상한 상의 시상 기관은 어디입니까?",
    "P39→P1001": "{subj}가 역임한 직위의 관할 지역은 어디입니까?",
    # Film
    "P57→P27":   "{subj}의 감독의 국적은 무엇입니까?",
    "P57→P19":   "{subj}의 감독의 출생지는 어디입니까?",
    "P57→P166":  "{subj}의 감독이 수상한 상은 무엇입니까?",
    "P57→P69":   "{subj}의 감독이 졸업한 학교는 어디입니까?",
    "P161→P27":  "{subj}에 출연한 배우의 국적은 무엇입니까?",
    "P161→P19":  "{subj}에 출연한 배우의 출생지는 어디입니까?",
    "P161→P166": "{subj}에 출연한 배우가 수상한 상은 무엇입니까?",
    "P272→P17":  "{subj}의 제작사가 위치한 국가는 어디입니까?",
    "P750→P17":  "{subj}의 배급사가 위치한 국가는 어디입니까?",
    "P495→P30":  "{subj}의 제작 국가가 속한 대륙은 어디입니까?",
    # Sports
    "P54→P17":   "{subj}의 소속 팀이 속한 국가는 어디입니까?",
    "P54→P118":  "{subj}의 소속 팀이 참가하는 리그는 무엇입니까?",
    "P118→P17":  "{subj}가 참가하는 리그가 속한 국가는 어디입니까?",
    "P1344→P276":"{subj}가 참가한 대회의 개최지는 어디입니까?",
    "P1344→P17": "{subj}가 참가한 대회가 열린 국가는 어디입니까?",
    "P115→P131": "{subj}의 홈 경기장이 위치한 행정구역은 어디입니까?",
    "P276→P17":  "{subj}의 개최지가 속한 국가는 어디입니까?",
    # Organization / Company
    "P159→P17":  "{subj}의 본사가 위치한 국가는 어디입니까?",
    "P159→P131": "{subj}의 본사가 위치한 행정구역은 어디입니까?",
    "P112→P27":  "{subj}의 설립자의 국적은 무엇입니까?",
    "P112→P19":  "{subj}의 설립자의 출생지는 어디입니까?",
    "P112→P166": "{subj}의 설립자가 수상한 상은 무엇입니까?",
    "P112→P69":  "{subj}의 설립자가 졸업한 학교는 어디입니까?",
    "P749→P159": "{subj}의 모기업의 본사는 어디입니까?",
    "P123→P17":  "{subj}의 출판사 또는 배급사가 위치한 국가는 어디입니까?",
    "P123→P112": "{subj}의 출판사 또는 배급사의 설립자는 누구입니까?",
    # Tech / Software / Product
    "P178→P159": "{subj}의 개발사의 본사는 어디입니까?",
    "P178→P17":  "{subj}의 개발사가 위치한 국가는 어디입니까?",
    "P178→P112": "{subj}의 개발사를 설립한 사람은 누구입니까?",
    "P277→P287": "{subj}에 사용된 프로그래밍 언어의 설계자는 누구입니까?",
    "P287→P27":  "{subj}의 설계자의 국적은 무엇입니까?",
    "P287→P19":  "{subj}의 설계자의 출생지는 어디입니까?",
    "P287→P69":  "{subj}의 설계자가 졸업한 학교는 어디입니까?",
    "P176→P17":  "{subj}의 제조사가 위치한 국가는 어디입니까?",
    "P176→P112": "{subj}의 제조사를 설립한 사람은 누구입니까?",
    "P61→P27":   "{subj}의 발명자의 국적은 무엇입니까?",
    # Literature
    "P50→P27":   "{subj}의 저자의 국적은 무엇입니까?",
    "P50→P19":   "{subj}의 저자의 출생지는 어디입니까?",
    "P50→P166":  "{subj}의 저자가 수상한 상은 무엇입니까?",
    # Music
    "P175→P27":  "{subj}의 아티스트의 국적은 무엇입니까?",
    "P175→P166": "{subj}의 아티스트가 수상한 상은 무엇입니까?",
    "P264→P17":  "{subj}의 음반사가 위치한 국가는 어디입니까?",
    # Award / Politics / Architecture / Infrastructure / Education
    "P1027→P159":"{subj}의 시상 기관의 본사는 어디입니까?",
    "P102→P17":  "{subj}의 소속 정당이 속한 국가는 어디입니까?",
    "P84→P27":   "{subj}의 건축가의 국적은 무엇입니까?",
    "P84→P166":  "{subj}의 건축가가 수상한 상은 무엇입니까?",
    "P137→P17":  "{subj}의 운영 기관이 위치한 국가는 어디입니까?",
    "P17→P30":   "{subj}가 위치한 국가가 속한 대륙은 어디입니까?",
}

# schema_id 수준 재정의 — P123(출판사/배급사), 음악 공연자, 스포츠 등 문맥 의존적 schema
_SCHEMA_OVERRIDES: dict[str, str] = {
    "newspaper_publisher_country":   "{subj}의 발행사가 위치한 국가는 어디입니까?",
    "book_publisher_country":        "{subj}의 출판사가 위치한 국가는 어디입니까?",
    "game_publisher_country":        "{subj}의 배급사가 위치한 국가는 어디입니까?",
    "newspaper_publisher_founder":   "{subj}의 발행사를 설립한 사람은 누구입니까?",
    "game_publisher_founder":        "{subj}의 배급사를 설립한 사람은 누구입니까?",
    "song_performer_country":        "{subj}을 부른 아티스트의 국적은 무엇입니까?",
    "album_performer_country":       "{subj}의 아티스트의 국적은 무엇입니까?",
    "album_performer_award":         "{subj}의 아티스트가 수상한 상은 무엇입니까?",
    "team_league_country":           "{subj}가 참가하는 리그가 속한 국가는 어디입니까?",
    "athlete_team_league":           "{subj}의 소속 팀이 참가하는 리그는 무엇입니까?",
    "athlete_team_league_domestic":  "{subj}의 소속 팀이 참가하는 리그는 무엇입니까?",
    "university_country_continent":  "{subj}가 위치한 국가가 속한 대륙은 어디입니까?",
}

# auto + relation-pair override 합산 맵 (schema override는 get_question에서 별도 처리)
QUESTION_MAP: dict[str, str] = {**_build_question_map(), **_OVERRIDES}


def get_question(path: dict, subj_label: str) -> str:
    # 우선순위: schema_id > relation_pair > auto
    schema_id = extract_schema_id(path.get("query_type", ""))
    if schema_id in _SCHEMA_OVERRIDES:
        return _SCHEMA_OVERRIDES[schema_id].format(subj=subj_label)
    relations = path.get("relations", [])
    if len(relations) >= 2:
        key = f"{relations[0]}→{relations[1]}"
        tmpl = QUESTION_MAP.get(key, "{subj}에 대한 질문의 답은 무엇입니까?")
        return tmpl.format(subj=subj_label)
    return f"{subj_label}에 대한 질문의 답은 무엇입니까?"


# ── 텍스트 정규화 & gold match ─────────────────────────────────────────────────

def _norm(text: str) -> str:
    text = re.sub(r"\([^)]*\)", "", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\s\-·,.\[\]\"\']+", "", text)
    return text


def parse_bool(val) -> bool:
    """JSON 응답의 boolean 필드를 파싱한다. 문자열 "false"가 True로 처리되는 오류를 방지."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() not in ("false", "no", "0", "")
    return bool(val)


def gold_match(predicted: str, aliases: list[str]) -> str | None:
    """매치 유형을 반환한다: 'exact' | 'alias_in_pred' | 'pred_in_alias_partial' | None."""
    pred_n = _norm(predicted)
    if not pred_n:
        return None
    for alias in aliases:
        alias_n = _norm(alias)
        min_len = 2 if any(ord(c) > 127 for c in alias_n) else 3
        if len(alias_n) < min_len:
            continue
        if pred_n == alias_n:
            return "exact"
        if alias_n in pred_n:
            return "alias_in_pred"
        if pred_n in alias_n:
            return "pred_in_alias_partial"
    return None


def build_gold_aliases(hop: dict, alias_cache: dict[str, list[str]] | None = None) -> list[str]:
    candidates = [
        hop.get("label_ko", ""),
        hop.get("label_en", ""),
        (hop.get("wiki_title") or "").replace("_", " "),
    ]
    if alias_cache:
        candidates.extend(alias_cache.get(hop.get("qid", ""), []))
    return list(dict.fromkeys(a for a in candidates if a))  # 순서 유지 중복 제거


# ── Oracle 프롬프트 ───────────────────────────────────────────────────────────

_ORACLE_PROMPT = """\
당신은 두-홉 멀티홉 질문을 판정하는 오라클입니다.
증거 이외의 배경 지식은 절대 사용하지 마십시오.

[두-홉 구조]
이 질문은 두 단계로 답이 도출됩니다:
  1단계: Hop-0 Evidence → 주어 엔티티에서 중간 엔티티(bridge)를 확인
  2단계: Hop-1 Evidence → 중간 엔티티에서 최종 답을 확인

[판정 기준]
- hop0_bridge_supported: Hop-0 Evidence가 "주어 → 중간 엔티티" 연결을 명시적으로 지지하는가?
  (중간 엔티티가 최종 답일 필요는 없다. 연결 관계가 증거에 나타나는지만 확인한다.)
- hop1_answer_supported: Hop-1 Evidence가 "중간 엔티티 → 최종 답" 연결을 명시적으로 지지하는가?
- 두 조건이 모두 참이고 최종 답을 명확히 하나로 도출할 수 있으면 ANSWERABLE.
- 증거가 서로 다른 최종 답 둘 이상을 지지하면 AMBIGUOUS.
- 그 외(증거 부족, 연결 불확실)에는 UNANSWERABLE.

[질문]
{question}

[Hop-0 Evidence] (주어 엔티티 Wikipedia 발췌)
{hop0_evidence}

[Hop-1 Evidence] (중간 엔티티 Wikipedia 발췌)
{hop1_evidence}

다음 JSON만 출력하고 다른 텍스트는 일절 포함하지 마십시오:
{{
  "verdict": "ANSWERABLE" | "UNANSWERABLE" | "AMBIGUOUS",
  "final_answer": "<최종 답 문자열; UNANSWERABLE/AMBIGUOUS이면 빈 문자열>",
  "hop0_bridge_supported": true | false,
  "hop0_bridge_support": "<Hop-0 Evidence에서 주어→중간 엔티티 연결 근거 문장; 없으면 빈 문자열>",
  "hop1_answer_supported": true | false,
  "hop1_answer_support": "<Hop-1 Evidence에서 중간 엔티티→최종 답 근거 문장; 없으면 빈 문자열>",
  "rationale": "<한두 문장: hop0에서 어떤 연결을, hop1에서 어떤 연결을 확인했는지 명시>"
}}"""


def extract_evidence_window(
    text: str,
    aliases: list[str],
    before: int = 400,
    after: int = 2000,
    max_chars: int = 3000,
) -> str:
    """alias 주변 window를 추출한다. 매치 없으면 앞 max_chars자를 반환."""
    text_lower = text.lower()
    for alias in aliases:
        alias_lower = alias.lower()
        if len(alias_lower) < 2:
            continue
        pos = text_lower.find(alias_lower)
        if pos >= 0:
            start = max(0, pos - before)
            end = min(len(text), pos + len(alias) + after)
            return text[start:end]
    return text[:max_chars]


def make_prompt(
    question: str,
    hop0_text: str,
    hop1_text: str,
    h1_aliases: list[str],
    h2_aliases: list[str],
) -> str:
    hop0_evidence = extract_evidence_window(hop0_text, h1_aliases)
    hop1_evidence = extract_evidence_window(hop1_text, h2_aliases)
    return _ORACLE_PROMPT.format(
        question=question,
        hop0_evidence=hop0_evidence,
        hop1_evidence=hop1_evidence,
    )


# ── 응답 파싱 & 판정 ──────────────────────────────────────────────────────────

def parse_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    return json.loads(text)


def determine_verdict(
    parsed: dict, gold_aliases: list[str]
) -> tuple[str, str | None, str | None]:
    """(gate_verdict, fail_reason, match_type) 반환. match_type은 PASS일 때만 non-None."""
    verdict = str(parsed.get("verdict", "")).upper()
    hop0_bridge = parse_bool(parsed.get("hop0_bridge_supported"))
    hop1_answer = parse_bool(parsed.get("hop1_answer_supported"))
    predicted = str(parsed.get("final_answer") or "")

    if verdict == "AMBIGUOUS":
        return "FAIL", "ORACLE_FAIL_AMBIGUOUS", None
    if verdict == "UNANSWERABLE":
        if not hop0_bridge:
            return "FAIL", "ORACLE_FAIL_HOP0_BRIDGE_UNSUPPORTED", None
        if not hop1_answer:
            return "FAIL", "ORACLE_FAIL_HOP1_ANSWER_UNSUPPORTED", None
        return "FAIL", "ORACLE_FAIL_FINAL_NOT_DERIVABLE", None
    if verdict == "ANSWERABLE":
        if not hop0_bridge:
            return "FAIL", "ORACLE_FAIL_HOP0_BRIDGE_UNSUPPORTED", None
        if not hop1_answer:
            return "FAIL", "ORACLE_FAIL_HOP1_ANSWER_UNSUPPORTED", None
        match_type = gold_match(predicted, gold_aliases)
        if not match_type:
            return "FAIL", "ORACLE_FAIL_WRONG_ANSWER", None
        return "PASS", None, match_type
    return "FAIL", "ORACLE_FAIL_PARSE_ERROR", None


# ── Anthropic API 호출 ────────────────────────────────────────────────────────

async def call_llm(client, model: str, prompt: str, return_usage: bool = False):
    """Anthropic 또는 OpenAI 클라이언트를 통일된 인터페이스로 호출한다.

    return_usage=True 면 (text, usage dict) 를 돌려준다. 호출 파라미터를 한곳에 유지한 채
    토큰 사용량만 추가로 기록하기 위한 경로다 (기본 False — 기존 호출부 동작 불변).
    """
    if model.startswith("gpt") or model.startswith("o"):
        response = await client.chat.completions.create(
            model=model,
            max_completion_tokens=512,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return (text, _usage_dict(getattr(response, "usage", None))) if return_usage else text
    msg = await client.messages.create(
        model=model,
        max_tokens=512,
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    return (text, _usage_dict(getattr(msg, "usage", None))) if return_usage else text


def _usage_dict(usage) -> dict:
    """SDK usage 객체를 {input,output,reasoning} 로 평탄화한다 (없으면 빈 dict)."""
    if usage is None:
        return {}
    get = lambda *names: next(  # noqa: E731
        (v for n in names if (v := getattr(usage, n, None)) is not None), None)
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens":  get("prompt_tokens", "input_tokens"),
        "output_tokens": get("completion_tokens", "output_tokens"),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
    }


async def process_item(
    path: dict,
    passage_index: dict[str, str],
    client,
    model: str,
    semaphore: asyncio.Semaphore,
    alias_cache: dict[str, list[str]] | None = None,
) -> dict:
    h0, h1, h2 = path["hops"][0], path["hops"][1], path["hops"][2]
    subj_label = h0.get("label_ko") or h0.get("label_en") or "?"
    gold_aliases = build_gold_aliases(h2, alias_cache)
    h1_aliases = build_gold_aliases(h1, alias_cache)
    question = get_question(path, subj_label)
    cache_key = make_cache_key(path, model)

    hop0_text = passage_index.get(path.get("hop0_selected_passage_id") or "", "")
    hop1_text = passage_index.get(path.get("hop1_selected_passage_id") or "", "")

    prompt = make_prompt(question, hop0_text, hop1_text, h1_aliases, gold_aliases)
    raw_text = None
    last_error = None
    parsed = None

    async with semaphore:
        for attempt in range(2):
            try:
                raw_text = await call_llm(client, model, prompt)
                parsed = parse_response(raw_text)
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt == 1:
                    parsed = None

    if parsed is None:
        return {
            **path,
            "cache_key": cache_key,
            "oracle_verdict": "FAIL",
            "oracle_predicted_answer": None,
            "oracle_gold_match": False,
            "oracle_gold_match_type": None,
            "oracle_hop0_bridge_supported": False,
            "oracle_hop0_bridge_support": "",
            "oracle_hop1_answer_supported": False,
            "oracle_hop1_answer_support": "",
            "oracle_fail_reason": "ORACLE_FAIL_PARSE_ERROR",
            "oracle_explanation": raw_text or last_error,
            "oracle_question": question,
            "oracle_model": model,
        }

    gate_verdict, fail_reason, match_type = determine_verdict(parsed, gold_aliases)
    hop0_bridge_support = str(parsed.get("hop0_bridge_support") or "").strip()
    hop1_answer_support = str(parsed.get("hop1_answer_support") or "").strip()
    return {
        **path,
        "cache_key": cache_key,
        "oracle_verdict": gate_verdict,
        "oracle_predicted_answer": parsed.get("final_answer"),
        "oracle_gold_match": match_type is not None,
        "oracle_gold_match_type": match_type,
        "oracle_hop0_bridge_supported": parse_bool(parsed.get("hop0_bridge_supported")),
        "oracle_hop0_bridge_support": hop0_bridge_support,
        "oracle_hop1_answer_supported": parse_bool(parsed.get("hop1_answer_supported")),
        "oracle_hop1_answer_support": hop1_answer_support,
        "oracle_support_complete": bool(hop0_bridge_support and hop1_answer_support),
        "oracle_fail_reason": fail_reason,
        "oracle_explanation": parsed.get("rationale"),
        "oracle_question": question,
        "oracle_model": model,
    }


# ── Cache key ────────────────────────────────────────────────────────────────

def make_cache_key(path: dict, model: str) -> str:
    """프롬프트 버전 · 모델 · path_id · passage IDs · 질문을 포함한 복합 캐시 키."""
    h0 = path["hops"][0]
    subj_label = h0.get("label_ko") or h0.get("label_en") or "?"
    question = get_question(path, subj_label)
    key_str = "|".join([
        PROMPT_VERSION,
        model,
        path["path_id"],
        path.get("hop0_selected_passage_id") or "",
        path.get("hop1_selected_passage_id") or "",
        question,
    ])
    return hashlib.sha256(key_str.encode()).hexdigest()


# ── 데이터 로드 ───────────────────────────────────────────────────────────────

def load_passage_index(passages_path: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    with open(passages_path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            index[p["passage_id"]] = p["text"]
    return index


def load_lexical_pass(gated_path: Path) -> list[dict]:
    rows = []
    with open(gated_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("lexical_pass"):
                rows.append(d)
    return rows


def load_cache(cache_path: Path) -> dict[str, dict]:
    """cache_key 필드가 있는 항목만 로드한다. 구버전(path_id 키) 항목은 무시."""
    cache: dict[str, dict] = {}
    if not cache_path.exists():
        return cache
    with open(cache_path, encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if "cache_key" in d:
                    cache[d["cache_key"]] = d
            except Exception:
                pass
    return cache


# ── 리포트 ────────────────────────────────────────────────────────────────────

def build_report(results: list[dict]) -> dict:
    total = len(results)
    n_pass = sum(1 for r in results if r["oracle_verdict"] == "PASS")

    by_pattern: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    by_schema: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    fail_reasons: Counter = Counter()

    for r in results:
        pat = r.get("pattern", "?")
        schema = r.get("query_type", "?")
        by_pattern[pat]["total"] += 1
        by_schema[schema]["total"] += 1
        if r["oracle_verdict"] == "PASS":
            by_pattern[pat]["pass"] += 1
            by_schema[schema]["pass"] += 1
        else:
            fail_reasons[r.get("oracle_fail_reason", "UNKNOWN")] += 1

    feasibility = {
        pat: {
            "pass": by_pattern[pat]["pass"],
            "feasible_250": by_pattern[pat]["pass"] >= 250,
        }
        for pat in "abcd"
    }
    feasibility["main_pool_1000"] = all(by_pattern[p]["pass"] >= 250 for p in "abcd")

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": total,
        "n_pass": n_pass,
        "n_fail": total - n_pass,
        "pass_rate": round(n_pass / total, 4) if total else 0,
        "by_pattern": {k: v for k, v in sorted(by_pattern.items())},
        "by_schema": {k: v for k, v in sorted(by_schema.items())},
        "fail_reasons": dict(fail_reasons.most_common()),
        "feasibility": feasibility,
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Oracle Gate 결과  ({report['generated_at']})")
    print(f"총 {report['total']:,}개 → PASS {report['n_pass']:,} ({report['pass_rate']*100:.1f}%)")
    print("\n패턴별:")
    for pat in "abcd":
        v = report["by_pattern"].get(pat, {"pass": 0, "total": 0})
        rate = v["pass"] / v["total"] * 100 if v["total"] else 0
        mark = "✅" if v["pass"] >= 250 else "❌"
        print(f"  패턴 {pat}: {v['pass']:,} / {v['total']:,} ({rate:.1f}%) {mark}")
    print(f"\nMain pool 1,000 가능: {'✅' if report['feasibility']['main_pool_1000'] else '❌'}")
    print("\nFail 사유:")
    for reason, cnt in report["fail_reasons"].items():
        print(f"  {reason}: {cnt:,}")


# ── --dump-templates ──────────────────────────────────────────────────────────

def dump_templates() -> None:
    try:
        from utils.schema_registry import SCHEMAS
    except ImportError:
        print("schema_registry를 찾을 수 없습니다.")
        return

    print(f"# Oracle Gate 질문 템플릿 (PROMPT_VERSION={PROMPT_VERSION})")
    print(f"# {'schema_id':<48} {'source':<10} {'relation_pair':<14} 최종 질문 템플릿")
    print("-" * 120)

    for sid in sorted(SCHEMAS):
        s = SCHEMAS[sid]
        p1, p2 = s["path"]
        rel_key = f"{p1}→{p2}"
        schema_id = extract_schema_id(sid)

        if schema_id in _SCHEMA_OVERRIDES:
            source = "schema"
            tmpl = _SCHEMA_OVERRIDES[schema_id]
        elif rel_key in _OVERRIDES:
            source = "relation"
            tmpl = _OVERRIDES[rel_key]
        else:
            source = "auto"
            tmpl = QUESTION_MAP.get(rel_key, _auto_question(p1, p2))

        filled = tmpl.format(subj="<주어>")
        print(f"  {sid:<48} [{source:<8}] {rel_key:<14} {filled}")


