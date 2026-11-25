"""
utils/retrieval_success.py
===========================
alias-aware answer_bearing_hit_alias/v3 계산 유틸리티.

외부에서 import해서 쓰거나 analysis/09_alias_retrieval.py 에서 직접 실행.
"""

import re

MIN_ALIAS_LEN = 4  # false positive 방지: 이 길이 이하 alias 매칭 제외


# ─── alias set 구성 ──────────────────────────────────────────────────────────

def build_alias_set(entry: dict) -> dict[str, list[str]]:
    """
    alias_cache 엔트리 → {'ko': [...], 'en': [...], 'all': [...]} 형태.
    각 언어별 alias set과 전체 alias set(중복 제거, MIN_ALIAS_LEN 필터)을 반환.
    """
    ko_candidates = [
        entry.get("label_ko", ""),
        entry.get("wiki_title_ko", ""),
        *entry.get("aliases_ko", []),
    ]
    en_candidates = [
        entry.get("label_en", ""),
        entry.get("wiki_title_en", ""),
        *entry.get("aliases_en", []),
    ]

    def _filter(cands: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for c in cands:
            c = c.strip()
            if len(c) >= MIN_ALIAS_LEN and c.lower() not in seen:
                seen.add(c.lower())
                result.append(c)
        return result

    ko = _filter(ko_candidates)
    en = _filter(en_candidates)

    # 'all': ko + en에서 추가 중복 제거
    seen_all: set[str] = set()
    all_aliases = []
    for a in ko + en:
        if a.lower() not in seen_all:
            seen_all.add(a.lower())
            all_aliases.append(a)

    return {"ko": ko, "en": en, "all": all_aliases}


# ─── 매칭 ────────────────────────────────────────────────────────────────────

def _alias_type(alias: str, entry: dict) -> str:
    """매칭된 alias가 label/wiki_title/alias 중 어느 것인지 판정."""
    a_lower = alias.lower()
    for field in ("label_ko", "label_en"):
        if entry.get(field, "").lower() == a_lower:
            return "label"
    for field in ("wiki_title_ko", "wiki_title_en"):
        if entry.get(field, "").lower() == a_lower:
            return "wiki_title"
    return "alias"


def match_passages(passages: list[str], alias_set: list[str],
                   entry: dict) -> tuple[bool, str, str]:
    """
    passages × alias_set에서 첫 번째 hit를 반환.

    Returns:
        (success: bool, matched_alias: str, match_type: 'label'|'alias'|'wiki_title'|'none')
    """
    for alias in alias_set:
        a_lower = alias.lower()
        for p in passages:
            if a_lower in p.lower():
                return True, alias, _alias_type(alias, entry)
    return False, "", "none"


# ─── 단일 레코드 v2 계산 ─────────────────────────────────────────────────────

def compute_v2_record(record: dict, passages: list[str],
                      cache: dict) -> dict:
    """
    hop0 레코드에 v2 필드를 추가한 dict를 반환.

    추가 필드:
      answer_bearing_hit_alias   : bool
      v2_matched_alias       : str or ""
      v2_match_type          : "label"|"alias"|"wiki_title"|"none"
      v2_alias_set_size      : int
      v2_alias_set           : list[str]
    """
    qid_ent = record.get("gold_entity_qid", "")
    entry   = cache.get(qid_ent, {})
    aliases = build_alias_set(entry)["all"] if entry else []

    hit, matched, match_type = match_passages(passages, aliases, entry)

    return {
        **record,
        "answer_bearing_hit_alias":  hit,
        "v2_matched_alias":      matched,
        "v2_match_type":         match_type,
        "v2_alias_set_size":     len(aliases),
        "v2_alias_set":          aliases,
    }


# ─── v3: normalization 기반 매칭 ──────────────────────────────────────────────

# 긴 조사를 먼저 매칭해야 오매칭 방지 (예: '으로' 전에 '로' 매칭되면 안 됨)
_KO_PARTICLES = ['에서', '에게', '한테', '으로', '는', '은', '이', '가', '을', '를', '의', '에', '와', '과', '로']


def _has_korean(text: str) -> bool:
    return any('가' <= c <= '힣' for c in text)


def normalize_ko(text: str) -> str:
    """
    한국어 텍스트 정규화.
    처리 순서: 소문자화 → 특수문자/구두점 제거 → 공백 정규화 → 한국어 조사 제거.
    조사 제거는 각 어절(공백 구분 토큰) 단위로, 토큰 뒤에 붙은 조사만 제거.
    """
    text = text.lower()
    # 한국어·영문자·숫자·공백 이외 제거 (한자 포함 유지)
    text = re.sub(r"[^\w\s가-힣]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    # 각 어절 뒤 조사 제거
    tokens = text.split(" ")
    result = []
    for tok in tokens:
        for particle in _KO_PARTICLES:
            if tok.endswith(particle) and len(tok) > len(particle):
                tok = tok[: -len(particle)]
                break  # 한 번만 제거
        result.append(tok)
    return " ".join(result)


def normalize_en(text: str) -> str:
    """
    영어 텍스트 정규화.
    처리 순서: 소문자화 → 하이픈→공백 → 구두점 제거 → 공백 정규화.
    """
    text = text.lower()
    text = text.replace("-", " ").replace("–", " ").replace("—", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize(text: str) -> tuple[str, str]:
    """(normalize_ko 결과, normalize_en 결과) 반환."""
    return normalize_ko(text), normalize_en(text)


def build_alias_surfaces(qid: str, cache: dict) -> list[str]:
    """
    v2 alias_set에 normalized variant를 추가한 리스트 반환.
    각 alias에 대해 normalize_ko/normalize_en 적용 후 원본 + normalized 둘 다 포함.
    중복 제거, MIN_ALIAS_LEN 필터 유지.
    """
    entry = cache.get(qid, {})
    alias_surfaces = build_alias_set(entry)["all"] if entry else []

    seen: set[str] = set()
    result: list[str] = []

    def _add(s: str) -> None:
        s = s.strip()
        if len(s) >= MIN_ALIAS_LEN and s not in seen:
            seen.add(s)
            result.append(s)

    for alias in alias_surfaces:
        _add(alias)
        norm_ko = normalize_ko(alias)
        norm_en = normalize_en(alias)
        _add(norm_ko)
        _add(norm_en)

    return result


def check_answer_bearing_hit(
    passages: list[str],
    qid: str,
    cache: dict,
    gold_lang: str,
) -> dict:
    """
    v2 매칭 후 실패 시 normalized alias × normalized passage 재시도.

    반환:
        answer_bearing_hit : bool
        match_method      : "alias_exact" | "surface_normalized" | "none"
        hit_matched_alias     : str or None
    """
    entry = cache.get(qid, {})
    alias_surfaces = build_alias_set(entry)["all"] if entry else []

    # ── 1단계: v2 방식 (원본 alias × 원본 passage, case-insensitive) ──────────
    for alias in alias_surfaces:
        a_lower = alias.lower()
        for p in passages:
            if a_lower in p.lower():
                return {
                    "answer_bearing_hit": True,
                    "match_method": "alias_exact",
                    "hit_matched_alias": alias,
                }

    # ── 2단계: normalized alias × normalized passage ───────────────────────────
    # 각 passage를 ko/en 양쪽으로 미리 정규화
    norm_passages_ko = [normalize_ko(p) for p in passages]
    norm_passages_en = [normalize_en(p) for p in passages]

    hit_aliases = build_alias_surfaces(qid, cache)
    # v2에 이미 없는 normalized variant만 2단계에서 사용
    v2_set_lower = {a.lower() for a in alias_surfaces}
    new_aliases = [a for a in hit_aliases if a.lower() not in v2_set_lower]

    for alias in new_aliases:
        # alias가 한국어면 ko-normalized passage와, 영어면 en-normalized passage와 비교
        if _has_korean(alias):
            norm_alias = normalize_ko(alias)
            for np in norm_passages_ko:
                if norm_alias in np:
                    return {
                        "answer_bearing_hit": True,
                        "match_method": "surface_normalized",
                        "hit_matched_alias": alias,
                    }
        else:
            norm_alias = normalize_en(alias)
            for np in norm_passages_en:
                if norm_alias in np:
                    return {
                        "answer_bearing_hit": True,
                        "match_method": "surface_normalized",
                        "hit_matched_alias": alias,
                    }

    return {
        "answer_bearing_hit": False,
        "match_method": "none",
        "hit_matched_alias": None,
    }
