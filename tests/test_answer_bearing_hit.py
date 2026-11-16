"""
unit tests for retrieval_success.py v3 functions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.retrieval_success import (
    normalize_ko,
    normalize_en,
    build_alias_surfaces,
    check_answer_bearing_hit,
)


# ── normalize_ko ─────────────────────────────────────────────────────────────

def test_ko_particle_는():
    """'김연아는' → '김연아' → alias '김연아' 매칭 성공."""
    result = normalize_ko("김연아는")
    assert result == "김연아", f"got: {result!r}"


def test_ko_particle_에서():
    """'서울대학교에서' → '서울대학교'."""
    result = normalize_ko("서울대학교에서")
    assert result == "서울대학교", f"got: {result!r}"


def test_ko_particle_이():
    """'이화여대이' 같은 경우도 처리."""
    result = normalize_ko("이화여대이")
    assert result == "이화여대", f"got: {result!r}"


def test_ko_particle_을():
    result = normalize_ko("한양대학교를")
    assert result == "한양대학교", f"got: {result!r}"


def test_ko_particle_으로():
    result = normalize_ko("서울로")
    # '로' 제거 → '서울'
    assert result == "서울", f"got: {result!r}"


def test_ko_no_strip_when_short():
    """어절이 조사만 남게 되는 경우 제거 안 함."""
    result = normalize_ko("가")
    assert result == "가", f"got: {result!r}"


def test_ko_lowercase_and_punct():
    result = normalize_ko("서울大學校!")
    assert "!" not in result
    assert "서울" in result


def test_ko_sentence():
    """문장 내 여러 어절에 조사 제거 적용."""
    result = normalize_ko("김연아는 한국에서 태어났다")
    # '김연아는' → '김연아', '한국에서' → '한국'
    assert "김연아" in result
    assert "한국" in result
    assert "는" not in result.split()
    assert "에서" not in result


# ── normalize_en ─────────────────────────────────────────────────────────────

def test_en_lowercase():
    """'MIT' → 'mit'."""
    assert normalize_en("MIT") == "mit"


def test_en_hyphen_to_space():
    """'Kim Woo-min' → 'kim woo min'."""
    assert normalize_en("Kim Woo-min") == "kim woo min"


def test_en_emdash():
    """em dash도 공백으로 변환."""
    result = normalize_en("badminton at the 2014 Asian Games – women's team")
    assert "–" not in result
    assert "women" in result


def test_en_punct_removed():
    result = normalize_en("ladies' singles")
    assert "'" not in result


# ── 서울대 축약형 (v2에서도 substring으로 성공해야 함) ────────────────────────

def test_seoul_univ_abbreviation_v2():
    """
    passage에 '서울대' 있고 alias가 '서울대학교'이면
    v2 substring 매칭: '서울대학교'.lower() in '서울대'.lower() → 실패 (부분 매칭 방향 반대).
    v3 normalized: '서울대' in '서울대학교' → 역방향이므로 여전히 실패.
    즉 이건 v2/v3 모두 실패 케이스.
    """
    cache = {"Q_test": {
        "label_ko": "서울대학교",
        "wiki_title_ko": "서울대학교",
        "aliases_ko": [],
        "label_en": "Seoul National University",
        "wiki_title_en": "Seoul National University",
        "aliases_en": [],
    }}
    passages = ["서울대 출신의 연구자"]
    result = check_answer_bearing_hit(passages, "Q_test", cache, "ko")
    # v2: '서울대학교' not in '서울대 출신의 연구자' → fail
    # v3 normalized: normalize_ko('서울대학교') = '서울대학교', normalize_ko(passage) = '서울대 출신의 연구자'
    #               '서울대학교' not in '서울대 출신의 연구자' → fail
    assert result["answer_bearing_hit"] is False, (
        "서울대 축약형은 v3 범위 밖 — false positive 아님을 확인"
    )


# ── KO 조사 제거로 v3 회수 ────────────────────────────────────────────────────

def test_ko_particle_recovery():
    """passage에 '이화여대는' 포함 → alias '이화여대' 매칭 성공 (alias_exact: substring 포함)."""
    cache = {"Q_ewha": {
        "label_ko": "이화여자대학교",   # 7자 → MIN_ALIAS_LEN 통과
        "wiki_title_ko": "이화여대",    # 4자 → MIN_ALIAS_LEN 통과
        "aliases_ko": [],
        "label_en": "Ewha Womans University",
        "wiki_title_en": "Ewha Womans University",
        "aliases_en": [],
    }}
    passages = ["이화여대는 서울에 위치한 대학교이다."]
    result = check_answer_bearing_hit(passages, "Q_ewha", cache, "ko")
    # v2: '이화여대' in '이화여대는 서울에...' → substring 포함 → 성공
    assert result["answer_bearing_hit"] is True
    assert result["match_method"] == "alias_exact"


def test_ko_particle_only_in_normalized():
    """
    alias가 '한양대학교를'처럼 조사 붙은 형태만 passage에 없고,
    normalized '한양대학교'는 passage에 있어서 v2 실패 → v3 성공하는 경우.
    실제로는 v2가 '한양대학교' alias 자체를 가지므로, v2에서 성공.
    이 테스트는 normalized variant 경로를 직접 검증한다.
    """
    # alias를 굳이 조사 붙은 형태만 넣어서 v2 강제 실패, v3만 성공하게
    cache = {"Q_hanyang": {
        "label_ko": "한양대학교를",   # 비정상적 alias지만 테스트 목적
        "wiki_title_ko": "",
        "aliases_ko": [],
        "label_en": "Hanyang University",
        "wiki_title_en": "Hanyang University",
        "aliases_en": [],
    }}
    passages = ["그는 한양대학교 출신이다."]
    result = check_answer_bearing_hit(passages, "Q_hanyang", cache, "ko")
    # v2: '한양대학교를' not in passage → fail
    # v3: normalize_ko('한양대학교를') = '한양대학교', normalize_ko(passage) 에 '한양대학교' 포함 → success
    assert result["answer_bearing_hit"] is True
    assert result["match_method"] == "surface_normalized"


# ── EN 하이픈 정규화 (v3 범위 밖 케이스 기록) ────────────────────────────────

def test_en_hyphen_not_fuzzy():
    """
    'Kim Woo-min' normalize → 'kim woo min'
    passage 'Kim Woomin' normalize → 'kim woomin'
    → 여전히 불일치. fuzzy match 범위이므로 v3 실패.
    """
    norm_alias   = normalize_en("Kim Woo-min")   # 'kim woo min'
    norm_passage = normalize_en("Kim Woomin")    # 'kim woomin'
    assert norm_alias != norm_passage, (
        "하이픈 합성 불일치는 fuzzy match 영역 — v3 범위 밖 확인"
    )


def test_en_case_insensitive():
    """'KAIST' → v2 lowercase 매칭으로 성공."""
    cache = {"Q_kaist": {
        "label_ko": "한국과학기술원",
        "wiki_title_ko": "한국과학기술원",
        "aliases_ko": [],
        "label_en": "KAIST",
        "wiki_title_en": "KAIST",
        "aliases_en": ["Korea Advanced Institute of Science and Technology"],
    }}
    passages = ["He is a researcher at kaist in Daejeon."]
    result = check_answer_bearing_hit(passages, "Q_kaist", cache, "en")
    assert result["answer_bearing_hit"] is True
    assert result["match_method"] == "alias_exact"


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {fn.__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
