"""
unit tests for utils/lang_switch.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.lang_switch import compute_lang_switch


def test_pattern_a():
    """a: KO→KO→KO, QA-홉 gold_lang [ko, ko] → 전환 없음."""
    assert compute_lang_switch(["ko", "ko"]) == [False, False]


def test_pattern_b():
    """b: KO→EN→KO, QA-홉 gold_lang [en, ko] → 두 홉 모두 전환."""
    assert compute_lang_switch(["en", "ko"]) == [True, True]


def test_pattern_c():
    """c: KO→KO→EN, QA-홉 gold_lang [ko, en] → hop1만 전환."""
    assert compute_lang_switch(["ko", "en"]) == [False, True]


def test_pattern_d():
    """d: KO→EN→EN, QA-홉 gold_lang [en, en] → hop0만 전환."""
    assert compute_lang_switch(["en", "en"]) == [True, False]


def test_hop0_matches_old_definition():
    """hop0은 항상 (gold_lang0 != question_lang) 와 일치."""
    for seq in (["ko", "ko"], ["en", "ko"], ["ko", "en"], ["en", "en"]):
        assert compute_lang_switch(seq)[0] == (seq[0] != "ko")


def test_hop0_single_en():
    assert compute_lang_switch(["en"], question_lang="ko") == [True]


def test_hop0_single_ko():
    assert compute_lang_switch(["ko"], question_lang="ko") == [False]


def test_empty():
    assert compute_lang_switch([]) == []


def test_custom_question_lang():
    assert compute_lang_switch(["en", "en"], question_lang="en") == [False, False]
