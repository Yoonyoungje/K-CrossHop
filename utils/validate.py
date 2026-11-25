"""
utils/validate.py
=================
경로 텍스트 정합성 검증 (조건 B).
"""

import re


def is_valid_hit(passage_text: str, entity_label: str, entity_prop: str = None) -> bool:
    """
    텍스트 내에 엔티티 라벨이 정상적으로 언급되었는지 검증.

    :param passage_text: 위키피디아 패시지 본문
    :param entity_label: 찾고자 하는 다음 홉의 엔티티 이름 (Wikidata Label)
    :param entity_prop: (Optional) 현재 홉의 관계 속성 (예: 'P17', 'P131')
    """
    label = entity_label.strip()
    if not label:
        return False

    # 지명·국가명 속성은 2자 한글도 고유명사로 간주 (조사 컨텍스트 불요)
    # P166 제외: award 이름은 길이 검증 적용 ("대상", "금상" false positive 방지)
    is_safe_property = entity_prop in ['P17', 'P131', 'P27', 'P19', 'P159', 'P495']

    if not is_safe_property:
        if re.fullmatch(r'\d+(년)?', label):
            return False
        if len(label) < 2:
            return False

    # [A] 순수 영문 엔티티
    # \b 대신 lookaround 사용: Python 3에서 \w가 한글 포함이라
    # \bApple\b가 "Apple은" 같은 한-영 혼재 텍스트에서 매칭 실패함
    if re.fullmatch(r'[a-zA-Z\s\.]+', label):
        pattern = rf'(?<![a-zA-Z]){re.escape(label)}(?![a-zA-Z])'
        if len(label) <= 3 and label.isupper():
            return bool(re.search(pattern, passage_text))
        else:
            return bool(re.search(pattern, passage_text, re.IGNORECASE))

    # [B] 한국어 또는 다국어 혼합 엔티티
    else:
        if len(label) == 2 and not is_safe_property:
            pattern = rf'{re.escape(label)}([\s\.,!?\(\)은는이가을를의에와과로]|$)'
            return bool(re.search(pattern, passage_text))
        else:
            return label in passage_text


def validate_text_coherence(path: dict, entity_texts: dict[str, str]) -> dict:
    """
    조건 B: 각 홉의 패시지 텍스트에 다음 홉 엔티티 레이블이 포함되는지 검증.

    hop-0: subj 패시지에 mid 엔티티 레이블 포함 여부
    hop-1: mid 패시지에 obj 엔티티 레이블 포함 여부

    is_valid_hit으로 매칭 (word boundary, 연도 필터, P17/P131 allowlist).
    언어 선택은 primary_lang 로직과 동일.

    Returns:
        {hop0_pass, hop1_pass, pass, hop0_matched_label, hop1_matched_label}
    """
    hops = path["hops"]
    relations = path.get("relations", [])
    prop_hop0 = relations[0] if len(relations) > 0 else None
    prop_hop1 = relations[1] if len(relations) > 1 else None

    def _lang(hop):
        return "ko" if hop["primary_lang"] in ("ko", "both") else "en"

    subj_text = entity_texts.get(f"{hops[0]['qid']}_{_lang(hops[0])}", "")
    mid_text  = entity_texts.get(f"{hops[1]['qid']}_{_lang(hops[1])}", "")

    mid_labels = [l for l in [hops[1]["label_ko"], hops[1]["label_en"]] if l]
    obj_labels = [l for l in [hops[2]["label_ko"], hops[2]["label_en"]] if l]

    hop0_pass, hop0_matched = False, None
    for label in mid_labels:
        if is_valid_hit(subj_text, label, entity_prop=prop_hop0):
            hop0_pass, hop0_matched = True, label
            break

    hop1_pass, hop1_matched = False, None
    for label in obj_labels:
        if is_valid_hit(mid_text, label, entity_prop=prop_hop1):
            hop1_pass, hop1_matched = True, label
            break

    return {
        "hop0_pass": hop0_pass,
        "hop1_pass": hop1_pass,
        "pass": hop0_pass and hop1_pass,
        "hop0_matched_label": hop0_matched,
        "hop1_matched_label": hop1_matched,
    }
