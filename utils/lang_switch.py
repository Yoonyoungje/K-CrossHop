"""
utils/lang_switch.py
====================
lang_switch(언어전환) 라벨 단일 소스.

정의: 한 홉의 retrieval gold_lang이 직전 홉의 gold_lang과 다르면 switch.
      hop0은 질문 언어(기본 "ko")와 비교한다.
"""


def compute_lang_switch(gold_langs: list[str], question_lang: str = "ko") -> list[bool]:
    """홉별 언어전환 플래그.

    lang_switch@hop_k = (gold_lang[k] != gold_lang[k-1]),  hop0은 question_lang과 비교.

    예시 (a/b/c/d 패턴의 QA-홉 gold_lang 시퀀스):
      [ko, ko] → [False, False]   # a: KO→KO→KO
      [en, ko] → [True,  True]    # b: KO→EN→KO
      [ko, en] → [False, True]    # c: KO→KO→EN
      [en, en] → [True,  False]   # d: KO→EN→EN
    """
    out, prev = [], question_lang
    for g in gold_langs:
        out.append(g != prev)
        prev = g
    return out
