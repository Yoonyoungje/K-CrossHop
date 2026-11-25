"""
utils/answer_parser.py
=======================
에이전트 출력 파서 — pipeline/06_annotate.py와 진단 스크립트 공유.
"""
import re
from dataclasses import dataclass

PARSER_VERSION = "v1"


@dataclass
class ParseResult:
    claims: list[str]
    source: str            # 'answer_section' | 'fallback_cleaned' | 'fallback_whole' | 'none'
    abstained: bool
    answer_marker_count: int
    ambiguous_answer_marker: bool
    parse_issue: str | None
    multi_answer: bool = False  # 단일 마커 내 복수 엔티티 나열


_META_LINE = re.compile(
    r"^(생각|검색어|질문|현재\s*단계|검색된\s*문서|이전\s*단계\s*답변|문서|답변)\s*[:：]"
)

_ANSWER_MARKER = re.compile(r"답변\s*[:：]")

_CLEAR_ABSTAIN = re.compile(
    r"(모르겠습니다|모릅니다|정보가 없습니다|찾을 수 없습니다|알 수 없습니다"
    r"|확인할 수 없습니다|제공된 문서에서 확인할 수 없습니다"
    r"|I don't know|cannot determine|not enough information)"
)

_SOFT_ABSTAIN_NORMALIZED = {"없음", "없습니다", "해당없음", "답변없음", "n/a", "모릅니다"}

ABSTAIN_KWS = ["모르겠", "모릅니다", "알 수 없", "정보가 없", "정보 없",
               "확인할 수 없", "찾을 수 없", "파악할 수 없",
               "제공되지 않", "언급되지 않",
               "찾지 못", "명시되지 않", "포함되어 있지 않",
               "기재되어 있지 않", "확인되지 않", "답변드리기 어렵",
               "나와 있지 않",
               "I don't know", "cannot determine", "not enough information",
               "unknown", "N/A", "unable to find"]

# 영어 기권 표현은 부분문자열 오탐("The Unknown Soldier" 등)을 막기 위해
# 단어경계(word-boundary) 정규식으로 별도 매칭한다.
# frontier 모델은 8B보다 장황한 영어 거절을 내므로 고빈도 패턴을 추가 커버한다
# ("the documents do not mention", "insufficient information" 등). 단방향(abstain↑).
_EN_ABSTAIN_RE = re.compile(
    r"\b("
    # bare "unknown"은 뒤에 대문자 고유명사가 이어지면 제외 (예: "Unknown Worlds
    # Entertainment"=정답 → 기권 오탐 방지). 룩어헤드는 case-sensitive((?-i:))로 대문자만 본다.
    r"unknown(?!\s+(?-i:[A-Z]))|N/A"
    r"|I\s+don'?t\s+know"
    r"|(?:not\s+enough|insufficient)\s+information"
    r"|no\s+(?:information|mention)"
    r"|unable\s+to\s+(?:find|determine|answer)"
    r"|cannot\s+(?:determine|be\s+determined)"
    r"|not\s+possible\s+to\s+determine"
    r"|do(?:es)?\s+not\s+(?:contain|mention|provide|specify)"
    r")\b",
    re.IGNORECASE,
)
_KO_ABSTAIN_KWS = [kw for kw in ABSTAIN_KWS if any(ord(c) > 0x7F for c in kw)]

# 한글 음절 범위
_KO_SYLLABLE = re.compile(r'[가-힣]')


def _is_artifact(text: str) -> bool:
    """한국어·영어(ASCII)·한자 이외 문자가 지배적이면 True (키릴, 아랍, 말라얄람 등)."""
    core = text.replace(' ', '').replace('\n', '')
    if not core:
        return False
    # 한자(CJK Unified Ideographs)는 인명·지명에서 정상 출력이므로 유효 문자로 포함
    ko_en = sum(1 for c in core if '가' <= c <= '힣' or c.isascii() or '一' <= c <= '鿿')
    return ko_en / len(core) < 0.3


def is_judge_target(parse_result: "ParseResult", hop_data: dict) -> bool:
    """LLM judge 호출 대상 여부. 기권·애매·실행 오류·claim 없음 홉 제외."""
    execution_artifact = (
        hop_data.get("api_error", False)
        or hop_data.get("empty_content", False)
        or hop_data.get("empty_content_with_reasoning", False)
        or hop_data.get("thinking_tag_leak", False)
    )
    return (
        not parse_result.abstained
        and not parse_result.ambiguous_answer_marker
        and not execution_artifact
        and len(parse_result.claims) > 0
    )


def _is_soft_abstain(answer_text: str) -> bool:
    """answer_text 전체가 soft abstain 패턴과 정확히 매칭되면 True (단독 구성 시만)."""
    normalized = re.sub(r"[\s.()\[\]!?]", "", answer_text).lower()
    return normalized in _SOFT_ABSTAIN_NORMALIZED


def _extract_sentences(text: str, min_len: int = 10) -> list[str]:
    sentences = re.split(r"(?<=[.!?。])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > min_len]


def _check_abstain(answer: str) -> bool:
    """모델이 기권 표현을 했는지 휴리스틱 감지.

    한국어 키워드: 부분문자열 매칭 (조사 결합 형태 처리를 위해 그대로 유지).
    영어 키워드: 단어경계 정규식 매칭 ("The Unknown Soldier" 등 오탐 방지).
    """
    a_lower = answer.lower()
    if any(kw in a_lower for kw in _KO_ABSTAIN_KWS):
        return True
    return bool(_EN_ABSTAIN_RE.search(answer))


_MULTI_ANSWER_SEP = re.compile(r",\s+| 와 | 과 | 및 ")


def _is_multi_answer(answer_text: str) -> bool:
    """단일 답변 섹션 내에 복수 엔티티가 나열됐는지 감지."""
    segments = [s.strip() for s in _MULTI_ANSWER_SEP.split(answer_text)]
    return sum(1 for s in segments if len(s) >= 5) >= 2


def normalize_entity(text: str) -> str:
    """alias 비교용 정규화: 소문자, 공백·구두점 제거."""
    return re.sub(r"[^\w가-힣]", "", text.lower())


def parse_generated_answer(text: str) -> ParseResult:
    """에이전트 출력을 파싱해 claim 목록과 파서 메타데이터를 반환.

    답변 마커(답변:, 답변 :, 답변：) 수에 따른 보수적 정책:
      0개 — fallback: meta 라인 제거 후 잔여 텍스트에서 추출
      1개 — 최종 블록이 기권이면 abstained; 아니면 claim 추출
      ≥2개 — 최종 블록이 기권이면 abstained;
              비기권이면 ambiguous_answer_marker=True, no judge call

    source 값:
      answer_section / answer_section_short / answer_section_abstain
      fallback_cleaned / fallback_whole / fallback_short / fallback_abstain
      artifact  — 한국어·영어 이외 문자 지배적 출력 (judge 대상 제외)
      none      — 추출 불가
    """
    marker_count = len(_ANSWER_MARKER.findall(text))

    # artifact: 비KO/비EN 출력 (키릴·아랍·말라얄람 등)
    if _is_artifact(text):
        return ParseResult([], "artifact", False, marker_count, False, "artifact_output")

    if marker_count == 0:
        lines = [l for l in text.splitlines() if not _META_LINE.match(l.strip())]
        fallback_text = " ".join(l.strip() for l in lines if l.strip()).strip()
        fallback_text = _ANSWER_MARKER.sub("", fallback_text).strip()
        if not fallback_text:
            return ParseResult([], "none", False, 0, False, None)
        # 기권: marker_count==1 경로와 동일 기준(무게이트). 마커(답변:) 유무라는
        # 표면 토큰으로 같은 내용이 abstain/claim으로 갈리던 비대칭을 제거한다.
        if _check_abstain(fallback_text) or _is_soft_abstain(fallback_text):
            return ParseResult([], "fallback_abstain", True, 0, False, None)
        claims = _extract_sentences(fallback_text, min_len=5)
        if claims:
            return ParseResult(claims, "fallback_cleaned", False, 0, False, None)
        if len(fallback_text) >= 30:
            return ParseResult([fallback_text], "fallback_whole", False, 0, False, None)
        # 짧은 bare 답변 (v2.3: 명사구만 출력하는 케이스)
        return ParseResult([fallback_text], "fallback_short", False, 0, False, None)

    answer_section = _ANSWER_MARKER.split(text)[-1].strip()
    is_abstain = _check_abstain(answer_section) or _is_soft_abstain(answer_section)

    if marker_count == 1:
        if is_abstain:
            return ParseResult([], "answer_section_abstain", True, 1, False, None)
        claims = _extract_sentences(answer_section, min_len=10)
        multi = _is_multi_answer(answer_section)
        if claims:
            return ParseResult(claims, "answer_section", False, 1, False, None, multi)
        if answer_section:
            return ParseResult([answer_section], "answer_section_short", False, 1, False, None, multi)
        return ParseResult([], "none", False, 1, False, None)

    # marker_count >= 2
    if is_abstain:
        return ParseResult([], "answer_section_abstain", True, marker_count, False, None)
    return ParseResult([], "multi_answer_ambiguous", False, marker_count, True, "multi_answer_non_abstain")
