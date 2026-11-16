"""
utils/wiki_api.py
Wikipedia 패시지 fetcher
- KO / EN Wikipedia API 래퍼
- 엔티티 → 패시지 목록(chunked) 반환
"""

import os
import re
import time
import threading
import logging
from dataclasses import dataclass, asdict
import wikipediaapi
from wikipediaapi.exceptions import WikiRateLimitError

logger = logging.getLogger(__name__)



@dataclass
class Passage:
    passage_id: str       # "{entity_qid}_{lang}_{chunk_idx}"
    entity_qid: str       # Wikidata QID (e.g., "Q123456")
    entity_label: str     # 엔티티 레이블
    lang: str             # "ko" | "en"
    wiki_title: str       # Wikipedia 페이지 제목
    section: str          # 섹션 제목 (빈 문자열 = 서문)
    text: str             # 패시지 텍스트
    chunk_idx: int        # 페이지 전역 청크 순서 (섹션 경계 무관)

    def to_dict(self) -> dict:
        return asdict(self)


class WikiPassageFetcher:
    """KO/EN Wikipedia에서 엔티티 패시지를 추출한다."""

    def __init__(self, chunk_size: int = 350, overlap: int = 50, min_len: int = 80):
        self.chunk_size = chunk_size   # 단어 기준 (토큰 근사)
        self.overlap = overlap
        self.min_len = min_len

        ua = os.getenv("WIKI_USER_AGENT", "K-CrossHop-Research/1.0 (research use)")
        # max_retries=5: 429·5xx·timeout 시 최대 5회 재시도
        # retry_wait=2: 기본 대기 2s → exponential backoff (2, 4, 8, 16, 32s)
        # 429에는 Retry-After 헤더 값이 우선 적용됨
        self._apis = {
            lang: wikipediaapi.Wikipedia(
                language=lang, user_agent=ua,
                max_retries=5, retry_wait=2,
            )
            for lang in ("ko", "en")
        }
        # per-instance rate limit: 여러 인스턴스를 병렬 사용할 때 독립적으로 동작
        self._rate_lock = threading.Lock()
        self._last_req_time = 0.0

    def _rate_limit(self, min_interval: float = 1.2) -> None:
        """인스턴스별 락으로 요청 간격을 보장."""
        with self._rate_lock:
            elapsed = time.time() - self._last_req_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            self._last_req_time = time.time()

    def fetch_passages(self, qid: str, label: str,
                       wiki_title: str, lang: str) -> tuple[list[Passage], str]:
        """
        단일 엔티티의 Wikipedia 페이지를 청크로 나눠 반환.

        반환: (passages, status)
          status: "ok" | "not_found"
          429 소진 시 WikiRateLimitError를 호출자에게 전파.
        """
        api = self._apis.get(lang)
        if api is None:
            raise ValueError(f"지원하지 않는 언어: {lang}")

        self._rate_limit()
        page = api.page(wiki_title)

        if not page.exists():
            logger.warning(f"[{lang}] 페이지 없음: {wiki_title!r}")
            return [], "not_found"

        passages: list[Passage] = []
        chunk_global = 0

        sections = [("", page.summary)] + [
            (s.title, s.text) for s in page.sections
        ]

        for sec_title, sec_text in sections:
            sec_text = _clean_text(sec_text)
            if not sec_text:
                continue
            is_summary = (sec_title == "")
            chunks = _chunk_words(sec_text, self.chunk_size, self.overlap)
            for chunk in chunks:
                if not is_summary and len(chunk.split()) < self.min_len:
                    continue
                passages.append(Passage(
                    passage_id=f"{qid}_{lang}_{chunk_global:04d}",
                    entity_qid=qid,
                    entity_label=label,
                    lang=lang,
                    wiki_title=page.title,
                    section=sec_title,
                    text=chunk,
                    chunk_idx=chunk_global,
                ))
                chunk_global += 1

        logger.info(f"[{lang}] {wiki_title!r}: {chunk_global}개 패시지 추출")
        return passages, "ok"


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """위키피디아 마크업 잔재 제거 및 공백 정리."""
    text = re.sub(r"\[\d+\]", "", text)           # 각주 번호
    text = re.sub(r"={2,}.*?={2,}", "", text)      # == 섹션 헤더 ==
    text = re.sub(r"\s+", " ", text)               # 연속 공백
    return text.strip()


def _chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """
    단어(space-split) 기준 청킹.
    영어는 단어 단위 OK; 한국어는 어절 단위 근사(어차피 BGE-M3 입력이 토큰이므로 무방).
    """
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += size - overlap
    return chunks
