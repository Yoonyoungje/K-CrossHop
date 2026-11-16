#!/usr/bin/env python3
"""
analysis/09_alias_retrieval.py
=========================
alias-aware answer_bearing_hit_alias 계산.

1. gold_entity_qid 목록을 Wikidata wbgetentities API로 배치 조회
2. alias set (label + wiki_title + aliases, ko/en) 를 data/v6/alias_cache.json 에 캐싱
3. retrieved passages 에서 alias hit 여부로 answer_bearing_hit_alias 계산
4. hit_hop0_retrieval.jsonl 에 v2 필드 추가 (in-place update)
5. v1 vs v2 비교 요약 출력

실행:
  python analysis/09_alias_retrieval.py
  python analysis/09_alias_retrieval.py --rebuild-cache   # 캐시 무시하고 재조회
  python analysis/09_alias_retrieval.py --dry-run          # 캐시만 구축, JSONL 수정 안 함
"""

import os
import argparse
import json
import time
import urllib.parse
import urllib.request
import yaml
from collections import defaultdict
from pathlib import Path

# ─── 상수 ────────────────────────────────────────────────────────────────────

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT   = os.getenv("WIKI_USER_AGENT", "K-CrossHop-Research/1.0 (research use)")
BATCH_SIZE   = 50      # wbgetentities 최대 50 QIDs/요청
RATE_DELAY   = 1.0     # 배치 간 대기 (초)
MIN_ALIAS_LEN = 4      # false positive 방지: 이 길이 이하 alias는 매칭에서 제외

DEFAULT_HOP0_PATH  = "data/retrieval/hop0_retrieval_aya_ecological.jsonl"   # 08 기본 출력
DEFAULT_CACHE_PATH = "data/retrieval/alias_cache.json"

# ─── Wikidata API ─────────────────────────────────────────────────────────────

def _fetch_batch(qids: list[str]) -> dict:
    """wbgetentities 단일 배치 조회. {qid: entity_data} 반환."""
    params = urllib.parse.urlencode({
        "action":     "wbgetentities",
        "ids":        "|".join(qids),
        "props":      "labels|aliases|sitelinks",
        "languages":  "ko|en",
        "sitefilter": "kowiki|enwiki",
        "format":     "json",
    })
    req = urllib.request.Request(
        f"{WIKIDATA_API}?{params}",
        headers={"User-Agent": USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read())["entities"]


def _parse_entity(qid: str, ent: dict) -> dict:
    """Wikidata entity → alias set dict."""
    def label(lang):
        return ent.get("labels", {}).get(lang, {}).get("value", "")

    def aliases(lang):
        return [a["value"] for a in ent.get("aliases", {}).get(lang, [])]

    def wiki_title(site):
        return ent.get("sitelinks", {}).get(site, {}).get("title", "")

    return {
        "qid":          qid,
        "label_ko":     label("ko"),
        "label_en":     label("en"),
        "wiki_title_ko": wiki_title("kowiki"),
        "wiki_title_en": wiki_title("enwiki"),
        "aliases_ko":   aliases("ko"),
        "aliases_en":   aliases("en"),
    }


def build_cache(qids: list[str], cache_path: Path,
                rebuild: bool = False) -> dict:
    """alias cache 구축. 기존 캐시가 있으면 미조회 QID만 추가."""
    cache: dict = {}
    if cache_path.exists() and not rebuild:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"캐시 로드: {len(cache)}개 (path={cache_path})")

    missing = [q for q in qids if q not in cache]
    if not missing:
        print("모든 QID가 캐시에 있음. API 조회 생략.")
        return cache

    print(f"Wikidata API 조회: {len(missing)}개 QID ({len(missing)//BATCH_SIZE + 1}배치)")
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        print(f"  배치 {i//BATCH_SIZE + 1}: {len(batch)}개 ... ", end="", flush=True)
        try:
            entities = _fetch_batch(batch)
            for qid in batch:
                ent = entities.get(qid, {})
                if "missing" in ent:
                    # Wikidata에 없는 QID → 빈 alias set
                    cache[qid] = {"qid": qid, "label_ko": "", "label_en": "",
                                  "wiki_title_ko": "", "wiki_title_en": "",
                                  "aliases_ko": [], "aliases_en": []}
                else:
                    cache[qid] = _parse_entity(qid, ent)
            print("완료")
        except Exception as e:
            print(f"실패: {e}")
        if i + BATCH_SIZE < len(missing):
            time.sleep(RATE_DELAY)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"캐시 저장: {cache_path} ({len(cache)}개)")
    return cache

# ─── Alias set 구성 ───────────────────────────────────────────────────────────

def _alias_set(entry: dict) -> list[str]:
    """alias set 구성 (MIN_ALIAS_LEN 초과 항목만, 중복 제거)."""
    candidates = [
        entry.get("label_ko", ""),
        entry.get("label_en", ""),
        entry.get("wiki_title_ko", ""),
        entry.get("wiki_title_en", ""),
        *entry.get("aliases_ko", []),
        *entry.get("aliases_en", []),
    ]
    seen: set[str] = set()
    result = []
    for c in candidates:
        c = c.strip()
        if len(c) >= MIN_ALIAS_LEN and c.lower() not in seen:
            seen.add(c.lower())
            result.append(c)
    return result


def _match(passages: list[str], aliases: list[str]) -> tuple[bool, str]:
    """
    passages 중 하나에 aliases 중 하나라도 포함되면 (True, matched_alias).
    모두 miss이면 (False, "").
    """
    for alias in aliases:
        alias_lower = alias.lower()
        for p in passages:
            if alias_lower in p.lower():
                return True, alias
    return False, ""

# ─── v2 계산 ─────────────────────────────────────────────────────────────────

def compute_v2(records: list[dict], cache: dict) -> list[dict]:
    updated = []
    for r in records:
        qid_ent = r.get("gold_entity_qid", "")
        entry = cache.get(qid_ent, {})
        aliases = _alias_set(entry) if entry else []

        passages = []
        # retrieved_passages는 hop0_retrieval.jsonl에는 없음 → 원본 trajectory 참조 불필요
        # 이 스크립트는 별도로 trajectory를 다시 읽어서 passages를 가져옴
        # (아래 main()에서 passages를 r에 임시 첨부)
        passages = r.get("_passages", [])

        hit, matched = _match(passages, aliases)

        r2 = dict(r)
        r2["answer_bearing_hit_alias"]   = hit
        r2["v2_aliases_tried"]       = aliases
        r2["v2_matched_alias"]       = matched
        r2["v2_alias_count"]         = len(aliases)
        # 임시 필드 제거
        r2.pop("_passages", None)
        updated.append(r2)
    return updated

# ─── 비교 보고 ────────────────────────────────────────────────────────────────

def print_comparison(records: list[dict]) -> None:
    total = len(records)

    def rate(lst, key):
        return sum(1 for r in lst if r.get(key)) / len(lst) if lst else float("nan")

    by_pat = defaultdict(list)
    for r in records:
        by_pat[r["pattern"]].append(r)

    print(f"\n{'='*55}")
    print(f"  Retrieval Success: v1 (string-match) vs v2 (alias-aware)")
    print(f"{'='*55}")
    print(f"{'Pattern':<10} {'N':>4}  {'v1':>6}  {'v2':>6}  {'Δ(v2-v1)':>9}")
    print(f"{'-'*55}")
    for pat in "abcd":
        recs = by_pat.get(pat, [])
        if not recs:
            continue
        v1 = rate(recs, "retrieval_success_any")
        v2 = rate(recs, "answer_bearing_hit_alias")
        delta = v2 - v1
        print(f"  {pat:<8} {len(recs):>4}  {v1:>6.1%}  {v2:>6.1%}  {delta:>+8.1%}")
    print(f"{'-'*55}")
    v1_all = rate(records, "retrieval_success_any")
    v2_all = rate(records, "answer_bearing_hit_alias")
    print(f"  {'ALL':<8} {total:>4}  {v1_all:>6.1%}  {v2_all:>6.1%}  {v2_all-v1_all:>+8.1%}")

    # trustworthy subset 비교
    tw = [r for r in records if r.get("trustworthy")]
    print(f"\n  [Trustworthy subset N={len(tw)} | gold_lang=ko, entity_len>3]")
    print(f"  v1={rate(tw, 'retrieval_success_any'):.1%}  v2={rate(tw, 'answer_bearing_hit_alias'):.1%}")

    # xling mismatch 케이스에서 v2 회수율
    mm = [r for r in records if r.get("xling_mismatch")]
    print(f"\n  [Cross-lingual mismatch N={len(mm)} | gold_lang=en, KO-label entity]")
    print(f"  v1={rate(mm, 'retrieval_success_any'):.1%}  v2={rate(mm, 'answer_bearing_hit_alias'):.1%}")
    print(f"  → v2로 회수된 케이스: {sum(1 for r in mm if r.get('answer_bearing_hit_alias'))}/{len(mm)}")

    # v2 alias 통계
    zero_alias = [r for r in records if r.get("v2_alias_count", 0) == 0]
    print(f"\n  alias set 비어있는 QID: {len(zero_alias)}/{total}")
    print(f"  평균 alias 수: {sum(r.get('v2_alias_count',0) for r in records)/total:.1f}")

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="config.yaml")
    parser.add_argument("--hop0-path",     default=None)
    parser.add_argument("--trajectories",  default=None)
    parser.add_argument("--cache-path",    default=None)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--dry-run",       action="store_true",
                        help="캐시만 구축하고 JSONL은 수정하지 않음")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ana = cfg.get("analysis", {})

    hop0_path  = Path(args.hop0_path  or ana.get("hop0_path",   DEFAULT_HOP0_PATH))
    cache_path = Path(args.cache_path or ana.get("alias_cache", DEFAULT_CACHE_PATH))
    traj_arg   = args.trajectories    or cfg.get("output", {}).get("trajectories")

    # hop0 레코드 로드
    records = [json.loads(l) for l in hop0_path.open(encoding="utf-8")]
    print(f"hop0 레코드 로드: {len(records)}개")

    # trajectory에서 hop0 passages 매핑 (qid → passages)
    traj_path = Path(traj_arg)
    passages_by_qid: dict[str, list[str]] = {}
    for line in traj_path.open(encoding="utf-8"):
        t = json.loads(line)
        hop0 = next((h for h in t.get("trajectory", []) if h["hop"] == 0), None)
        if hop0:
            passages_by_qid[t["qid"]] = hop0.get("retrieved_passages", [])
    print(f"trajectory passages 로드: {len(passages_by_qid)}개 QA")

    # passages를 레코드에 임시 첨부
    for r in records:
        r["_passages"] = passages_by_qid.get(r["qid"], [])

    # alias cache 구축
    all_qids = sorted(set(r["gold_entity_qid"] for r in records if r["gold_entity_qid"]))
    cache = build_cache(all_qids, cache_path, rebuild=args.rebuild_cache)

    if args.dry_run:
        print("--dry-run: JSONL 수정 생략.")
        # passages 임시 필드 제거
        for r in records:
            r.pop("_passages", None)
        return

    # v2 계산
    updated = compute_v2(records, cache)

    # JSONL 덮어쓰기
    with hop0_path.open("w", encoding="utf-8") as f:
        for r in updated:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {hop0_path} (v2 필드 추가됨)")

    print_comparison(updated)


if __name__ == "__main__":
    main()
