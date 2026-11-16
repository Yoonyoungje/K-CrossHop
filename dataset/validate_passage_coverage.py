#!/usr/bin/env python3
"""
dataset/validate_passage_coverage.py

QA pairs의 모든 gold_entity_qid × gold_lang 조합이 passages에 존재하는지 검증.
데이터셋 freeze 전, FAISS 빌드 전에 실행한다.

사용법:
  # 검증만
  python dataset/validate_passage_coverage.py \
    --qa-pairs  data/dataset/qa_pairs.jsonl \
    --passages  data/corpus/passages.jsonl

  # 검증 + 누락 패시지 자동 보완
  python dataset/validate_passage_coverage.py \
    --qa-pairs  data/dataset/qa_pairs.jsonl \
    --passages  data/corpus/passages.jsonl \
    --alias-cache data/retrieval/alias_cache.json \
    --fix
"""

import argparse
import dataclasses
import json
import sys
import yaml
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.wiki_api import WikiPassageFetcher


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def check_coverage(qa_items: list[dict], passages: list[dict]) -> list[tuple]:
    """누락된 (entity_qid, lang, gold_entity) 목록 반환."""
    has = {(p["entity_qid"], p["lang"]) for p in passages}
    missing = {}
    for qa in qa_items:
        for h in qa.get("hops", []):
            qid = h.get("gold_entity_qid", "")
            gl  = h.get("gold_lang", "")
            if qid and (qid, gl) not in has:
                missing[(qid, gl)] = h.get("gold_entity", "")
    return [(qid, gl, label) for (qid, gl), label in sorted(missing.items())]


def fetch_missing(missing: list[tuple], alias_cache: dict,
                  fetcher: WikiPassageFetcher) -> list[dict]:
    """누락 엔티티 패시지를 fetch해 dict 리스트로 반환."""
    new_passages = []
    errors = []
    for qid, lang, label in missing:
        wiki_title = alias_cache.get(qid, {}).get(f"wiki_title_{lang}", "")
        if not wiki_title:
            print(f"  [SKIP] {qid} {lang} '{label}': wiki_title 없음 (alias_cache 미등록)")
            errors.append((qid, lang, "wiki_title 없음"))
            continue
        print(f"  fetch {qid} {lang} '{wiki_title}' ...", end=" ", flush=True)
        try:
            result = fetcher.fetch_passages(qid, label, wiki_title, lang)
            plist = result[0] if isinstance(result, tuple) else result
            dicts = []
            for p in plist:
                if dataclasses.is_dataclass(p):
                    dicts.append(dataclasses.asdict(p))
                elif hasattr(p, "_asdict"):
                    dicts.append(p._asdict())
                else:
                    dicts.append(vars(p))
            new_passages.extend(dicts)
            print(f"{len(dicts)}개")
        except Exception as e:
            print(f"실패: {e}")
            errors.append((qid, lang, str(e)))

    if errors:
        print(f"\n  [경고] fetch 실패 {len(errors)}개:")
        for qid, lang, err in errors:
            print(f"    {qid} {lang}: {err}")
    return new_passages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-pairs",    required=True, help="qa_pairs.jsonl 경로")
    parser.add_argument("--passages",    required=True, help="passages.jsonl 경로")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--alias-cache", default=None,
                        help="alias_cache.json 경로 (--fix 시 필요)")
    parser.add_argument("--fix", action="store_true",
                        help="누락 패시지를 자동으로 fetch해 passages 파일에 추가")
    args = parser.parse_args()

    qa_path  = Path(args.qa_pairs)
    pas_path = Path(args.passages)

    qa_items = load_jsonl(qa_path)
    passages = load_jsonl(pas_path)

    total_hops = sum(len(qa.get("hops", [])) for qa in qa_items)
    print(f"QA pairs: {len(qa_items)}개 ({total_hops} hop)")
    print(f"Passages: {len(passages):,}개")

    missing = check_coverage(qa_items, passages)

    if not missing:
        print("\n✓ 누락 없음 — 모든 gold_entity_qid × gold_lang 커버됨")
        return 0

    # 영향받는 QA item 수
    has = {(p["entity_qid"], p["lang"]) for p in passages}
    affected = {qa["qid"] for qa in qa_items
                for h in qa.get("hops", [])
                if h.get("gold_entity_qid") and
                (h["gold_entity_qid"], h.get("gold_lang","")) not in has}

    print(f"\n✗ 누락 {len(missing)}개 (영향 QA: {len(affected)}개 / {len(qa_items)}개)")
    for qid, lang, label in missing:
        print(f"  {qid:12s} {lang}  '{label}'")

    if not args.fix:
        print("\n→ --fix 플래그를 추가하면 자동으로 보완합니다.")
        return 1

    # --fix: 누락 패시지 fetch
    if not args.alias_cache:
        print("\n[ERROR] --fix 사용 시 --alias-cache 필요")
        return 1

    alias_cache = json.loads(Path(args.alias_cache).read_text(encoding="utf-8"))
    cfg         = yaml.safe_load(Path(args.config).read_text())
    wiki_cfg    = cfg["wikipedia"]
    fetcher     = WikiPassageFetcher(
        chunk_size=wiki_cfg.get("chunk_size", 350),
        overlap=wiki_cfg.get("chunk_overlap", 50),
        min_len=wiki_cfg.get("min_len", 80),
    )

    print(f"\n누락 {len(missing)}개 fetch 시작...")
    new_passages = fetch_missing(missing, alias_cache, fetcher)

    if new_passages:
        with pas_path.open("a", encoding="utf-8") as f:
            for p in new_passages:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\n{len(new_passages)}개 패시지 추가 → {pas_path}")

    # 재검증
    passages_updated = load_jsonl(pas_path)
    remaining = check_coverage(qa_items, passages_updated)
    if not remaining:
        print("✓ 재검증 완료 — 누락 없음")
        return 0
    else:
        print(f"✗ 여전히 {len(remaining)}개 누락 (wiki_title 없거나 fetch 실패)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
