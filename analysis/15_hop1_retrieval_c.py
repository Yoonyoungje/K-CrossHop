#!/usr/bin/env python3
"""
13_hop1_retrieval_c.py
======================
패턴 c hop1 answer_bearing_hit 프록시 계산.

- 대상: 패턴 c hop1 (gold_lang=en, lang_switch=True)
- hop1 gold_entity_qid → Wikidata alias 조회 → 기존 cache에 추가
- hop1 retrieved_passages × alias 매칭 → answer_bearing_hit 계산
- 출력: data/retrieval/hop1c_retrieval_v3_{model}.jsonl

실행:
  python 13_hop1_retrieval_c.py \\
    --trajectories data/trajectories/aya_ecological.jsonl \\
    --qa-pairs     data/dataset/qa_pairs.jsonl \\
    --cache        data/retrieval/alias_cache.json \\
    --output-dir   data/retrieval/
"""

import os
import argparse
import json
import time
import urllib.parse
import urllib.request
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)
from utils.retrieval_success import check_answer_bearing_hit

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
USER_AGENT   = os.getenv("WIKI_USER_AGENT", "K-CrossHop-Research/1.0 (research use)")
BATCH_SIZE   = 50
RATE_DELAY   = 1.0

DEFAULT_CACHE = "data/retrieval/alias_cache.json"
DEFAULT_OUT   = "data/retrieval/"


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _fetch_batch(qids: list[str]) -> dict:
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
    def label(lang):
        return ent.get("labels", {}).get(lang, {}).get("value", "")
    def aliases(lang):
        return [a["value"] for a in ent.get("aliases", {}).get(lang, [])]
    def wiki_title(site):
        return ent.get("sitelinks", {}).get(site, {}).get("title", "")
    return {
        "qid":           qid,
        "label_ko":      label("ko"),
        "label_en":      label("en"),
        "wiki_title_ko": wiki_title("kowiki"),
        "wiki_title_en": wiki_title("enwiki"),
        "aliases_ko":    aliases("ko"),
        "aliases_en":    aliases("en"),
    }


def update_cache(entity_qids: list[str], cache_path: Path) -> dict:
    """캐시에 없는 entity_qids만 Wikidata에서 조회해 캐시에 추가."""
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"캐시 로드: {len(cache)}개 엔티티")

    missing = [q for q in entity_qids if q not in cache]
    if not missing:
        print(f"모든 hop1 엔티티 캐시 커버됨 ({len(entity_qids)}개)")
        return cache

    n_unique = len(set(missing))
    print(f"Wikidata 조회 필요: {n_unique}개 엔티티 ({(n_unique-1)//BATCH_SIZE + 1}배치)")
    unique_missing = list(dict.fromkeys(missing))  # 순서 유지 중복 제거
    for i in range(0, len(unique_missing), BATCH_SIZE):
        batch = unique_missing[i:i + BATCH_SIZE]
        print(f"  배치 {i//BATCH_SIZE + 1}/{(len(unique_missing)-1)//BATCH_SIZE + 1}: {len(batch)}개 ... ", end="", flush=True)
        try:
            entities = _fetch_batch(batch)
            for qid in batch:
                ent = entities.get(qid, {})
                if ent.get("missing"):
                    cache[qid] = {"qid": qid, "label_ko": "", "label_en": "", "wiki_title_ko": "", "wiki_title_en": "", "aliases_ko": [], "aliases_en": []}
                else:
                    cache[qid] = _parse_entity(qid, ent)
            print("OK")
        except Exception as e:
            print(f"오류: {e}")
        if i + BATCH_SIZE < len(unique_missing):
            time.sleep(RATE_DELAY)

    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"캐시 저장: {len(cache)}개 → {cache_path}")
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--trajectories", default=None)
    parser.add_argument("--qa-pairs",     default=None)
    parser.add_argument("--cache",        default=None)
    parser.add_argument("--output-dir",   default=None)
    parser.add_argument("--all-patterns", action="store_true",
                        help="모든 패턴 hop1 v3 산출 (기본: 패턴 c 전용). "
                             "gold_lang은 각 QA hops[1] 기준 (a/b=ko, c/d=en).")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ana = cfg.get("analysis", {})
    out_cfg = cfg.get("output", {})

    traj_path  = Path(args.trajectories or out_cfg.get("trajectories", DEFAULT_CACHE))
    cache_path = args.cache      or ana.get("alias_cache",  DEFAULT_CACHE)
    out_dir    = args.output_dir or ana.get("analysis_dir", DEFAULT_OUT)

    model = traj_path.stem.replace("trajectories_", "")
    out_name = "hop1_retrieval_v3" if args.all_patterns else "hop1c_retrieval_v3"
    out_path = Path(out_dir) / f"{out_name}_{model}.jsonl"

    # QA pairs: hop1 gold_entity_qid 수집 (--all-patterns면 전 패턴, 아니면 c 전용)
    qa_records = load_jsonl(args.qa_pairs)
    qa_map = {r["qid"]: r for r in qa_records}
    target_pats = set("abcd") if args.all_patterns else {"c"}
    hop1_qids = {
        r["qid"]: r["hops"][1]["gold_entity_qid"]
        for r in qa_records
        if r["pattern"] in target_pats and len(r["hops"]) >= 2
    }
    hop1_lang = {
        r["qid"]: r["hops"][1].get("gold_lang", "en")
        for r in qa_records
        if r["pattern"] in target_pats and len(r["hops"]) >= 2
    }
    print(f"대상 패턴 {sorted(target_pats)} QA: {len(hop1_qids)}개, "
          f"고유 hop1 엔티티: {len(set(hop1_qids.values()))}개")

    # alias cache 업데이트
    cache = update_cache(list(hop1_qids.values()), Path(cache_path))

    # trajectories: 대상 패턴 hop1 passages 로드 + answer_bearing_hit 계산
    traj_records = load_jsonl(args.trajectories)
    sel_traj = [r for r in traj_records if r["pattern"] in target_pats]
    print(f"\n대상 trajectory: {len(sel_traj)}개 (model={model})")

    out_records = []
    n_succ = n_total = 0

    for traj in sel_traj:
        qid = traj["qid"]
        hop1_ent_qid = hop1_qids.get(qid)
        if hop1_ent_qid is None:
            continue
        gold_lang = hop1_lang.get(qid, "en")

        # hop1 retrieved passages
        hop1_data = next((h for h in traj["trajectory"] if h["hop"] == 1), None)
        if hop1_data is None:
            continue
        passages = hop1_data.get("retrieved_passages", [])
        if not isinstance(passages, list):
            passages = []

        hit_result = check_answer_bearing_hit(passages, hop1_ent_qid, cache, gold_lang)

        gold_entity = qa_map[qid]["hops"][1].get("gold_entity", "")
        n_total += 1
        if hit_result["answer_bearing_hit"]:
            n_succ += 1

        out_records.append({
            "qid":                    qid,
            "pattern":                traj["pattern"],
            "hop":                    1,
            "gold_lang":              gold_lang,
            "gold_entity":            gold_entity,
            "gold_entity_qid":        hop1_ent_qid,
            "n_passages":             len(passages),
            "answer_bearing_hit":   hit_result["answer_bearing_hit"],
            "match_method":        hit_result["match_method"],
            "hit_matched_alias":       hit_result["hit_matched_alias"],
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n결과: {n_succ}/{n_total} = {n_succ/n_total:.1%} (answer_bearing_hit=True)")
    print(f"출력: {out_path}")


if __name__ == "__main__":
    main()
