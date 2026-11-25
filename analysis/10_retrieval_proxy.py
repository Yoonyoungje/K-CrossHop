"""
analysis/10_retrieval_proxy.py
========================
hit_hop0_retrieval_v2.jsonl + trajectory passages → hit_hop0_retrieval_v3.jsonl

v2 alias 매칭 실패 레코드에 대해 normalized alias × normalized passage 재시도.
"""

import argparse
import json
import yaml
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.retrieval_success import check_answer_bearing_hit

DEFAULT_TRAJ_PATH  = "data/trajectories/aya_ecological.jsonl"
DEFAULT_V2_PATH    = "data/retrieval/hop0_retrieval_aya_ecological.jsonl"   # 08→09 기본 출력
DEFAULT_CACHE_PATH = "data/retrieval/alias_cache.json"
# 정본 hit 파일을 덮어쓰지 않도록 다른 이름에 쓴다
DEFAULT_OUT_PATH   = "data/retrieval/hits_rebuilt_aya_ecological.jsonl"


def load_traj_passages(path: Path) -> dict[str, dict[int, list[str]]]:
    """qid → {hop_idx: retrieved_passages} (전 홉)."""
    result: dict[str, dict[int, list[str]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        t = json.loads(line)
        result[t["qid"]] = {h["hop"]: h.get("retrieved_passages", [])
                            for h in t.get("trajectory", [])}
    return result


def load_qa_gold(path: Path) -> dict[str, dict[int, tuple[str, str]]]:
    """qid → {hop_idx: (gold_entity_qid, gold_lang)}. hop1 gold entity 출처(trajectory엔 없음)."""
    result: dict[str, dict[int, tuple[str, str]]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        qa = json.loads(line)
        result[qa["qid"]] = {h["hop"]: (h.get("gold_entity_qid", ""), h.get("gold_lang", "ko"))
                             for h in qa.get("hops", [])}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--trajectories", default=None)
    parser.add_argument("--v2-path",      default=None)
    parser.add_argument("--qa-pairs",     default=None,
                        help="hop1 gold_entity_qid 출처 (미지정 시 config output.qa_pairs)")
    parser.add_argument("--cache-path",   default=None)
    parser.add_argument("--output",       default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ana = cfg.get("analysis", {})

    traj_path  = Path(args.trajectories or cfg.get("output", {}).get("trajectories", DEFAULT_TRAJ_PATH))
    v2_path    = Path(args.v2_path      or DEFAULT_V2_PATH)
    qa_path    = Path(args.qa_pairs     or cfg.get("output", {}).get("qa_pairs"))
    cache_path = Path(args.cache_path   or ana.get("alias_cache", DEFAULT_CACHE_PATH))
    out_path   = Path(args.output       or DEFAULT_OUT_PATH)

    cache      = json.loads(cache_path.read_text())
    v2_records = [json.loads(l) for l in v2_path.read_text().splitlines() if l.strip()]
    traj_map   = load_traj_passages(traj_path)
    qa_gold    = load_qa_gold(qa_path)

    out_records = []
    # ── hop0: 기존 v2 레코드(alias v2 필드 보존) + hop0 passages로 v3 ──────────
    for rec in v2_records:
        qid        = rec["qid"]
        entity_qid = rec.get("gold_entity_qid", "")
        gold_lang  = rec.get("gold_lang", "ko")
        passages   = traj_map.get(qid, {}).get(0, [])
        v3 = check_answer_bearing_hit(passages, entity_qid, cache, gold_lang)
        out_records.append({**rec, **v3, "hop": 0})

    # ── hop1: qa(gold entity) + trajectory(passages)로 v3 (별도 플래그 없이 항상 산출) ──
    for rec in v2_records:
        qid = rec["qid"]
        h1 = qa_gold.get(qid, {}).get(1)
        if h1 is None:
            continue
        entity_qid, gold_lang = h1
        passages = traj_map.get(qid, {}).get(1, [])
        v3 = check_answer_bearing_hit(passages, entity_qid, cache, gold_lang)
        out_records.append({"qid": qid, "hop": 1, "pattern": rec.get("pattern"),
                            "gold_lang": gold_lang, "gold_entity_qid": entity_qid, **v3})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_records))
    n0 = sum(1 for r in out_records if r["hop"] == 0)
    n1 = sum(1 for r in out_records if r["hop"] == 1)
    print(f"저장: {out_path}  ({len(out_records)}건 = hop0 {n0} + hop1 {n1})")

    # ── 비교 표: hop0 (v2 vs v3) + hop1 (v3) ─────────────────────────────────
    hop0 = [r for r in out_records if r["hop"] == 0]
    hop1 = [r for r in out_records if r["hop"] == 1]
    patterns = sorted({r["pattern"] for r in hop0})

    # v2 필드가 없는 운반체(qa_pairs 에서 만든 qid·gold 목록)면 v2 비교 열을 생략한다.
    has_v2 = all("answer_bearing_hit_alias" in r for r in hop0)
    print(f"\n[hop0]  {'패턴':^6} " + (f"{'v2%':>7} " if has_v2 else "") + f"{'v3%':>7}" + (f" {'Δ':>6}" if has_v2 else ""))
    print("─" * (32 if has_v2 else 22))
    for pat in patterns + ["전체"]:
        subset = hop0 if pat == "전체" else [r for r in hop0 if r["pattern"] == pat]
        n = len(subset) or 1
        pct_hit = sum(1 for r in subset if r["answer_bearing_hit"]) / n * 100
        if has_v2:
            pct_v2 = sum(1 for r in subset if r["answer_bearing_hit_alias"]) / n * 100
            print(f"  {pat:^4}  {pct_v2:6.1f}%  {pct_hit:6.1f}%  {pct_hit - pct_v2:+5.1f}%")
        else:
            print(f"  {pat:^4}  {pct_hit:6.1f}%")

    print(f"\n[hop1]  {'패턴':^6} {'v3%':>7}")
    print("─" * 22)
    for pat in patterns + ["전체"]:
        subset = hop1 if pat == "전체" else [r for r in hop1 if r["pattern"] == pat]
        n = len(subset) or 1
        pct_hit = sum(1 for r in subset if r["answer_bearing_hit"]) / n * 100
        print(f"  {pat:^4}  {pct_hit:6.1f}%")

    # match_method 분포 (전 홉)
    method_counter: Counter = Counter(r["match_method"] for r in out_records)
    print("\nmatch_method 분포 (전 홉):")
    for method, cnt in sorted(method_counter.items()):
        print(f"  {method}: {cnt}건")

    # false positive 경고 (hop0 기준)
    if hop0 and has_v2:
        v3_ok = sum(1 for r in hop0 if r["answer_bearing_hit"])
        v2_ok = sum(1 for r in hop0 if r["answer_bearing_hit_alias"])
        delta_pct = (v3_ok - v2_ok) / len(hop0) * 100
        if delta_pct >= 20:
            print(f"\n⚠ hop0 v3 success 급증 ({delta_pct:+.1f}%p) — false positive 재점검 필요")


if __name__ == "__main__":
    main()
