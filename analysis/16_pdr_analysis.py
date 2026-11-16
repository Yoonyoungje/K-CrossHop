#!/usr/bin/env python3
"""
analysis/16_pdr_analysis.py
==================
PDR (Parametric Dependency Ratio) — 연구계획서 §10.10 정의.

PDR = P(gold entity in generated_text | answer_bearing_hit=False, non-abstained)
    = 검색 실패 홉 중 생성 텍스트에 gold entity(alias 포함)가 언급된 비율

ΔPDR = PDR(cross-index) − PDR(same-index)
     → 양수 = cross-index에서 파라메트릭 지식 의존도 높음

기존 analysis/07_analyze.py의 PDR(unsupported/total)과 별개 지표.

입력:
  --trajectories   data/trajectories/trajectories_{model}_controlled.jsonl
  --retrieval-hits        data/retrieval/retrieval_v3_{model}_controlled.jsonl
  --alias-cache    data/retrieval/alias_cache.json
  --annotated      data/annotated/{model}_controlled.jsonl  (abstain 판단용)

실행:
  python analysis/16_pdr_analysis.py \\
    --trajectories data/trajectories/trajectories_aya_controlled.jsonl \\
    --retrieval-hits      data/retrieval/retrieval_v3_aya_controlled.jsonl \\
    --alias-cache  data/retrieval/alias_cache.json \\
    --annotated    data/annotated/aya_controlled.jsonl
"""

import argparse
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def build_alias_surfaces(qid: str, alias_cache: dict, gold_entity: str) -> set[str]:
    """QID → 정규화된 alias 집합. cache miss 시 gold_entity 단독 사용."""
    entry = alias_cache.get(qid)
    if not entry:
        return {normalize(gold_entity)} if gold_entity else set()
    names = []
    for key in ("label_ko", "label_en", "wiki_title_ko", "wiki_title_en"):
        v = entry.get(key)
        if v:
            names.append(v)
    names.extend(entry.get("aliases_ko", []))
    names.extend(entry.get("aliases_en", []))
    return {normalize(n) for n in names if n}


def entity_in_text(aliases: set[str], text: str) -> bool:
    t = normalize(text)
    return any(a in t for a in aliases if a)


def section(title: str):
    print(f"\n{'='*62}")
    print(title)
    print("=" * 62)


def fmt(val, n=None) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "  N/A  "
    s = f"{val:.3f}"
    if n is not None:
        s += f" (n={n})"
    return s


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--retrieval-hits",      required=True)
    parser.add_argument("--alias-cache",  required=True)
    parser.add_argument("--annotated",    required=True)
    parser.add_argument("--out", default=None,
                        help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    args = parser.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    alias_cache  = json.load(open(args.alias_cache, encoding="utf-8"))
    traj_records = load_jsonl(args.trajectories)
    hit_records   = load_jsonl(args.retrieval_hits)
    ann_records  = load_jsonl(args.annotated)

    # ── 인덱스 구축 ────────────────────────────────────────────────────────────
    # qid → hop → generated_text
    gen_text: dict[str, dict[int, str]] = defaultdict(dict)
    for r in traj_records:
        for step in r["trajectory"]:
            gen_text[r["qid"]][step["hop"]] = step.get("generated_text", "")

    # qid → hop → abstained
    abstained_map: dict[str, dict[int, bool]] = defaultdict(dict)
    for r in ann_records:
        for h in r["hops"]:
            abstained_map[r["qid"]][h["hop"]] = bool(
                h.get("abstained") or h.get("ambiguous_answer_marker")
            )

    # (qid, hop) → v3 레코드
    hit_map: dict[tuple, dict] = {(r["qid"], r["hop"]): r for r in hit_records}

    model = Path(args.trajectories).stem.replace("trajectories_", "")
    print(f"\n모델: {model}")
    print(f"  trajectory={len(traj_records)}, v3={len(hit_records)}, annotated={len(ann_records)}")

    # ── hop별 PDR 집계 ─────────────────────────────────────────────────────────
    # key: "hop0" / "hop1" / "all"  →  {"same": [correct, total], "cross": [correct, total]}
    counts: dict[str, dict[str, list[int]]] = {
        k: {"same": [0, 0], "cross": [0, 0]}
        for k in ("hop0", "hop1", "all")
    }

    for (qid, hop), v3 in hit_map.items():
        if v3.get("answer_bearing_hit", True):   # v3 success → PDR 분모 아님
            continue
        if abstained_map.get(qid, {}).get(hop, False):   # 기권 홉 제외
            continue

        # 빈 gen_text도 분모에 포함(correct=0). §10.10 조건집합={v3=False, non-abstained}을
        # 그대로 따르며 13_bootstrap_ci.build_pdr_data와 정의를 일치시킨다.
        gt = gen_text.get(qid, {}).get(hop, "")

        aliases = build_alias_surfaces(
            v3.get("gold_entity_qid", ""), alias_cache, v3.get("gold_entity", "")
        )
        correct = int(entity_in_text(aliases, gt))
        idx = "cross" if v3.get("gold_lang") == "en" else "same"
        hop_key = f"hop{hop}"

        counts[hop_key][idx][0] += correct
        counts[hop_key][idx][1] += 1
        counts["all"][idx][0] += correct
        counts["all"][idx][1] += 1

    # ── 출력 ───────────────────────────────────────────────────────────────────
    section("PDR: P(gold entity in gen_text | v3=False, non-abstained)  (§10.10)")
    print("  정의: 검색 실패(v3=False) + 비기권 홉 중 생성 텍스트에 gold entity/alias 포함 비율")
    print("  ΔPDR = PDR(cross-index) − PDR(same-index)")
    print()
    print(f"  {'구분':<8} {'same-index PDR':>20} {'cross-index PDR':>20} {'ΔPDR':>8}")
    print("  " + "-" * 60)

    for key in ("hop0", "hop1", "all"):
        sc, st = counts[key]["same"]
        cc, ct = counts[key]["cross"]
        same_pdr  = sc / st if st > 0 else float("nan")
        cross_pdr = cc / ct if ct > 0 else float("nan")
        dpdr = (cross_pdr - same_pdr
                if not (math.isnan(same_pdr) or math.isnan(cross_pdr))
                else float("nan"))
        label = "전체" if key == "all" else key
        dpdr_s = f"{dpdr:+.3f}" if not math.isnan(dpdr) else "  N/A  "
        print(f"  {label:<8} {fmt(same_pdr, st):>20} {fmt(cross_pdr, ct):>20} {dpdr_s:>8}")

    print()
    print("  ※ alias-aware: Wikidata label/wiki_title/aliases (KO+EN)")
    print("  ※ 기권(abstained / ambiguous_answer_marker) 홉 제외")


if __name__ == "__main__":
    main()
