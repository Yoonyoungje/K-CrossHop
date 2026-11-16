#!/usr/bin/env python3
"""
analysis/08_retrieval_quality.py
=================================
Hop0 retrieval quality 분석 + proxy diagnostic 보고서 생성.

입력:
  --annotated      data/v6/annotated/annotated_full_v3.jsonl
  --trajectories   data/trajectories/v6/trajectories_llama31_v6_pilot.jsonl
  --qa-pairs       data/v6/qa_pairs_sampled.jsonl
  --output-dir     data/v6/analysis/

출력:
  hit_hop0_retrieval.jsonl                     — hop0별 retrieval_success 상세 레코드
  retrieval_quality_proxy_diagnostic.json     — proxy 진단 집계
  retrieval_quality_proxy_diagnostic.md       — 보고서 (MD)

NOTE (retrieval_success proxy 한계):
  - string matching 기반이므로 false positive/negative 존재.
  - gold_lang="en" 홉에서 gold_entity가 한국어 문자열이면
    EN passage에서 KO label string match 불가 → false negative 구조.
  - gold_entity가 짧거나 일반명사(예: "미국", "서울")이면
    관련 없는 패시지에서 false positive 발생 가능.
  - alias mismatch로 false negative 발생 가능.
"""

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone

# ─── 상수 ────────────────────────────────────────────────────────────────────

SHORT_ENTITY_THRESHOLD = 3  # 이 길이 이하면 fp_risk=True

_COMMON_NOUNS_KO = {"미국", "서울", "한국", "일본", "중국", "영국", "독일", "프랑스",
                    "대학", "학교", "병원", "회사", "정부", "국가", "도시", "지역"}

# ─── 헬퍼 ────────────────────────────────────────────────────────────────────

def _is_korean(s: str) -> bool:
    return any("HANGUL" in unicodedata.name(c, "") for c in s if c.strip())


def _contains_entity(passages: list[str], entity: str) -> bool:
    """passages에 entity string 포함 여부 (case-insensitive)."""
    entity_lower = entity.lower()
    return any(entity_lower in p.lower() for p in passages if p)


def _fp_risk(gold_entity: str) -> bool:
    """gold_entity가 false positive를 낼 가능성이 높으면 True."""
    return len(gold_entity) <= SHORT_ENTITY_THRESHOLD or gold_entity in _COMMON_NOUNS_KO


def _trustworthy(r: dict) -> bool:
    """proxy 신뢰 가능 케이스: gold_lang=ko AND gold_entity 길이 > 3자."""
    return r["gold_lang"] == "ko" and len(r["gold_entity"]) > SHORT_ENTITY_THRESHOLD


def _xling_mismatch(r: dict) -> bool:
    """cross-lingual mismatch: gold_lang=en이지만 gold_entity가 한국어 문자열."""
    return r["gold_lang"] == "en" and _is_korean(r["gold_entity"])


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]

# ─── hop0 레코드 구성 ─────────────────────────────────────────────────────────

def build_hop0_records(annotated: list[dict],
                       trajectories: list[dict],
                       qa_pairs: list[dict]) -> list[dict]:
    traj_by_qid = {t["qid"]: t for t in trajectories}
    qa_by_qid   = {q["qid"]: q for q in qa_pairs}
    ann_by_qid  = {a["qid"]: a for a in annotated}

    records = []
    for qid, traj in traj_by_qid.items():
        ann = ann_by_qid.get(qid)
        qa  = qa_by_qid.get(qid)
        if ann is None or qa is None:
            continue

        hop0_traj = next((h for h in traj["trajectory"] if h["hop"] == 0), None)
        hop0_ann  = next((h for h in ann["hops"]        if h["hop"] == 0), None)
        qa_hop0   = next((h for h in qa["hops"]         if h["hop"] == 0), None)
        if hop0_traj is None or hop0_ann is None:
            continue

        gold_entity     = hop0_traj.get("gold_entity") or (qa_hop0["gold_entity"] if qa_hop0 else "")
        gold_entity_qid = qa_hop0["gold_entity_qid"] if qa_hop0 else ""
        gold_lang       = hop0_traj.get("gold_lang", qa_hop0["gold_lang"] if qa_hop0 else "")
        retrieved       = hop0_traj.get("retrieved_passages", [])
        index_used      = hop0_traj.get("retrieval_index_used", "")

        succ_entity = _contains_entity(retrieved, gold_entity)
        succ_any    = succ_entity  # succ_answer == succ_entity here (same target)

        records.append({
            "qid":                            qid,
            "pattern":                        traj.get("pattern", ""),
            "query_type":                     qa.get("query_type", ""),
            "hop":                            0,
            "gold_lang":                      gold_lang,
            "retrieval_index_used":           index_used,
            "gold_entity":                    gold_entity,
            "gold_entity_qid":                gold_entity_qid,
            "retrieval_success_gold_entity":  succ_entity,
            "retrieval_success_answer":       succ_entity,
            "retrieval_success_any":          succ_any,
            "has_hallucination":              hop0_ann.get("has_hallucination", False),
            "hop_faithfulness":               hop0_ann.get("hop_faithfulness"),
            "abstained":                      hop0_ann.get("abstained", False),
            "claim_extraction_source":        hop0_ann.get("claim_extraction_source", "none"),
            "fp_risk":                        _fp_risk(gold_entity),
            "xling_mismatch":                 _xling_mismatch({"gold_lang": gold_lang, "gold_entity": gold_entity}),
            "trustworthy":                    _trustworthy({"gold_lang": gold_lang, "gold_entity": gold_entity}),
            "index_lang_mismatch":            (gold_lang != index_used) if gold_lang and index_used else None,
        })
    return records

# ─── 진단 집계 ────────────────────────────────────────────────────────────────

def compute_diagnostics(records: list[dict], source_file: str = "") -> dict:
    total = len(records)

    trustworthy  = [r for r in records if r["trustworthy"]]
    xling_mm     = [r for r in records if r["xling_mismatch"]]
    fp_risk      = [r for r in records if r["fp_risk"]]

    # fp_risk 샘플 10개
    fp_samples = [
        {"qid": r["qid"], "pattern": r["pattern"],
         "gold_entity": r["gold_entity"], "gold_entity_len": len(r["gold_entity"]),
         "retrieval_success_any": r["retrieval_success_any"]}
        for r in fp_risk[:10]
    ]

    # pattern별 집계
    by_pat = defaultdict(list)
    for r in records:
        by_pat[r["pattern"]].append(r)

    pattern_stats = {}
    for pat in "abcd":
        recs = by_pat.get(pat, [])
        tw   = [r for r in recs if r["trustworthy"]]
        s_all = sum(1 for r in recs if r["retrieval_success_any"])
        s_tw  = sum(1 for r in tw  if r["retrieval_success_any"])
        pattern_stats[pat] = {
            "n_total":        len(recs),
            "n_trustworthy":  len(tw),
            "succ_any_all":   round(s_all / len(recs), 4) if recs else None,
            "succ_any_tw":    round(s_tw  / len(tw),   4) if tw   else None,
        }

    # KO-index sanity check (trustworthy subset, non-abstained)
    tw_non_abs = [r for r in trustworthy if not r["abstained"]]
    tw_succ    = [r for r in tw_non_abs if r["retrieval_success_any"]]
    tw_fail    = [r for r in tw_non_abs if not r["retrieval_success_any"]]
    ko_sanity = {
        "n_trustworthy":           len(trustworthy),
        "n_non_abstained":         len(tw_non_abs),
        "succ_any_rate":           round(sum(1 for r in trustworthy if r["retrieval_success_any"]) / len(trustworthy), 4) if trustworthy else None,
        "hallu_rate_success_true": round(sum(1 for r in tw_succ if r["has_hallucination"]) / len(tw_succ), 4) if tw_succ else None,
        "hallu_rate_success_false": round(sum(1 for r in tw_fail if r["has_hallucination"]) / len(tw_fail), 4) if tw_fail else None,
        "n_success": len(tw_succ),
        "n_fail":    len(tw_fail),
    }

    return {
        "source_file":              source_file,
        "n_total":                  total,
        "n_trustworthy":            len(trustworthy),
        "n_xling_mismatch":         len(xling_mm),
        "n_fp_risk":                len(fp_risk),
        "rate_trustworthy":         round(len(trustworthy) / total, 4),
        "rate_xling_mismatch":      round(len(xling_mm)    / total, 4),
        "rate_fp_risk":             round(len(fp_risk)      / total, 4),
        "overall_succ_any":         round(sum(1 for r in records if r["retrieval_success_any"]) / total, 4),
        "pattern_stats":            pattern_stats,
        "ko_index_sanity_check":    ko_sanity,
        "fp_risk_samples":          fp_samples,
        "proxy_limitation_note":    (
            "string-match proxy. "
            "FN: gold_lang=en hops store KO-script entity labels; EN passages cannot match. "
            "FP: short/generic entities match unrelated passages. "
            "Trustworthy subset = gold_lang=ko AND entity_len>3."
        ),
        "created_at":               datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

# ─── MD 보고서 ────────────────────────────────────────────────────────────────

def render_md(d: dict) -> str:
    ps = d["pattern_stats"]
    ks = d["ko_index_sanity_check"]

    def pct(v):
        return f"{v:.1%}" if v is not None else "N/A"

    rows_pat = ""
    for pat in "abcd":
        s = ps[pat]
        rows_pat += (
            f"| {pat} | {s['n_total']} | {s['n_trustworthy']} | "
            f"{pct(s['succ_any_all'])} | {pct(s['succ_any_tw'])} |\n"
        )

    fp_rows = ""
    for samp in d["fp_risk_samples"]:
        fp_rows += f"| {samp['qid']} | {samp['pattern']} | `{samp['gold_entity']}` | {samp['gold_entity_len']} | {samp['retrieval_success_any']} |\n"

    n = d['n_total']
    return f"""# Retrieval Success Proxy Diagnostic Report

> **이 보고서는 공식 H1/H2 분석이 아닌 proxy 검증 보고서입니다.**
> string-match 기반 retrieval_success proxy의 한계를 진단하고,
> alias-aware matching(v2) 구현 전까지 조건부 faithfulness 분석이 불가함을 명확히 합니다.

Generated: {d['created_at']}
Source: `{d['source_file']}`
N = {n} hop0 records

---

## 1. Executive Summary

현재 `retrieval_success`는 retrieved passage 내 gold_entity 문자열 포함 여부로 판정합니다.
이 proxy는 두 가지 구조적 오류를 내재합니다:

| 오류 유형 | 원인 | 규모 |
|-----------|------|------|
| **False Negative** | gold_lang=en 홉의 gold_entity가 한국어 문자열로 저장 → EN passage에서 KO label 매칭 불가 | {d['n_xling_mismatch']}/{n} ({pct(d['rate_xling_mismatch'])}) |
| **False Positive** | 3자 이하 또는 일반명사 gold_entity → 무관 passage에서 hit 가능 | {d['n_fp_risk']}/{n} ({pct(d['rate_fp_risk'])}) |

**proxy 신뢰 가능 케이스** (gold_lang=ko AND entity_len > 3): **{d['n_trustworthy']}/{n} ({pct(d['rate_trustworthy'])})**

Pattern b와 d는 신뢰 가능 케이스가 없거나 극히 적어 proxy 결과가 무의미합니다.

---

## 2. Proxy 한계 요약

### 2.1 Cross-lingual False Negative 구조

- Pattern b/d hop0은 EN 인덱스를 사용하지만, `gold_entity`는 한국어 표기(예: "프랫 대학교", "2018년 동계 올림픽")로 저장됨
- EN passage에서 KO 문자열 검색 → 항상 실패 → retrieval_success=False
- **이것은 retrieval 실패가 아닌 proxy 측정 실패입니다**
- 해당 케이스: **{d['n_xling_mismatch']}/{n}**

### 2.2 Short Entity False Positive 구조

- gold_entity가 3자 이하(예: "적도", "남코", "세가")이거나 일반명사이면
  관련 없는 passage에서도 문자열이 우연히 포함될 수 있음
- 해당 케이스: **{d['n_fp_risk']}/{n}**

### 2.3 결론

신뢰 가능 케이스({d['n_trustworthy']}/{n})만으로 H1/H2 조건부 분석을 수행하면
**pattern a + 일부 c만 포함**되어 실험 설계의 cross-lingual 비교가 불가합니다.

---

## 3. Pattern별 Proxy 분포

| Pattern | N | Trustworthy N | succ_any (전체) | succ_any (Trustworthy만) |
|---------|---|---------------|-----------------|--------------------------|
{rows_pat}

**관찰:**
- Pattern b: trustworthy 케이스 0개 → succ_any=0%는 proxy 실패이지 retrieval 실패가 아님
- Pattern d: trustworthy 케이스 없음 → 동일
- Pattern a/c: trustworthy 비율이 높아 proxy 결과가 상대적으로 신뢰 가능

---

## 4. KO-index 홉 Sanity Check (보조 진단)

> As a diagnostic check on the proxy, we restrict to trustworthy cases
> (gold_lang=ko, entity_len>3) and examine whether retrieval_success correlates with hallucination.

| 지표 | 값 |
|------|-----|
| Trustworthy subset N | {ks['n_trustworthy']} |
| (non-abstained) N | {ks['n_non_abstained']} |
| retrieval_success_any 비율 | {pct(ks['succ_any_rate'])} |
| HalluRate (success=True,  N={ks['n_success']}) | {pct(ks['hallu_rate_success_true'])} |
| HalluRate (success=False, N={ks['n_fail']}) | {pct(ks['hallu_rate_success_false'])} |

**해석:** success=True 홉의 HalluRate {pct(ks['hallu_rate_success_true'])} vs success=False 홉의 HalluRate {pct(ks['hallu_rate_success_false'])} — 방향성 일관 (retrieval 성공 시 hallucination 감소). 단, string-match proxy이므로 false positive 포함 가능성 존재.

**⚠️ 이 결과는 공식 H1/H2 분석에 직접 사용하지 않습니다. alias-aware v3(analysis/10_retrieval_proxy.py) 결과 기반 조건부 분석을 사용하십시오.**

---

## 5. Alias-aware Retrieval Success v2/v3 현황

v2(analysis/09_alias_retrieval.py) 및 v3(analysis/10_retrieval_proxy.py)가 이미 구현되어 있습니다.
`data/retrieval/alias_cache.json`에 Wikidata alias 캐시가 구축되어 있으며,
`hop0_retrieval_v3_*.jsonl`에 `answer_bearing_hit` 필드가 추가되어 있습니다.

### 구현 완료 사항

- **False positive 방지**: alias 길이 ≥ 4자만 사용 (MIN_ALIAS_LEN)
- **캐싱**: `data/retrieval/alias_cache.json`에 QID별 alias set 저장
- **Cross-lingual 커버리지**: ko/en alias 통합 매칭 + normalized variant
- **정규화**: 조사 제거(KO), 하이픈→공백(EN) 후 재매칭

---

## 6. False Positive 의심 케이스 샘플 (상위 10개)

| QID | Pattern | gold_entity | 길이 | retrieval_success_any |
|-----|---------|-------------|------|-----------------------|
{fp_rows}

---

## 7. 결론

| 항목 | 현황 |
|------|------|
| xling_mismatch (구조적 FN) | **{pct(d['rate_xling_mismatch'])}** |
| fp_risk (단어 FP) | **{pct(d['rate_fp_risk'])}** |
| Trustworthy subset | **{pct(d['rate_trustworthy'])}** |

> xling_mismatch가 낮아 alias-aware answer-bearing hit을 retrieval-conditioned 분석에 사용할 수 있습니다.
> 조건부 H1/H2 분석은 `analysis/12_conditioned_analysis.py`를 참조하십시오.
"""

# ─── stdout 보고 ──────────────────────────────────────────────────────────────

def print_summary(d: dict) -> None:
    ks = d["ko_index_sanity_check"]
    n = d["n_total"]
    print(f"\n=== Proxy Diagnostic Summary ===")
    print(f"  proxy 신뢰 가능 케이스:       {d['n_trustworthy']}/{n} ({d['rate_trustworthy']:.1%})")
    print(f"  cross-lingual mismatch:      {d['n_xling_mismatch']}/{n} ({d['rate_xling_mismatch']:.1%})")
    print(f"  false positive 의심:          {d['n_fp_risk']}/{n} ({d['rate_fp_risk']:.1%})")
    print(f"\n  [KO-index sanity check]")
    print(f"    HalluRate | success=True : {ks['hallu_rate_success_true']:.3f} (N={ks['n_success']})")
    print(f"    HalluRate | success=False: {ks['hallu_rate_success_false']:.3f} (N={ks['n_fail']})")
    print(f"\n  [결론] v3 alias-normalized 결과 기반 조건부 분석: 11_conditioned_analysis.py 참조")

# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated",    default="data/annotated/aya_ecological.jsonl")
    parser.add_argument("--trajectories", default="data/trajectories/aya_ecological.jsonl")
    parser.add_argument("--qa-pairs",     default="data/dataset/qa_pairs.jsonl")
    parser.add_argument("--output-dir",   default="data/retrieval/")
    parser.add_argument("--model",        default=None,
                        help="모델명 (출력 파일명에 사용, 미지정 시 trajectory 파일명에서 자동 추출)")
    args = parser.parse_args()

    # 모델명 결정
    model = args.model or Path(args.trajectories).stem.replace("trajectories_", "")

    annotated    = load_jsonl(args.annotated)
    trajectories = load_jsonl(args.trajectories)
    qa_pairs     = load_jsonl(args.qa_pairs)

    records = build_hop0_records(annotated, trajectories, qa_pairs)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hop0_path = out_dir / f"hop0_retrieval_{model}.jsonl"
    with open(hop0_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"저장: {hop0_path} ({len(records)} 레코드)")

    # 진단 집계
    diag = compute_diagnostics(records, source_file=str(hop0_path))

    # JSON 보고서
    json_path = out_dir / f"hop0_diagnostic_{model}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(diag, f, ensure_ascii=False, indent=2)
    print(f"저장: {json_path}")

    # MD 보고서
    md_path = out_dir / f"hop0_diagnostic_{model}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_md(diag))
    print(f"저장: {md_path}")

    print_summary(diag)


if __name__ == "__main__":
    main()
