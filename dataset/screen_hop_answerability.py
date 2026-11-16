#!/usr/bin/env python3
"""
dataset/screen_hop_answerability.py

홉 단위 EN answerability 스크리닝 (Paper 2 Stage 0 게이트).

배경:
  Paper 2 의 A1~A3 arm 은 KO evidence 홉에도 EN sub-question 을 쓴다. 그 EN 질문 1,000홉은
  정본에 없던 신규 산출물이라 answerability 가 검증된 적이 없다. 검증하지 않으면 Δy 하락이
  "언어 결정 실패" 때문인지 "EN 템플릿이 애초에 못 푸는 질문" 때문인지 구분되지 않는다.

  dataset/oracle_gate.py 는 (i) 구축 단계 감사 입력 하드와이어, (ii) KO 질문 템플릿 전용,
  (iii) 2-hop path 단위라 그대로 못 쓴다. 이 스크립트는 홉 단위 EN 어댑터이며 질문 템플릿을
  제외한 나머지 — 프롬프트 골격 · evidence window · 파싱 · 정답 매칭 · alias 구성 · 캐시 구조 ·
  호출 파라미터 — 를 전부 oracle_gate_lib / oracle_gate 에서 재사용한다.

설계 (계획서 §14.1 사전등록):
  * 신규군(new_ko_evidence)  : gold_lang=ko 홉의 EN sub-question — 검증 대상
  * 비교군(control_en_evidence): gold_lang=en 홉의 EN sub-question — 정본에 이미 있던 질문
    두 군을 **같은 프롬프트**로 함께 판정하므로 프롬프트 신규성 효과가 상쇄된다.
  * 판정 모델 gpt-5.4-mini(정본 oracle 실측), max_completion_tokens=512, temperature=0.

대응 구조 (구현 중 실측):
  250 path_cluster × 2 hop = 500 셀 각각에서 4패턴이 같은 EN 질문 · 같은 carrier · 같은 target 을
  공유하고 gold_lang 만 2:2 로 갈린다. 즉 두 군은 **같은 질문을 KO evidence 에 물었는가 EN
  evidence 에 물었는가** 만 다른 대응표본이다. 따라서:
    - 판정 단위는 홉(2,000)이 아니라 고유 프롬프트(912 = 군당 456). 나머지는 완전 중복이라
      내용 기반 캐시가 흡수한다 (정본 oracle 과 같은 방식).
    - 주 검정은 456 쌍의 McNemar (대응). 보조로 비대응 2-비율 검정을 함께 보고한다.

사용법 (비용 발생 — 사용자가 직접 실행). 키는 프로젝트 루트 .env 의 OPENAI_API_KEY 를 쓴다:
  python3 dataset/screen_hop_answerability.py --dry-run
  python3 dataset/screen_hop_answerability.py
  python3 dataset/screen_hop_answerability.py --report-only     # 캐시만으로 재집계
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_gate_lib as gate            # noqa: E402  프롬프트 골격 · 파싱 · 정답 매칭 · 호출
import oracle_gate as og                  # noqa: E402  alias / evidence 구성 헬퍼
from entity_cache import build_passage_index  # noqa: E402

PROMPT_VERSION = "p2-hop-1.0"

# 질문 필드 → (evidence 언어 → 군 이름). sub_q_en 이 기본(Paper 2 Stage 0)이다.
# sub_q 는 실험에서 실제로 던진 한국어 질문을 양 언어 evidence 로 판정하는 전 홉 감사이며,
# 군 이름은 한국어 기준선(stage0_ko_baseline.py) 캐시와 맞춘다.
GROUPS = {
    "sub_q_en": {"ko": "new_ko_evidence", "en": "control_en_evidence"},
    "sub_q":    {"ko": "ko_question_ko_evidence", "en": "ko_question_en_evidence"},
}

# 정본 _ORACLE_PROMPT 에서 2홉 구조만 걷어낸 단일 홉 판정 프롬프트.
# 지시문을 KO 로 유지하는 것은 정본 판정 특성을 보존하기 위해서다 — 두 군에 동일 적용되므로
# 군 간 비교(=이 게이트의 판정 대상)는 지시문 언어에 교락되지 않는다.
SINGLE_HOP_PROMPT = """\
당신은 단일 홉 질문을 판정하는 오라클입니다.
증거 이외의 배경 지식은 절대 사용하지 마십시오.

[판정 기준]
- answer_supported: Evidence가 질문의 답을 명시적으로 지지하는가?
- 답을 명확히 하나로 도출할 수 있으면 ANSWERABLE.
- 증거가 서로 다른 답 둘 이상을 지지하면 AMBIGUOUS.
- 그 외(증거 부족, 연결 불확실)에는 UNANSWERABLE.

[질문]
{question}

[Evidence] (주어 엔티티 Wikipedia 발췌)
{evidence}

다음 JSON만 출력하고 다른 텍스트는 일절 포함하지 마십시오:
{{
  "verdict": "ANSWERABLE" | "UNANSWERABLE" | "AMBIGUOUS",
  "final_answer": "<최종 답 문자열; UNANSWERABLE/AMBIGUOUS이면 빈 문자열>",
  "answer_supported": true | false,
  "rationale": "<2문장 이내 근거>"
}}"""


# ── 판정 단위 조립 ────────────────────────────────────────────────────────────

def build_units(core: list[dict], paths: dict, passage_index: dict,
                alias_cache: dict, model: str, question_field: str = "sub_q_en",
                prompt_version: str = PROMPT_VERSION) -> tuple[list[dict], list[dict]]:
    """홉 2,000개를 고유 프롬프트 단위로 접는다. (units, hops) 반환.

    hops 는 홉↔unit 매핑(보고를 홉 수준으로 되펼치기 위함)이고, units 가 실제 판정 대상이다.
    """
    units: dict[str, dict] = {}
    hops: list[dict] = []
    for r in core:
        ph = paths[r["source_path_id"]]["hops"]
        for i, h in enumerate(r["hops"]):
            carrier = ph[i]        # hop0=subject, hop1=bridge
            target = ph[i + 1]     # 이 홉이 찾아야 할 엔티티
            lang = h["gold_lang"]  # evidence 언어 = 그 홉의 gold_lang
            question = h[question_field]
            aliases = og.build_aliases_for_qid(target["qid"], alias_cache) or [
                a for a in (target.get("label_ko"), target.get("label_en")) if a
            ]
            evidence = og.build_evidence_text(passage_index, carrier["qid"], lang, aliases)
            key = hashlib.sha256("|".join([
                prompt_version, model, carrier["qid"], target["qid"], lang, question,
            ]).encode()).hexdigest()

            if key not in units:
                units[key] = {
                    "cache_key": key,
                    "group": GROUPS[question_field][lang],
                    "evidence_lang": lang,
                    "carrier_qid": carrier["qid"],
                    "target_qid": target["qid"],
                    "question": question,
                    "gold_aliases": aliases,
                    "evidence": evidence,
                    "relation": (r.get("relations") or [None, None])[i],
                    "query_type": r.get("query_type"),
                    "n_hops": 0,
                    "oracle_model": model,
                }
            units[key]["n_hops"] += 1
            hops.append({
                "qid": r["qid"], "hop": h["hop"], "pattern": r["pattern"],
                "path_cluster_id": r["path_cluster_id"], "cache_key": key,
            })
    return list(units.values()), hops


def pair_units(units: list[dict]) -> list[tuple[dict, dict]]:
    """같은 (질문, carrier, target) 을 KO/EN evidence 로 물은 짝을 맞춘다."""
    by_key: dict[tuple, dict] = {}
    for u in units:
        by_key[(u["question"], u["carrier_qid"], u["target_qid"], u["evidence_lang"])] = u
    pairs = []
    for (q, c, t, lang), u in by_key.items():
        if lang != "ko":
            continue
        mate = by_key.get((q, c, t, "en"))
        if mate:
            pairs.append((u, mate))
    return pairs


# ── 판정 ─────────────────────────────────────────────────────────────────────

def determine_hop_verdict(parsed: dict, gold_aliases: list[str]):
    """정본 determine_verdict 의 홉 단위 판정. (verdict, fail_reason, match_type)."""
    verdict = str(parsed.get("verdict", "")).upper()
    supported = gate.parse_bool(parsed.get("answer_supported"))
    predicted = str(parsed.get("final_answer") or "")

    if verdict == "AMBIGUOUS":
        return "FAIL", "ORACLE_FAIL_AMBIGUOUS", None
    if verdict == "UNANSWERABLE":
        return "FAIL", ("ORACLE_FAIL_ANSWER_UNSUPPORTED" if not supported
                        else "ORACLE_FAIL_NOT_DERIVABLE"), None
    if verdict == "ANSWERABLE":
        if not supported:
            return "FAIL", "ORACLE_FAIL_ANSWER_UNSUPPORTED", None
        match_type = gate.gold_match(predicted, gold_aliases)
        if not match_type:
            return "FAIL", "ORACLE_FAIL_WRONG_ANSWER", None
        return "PASS", None, match_type
    return "FAIL", "ORACLE_FAIL_PARSE_ERROR", None


async def judge_unit(unit: dict, client, model: str, semaphore: asyncio.Semaphore) -> dict:
    base = {k: unit[k] for k in ("cache_key", "group", "evidence_lang", "carrier_qid",
                                 "target_qid", "question", "relation", "query_type",
                                 "n_hops", "oracle_model")}
    if not unit["evidence"]:
        return {**base, "oracle_verdict": "FAIL",
                "oracle_fail_reason": "ORACLE_FAIL_NO_EVIDENCE_TEXT",
                "oracle_predicted_answer": None, "oracle_answer_supported": False,
                "oracle_gold_match_type": None, "oracle_explanation": None, "usage": {}}

    prompt = SINGLE_HOP_PROMPT.format(question=unit["question"], evidence=unit["evidence"])
    raw, parsed, usage, last_error = None, None, {}, None
    async with semaphore:
        for attempt in range(2):
            try:
                raw, usage = await gate.call_llm(client, model, prompt, return_usage=True)
                parsed = gate.parse_response(raw)
                break
            except Exception as e:                      # 정본과 동일한 2회 재시도 정책
                last_error = f"{type(e).__name__}: {e}"
                parsed = None

    if parsed is None:
        return {**base, "oracle_verdict": "FAIL",
                "oracle_fail_reason": "ORACLE_FAIL_PARSE_ERROR",
                "oracle_predicted_answer": None, "oracle_answer_supported": False,
                "oracle_gold_match_type": None, "oracle_explanation": raw or last_error,
                "usage": usage}

    verdict, reason, match_type = determine_hop_verdict(parsed, unit["gold_aliases"])
    return {**base, "oracle_verdict": verdict, "oracle_fail_reason": reason,
            "oracle_predicted_answer": parsed.get("final_answer"),
            "oracle_answer_supported": gate.parse_bool(parsed.get("answer_supported")),
            "oracle_gold_match_type": match_type,
            "oracle_explanation": parsed.get("rationale"), "usage": usage}


# ── 검정 ─────────────────────────────────────────────────────────────────────

def mcnemar(b: int, c: int) -> tuple[float, float]:
    """연속성 보정 McNemar. (chi2, p) — b/c 는 불일치 쌍 수."""
    from math import erfc, sqrt
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    p = erfc(sqrt(chi2 / 2))          # chi2(df=1) 상측확률 = 2*(1-Φ(√chi2))
    return chi2, p


def two_proportion(x1: int, n1: int, x2: int, n2: int) -> tuple[float, float]:
    """비대응 2-비율 z 검정 (pooled). (z, p)."""
    from math import erfc, sqrt
    if not n1 or not n2:
        return 0.0, 1.0
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (x1 / n1 - x2 / n2) / se
    return z, erfc(abs(z) / sqrt(2))


# ── 보고 ─────────────────────────────────────────────────────────────────────

def _order_only_match(predicted: str | None, aliases: list[str]) -> bool:
    """어순만 다른 동일 엔티티인가 (예: 'Chiba Shinichi' vs 'Shinichi Chiba')."""
    def toks(s):
        s = re.sub(r"\([^)]*\)", "", s or "").lower()
        return frozenset(t for t in re.split(r"[\s\-·,.\[\]\"']+", s) if len(t) > 1)
    p = toks(predicted)
    return bool(p) and any(p == toks(a) for a in aliases)


def report(results: dict[str, dict], units: list[dict], hops: list[dict],
           question_field: str = "sub_q_en") -> None:
    ko_g, en_g = GROUPS[question_field]["ko"], GROUPS[question_field]["en"]
    print("홉 단위 EN answerability 스크리닝  (Paper 2 Stage 0)")
    print("=" * 70)
    judged = [results[u["cache_key"]] for u in units if u["cache_key"] in results]
    alias_of = {u["cache_key"]: u["gold_aliases"] for u in units}
    print(f"  판정 단위 {len(judged)}/{len(units)}  (홉 {len(hops)}개를 고유 프롬프트로 접은 것)")
    if not judged:
        print("  판정 결과 없음 — --api-key 로 실행하세요.")
        return

    print("\n1. 군별 탈락률 (판정 단위 기준)")
    print(f"  {'군':22} {'n':>5} {'PASS':>6} {'FAIL':>6} {'탈락률':>8}")
    print("  " + "-" * 52)
    stat = {}
    for grp in (ko_g, en_g):
        rows = [r for r in judged if r["group"] == grp]
        f = sum(r["oracle_verdict"] == "FAIL" for r in rows)
        stat[grp] = (f, len(rows))
        rate = f / len(rows) if rows else 0
        print(f"  {grp:22} {len(rows):>5} {len(rows)-f:>6} {f:>6} {rate:>7.1%}")

    (f1, n1), (f2, n2) = stat[ko_g], stat[en_g]
    print(f"\n  탈락률 차이 (신규 − 비교군): {f1/n1 - f2/n2:+.1%}" if n1 and n2 else "")

    print("\n2. 주 검정 — McNemar (대응: 같은 EN 질문 × KO/EN evidence)")
    pairs = [(a, b) for a, b in pair_units(units)
             if a["cache_key"] in results and b["cache_key"] in results]
    b_cnt = sum(1 for a, m in pairs
                if results[a["cache_key"]]["oracle_verdict"] == "FAIL"
                and results[m["cache_key"]]["oracle_verdict"] == "PASS")
    c_cnt = sum(1 for a, m in pairs
                if results[a["cache_key"]]["oracle_verdict"] == "PASS"
                and results[m["cache_key"]]["oracle_verdict"] == "FAIL")
    chi2, p = mcnemar(b_cnt, c_cnt)
    print(f"  쌍 {len(pairs)}개  |  KO만 탈락 {b_cnt}  EN만 탈락 {c_cnt}  일치 {len(pairs)-b_cnt-c_cnt}")
    print(f"  chi2={chi2:.3f}  p={p:.4f}  → {'유의한 차이 (STOP 검토)' if p < 0.05 else 'PASS (유의차 없음)'}")

    print("\n3. 보조 검정 — 비대응 2-비율 z")
    z, pz = two_proportion(f1, n1, f2, n2)
    print(f"  z={z:+.3f}  p={pz:.4f}")

    print("\n4. 탈락 사유 분포")
    for grp in (ko_g, en_g):
        c = Counter(r["oracle_fail_reason"] for r in judged
                    if r["group"] == grp and r["oracle_verdict"] == "FAIL")
        print(f"  [{grp}] " + (", ".join(f"{k}={v}" for k, v in c.most_common()) or "없음"))

    print("\n5. relation별 탈락 — 대응 쌍 분해 (계획서 §7: 신규군 고유인지 양쪽 공통인지 구분)")
    print("   같은 EN 질문을 KO evidence 로 물었을 때와 EN evidence 로 물었을 때를 나란히 본다.")
    print("   양쪽이 같이 높으면 EN 템플릿 품질이 아니라 그 홉들 자체의 성질이다.")
    rel = defaultdict(lambda: [0, 0, 0])          # relation -> [ko_fail, en_fail, n]
    for ko_u, en_u in pairs:
        k, e = results[ko_u["cache_key"]], results[en_u["cache_key"]]
        v = rel[ko_u["relation"]]
        v[0] += k["oracle_verdict"] == "FAIL"
        v[1] += e["oracle_verdict"] == "FAIL"
        v[2] += 1
    print(f"  {'relation':10} {'쌍':>4} {'KO탈락':>11} {'EN탈락':>11}   구분")
    print("  " + "-" * 58)
    for k, (kf, ef, n) in sorted(rel.items(), key=lambda x: -(x[1][0] - x[1][1])):
        if not (kf or ef):
            continue
        tag = "신규군 고유" if kf - ef >= 3 else ("양쪽 공통" if min(kf, ef) >= max(1, n * 0.3) else "")
        print(f"  {str(k):10} {n:>4} {kf:>3}({kf/n:>5.1%}) {ef:>3}({ef/n:>5.1%})   {tag}")

    print("\n6. 표면형 아티팩트 점검 — WRONG_ANSWER 중 '어순만 다른 동일 엔티티'")
    print("   KO evidence 는 인명을 성-이름 순으로 로마자화해 alias(서구식 어순)와 어긋날 수 있다.")
    art = {g: [0, 0] for g in (ko_g, en_g)}
    art_keys = set()
    for r in judged:
        if r["oracle_fail_reason"] != "ORACLE_FAIL_WRONG_ANSWER":
            continue
        aliases = alias_of.get(r["cache_key"], [])
        is_art = _order_only_match(r.get("oracle_predicted_answer"), aliases)
        art[r["group"]][0] += is_art
        art[r["group"]][1] += 1
        if is_art:
            art_keys.add(r["cache_key"])
    for g, (a, n) in art.items():
        print(f"  {g:22} {a}/{n}")

    if art_keys:
        b2 = sum(1 for a, m in pairs
                 if results[a["cache_key"]]["oracle_verdict"] == "FAIL"
                 and a["cache_key"] not in art_keys
                 and (results[m["cache_key"]]["oracle_verdict"] == "PASS"
                      or m["cache_key"] in art_keys))
        c2 = sum(1 for a, m in pairs
                 if results[m["cache_key"]]["oracle_verdict"] == "FAIL"
                 and m["cache_key"] not in art_keys
                 and (results[a["cache_key"]]["oracle_verdict"] == "PASS"
                      or a["cache_key"] in art_keys))
        chi2b, pb = mcnemar(b2, c2)
        print(f"  민감도 — 아티팩트를 PASS 로 돌린 McNemar: KO만 탈락 {b2} / EN만 탈락 {c2}, "
              f"p={pb:.4f}")
        print("  (사전등록 주 검정은 위 2번이다. 이건 결론이 아티팩트에 뒤집히지 않는지 보는 보조 확인.)")

    used = [r["usage"] for r in judged if r.get("usage")]
    if used:
        ti = sum(u.get("input_tokens") or 0 for u in used)
        to = sum(u.get("output_tokens") or 0 for u in used)
        tr = sum(u.get("reasoning_tokens") or 0 for u in used)
        print(f"\n7. 토큰 사용량 (실측 {len(used)}콜): 입력 {ti:,} / 출력 {to:,} / 추론 {tr:,}")


# ── main ─────────────────────────────────────────────────────────────────────

async def run(units, cache_path: Path, client, model: str, concurrency: int) -> None:
    sem = asyncio.Semaphore(concurrency)
    tasks = [judge_unit(u, client, model, sem) for u in units]
    done = 0
    with cache_path.open("a", encoding="utf-8") as f:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()
            done += 1
            if done % 50 == 0 or done == len(tasks):
                print(f"  판정 {done}/{len(tasks)}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="paper2/data/qa_pairs_core_bothlangs.jsonl")
    ap.add_argument("--paths", default="data/dataset/paths.jsonl")
    ap.add_argument("--passages", default="data/corpus/passages.jsonl")
    ap.add_argument("--alias-cache", default="data/retrieval/alias_cache.json")
    ap.add_argument("--cache", default="paper2/work/stage0/answerability_cache.jsonl")
    ap.add_argument("--output", default="paper2/work/stage0/answerability_hops.jsonl")
    ap.add_argument("--model", default="gpt-5.4-mini-2026-03-17")
    ap.add_argument("--api-key", default=None,
                    help="미지정 시 .env / 환경변수의 OPENAI_API_KEY 사용")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="판정할 단위 수 상한 (시범 실행용)")
    ap.add_argument("--dry-run", action="store_true", help="프롬프트만 조립하고 콜 수·토큰 보고")
    ap.add_argument("--report-only", action="store_true", help="캐시만으로 재집계")
    ap.add_argument("--question-field", choices=sorted(GROUPS), default="sub_q_en",
                    help="판정할 질문 필드. sub_q = 실험의 한국어 질문(전 홉 감사)")
    ap.add_argument("--prompt-version", default=PROMPT_VERSION,
                    help="cache_key 에 들어가는 버전 문자열. sub_q 감사는 p2-hop-ko-1.0 으로 "
                         "한국어 기준선 캐시(KO evidence 456단위)를 재사용한다")
    args = ap.parse_args()

    try:                                    # 05/06 과 같은 관례 — 루트 .env 자동 로드
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")

    root = REPO_ROOT
    core = [json.loads(l) for l in (root / args.core).read_text(encoding="utf-8").splitlines() if l.strip()]
    paths = {}
    for line in (root / args.paths).read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line)
            paths[p["path_id"]] = p
    alias_cache = json.loads((root / args.alias_cache).read_text(encoding="utf-8"))

    needed = {h["qid"] for r in core for h in paths[r["source_path_id"]]["hops"]}
    passage_index = build_passage_index(needed, root / args.passages)

    units, hops = build_units(core, paths, passage_index, alias_cache, args.model,
                              args.question_field, args.prompt_version)
    grp = GROUPS[args.question_field]
    print(f"로드: 홉 {len(hops)} → 고유 판정 단위 {len(units)}"
          f" ({grp['ko']} {sum(u['group']==grp['ko'] for u in units)} /"
          f" {grp['en']} {sum(u['group']==grp['en'] for u in units)}),"
          f" 대응 쌍 {len(pair_units(units))}", file=sys.stderr)

    cache_path = root / args.cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                results[d["cache_key"]] = d
        print(f"캐시 히트: {len(results)}건", file=sys.stderr)

    pending = [u for u in units if u["cache_key"] not in results]
    if args.limit:
        pending = pending[:args.limit]

    if args.dry_run:
        no_ev = sum(1 for u in units if not u["evidence"])
        chars = sum(len(SINGLE_HOP_PROMPT.format(question=u["question"], evidence=u["evidence"]))
                    for u in pending)
        print(f"\n[dry-run] 신규 호출 대상 {len(pending)}건, evidence 결손 {no_ev}건, "
              f"프롬프트 총 {chars:,}자")
        if pending:
            u = pending[0]
            print("\n--- 프롬프트 예시 ---")
            print(SINGLE_HOP_PROMPT.format(question=u["question"], evidence=u["evidence"])[:1200])
        return

    if pending and not args.report_only:
        if not api_key:
            print(f"미판정 {len(pending)}건 — OPENAI_API_KEY 를 .env 에 두거나 --api-key 로 "
                  f"넘기세요 (--dry-run 으로 먼저 확인하세요).", file=sys.stderr)
            raise SystemExit(1)
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        asyncio.run(run(pending, cache_path, client, args.model, args.concurrency))
        results = {}
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                results[d["cache_key"]] = d

    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    carry = ("group", "evidence_lang", "oracle_verdict", "oracle_fail_reason", "relation")
    with out.open("w", encoding="utf-8") as f:
        for h in hops:
            r = results.get(h["cache_key"])
            row = {**h, **{k: r[k] for k in carry}} if r else h
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"저장: {args.output} ({len(hops)} 홉)", file=sys.stderr)

    report(results, units, hops, args.question_field)


if __name__ == "__main__":
    main()
