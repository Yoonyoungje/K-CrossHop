#!/usr/bin/env python3
"""
dataset/oracle_gate.py

Oracle Solvability Gate — 4조건(a_ko_ko/b_en_ko/c_ko_en/d_en_en) per triple.

배경 (data/audit/ROADMAP_DECISIONS.md 결정 1):
  lexical-pass(alias가 evidence에 등장)만으로는 그 evidence로 hop을 실제로 풀 수
  있는지(relation이 서술 형태로 존재하는지)를 보장하지 못한다. v9의 oracle-pass는
  a_ko_ko(KO evidence)만 검증했으므로, v10이 새로 도입하는 EN evidence 조건
  (b/c/d)은 한 번도 oracle로 검증된 적이 없다. 이 스크립트는 그 갭을 채운다.

조건별 evidence carrier (hop0=Subject S, hop1=Bridge B):
  a_ko_ko : hop0=S_KO, hop1=B_KO  → 선행 KO-only oracle 결과에서 이미 PASS로 확인됨.
            여기서는 LLM을 다시 호출하지 않고 그 결과를 그대로 가져온다 (비용 절감).
  b_en_ko : hop0=S_EN, hop1=B_KO  → 신규 LLM 호출
  c_ko_en : hop0=S_KO, hop1=B_EN  → 신규 LLM 호출
  d_en_en : hop0=S_EN, hop1=B_EN  → 신규 LLM 호출

질문 템플릿, 프롬프트, 정답 매칭, 파싱/판정 로직은 dataset/oracle_gate_lib.py를
import 해서 재사용한다 (선행 KO-only gate와 동일 기준 유지).

Input:
  data/audit/base_reuse_a644_triple_audit_refetched.jsonl  (eligible_full_2x2_lexical=True)
  data/oracle_pass_base.jsonl                              (a_ko_ko 재사용)
  data/passages.jsonl + data/raw/passages_refetch.jsonl (evidence text)
  data/audit/entity_cache.json           (gold alias)

Output:
  data/oracle/oracle_cache.jsonl   (재실행 캐시, append-only)
  data/construction/gate_decisions.jsonl   (전체 결과, a 포함 4조건 x N triple)
  data/oracle/oracle_pass.jsonl    (PASS만)
  data/oracle/oracle_report.json   (조건별 pass rate, fail reason 분포)

사용법 (비용 발생 — 사용자가 직접 실행):
  python dataset/oracle_gate.py --dump-templates
  python dataset/oracle_gate.py --dry-run --pending-limit 5
  python dataset/oracle_gate.py --api-key $ANTHROPIC_API_KEY
  python dataset/oracle_gate.py --api-key $ANTHROPIC_API_KEY --pending-limit 50
"""

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 질문 템플릿 · 프롬프트 · 판정 로직
import oracle_gate_lib as gate  # noqa: E402
from entity_cache import build_passage_index  # noqa: E402

PROMPT_VERSION = "1.1"

DATA_DIR = REPO_ROOT / "data"
BASE_DIR = DATA_DIR / "base"
BUILD_DIR = DATA_DIR / "build"
AUDIT_DIR = BUILD_DIR / "audit"
ORACLE_DIR = BUILD_DIR / "oracle"
ORACLE_DIR.mkdir(parents=True, exist_ok=True)

REFETCHED_AUDIT_PATH = AUDIT_DIR / "base_reuse_a644_triple_audit_refetched.jsonl"
PLAIN_AUDIT_PATH = AUDIT_DIR / "base_reuse_a644_triple_audit.jsonl"  # refetch 전 fallback
PHASE2_AUDIT_PATH = AUDIT_DIR / "phase2_candidates_triple_audit.jsonl"
PHASE2_REFETCHED_AUDIT_PATH = AUDIT_DIR / "phase2_candidates_triple_audit_refetched.jsonl"
PHASE3_AUDIT_PATH = AUDIT_DIR / "phase3_candidates_triple_audit.jsonl"
PHASE3_REFETCHED_AUDIT_PATH = AUDIT_DIR / "phase3_candidates_triple_audit_refetched.jsonl"
PHASE4_AUDIT_PATH = AUDIT_DIR / "phase4_candidates_triple_audit.jsonl"
PHASE4_REFETCHED_AUDIT_PATH = AUDIT_DIR / "phase4_candidates_triple_audit_refetched.jsonl"
PHASE4_REFETCH_PASSAGES_PATH = BUILD_DIR / "raw" / "passages_refetch_phase4.jsonl"
ORACLE_PASS_A_PATH = BASE_DIR / "oracle_pass_base.jsonl"
BASE_PASSAGES_PATH = BASE_DIR / "passages.jsonl"
REFETCH_PASSAGES_PATH = BUILD_DIR / "raw" / "passages_refetch.jsonl"
WIKIDATA_CACHE_PATH = AUDIT_DIR / "entity_cache.json"

CACHE_PATH = ORACLE_DIR / "oracle_cache.jsonl"
OUT_ALL = ORACLE_DIR / "oracle_gated.jsonl"
OUT_PASS = ORACLE_DIR / "oracle_pass.jsonl"
OUT_REPORT = ORACLE_DIR / "oracle_report.json"

CONDITIONS = {
    # condition_id: (hop0_lang, hop1_lang)
    "a_ko_ko": ("ko", "ko"),
    "b_en_ko": ("en", "ko"),
    "c_ko_en": ("ko", "en"),
    "d_en_en": ("en", "en"),
}


# ── 데이터 로드 ────────────────────────────────────────────────────────────────

def load_eligible_triples() -> list[dict]:
    """Phase 1(pattern=a 기반, refetch 포함), Phase 2(다양화 schema), Phase 3(person/org
    답변형 schema) audit 결과를 합쳐서 eligible_full_2x2_lexical=True인 triple만 반환.
    세 파일은 triple_id prefix가 달라(p1_/p2_/p3_) 서로 겹치지 않는다
    (06/08_audit_*_candidates.py가 이전 phase의 audit 완료분을 이미 제외)."""
    paths = []
    p1 = REFETCHED_AUDIT_PATH if REFETCHED_AUDIT_PATH.exists() else PLAIN_AUDIT_PATH
    paths.append(p1)
    p2 = PHASE2_REFETCHED_AUDIT_PATH if PHASE2_REFETCHED_AUDIT_PATH.exists() else PHASE2_AUDIT_PATH
    if p2.exists():
        paths.append(p2)
    p3 = PHASE3_REFETCHED_AUDIT_PATH if PHASE3_REFETCHED_AUDIT_PATH.exists() else PHASE3_AUDIT_PATH
    if p3.exists():
        paths.append(p3)
    p4 = PHASE4_REFETCHED_AUDIT_PATH if PHASE4_REFETCHED_AUDIT_PATH.exists() else PHASE4_AUDIT_PATH
    if p4.exists():
        paths.append(p4)

    rows = []
    for path in paths:
        n_before = len(rows)
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d.get("eligible_full_2x2_lexical"):
                    rows.append(d)
        print(f"audit 입력: {path} -> eligible {len(rows) - n_before}개")
    print(f"합산 eligible triple: {len(rows)}개")
    return rows


def load_a_ko_ko_oracle_results() -> dict[tuple, dict]:
    """선행 KO-only oracle pass(pattern=a)에서 (S,B,A) -> oracle 결과 재사용."""
    results = {}
    with ORACLE_PASS_A_PATH.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("pattern") != "a":
                continue
            key = (d["hops"][0]["qid"], d["hops"][1]["qid"], d["hops"][2]["qid"])
            results[key] = d  # 이미 PASS인 것만 oracle_pass_base.jsonl에 들어있음
    return results


PHASE2_REFETCH_PASSAGES_PATH = BUILD_DIR / "raw" / "passages_refetch_phase2.jsonl"
PHASE3_REFETCH_PASSAGES_PATH = BUILD_DIR / "raw" / "passages_refetch_phase3.jsonl"


def load_merged_passage_index(needed_qids: set[str]) -> dict[tuple, list[str]]:
    """기본 passages + 회차별 refetch passages를 병합."""
    index = build_passage_index(needed_qids, BASE_PASSAGES_PATH)
    for refetch_path in (REFETCH_PASSAGES_PATH, PHASE2_REFETCH_PASSAGES_PATH, PHASE3_REFETCH_PASSAGES_PATH, PHASE4_REFETCH_PASSAGES_PATH):
        if refetch_path.exists():
            with refetch_path.open(encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line)
                    if p["entity_qid"] in needed_qids:
                        index[(p["entity_qid"], p["lang"])].append(p["text"])
    return index


def load_wikidata_cache() -> dict:
    return json.loads(WIKIDATA_CACHE_PATH.read_text(encoding="utf-8"))


# ── alias / evidence 구성 (oracle_gate_lib 함수 재사용) ──────────────────────

def build_aliases_for_qid(qid: str, cache: dict) -> list[str]:
    entry = cache.get(qid, {})
    candidates = [
        entry.get("label_ko", ""), entry.get("label_en", ""),
        entry.get("wiki_title_ko", ""), entry.get("wiki_title_en", ""),
        *entry.get("aliases_ko", []), *entry.get("aliases_en", []),
    ]
    return list(dict.fromkeys(a for a in candidates if a))


def build_evidence_text(passage_index: dict, qid: str, lang: str, h_aliases: list[str]) -> str:
    texts = passage_index.get((qid, lang), [])
    if not texts:
        return ""
    joined = "\n\n".join(texts)
    return gate.extract_evidence_window(joined, h_aliases)


def subj_label_of(subj_qid: str, wd_cache: dict) -> str:
    """642건(pattern=a 기반)은 hop_s에 label이 있었지만, phase2 신규 145건은 audit row에
    qid만 있다. entity_cache.json은 두 풀의 모든 qid를 이미 커버하므로
    (06_audit_phase2_candidates.py가 fetch_wikidata_cache로 채워둠) 이걸 단일 source로 쓴다."""
    entry = wd_cache.get(subj_qid, {})
    return entry.get("label_ko") or entry.get("label_en") or "?"


def make_question(query_type: str, relations: list, subj_label: str) -> str:
    """oracle_gate_v9.get_question()이 기대하는 {"query_type","relations"} dict로 감싸서 재사용."""
    fake_path = {"query_type": query_type, "relations": relations}
    return gate.get_question(fake_path, subj_label)


# ── cache key ────────────────────────────────────────────────────────────────

def make_cache_key_v10(triple_id: str, condition: str, model: str,
                       hop0_evidence: str, hop1_evidence: str, question: str) -> str:
    key_str = "|".join([
        PROMPT_VERSION, model, triple_id, condition,
        hashlib.sha256(hop0_evidence.encode()).hexdigest()[:16],
        hashlib.sha256(hop1_evidence.encode()).hexdigest()[:16],
        question,
    ])
    return hashlib.sha256(key_str.encode()).hexdigest()


def load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not CACHE_PATH.exists():
        return cache
    with CACHE_PATH.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if "cache_key" in d:
                    cache[d["cache_key"]] = d
            except Exception:
                pass
    return cache


# ── 단일 (triple, condition) 처리 ────────────────────────────────────────────

async def process_condition(
    triple_row: dict,
    condition: str,
    passage_index: dict,
    wd_cache: dict,
    client,
    model: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    s, b, a = triple_row["subj_qid"], triple_row["bridge_qid"], triple_row["answer_qid"]
    hop0_lang, hop1_lang = CONDITIONS[condition]

    subj_label = subj_label_of(s, wd_cache)
    question = make_question(triple_row["query_type"], triple_row["relations"], subj_label)

    b_aliases = build_aliases_for_qid(b, wd_cache)  # hop0 evidence 안에서 찾을 대상 (bridge)
    a_aliases = build_aliases_for_qid(a, wd_cache)  # hop1 evidence 안에서 찾을 대상 (answer)

    hop0_evidence = build_evidence_text(passage_index, s, hop0_lang, b_aliases)
    hop1_evidence = build_evidence_text(passage_index, b, hop1_lang, a_aliases)

    triple_id = triple_row["triple_id"]
    cache_key = make_cache_key_v10(triple_id, condition, model, hop0_evidence, hop1_evidence, question)

    base = {
        "triple_id": triple_id, "condition": condition,
        "subj_qid": s, "bridge_qid": b, "answer_qid": a,
        "hop0_lang": hop0_lang, "hop1_lang": hop1_lang,
        "query_type": triple_row["query_type"], "relations": triple_row["relations"],
        "cache_key": cache_key, "oracle_question": question, "oracle_model": model,
    }

    if not hop0_evidence or not hop1_evidence:
        return {**base, "oracle_verdict": "FAIL", "oracle_fail_reason": "ORACLE_FAIL_NO_EVIDENCE_TEXT",
                "oracle_predicted_answer": None, "oracle_gold_match": False, "oracle_gold_match_type": None,
                "oracle_hop0_bridge_supported": False, "oracle_hop1_answer_supported": False,
                "oracle_explanation": None}

    prompt = gate.make_prompt(question, hop0_evidence, hop1_evidence, b_aliases, a_aliases)

    raw_text, parsed, last_error = None, None, None
    async with semaphore:
        for attempt in range(2):
            try:
                raw_text = await gate.call_llm(client, model, prompt)
                parsed = gate.parse_response(raw_text)
                break
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                parsed = None

    if parsed is None:
        return {**base, "oracle_verdict": "FAIL", "oracle_fail_reason": "ORACLE_FAIL_PARSE_ERROR",
                "oracle_predicted_answer": None, "oracle_gold_match": False, "oracle_gold_match_type": None,
                "oracle_hop0_bridge_supported": False, "oracle_hop1_answer_supported": False,
                "oracle_explanation": raw_text or last_error}

    gate_verdict, fail_reason, match_type = gate.determine_verdict(parsed, a_aliases)
    return {
        **base,
        "oracle_verdict": gate_verdict,
        "oracle_predicted_answer": parsed.get("final_answer"),
        "oracle_gold_match": match_type is not None,
        "oracle_gold_match_type": match_type,
        "oracle_hop0_bridge_supported": gate.parse_bool(parsed.get("hop0_bridge_supported")),
        "oracle_hop1_answer_supported": gate.parse_bool(parsed.get("hop1_answer_supported")),
        "oracle_fail_reason": fail_reason,
        "oracle_explanation": parsed.get("rationale"),
    }


def reuse_a_ko_ko_result(triple_row: dict, wd_cache: dict, a_results: dict, model: str) -> dict | None:
    """선행 KO-only oracle pass(pattern=a)에 이 triple이 실제로 있을 때만 재사용한다.
    버그 수정 기록: 이전 버전은 a_results에 키가 없어도 무조건 PASS를 반환했는데, 이는
    642건(pattern=a 기반)에만 맞고 2차 수집 신규 145건(선행 oracle 검증을 받은 적이 없음)에는
    틀린 동작이었다. 이제 키가 없으면 None을 반환해 호출자가 a_ko_ko도 일반 LLM job으로
    큐에 넣도록 한다."""
    s, b, a = triple_row["subj_qid"], triple_row["bridge_qid"], triple_row["answer_qid"]
    key = (s, b, a)
    if key not in a_results:
        return None
    src = a_results[key]
    subj_label = subj_label_of(s, wd_cache)
    question = make_question(triple_row["query_type"], triple_row["relations"], subj_label)
    cache_key = make_cache_key_v10(triple_row["triple_id"], "a_ko_ko", model, "REUSED_FROM_V9", "REUSED_FROM_V9", question)
    return {
        "triple_id": triple_row["triple_id"], "condition": "a_ko_ko",
        "subj_qid": s, "bridge_qid": b, "answer_qid": a,
        "hop0_lang": "ko", "hop1_lang": "ko",
        "query_type": triple_row["query_type"], "relations": triple_row["relations"],
        "cache_key": cache_key, "oracle_question": question, "oracle_model": "REUSED_FROM_EARLIER_RUN",
        "oracle_verdict": "PASS",
        "oracle_predicted_answer": src.get("oracle_predicted_answer"),
        "oracle_gold_match": src.get("oracle_gold_match"),
        "oracle_gold_match_type": src.get("oracle_gold_match_type"),
        "oracle_hop0_bridge_supported": src.get("oracle_hop0_bridge_supported"),
        "oracle_hop1_answer_supported": src.get("oracle_hop1_answer_supported"),
        "oracle_fail_reason": None,
        "oracle_explanation": "reused from data/oracle_pass_base.jsonl (pattern=a)",
        "reused_v9_path_id": src.get("path_id"),
    }


# ── 리포트 ────────────────────────────────────────────────────────────────────

def build_report(results: list[dict]) -> dict:
    total = len(results)
    n_pass = sum(1 for r in results if r["oracle_verdict"] == "PASS")
    by_condition: dict[str, dict] = defaultdict(lambda: {"pass": 0, "total": 0})
    fail_reasons: Counter = Counter()
    for r in results:
        cond = r["condition"]
        by_condition[cond]["total"] += 1
        if r["oracle_verdict"] == "PASS":
            by_condition[cond]["pass"] += 1
        else:
            fail_reasons[r.get("oracle_fail_reason", "UNKNOWN")] += 1

    by_triple: dict[str, set] = defaultdict(set)
    for r in results:
        if r["oracle_verdict"] == "PASS":
            by_triple[r["triple_id"]].add(r["condition"])
    n_full_2x2_oracle_pass = sum(1 for conds in by_triple.values() if conds == set(CONDITIONS))

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "total": total, "n_pass": n_pass, "n_fail": total - n_pass,
        "pass_rate": round(n_pass / total, 4) if total else 0,
        "by_condition": {k: v for k, v in sorted(by_condition.items())},
        "fail_reasons": dict(fail_reasons.most_common()),
        "n_triples_full_2x2_oracle_pass": n_full_2x2_oracle_pass,
    }


def print_report(report: dict) -> None:
    print("\n" + "=" * 60)
    print(f"Oracle Gate 결과  ({report['generated_at']})")
    print(f"총 {report['total']:,}개 → PASS {report['n_pass']:,} ({report['pass_rate']*100:.1f}%)")
    print("\n조건별:")
    for cond in CONDITIONS:
        v = report["by_condition"].get(cond, {"pass": 0, "total": 0})
        rate = v["pass"] / v["total"] * 100 if v["total"] else 0
        print(f"  {cond}: {v['pass']:,} / {v['total']:,} ({rate:.1f}%)")
    print(f"\nFull 2x2 oracle-pass triple 수: {report['n_triples_full_2x2_oracle_pass']:,}")
    print("\nFail 사유:")
    for reason, cnt in report["fail_reasons"].items():
        print(f"  {reason}: {cnt:,}")


# ── main ──────────────────────────────────────────────────────────────────────

async def main_async(args) -> None:
    model = args.model or "claude-haiku-4-5-20251001"

    is_openai = model.startswith("gpt") or model.startswith("o")
    if not args.dry_run and not args.dump_templates:
        if is_openai:
            if not args.openai_api_key:
                sys.exit("오류: OpenAI 모델에는 --openai-api-key 또는 OPENAI_API_KEY 환경변수 필요")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=args.openai_api_key)
        else:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=args.api_key)
    else:
        client = None

    eligible = load_eligible_triples()
    a_results = load_a_ko_ko_oracle_results()
    wd_cache = load_wikidata_cache()

    needed_qids = set()
    for row in eligible:
        needed_qids.add(row["subj_qid"])
        needed_qids.add(row["bridge_qid"])
    passage_index = load_merged_passage_index(needed_qids)

    if args.dry_run:
        print(f"\n[dry-run] 모델={model}, eligible triple={len(eligible)}개, "
              f"조건={list(CONDITIONS)} (a_ko_ko는 선행 oracle-pass에 있는 triple만 재사용, "
              f"나머지는 4조건 전부 신규 호출)")
        preview_n = args.pending_limit or 3
        shown = 0
        for row in eligible:
            for cond in ("b_en_ko", "c_ko_en", "d_en_en"):
                if shown >= preview_n:
                    break
                hop0_lang, hop1_lang = CONDITIONS[cond]
                subj_label = subj_label_of(row["subj_qid"], wd_cache)
                question = make_question(row["query_type"], row["relations"], subj_label)
                b_aliases = build_aliases_for_qid(row["bridge_qid"], wd_cache)
                a_aliases = build_aliases_for_qid(row["answer_qid"], wd_cache)
                hop0_ev = build_evidence_text(passage_index, row["subj_qid"], hop0_lang, b_aliases)[:300]
                hop1_ev = build_evidence_text(passage_index, row["bridge_qid"], hop1_lang, a_aliases)[:300]
                print(f"\n--- {row['triple_id']} [{cond}] ---")
                print(f"질문: {question}")
                print(f"정답 alias: {a_aliases[:5]}")
                print(f"hop0({hop0_lang}) evidence: {hop0_ev}...")
                print(f"hop1({hop1_lang}) evidence: {hop1_ev}...")
                shown += 1
            if shown >= preview_n:
                break
        return

    cache = load_cache()
    print(f"oracle cache 로드: {len(cache):,}개")

    # ── a_ko_ko: 선행 oracle-pass에 있는 triple만 재사용, 없으면 b/c/d처럼 일반 job으로 큐잉 ──
    reused_results = []
    pending_jobs = []
    cached_results = []
    all_conditions = ("a_ko_ko", "b_en_ko", "c_ko_en", "d_en_en")
    for row in eligible:
        key = (row["subj_qid"], row["bridge_qid"], row["answer_qid"])
        reused = reuse_a_ko_ko_result(row, wd_cache, a_results, model)
        if reused is not None:
            reused_results.append(reused)
            conditions_to_run = ("b_en_ko", "c_ko_en", "d_en_en")
        else:
            conditions_to_run = all_conditions

        for cond in conditions_to_run:
            hop0_lang, hop1_lang = CONDITIONS[cond]
            subj_label = subj_label_of(row["subj_qid"], wd_cache)
            question = make_question(row["query_type"], row["relations"], subj_label)
            b_aliases = build_aliases_for_qid(row["bridge_qid"], wd_cache)
            a_aliases = build_aliases_for_qid(row["answer_qid"], wd_cache)
            hop0_ev = build_evidence_text(passage_index, row["subj_qid"], hop0_lang, b_aliases)
            hop1_ev = build_evidence_text(passage_index, row["bridge_qid"], hop1_lang, a_aliases)
            ck = make_cache_key_v10(row["triple_id"], cond, model, hop0_ev, hop1_ev, question)
            if ck in cache:
                cached_results.append(cache[ck])
            else:
                pending_jobs.append((row, cond))

    if args.pending_limit:
        pending_jobs = pending_jobs[:args.pending_limit]

    cached_results = reused_results + cached_results
    print(f"pending(신규 LLM 호출 필요): {len(pending_jobs):,}  cached: {len(cached_results) - len(reused_results):,}  "
          f"(a_ko_ko 재사용: {len(reused_results):,})")

    semaphore = asyncio.Semaphore(args.concurrency)
    cache_fh = open(CACHE_PATH, "a", encoding="utf-8")
    new_results = []

    batch_size = 50
    for i in range(0, len(pending_jobs), batch_size):
        batch = pending_jobs[i:i + batch_size]
        tasks = [process_condition(row, cond, passage_index, wd_cache, client, model, semaphore)
                 for row, cond in batch]
        results = await asyncio.gather(*tasks)
        for r in results:
            cache_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        new_results.extend(results)
        n_pass_so_far = sum(1 for r in new_results if r["oracle_verdict"] == "PASS")
        print(f"  {len(new_results):,} / {len(pending_jobs):,} 처리 (신규 PASS: {n_pass_so_far:,})")

    cache_fh.close()

    all_results = cached_results + new_results

    with OUT_ALL.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with OUT_PASS.open("w", encoding="utf-8") as f:
        for r in all_results:
            if r["oracle_verdict"] == "PASS":
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = build_report(all_results)
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print_report(report)
    print(f"\n저장: {OUT_ALL.name}, {OUT_PASS.name}, {OUT_REPORT.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=None, help="Anthropic API key")
    import os
    ap.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--model", default=None,
                    help="모델 ID (기본: claude-haiku-4-5-20251001, oracle_gate_v9.py와 동일 기본값)")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--pending-limit", type=int, default=None,
                    help="신규 LLM 호출 개수 제한 (b/c/d만 해당, a는 항상 재사용, 테스트용)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dump-templates", action="store_true")
    args = ap.parse_args()

    if args.dump_templates:
        gate.dump_templates()
        return

    if not args.dry_run and not args.api_key and not args.openai_api_key:
        ap.error("--api-key (Anthropic) 또는 --openai-api-key (OpenAI) 필요 (또는 --dry-run)")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
