#!/usr/bin/env python3
"""
analysis/self_translate_check.py

self_translate 재작성 쿼리 점검 (모델 호출 0회). 파일럿 통과 기준과 전체 실행 사전 점검에 같이 쓴다.

축 (홉 단위, 분모 = 해당 조건의 전체 재작성 hop)
  1. 목표 언어 문자: 05 의 _detect_lang(검색 쿼리) == 인덱스 언어      → 파일럿 기준 ≥ 95%
  2. hop0 정답일 때 hop1 재작성 쿼리에 인덱스 언어 gold bridge label/alias 포함 → 보고만
  3. 빈 출력 / 설명문 섞임, 답 누출(V3)                                    → 참고 수치 2% (관문 아님)
     설명문 섞임 = 비어 있지 않은 줄이 2개 이상, 또는 첫 줄이 레이블·머리말로 시작
  (보고만) 검색 쿼리에 그 hop 의 gold 정답(alias 포함) 포함, 원 KO 쿼리의 같은 비율, 물음표 종결 비율

hop0 정답 = ¬기권(utils.answer_parser, 06 과 같은 파서) ∧ 응답에 gold alias
            (analysis/16_pdr_analysis 매처 — 정답 정확도 축과 같은 매처).

  python3 analysis/self_translate_check.py \
      --trajectories data/self_translate/trajectories_llama_self_tr_pilot20.jsonl \
      --out-csv data/self_translate/pilot_llama_rewrites.csv --examples 20
"""
import argparse
import csv
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.answer_parser import parse_generated_answer     # noqa: E402
from utils.retrieval_success import _KO_PARTICLES           # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pdr16 = _load("pdr16", "analysis/16_pdr_analysis.py")
_detect_lang = _load("runner05", "pipeline/05_pilot_runner.py")._detect_lang

HANGUL = re.compile(r"[가-힣]")
WORD = r"[0-9A-Za-z가-힣]"
_PARTICLES = "|".join(sorted(_KO_PARTICLES, key=len, reverse=True))
LATIN = re.compile(r"[A-Za-z]")
PREAMBLE = re.compile(r"^\s*(here|sure|rewritten|rewrite|search query|query|translation|translated|"
                      r"korean|english|note|번역|검색어|검색 쿼리|질문|다시 쓴|재작성)\b.{0,40}[:：]", re.I)


def hangul_ratio(text):
    h, l = len(HANGUL.findall(text)), len(LATIN.findall(text))
    return h / (h + l) if h + l else 0.0


def gold_aliases(entry: dict, fallback: str) -> list[str]:
    """gold 엔티티의 모든 언어 표면형 (누출 판정용)."""
    names = [entry.get("label_ko"), entry.get("label_en"),
             entry.get("wiki_title_ko"), entry.get("wiki_title_en"),
             *entry.get("aliases_ko", []), *entry.get("aliases_en", []), fallback]
    return sorted({n.strip().lower() for n in names if n and n.strip()})


def alias_hit(text: str, aliases: list[str], allow_particle: bool = False) -> str | None:
    """대소문자 무시 · 단어 경계 기준 매칭. allow_particle 이면 alias 뒤 한국어 조사 1개를 허용한다
    (라틴 alias 뒤에 조사가 붙는 "New Taipei, Taiwan가" 같은 경우도 포함)."""
    t = (text or "").lower()
    for a in aliases:
        tail = rf"(?:{_PARTICLES})?" if allow_particle else ""
        if re.search(rf"(?<!{WORD}){re.escape(a)}{tail}(?!{WORD})", t):
            return a
    return None


def carrier_surfaces(path_hop: dict, cache: dict, bridge_text: str) -> set[str]:
    """그 홉의 carrier(hop0=주어, hop1=bridge) 표면형 + 실제로 삽입된 bridge 텍스트.

    V3 규칙: 매칭된 gold 표면형이 이 집합의 일부이면 "원문에 이미 있던 엔티티의 번역·표준화"로 보고
    누출에서 뺀다 (재작성 프롬프트 문언과 같은 판정)."""
    out = {x.strip().lower() for x in (path_hop.get("label_ko"), path_hop.get("label_en")) if x}
    out |= set(gold_aliases(cache.get(path_hop.get("qid", ""), {}), ""))
    if bridge_text:
        out.add(bridge_text.strip().lower())
    return {x for x in out if x}


def index_lang_aliases(entry, lang):
    names = [entry.get(f"label_{lang}"), entry.get(f"wiki_title_{lang}"), *entry.get(f"aliases_{lang}", [])]
    return {pdr16.normalize(n) for n in names if n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+", required=True)
    ap.add_argument("--qa-pairs", default="data/dataset/qa_pairs.jsonl")
    ap.add_argument("--alias-cache", default="data/retrieval/alias_cache.json")
    ap.add_argument("--paths", default="data/dataset/paths.jsonl",
                    help="carrier(주어·bridge) 엔티티 라벨 출처 — V3 누출 규칙용")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--examples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qa = {json.loads(l)["qid"]: json.loads(l) for l in (ROOT / args.qa_pairs).open(encoding="utf-8") if l.strip()}
    cache = json.loads((ROOT / args.alias_cache).read_text(encoding="utf-8"))
    paths = {}
    for line in (ROOT / args.paths).open(encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            paths[p["path_id"]] = p

    rows = []
    for path in args.trajectories:
        model_file = Path(path).stem
        for line in (ROOT / path).open(encoding="utf-8"):
            if not line.strip():
                continue
            t = json.loads(line)
            steps = {h["hop"]: h for h in t["trajectory"]}
            q = qa[t["qid"]]
            h0 = steps[0]
            p0 = parse_generated_answer(h0.get("generated_text", "") or "")
            a0 = pdr16.build_alias_surfaces(q["hops"][0]["gold_entity_qid"], cache, q["hops"][0]["gold_entity"])
            hop0_correct = (not p0.abstained) and pdr16.entity_in_text(a0, h0.get("generated_text", "") or "")
            for h in t["trajectory"]:
                if "self_translate_status" not in h:
                    continue
                k = h["hop"]
                qh = q["hops"][k]
                raw = h["self_translate_raw_output"]
                query = h["retrieval_query"]
                src = h["self_translate_source_query"]
                target = h["self_translate_target_lang"]
                lines = [ln for ln in raw.strip().splitlines() if ln.strip()]
                gold_all = gold_aliases(cache.get(qh["gold_entity_qid"], {}), qh["gold_entity"])
                in_src = alias_hit(src, gold_all)               # V1: 단어 경계만
                in_qry = alias_hit(query, gold_all)
                in_src_p = alias_hit(src, gold_all, True)       # V3: + 조사 허용
                in_qry_p = alias_hit(query, gold_all, True)
                bridge_text = h.get("prev_answer_raw") or h.get("bridge_entity") or ""
                carriers = carrier_surfaces(paths[q["source_path_id"]]["hops"][k], cache, bridge_text)
                from_carrier = bool(in_qry_p) and any(in_qry_p in c for c in carriers)
                leak_v3 = bool(in_qry_p) and not in_src_p and not from_carrier
                leak_type = ("none" if not in_qry_p else
                             "already_in_source" if in_src_p else
                             "translation_of_carrier" if from_carrier else "added")
                usage = h.get("self_translate_usage") or {}
                row = {
                    "file": model_file, "qid": t["qid"], "pattern": t["pattern"], "hop": k,
                    "index_lang": target, "status": h["self_translate_status"],
                    "source_query": src, "raw_output": raw, "retrieval_query": query,
                    "generation_question": h.get("generation_question", ""),
                    "n_lines": len(lines), "detected_lang": _detect_lang(query),
                    "hangul_ratio": round(hangul_ratio(query), 3),
                    "script_ok": int(h["self_translate_status"] == "ok" and _detect_lang(query) == target),
                    "empty_or_error": int(h["self_translate_status"] != "ok"),
                    "explanation_mixed": int(h["self_translate_status"] == "ok"
                                             and (len(lines) > 1 or bool(PREAMBLE.match(lines[0])))),
                    "ends_with_question_mark": int(query.rstrip().endswith(("?", "？"))),
                    # 누출 = gold 답 표면형이 재작성에는 있고 원문 KO 쿼리에는 없음.
                    # 원문에 이미 있으면 "원문 포함"으로 따로 센다 (누출 아님).
                    # V3(주 규칙): 조사 허용 + carrier 이름의 번역·표준화는 제외
                    "leak_v3": int(leak_v3),
                    "leak_type": leak_type,
                    # V1(병기, 보수적 상한): 단어 경계만
                    "leak_v1": int(bool(in_qry) and not in_src),
                    "gold_in_source": int(bool(in_src_p)),
                    "matched_alias": in_qry_p or in_qry or "",
                    "finish_reason": usage.get("finish_reason", ""),
                    "output_tokens": usage.get("output_tokens", ""),
                    "hop0_correct": int(hop0_correct) if k == 1 else "",
                    "bridge_indexlang_in_query": "",
                }
                if k == 1:
                    entry = cache.get(q["hops"][0]["gold_entity_qid"], {})
                    row["bridge_indexlang_in_query"] = int(any(
                        a in pdr16.normalize(query) for a in index_lang_aliases(entry, target)))
                rows.append(row)

    out = ROOT / args.out_csv
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def rate(sel, key):
        n = len(sel)
        k = sum(int(r[key]) for r in sel)
        return f"{k}/{n} ({k / n:.1%})" if n else "0/0"

    rw = [r for r in rows if r["status"] != "skipped_ko_index"]   # 2차 절차: EN-index 홉만 재작성
    print(f"홉 {len(rows)} (재작성 {len(rw)} / KO-index 생략 {len(rows) - len(rw)})  →  CSV {out}")
    print("\n[기준 1] 목표 언어 문자 (_detect_lang == 인덱스 언어) — 기준 ≥ 95%, 재작성 홉 기준")
    print(f"  재작성 홉 {rate(rw, 'script_ok')}")
    for lang in ("ko", "en"):
        sel = [r for r in rw if r["index_lang"] == lang]
        if sel:
            print(f"  목표 {lang}: {rate(sel, 'script_ok')}")
    en_with_hangul = sum(1 for r in rows if r["index_lang"] == "en" and HANGUL.search(r["retrieval_query"]))
    print(f"  (참고) 목표 en 인데 한글이 한 글자라도 있는 쿼리: {en_with_hangul}")

    print("\n[기준 2] 빈 출력 / 설명문·서술문 섞임 — 기준 합계 ≤ 2%, 재작성 홉 기준")
    print(f"  ① 빈 출력·호출 오류(원 쿼리로 대체): {rate(rw, 'empty_or_error')}")
    print(f"  ② 여러 줄·머리말: {rate(rw, 'explanation_mixed')}")
    decl = [dict(r, d=int(not r["ends_with_question_mark"] and
                          bool(re.search(r"(다|니다|요)[.!]?$", r["retrieval_query"].strip())))) for r in rw]
    print(f"  ③ 답 서술문: {rate(decl, 'd')}")
    both = [dict(r, any_bad=int(r["empty_or_error"] or r["explanation_mixed"] or d["d"]))
            for r, d in zip(rw, decl)]
    print(f"  합집합: {rate(both, 'any_bad')}")

    print("\n[공변량 3] 답 누출 — 참고 수치 2%, 재작성 홉 기준")
    print(f"  V3(주 규칙, carrier 번역 제외): {rate(rw, 'leak_v3')}")
    print(f"  V1(병기, 단어 경계만 — 보수적 상한): {rate(rw, 'leak_v1')}")
    print(f"  유형 분포: {dict(Counter(r['leak_type'] for r in rw))}")

    fr = Counter(r["finish_reason"] for r in rw)
    n_len = sum(v for k, v in fr.items() if str(k).lower() in ("length", "max_tokens"))
    print(f"\n[보고] finish_reason: {dict(fr)}  → length 비율 "
          f"{n_len}/{len(rw)} ({n_len / len(rw):.1%})" if rw else "")

    print("\n[보고 2] hop0 정답일 때 hop1 재작성 쿼리에 인덱스 언어 gold bridge label/alias 포함")
    sel = [r for r in rows if r["hop"] == 1 and r["hop0_correct"] == 1]
    print(f"  전체 {rate(sel, 'bridge_indexlang_in_query')}")
    for lang in ("ko", "en"):
        print(f"  인덱스 {lang}: {rate([r for r in sel if r['index_lang'] == lang], 'bridge_indexlang_in_query')}")

    print("\n[보고] hop·인덱스별 누출 (V3)")
    for hop in (0, 1):
        for lang in ("ko", "en"):
            s = [r for r in rw if r["hop"] == hop and r["index_lang"] == lang]
            if s:
                print(f"  hop{hop} 인덱스 {lang}: 누출 {rate(s, 'leak_v3')}  "
                      f"원문 포함 {rate(s, 'gold_in_source')}")
    leaks = [r for r in rw if r["leak_v3"]]
    if leaks:
        print(f"\n[누출 전수 {len(leaks)}건]")
        for r in leaks:
            print(f"  {r['qid']} hop{r['hop']} 매칭={r['matched_alias']!r}\n     원문: {r['source_query'][:100]}"
                  f"\n     재작성: {r['retrieval_query'][:120]}")
    print(f"  물음표로 끝나는 검색 쿼리: {rate(rw, 'ends_with_question_mark')}")
    print(f"  status 분포: {dict(Counter(r['status'] for r in rows))}")

    if args.examples:
        rng = random.Random(args.seed)
        strata = defaultdict(list)
        for r in rows:
            strata[(r["hop"], r["index_lang"])].append(r)
        per = max(1, args.examples // len(strata))
        ex = [r for key in sorted(strata) for r in rng.sample(strata[key], min(per, len(strata[key])))]
        print(f"\n[예시 {len(ex)}] (hop, 인덱스 언어)별 층화 무작위, seed={args.seed}")
        print("| # | qid | 패턴 | hop | 인덱스 | 원 KO 쿼리 (= 생성 질문) | 모델 출력 원문 | 검색 쿼리 |")
        print("|-|-|-|-|-|-|-|-|")
        for i, r in enumerate(ex, 1):
            cell = lambda s: s.replace("|", "\\|").replace("\n", " ⏎ ")  # noqa: E731
            print(f"| {i} | {r['qid']} | {r['pattern']} | {r['hop']} | {r['index_lang']} | "
                  f"{cell(r['source_query'])} | {cell(r['raw_output'])} | {cell(r['retrieval_query'])} |")


if __name__ == "__main__":
    main()
