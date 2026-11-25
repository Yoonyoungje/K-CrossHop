#!/usr/bin/env python3
"""
analysis/e5_truncation_diag.py — E5 512 토큰 잘림 위험 진단 (E5 부록 truncation 문단). CPU, 토크나이저만 쓴다.
camera-ready 때 work/e5_gate/truncation_diag/truncation_diag.py 에서 옮겼다 (경로 기본값만 바꿈).

e5-large의 512 토큰 한계가 무엇을 위협하는지 정량화한다. 코퍼스는 건드리지 않는다.
tokenizer만 사용하며 2.2GB 가중치는 로드하지 않는다.

절단은 두 채널로 작용한다:
  (a) 인코더 — 잘린 꼬리가 벡터에 반영되지 않음 → 해당 passage가 검색되느냐
  (b) 판정기 — 잘린 꼬리를 hit 판정에서 못 봄 → 맞았다고 세느냐
C-0에서 (b)의 존재 여부를 먼저 확정한다.

근거 passage 정의 (페이지 단위 매핑을 청크로 내려서 직접 탐색):
  hop0 : subject 페이지(path.hops[0].qid) 청크에서 bridge 엔티티 표면형
  hop1 : bridge  페이지(path.hops[1].qid) 청크에서 answer 엔티티 표면형
  언어 : qa.hops[h].gold_lang (= 검색된 index 언어 = path.hops[h+1].gold_lang)

alias 표면형은 데이터셋 필터와 동일한 utils.retrieval_success 로직을 import 해서 쓴다.
"""

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.retrieval_success import build_alias_set, build_alias_surfaces  # noqa: E402

E5_MODEL = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "
LIMIT = 512


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── 정규화 + 원본 char index 매핑 ────────────────────────────────────────────
# utils.retrieval_success의 raw 매칭(1단계)으로 위치를 못 찾은 경우에만 쓰는 보조 경로.
# diag 계열 정규화(공백·문장부호 전부 제거)와 동일한 결과를 내되 원본 위치를 보존한다.
_KEEP = re.compile(r"[\w가-힣]")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    out, idx = [], []
    for i, ch in enumerate(text):
        for lc in ch.lower():
            if _KEEP.match(lc):
                out.append(lc)
                idx.append(i)
    return "".join(out), idx


def norm_plain(text: str) -> str:
    return "".join(lc for ch in text for lc in ch.lower() if _KEEP.match(lc))


def find_surface(text: str, aliases: list[str]) -> tuple[int, int, str, str] | None:
    """가장 이른 표면형 출현 (start_char, end_char, alias, method). 없으면 None."""
    low = text.lower()
    best = None
    for a in aliases:
        p = low.find(a.lower())
        if p >= 0 and (best is None or p < best[0]):
            best = (p, p + len(a), a, "raw")
    if best:
        return best

    norm, imap = normalize_with_map(text)
    for a in aliases:
        na = norm_plain(a)
        if not na:
            continue
        p = norm.find(na)
        if p >= 0:
            s_char = imap[p]
            e_char = imap[min(p + len(na) - 1, len(imap) - 1)] + 1
            if best is None or s_char < best[0]:
                best = (s_char, e_char, a, "normalized")
    return best


def visible_char_limit(tok, text: str) -> tuple[int, int]:
    """(절단 후 실제로 보이는 원본 char 상한, 전체 토큰 수)."""
    full = tok(text, add_special_tokens=True, truncation=False)["input_ids"]
    enc = tok(text, add_special_tokens=True, truncation=True, max_length=LIMIT,
              return_offsets_mapping=True)
    lim = max((e for _, e in enc["offset_mapping"]), default=0)
    return lim, len(full)


def pct(a: int, b: int) -> str:
    return f"{a/b:.2%}" if b else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths",       default="data/dataset/paths.jsonl")
    ap.add_argument("--qa-pairs",    default="data/dataset/qa_pairs.jsonl")
    ap.add_argument("--passages",    default="data/corpus/passages.jsonl")
    ap.add_argument("--alias-cache", default="data/retrieval/alias_cache.json")
    ap.add_argument("--out-dir",     default="results/camera_ready/retriever_robustness/e5_truncation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = {json.loads(l)["path_id"]: json.loads(l)
             for l in Path(args.paths).read_text("utf-8").splitlines() if l.strip()}
    qas = [json.loads(l) for l in Path(args.qa_pairs).read_text("utf-8").splitlines() if l.strip()]
    cache = json.loads(Path(args.alias_cache).read_text("utf-8"))

    # ── 필요한 페이지만 수집 ────────────────────────────────────────────────
    needed: set[tuple[str, str]] = set()
    jobs = []          # C-1 작업 목록
    own_pages: set[tuple[str, str]] = set()   # C-4
    for qa in qas:
        p = paths[qa["source_path_id"]]
        for h in (0, 1):
            lang = qa["hops"][h]["gold_lang"]
            ev_qid = p["hops"][h]["qid"]           # hop0=subject, hop1=bridge
            tgt_qid = qa["hops"][h]["gold_entity_qid"]
            needed.add((ev_qid, lang))
            own_pages.add((tgt_qid, lang))
            needed.add((tgt_qid, lang))
            jobs.append({"qid": qa["qid"], "pattern": qa["pattern"], "hop": h,
                         "lang": lang, "evidence_qid": ev_qid, "target_qid": tgt_qid,
                         "target_label": qa["hops"][h]["gold_entity"]})

    log(f"필요 페이지 {len(needed)}개 / 홉 작업 {len(jobs)}건")

    pages: dict[tuple[str, str], list[dict]] = defaultdict(list)
    corpus_lens: dict[str, list[int]] = defaultdict(list)
    log("passages 스트리밍 ...")
    tok = AutoTokenizer.from_pretrained(E5_MODEL)
    log(f"  tokenizer fast={tok.is_fast}")

    with open(args.passages, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            key = (p["entity_qid"], p.get("lang", "ko"))
            if key in needed:
                pages[key].append(p)
    log(f"  수집된 페이지 {len(pages)}개 / 청크 {sum(len(v) for v in pages.values())}개")

    # ── C-2 전체 코퍼스 기준 손실 분포 (Step 2 결과 재사용 대신 직접 재계산) ──
    log("전체 코퍼스 토큰 길이 재계산 ...")
    t0 = time.time()
    buf, buf_lang = [], []

    def flush_corpus():
        if not buf:
            return
        ids = tok([PASSAGE_PREFIX + t for t in buf], add_special_tokens=True,
                  truncation=False)["input_ids"]
        for lg, seq in zip(buf_lang, ids):
            corpus_lens[lg].append(len(seq))
        buf.clear()
        buf_lang.clear()

    with open(args.passages, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            buf.append(p["text"])
            buf_lang.append(p.get("lang", "ko"))
            if len(buf) >= 2000:
                flush_corpus()
    flush_corpus()
    log(f"  {time.time()-t0:.0f}s")

    # ── C-1 + C-3 ───────────────────────────────────────────────────────────
    alias_cache_surfaces: dict[str, list[str]] = {}

    def surfaces(qid: str) -> list[str]:
        if qid not in alias_cache_surfaces:
            entry = cache.get(qid, {})
            base = build_alias_set(entry)["all"] if entry else []
            extra = build_alias_surfaces(qid, cache)
            seen, merged = set(), []
            for a in list(base) + list(extra):
                if a and a.lower() not in seen:
                    seen.add(a.lower())
                    merged.append(a)
            alias_cache_surfaces[qid] = merged
        return alias_cache_surfaces[qid]

    log("근거 passage 탐색 + 표면형 위치 계산 ...")
    seen_pair: set[tuple[str, str]] = set()
    ev_records = []
    no_surface_pages = 0
    for j in jobs:
        chunks = pages.get((j["evidence_qid"], j["lang"]), [])
        al = surfaces(j["target_qid"]) or [j["target_label"]]
        found_any = False
        for ch in chunks:
            hit = find_surface(ch["text"], al)
            if not hit:
                continue
            found_any = True
            pair = (ch["passage_id"], j["target_qid"])
            if pair in seen_pair:
                continue
            seen_pair.add(pair)
            s, e, alias, method = hit
            prefixed = PASSAGE_PREFIX + ch["text"]
            lim, total_tok = visible_char_limit(tok, prefixed)
            visible = (e + len(PASSAGE_PREFIX)) <= lim
            ev_records.append({
                "passage_id": ch["passage_id"], "lang": j["lang"], "hop": j["hop"],
                "evidence_qid": j["evidence_qid"], "target_qid": j["target_qid"],
                "target_label": j["target_label"], "matched_alias": alias,
                "match_method": method, "total_tokens": total_tok,
                "surface_start_char": s, "surface_end_char": e,
                "visible_char_limit": lim - len(PASSAGE_PREFIX),
                "truncated": total_tok > LIMIT, "surface_visible": visible,
            })
        if not found_any:
            no_surface_pages += 1

    log(f"  근거 passage(중복 제거) {len(ev_records)}건 / 표면형 미발견 홉 {no_surface_pages}건")

    # ── C-4 자기 문서 첫 청크 ───────────────────────────────────────────────
    log("C-4 자기 문서 첫 청크 ...")
    own_records = []
    for qid, lang in sorted(own_pages):
        chunks = sorted(pages.get((qid, lang), []), key=lambda c: c.get("chunk_idx", 0))
        if not chunks:
            continue
        ch = chunks[0]
        al = surfaces(qid)
        if not al:
            continue
        hit = find_surface(ch["text"], al)
        prefixed = PASSAGE_PREFIX + ch["text"]
        lim, total_tok = visible_char_limit(tok, prefixed)
        own_records.append({
            "target_qid": qid, "lang": lang, "passage_id": ch["passage_id"],
            "total_tokens": total_tok, "truncated": total_tok > LIMIT,
            "surface_found": hit is not None,
            "surface_visible": bool(hit) and (hit[1] + len(PASSAGE_PREFIX)) <= lim,
        })

    # ── 집계 ────────────────────────────────────────────────────────────────
    def loss_stats(lengths: list[int]) -> dict:
        over = np.array([n for n in lengths if n > LIMIT])
        if over.size == 0:
            return {"n_truncated": 0}
        excess = over - LIMIT
        return {
            "n_truncated": int(over.size),
            "loss_tokens_median": float(np.median(excess)),
            "loss_tokens_p90": float(np.percentile(excess, 90)),
            "loss_ratio_median": round(float(np.median(excess / over)), 4),
        }

    report: dict = {
        "limit": LIMIT, "model": E5_MODEL, "passage_prefix": PASSAGE_PREFIX,
        "c0_channel_b_present": False,
        "c0_evidence": "04_build_faiss.py:60-69 meta['text']=passages 원문; "
                       "05_pilot_runner.py:88 / gold_entity_retrieval_diag.py:92 가 meta['text']를 판정에 사용. "
                       "절단은 model.encode 내부에서만 발생.",
        "mapping_granularity": "page-level (구축 단계 lexical 검사와 같은 페이지 단위 매핑)",
        "alias_logic": "utils.retrieval_success.build_alias_set / build_alias_surfaces",
        "c1": {}, "c2": {}, "c4": {},
        "n_hop_jobs": len(jobs),
        "n_hops_without_surface": no_surface_pages,
        # find_surface 의 두 경로를 고정 순서로 센다 (집합을 돌면 키 순서가 해시 시드에 따라 바뀐다)
        "match_method_counts": {m: n for m in ("raw", "normalized")
                                if (n := sum(1 for r in ev_records if r["match_method"] == m))},
    }

    # C-1 (전체 / hop별)
    for scope, sel in [("all", lambda r: True),
                       ("hop0", lambda r: r["hop"] == 0),
                       ("hop1", lambda r: r["hop"] == 1)]:
        d = {}
        for lang in ("ko", "en"):
            rs = [r for r in ev_records if r["lang"] == lang and sel(r)]
            trunc = [r for r in rs if r["truncated"]]
            at_risk = [r for r in trunc if not r["surface_visible"]]
            d[lang] = {
                "n_evidence_passages": len(rs),
                "n_truncated": len(trunc),
                "truncated_ratio": round(len(trunc) / len(rs), 4) if rs else None,
                "visible_among_truncated": round(1 - len(at_risk) / len(trunc), 4) if trunc else None,
                "n_at_risk": len(at_risk),
                "at_risk_ratio_of_evidence": round(len(at_risk) / len(rs), 4) if rs else None,
            }
        report["c1"][scope] = d

    # C-2
    for lang in ("ko", "en"):
        report["c2"][lang] = {
            "corpus": loss_stats(corpus_lens[lang]),
            "evidence": loss_stats([r["total_tokens"] for r in ev_records if r["lang"] == lang]),
        }

    # C-4
    for lang in ("ko", "en"):
        rs = [r for r in own_records if r["lang"] == lang]
        found = [r for r in rs if r["surface_found"]]
        report["c4"][lang] = {
            "n_own_first_chunks": len(rs),
            "n_surface_found": len(found),
            "surface_visible_ratio": round(sum(r["surface_visible"] for r in found) / len(found), 4) if found else None,
            "truncated_ratio": round(sum(r["truncated"] for r in rs) / len(rs), 4) if rs else None,
        }

    # ── 판정 ────────────────────────────────────────────────────────────────
    ko = report["c1"]["all"]["ko"]["at_risk_ratio_of_evidence"] or 0.0
    en = report["c1"]["all"]["en"]["at_risk_ratio_of_evidence"] or 0.0
    gap = abs(ko - en)
    if max(ko, en) < 0.05:
        verdict = "LOW"
    elif max(ko, en) <= 0.15 and gap <= 0.05:
        verdict = "MEDIUM"
    else:
        verdict = "HIGH"
    heavier = "KO" if ko > en else ("EN" if en > ko else "동률")
    report["verdict"] = verdict
    report["at_risk_ratio"] = {"ko": ko, "en": en, "gap_pp": round(gap * 100, 2)}
    report["heavier_side"] = heavier

    (out_dir / "truncation_diag.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), "utf-8")
    with open(out_dir / "at_risk_passages.jsonl", "w", encoding="utf-8") as f:
        for r in ev_records:
            if r["truncated"] and not r["surface_visible"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_dir / "evidence_passages.jsonl", "w", encoding="utf-8") as f:
        for r in ev_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 출력 ────────────────────────────────────────────────────────────────
    print("\n=== C-0 판정기 입력 경로 ===")
    print(f"채널 (b) 존재: {report['c0_channel_b_present']}")
    print(f"매핑 단위: {report['mapping_granularity']}")
    print(f"표면형 매칭 방법: {report['match_method_counts']}")

    print("\n=== C-1 answer-bearing 표면형 위치 ===")
    for scope in ("all", "hop0", "hop1"):
        print(f"\n[{scope}]")
        print(f"{'':34} {'KO':>12} {'EN':>12}")
        d = report["c1"][scope]
        rows = [("근거 passage 총 수", "n_evidence_passages", "d"),
                (">512 토큰 (절단 대상) 비율", "truncated_ratio", "p"),
                ("절단 대상 중 표면형 512 내 비율", "visible_among_truncated", "p"),
                ("표면형이 512 밖 (실질 위험 건수)", "n_at_risk", "d"),
                ("  ↳ 근거 passage 대비", "at_risk_ratio_of_evidence", "p")]
        for label, key, fmt in rows:
            vals = []
            for lang in ("ko", "en"):
                v = d[lang][key]
                vals.append("n/a" if v is None else (f"{v:,}" if fmt == "d" else f"{v:.2%}"))
            print(f"{label:34} {vals[0]:>12} {vals[1]:>12}")

    print("\n=== C-2 절단 심각도 ===")
    for base in ("corpus", "evidence"):
        print(f"\n[{base}]")
        print(f"{'index':6} {'절단 n':>9} {'손실 median':>12} {'p90':>8} {'손실 비율 median':>16}")
        for lang in ("ko", "en"):
            s = report["c2"][lang][base]
            if s["n_truncated"] == 0:
                print(f"{lang.upper():6} {0:>9}")
                continue
            print(f"{lang.upper():6} {s['n_truncated']:>9,} {s['loss_tokens_median']:>12.0f} "
                  f"{s['loss_tokens_p90']:>8.0f} {s['loss_ratio_median']:>15.2%}")

    print("\n=== C-4 자기 문서 첫 청크 (Step 3 민감도 대조) ===")
    print(f"{'index':6} {'n':>7} {'표면형 발견':>11} {'512 내 비율':>12} {'절단 비율':>10}")
    for lang in ("ko", "en"):
        s = report["c4"][lang]
        vr = "n/a" if s["surface_visible_ratio"] is None else f"{s['surface_visible_ratio']:.2%}"
        print(f"{lang.upper():6} {s['n_own_first_chunks']:>7,} {s['n_surface_found']:>11,} "
              f"{vr:>12} {s['truncated_ratio']:>9.2%}")

    print(f"\n판정: {verdict}  (KO {ko:.2%} / EN {en:.2%}, 차이 {gap*100:.2f}pp, 더 물리는 쪽: {heavier})")
    print(f"저장: {out_dir}")


if __name__ == "__main__":
    main()
