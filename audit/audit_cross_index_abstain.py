#!/usr/bin/env python3
"""
audit/audit_cross_index_abstain.py

answer-bearing hit proxy 감사용 후보 마크다운 생성.
cross-index hop0 중 proxy hit인데 모델이 기권한 케이스를 모델별로 모은다.

대상: hop0 gold_lang=en(cross-index) + answer_bearing_hit=True + abstained=True.
목적: gold entity가 passage에 있는데 모델이 기권한 게
  proxy-FP(entity는 등장하나 질문이 요구하는 relation이 명시 안 됨)인지
  과기권(relation이 명시됐는데도 기권)인지 사람이 판정.

출력의 판정 칸(`- [ ] 판정:`)은 비어 있다 — 사람이 채워야 한다(자동 판정 아님).
"""

import argparse
import json
from pathlib import Path


MODELS = {
    "aya":    ("data/retrieval/aya_ecological_hits.jsonl",
               "data/trajectories/aya_ecological.jsonl"),
    "exaone": ("data/retrieval/exaone_ecological_hits.jsonl",
               "data/trajectories/exaone_ecological.jsonl"),
    "llama":  ("data/retrieval/llama_ecological_hits.jsonl",
               "data/trajectories/llama_ecological.jsonl"),
    "qwen":   ("data/retrieval/qwen_ecological_hits.jsonl",
               "data/trajectories/qwen_ecological.jsonl"),
    "sonnet": ("data/retrieval/sonnet_ecological_hits.jsonl",
               "data/trajectories/sonnet_ecological.jsonl"),
    "gemini": ("data/retrieval/gemini_ecological_hits.jsonl",
               "data/trajectories/gemini_ecological.jsonl"),
}

CTX_WINDOW = 150   # 매칭 지점 앞뒤로 보여줄 글자 수
PLAIN_LEN  = 280   # 매칭 없는 passage를 자를 길이


def load_traj(traj_path: str) -> dict:
    trajs = {}
    for line in open(traj_path):
        it = json.loads(line)
        trajs[it["qid"]] = it
    return trajs


def hop0_of(traj_item: dict) -> dict:
    for h in traj_item.get("trajectory", []):
        if h.get("hop") == 0:
            return h
    return {}


def find_match_span(text: str, needle: str):
    if not needle:
        return None
    idx = text.lower().find(needle.lower())
    if idx == -1:
        return None
    return idx, idx + len(needle)


def render_passage(text: str, needle: str) -> tuple[str, bool]:
    span = find_match_span(text, needle)
    if span is None:
        snippet = text[:PLAIN_LEN]
        if len(text) > PLAIN_LEN:
            snippet += "…"
        return snippet, False

    start, end = span
    win_start = max(0, start - CTX_WINDOW)
    win_end = min(len(text), end + CTX_WINDOW)
    prefix = "…" if win_start > 0 else ""
    suffix = "…" if win_end < len(text) else ""
    snippet = f"{text[win_start:start]}**{text[start:end]}**{text[end:win_end]}"
    return f"{prefix}{snippet}{suffix}", True


def summary_stats(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0, "abstain_pct": 0.0, "hallu_pct": 0.0, "faith_pct": 0.0}
    abstain = sum(1 for r in rows if r.get("abstained"))
    hallu   = sum(1 for r in rows if not r.get("abstained") and r.get("has_hallucination"))
    faith   = n - abstain - hallu
    return {
        "n": n,
        "abstain_pct": 100 * abstain / n,
        "hallu_pct":   100 * hallu / n,
        "faith_pct":   100 * faith / n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--output", default=None,
                         help="미지정 시 data/retrieval/{model}/audit_cross_v3T_abstain.md")
    args = parser.parse_args()

    hits_path, traj_path = MODELS[args.model]
    hit_rows = [json.loads(l) for l in Path(hits_path).read_text().splitlines() if l.strip()]
    hop0_rows = [r for r in hit_rows if r.get("hop") == 0]

    same_v3t  = [r for r in hop0_rows if r["gold_lang"] == "ko" and r.get("answer_bearing_hit")]
    cross_v3t = [r for r in hop0_rows if r["gold_lang"] == "en" and r.get("answer_bearing_hit")]

    cases = [r for r in cross_v3t if r.get("abstained")]

    trajs = load_traj(traj_path)

    same_stats  = summary_stats(same_v3t)
    cross_stats = summary_stats(cross_v3t)

    lines = []
    lines.append(f"# Cross-index v3=True 기권 케이스 수동 감사 ({args.model} dynamic-controlled)")
    lines.append(f"> 자동 생성 (audit/audit_cross_index_abstain.py). "
                 f"hop0 gold_lang=en(cross-index) + hop0 answer_bearing_hit=True + 모델 기권 한 {len(cases)}건.")
    lines.append("> **목적**: gold entity가 passage에 (정확히) 있는데 모델이 기권한 이유를 사람이 판정 — "
                 "proxy-FP(entity 등장하나 관계 미진술) vs 진짜 과기권(관계 명시됐는데 기권) 구분.")
    lines.append("")
    lines.append("## 요약")
    lines.append("| index×v3 | n | 기권% | 환각% | faithful% |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| same(ko) v3=T | {same_stats['n']} | {same_stats['abstain_pct']:.1f}% | "
                 f"{same_stats['hallu_pct']:.1f}% | {same_stats['faith_pct']:.1f}% |")
    lines.append(f"| cross(en) v3=T | {cross_stats['n']} | **{cross_stats['abstain_pct']:.1f}%** | "
                 f"{cross_stats['hallu_pct']:.1f}% | {cross_stats['faith_pct']:.1f}% |")
    lines.append("")
    lines.append("> parser keyword-collision(예: `Unknown Worlds Entertainment`류 오탐) 재분류는 자동화하지 않음 — "
                 "판정 중 발견 시 케이스 하단에 `🔧 parser-fp` 로 표시.")
    lines.append("")
    lines.append("**감사 판정 칸**: 각 케이스 끝 `- [ ] 판정:` 에 `proxy-FP` / `과기권` / `모호` 기입.")
    lines.append("")
    lines.append("---")

    for i, r in enumerate(cases, 1):
        qid = r["qid"]
        traj_item = trajs.get(qid, {})
        h0 = hop0_of(traj_item)
        gen_text = h0.get("generated_text", "")
        sub_q = h0.get("retrieval_query", "")
        passages = h0.get("retrieved_passages", [])
        langs = h0.get("retrieved_langs", [])
        needle = r.get("hit_matched_alias") or r.get("gold_entity") or ""

        gold_idx = []
        rendered = []
        for pi, p in enumerate(passages):
            lang = langs[pi] if pi < len(langs) else "?"
            snippet, matched = render_passage(p, needle)
            tag = f"[{lang} ✅gold]" if matched else f"[{lang}]"
            if matched:
                gold_idx.append(f"P{pi+1}")
            rendered.append(f"  {pi+1}. {tag} {snippet}")

        lines.append(f"## {i}. `{qid}` — pattern {r.get('pattern','?')}  ")
        lines.append(f"- **gold_entity**: `{r.get('gold_entity','')}`  ·  match_method: `{r.get('match_method','')}`")
        lines.append(f"- **sub_q (hop0)**: {sub_q}")
        lines.append("- **모델 답변**:")
        lines.append("  ```")
        lines.append(f"  {gen_text}")
        lines.append("  ```")
        lines.append(f"- **gold entity 등장 passage**: {', '.join(gold_idx) if gold_idx else '(미검출)'}")
        lines.append(f"- **retrieved passages** ({len(passages)}개, gold 등장부 **굵게**):")
        lines.extend(rendered)
        lines.append("- [ ] **판정**: ")
        lines.append("")
        lines.append("---")

    out_path = Path(args.output) if args.output else Path(f"data/retrieval/{args.model}/audit_cross_v3T_abstain.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"{args.model}: cross(en) v3=True {cross_stats['n']}개 중 기권 {len(cases)}건 → {out_path}")


if __name__ == "__main__":
    main()
