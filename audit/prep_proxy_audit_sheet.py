#!/usr/bin/env python3
"""
audit/prep_proxy_audit_sheet.py

answer_bearing_hit proxy audit을 Human IAA(audit/sample_iaa.py)와 동일한
Google Sheets 양식(TSV, annotator 3인 label_A/B/C)으로 재구성한다.

배경: sonnet의 기존 판정(data/retrieval/sonnet/audit_cross_v3T_abstain_manual_reviewed.md)은
연구자가 LLM과 함께 임시로 붙인 판정이라 실제 human audit이 아니다. 이 스크립트는
6모델 전체 후보(audit/audit_cross_index_abstain.py로 생성된 케이스)를 pooling하고,
sonnet의 기존 판정을 draft_label(참고용, 숨김)로만 남긴 채 label_A/B/C를 비운 상태로
재배포한다 — sonnet 케이스도 다시 blind 판정 대상에 포함된다.

Calibration은 sonnet의 기존 판정에서 라벨 종류별로 뽑아 구성한다(답이 이미 있으므로
calibration 토론에 바로 쓸 수 있음). 그 케이스들은 main에서 제외한다.

출력:
  data/human_audit/proxy/proxy_audit_main.jsonl
  data/human_audit/proxy/proxy_audit_calibration.jsonl
  data/human_audit/proxy/proxy_audit_main_sheet.tsv
  data/human_audit/proxy/proxy_audit_calibration_sheet.tsv
"""

import argparse
import json
import random
import re
from collections import defaultdict, Counter
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

SONNET_REVIEWED_MD = "data/retrieval/sonnet/audit_cross_v3T_abstain_manual_reviewed.md"
QA_PAIRS = "data/dataset/qa_pairs.jsonl"

# --abstain-source annotated 용 정본 annotated.
# hit 파일의 abstained는 생성 시점 스냅샷이라 exaone이 기권 판정 일원화 이전 상태다
# (hop0 194 vs 정본 242) → 기본값 hits 는 기존 시트 재현용으로만 남긴다.
ANNOTATED = {
    "aya":    "data/annotated/aya_ecological.jsonl",
    "exaone": "data/annotated/exaone_ecological.jsonl",
    "llama":  "data/annotated/llama_ecological.jsonl",
    "qwen":   "data/annotated/qwen_ecological.jsonl",
    "sonnet": "data/annotated/sonnet_ecological.jsonl",
    "gemini": "data/annotated/gemini_ecological.jsonl",
}

LABEL_MAP_KO2CODE = {
    "proxy-FP":       "proxy_fp",
    "과기권":       "overabstain",
    "모호":        "ambiguous",
    "파서수정후 정답": "parser_fp",
}

# calibration 구성: sonnet 기존 판정에서 라벨별로 뽑을 개수
CALIB_SPEC = {"proxy_fp": 8, "overabstain": 4, "ambiguous": 4}
CTX_WINDOW = 150
PLAIN_LEN  = 280
LEAD_CHARS = 160   # 도입부에서 문장 끝을 못 찾을 때 쓸 길이
LEAD_MAX   = 400   # 첫 문장 탐색 상한
SEED = 42


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


def lead_of(text: str) -> str:
    """패시지 도입부(첫 문장). 위키 문서는 주어가 첫 문장에 오므로, 이게 잘리면
    라벨러는 그 passage가 무엇에 대한 글인지 알 수 없다."""
    m = re.search(r"[.。]\s", text[:LEAD_MAX])
    return text[:m.start() + 1] if m else text[:LEAD_CHARS]


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
    suffix = "…" if win_end < len(text) else ""
    # 도입부는 항상 보이게 한다. 창이 도입부에 닿으면 통째로 잇고, 멀면 도입부 + … + 창.
    lead = lead_of(text)
    if win_start <= len(lead):
        win_start = 0
    snippet = f"{text[win_start:start]}**{text[start:end]}**{text[end:win_end]}"
    if win_start > 0:
        snippet = f"{lead} … {snippet}"
    return f"{snippet}{suffix}", True


def load_ann_hop0_abstain(model: str) -> dict:
    """정본 annotated의 hop0 abstained. {qid: bool}"""
    out = {}
    for line in open(ANNOTATED[model], encoding="utf-8"):
        it = json.loads(line)
        for h in it.get("hops", []):
            if h.get("hop") == 0:
                out[it["qid"]] = bool(h.get("abstained"))
    return out


def load_qa_qids(qa_pairs: str) -> set:
    return {r["qid"] for r in map(json.loads, open(qa_pairs, encoding="utf-8"))}


def build_cases(model: str, abstain_source: str = "hits", qa_pairs: str | None = None) -> list[dict]:
    hits_path, traj_path = MODELS[model]
    hit_rows = [json.loads(l) for l in Path(hits_path).read_text().splitlines() if l.strip()]
    hop0_rows = [r for r in hit_rows if r.get("hop") == 0]
    if qa_pairs:
        keep = load_qa_qids(qa_pairs)
        hop0_rows = [r for r in hop0_rows if r["qid"] in keep]
    ann_abstain = load_ann_hop0_abstain(model) if abstain_source == "annotated" else None
    def abstained(r):
        return ann_abstain.get(r["qid"], False) if ann_abstain is not None else r.get("abstained")
    cases_raw = [r for r in hop0_rows if r["gold_lang"] == "en"
                 and r.get("answer_bearing_hit") and abstained(r)]

    trajs = load_traj(traj_path)
    out = []
    for r in cases_raw:
        qid = r["qid"]
        h0 = hop0_of(trajs.get(qid, {}))
        passages = h0.get("retrieved_passages", [])
        langs = h0.get("retrieved_langs", [])
        needle = r.get("hit_matched_alias") or r.get("gold_entity") or ""

        gold_idx, rendered = [], []
        for pi, p in enumerate(passages):
            lang = langs[pi] if pi < len(langs) else "?"
            snippet, matched = render_passage(p, needle)
            tag = f"[{lang} ✅gold]" if matched else f"[{lang}]"
            if matched:
                gold_idx.append(f"P{pi+1}")
            rendered.append(f"{tag} {snippet}")

        out.append({
            "model":            model,
            "qid":              qid,
            "pattern":          r.get("pattern", "?"),
            "gold_entity":      r.get("gold_entity", ""),
            "match_method":  r.get("match_method", ""),
            "sub_q":            h0.get("retrieval_query", ""),
            "generated_text":   h0.get("generated_text", ""),
            "passages_rendered": "\n---\n".join(rendered),
            "gold_passage_idx": ", ".join(gold_idx) if gold_idx else "(미검출)",
        })
    return out


def parse_sonnet_reviewed(path: str) -> dict:
    text = Path(path).read_text(encoding="utf-8")
    rows = {}
    row_re = re.compile(
        r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*(\w)\s*\|\s*`([^`]*)`\s*\|\s*\*\*([^*]+)\*\*\s*\|\s*(.+?)\s*\|\s*$",
        re.MULTILINE,
    )
    for m in row_re.finditer(text):
        qid, pattern, gold_entity, label_ko, reason = m.groups()
        code = LABEL_MAP_KO2CODE.get(label_ko.strip())
        if code is None:
            continue
        rows[qid] = {"label": code, "label_ko": label_ko.strip(), "reason": reason.strip()}
    return rows


def _sanitize(val) -> str:
    return str(val).replace("\n", " ↵ ").replace("\t", " ")


def write_tsv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = ["\t".join(headers)]
    for r in rows:
        lines.append("\t".join(_sanitize(r.get(h, "")) for h in headers))
    path.write_text("\n".join(lines), encoding="utf-8")


def to_sheet_row(c: dict, draft_label: str = "", calib_type: str = "") -> dict:
    return {
        "model":            c["model"],
        "qid":              c["qid"],
        "pattern":          c["pattern"],
        "gold_entity":      c["gold_entity"],
        "match_method":  c["match_method"],
        "sub_q":            c["sub_q"],
        "generated_text":   c["generated_text"],
        "gold_passage_idx": c["gold_passage_idx"],
        "passages":         c["passages_rendered"],
        "label_A": "", "reason_A": "",
        "label_B": "", "reason_B": "",
        "label_C": "", "reason_C": "",
        "draft_label_llm_참고용_판정에쓰지마세요": draft_label,
        "calib_type": calib_type,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/human_audit/proxy/")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--abstain-source", choices=["hits", "annotated"], default="hits",
                        help="기권 판정 출처. hits=answer-bearing hit 파일(기존 시트 재현), "
                             "annotated=정본 annotated(기권 판정 일원화 반영)")
    parser.add_argument("--exclude-sheet", nargs="*", default=[],
                        help="이미 라벨링된 시트 jsonl. 여기 있는 (model,qid)는 출력에서 뺀다")
    parser.add_argument("--include-draft-parser-fp", action="store_true",
                        help="sonnet draft parser_fp 2건을 pool에서 빼지 않는다 (인간 blind 판정 대상에 포함)")
    parser.add_argument("--prefix", default="proxy_audit", help="출력 파일 접두어")
    parser.add_argument("--qa-pairs", default=None,
                        help="분석 대상 QA JSONL (지정 시 이 파일의 qid 로 모집단 제한)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sonnet_reviewed = parse_sonnet_reviewed(SONNET_REVIEWED_MD)
    print(f"sonnet 기존(임시 LLM) 판정 파싱: {len(sonnet_reviewed)}건 "
          f"({dict(Counter(v['label'] for v in sonnet_reviewed.values()))})")

    all_cases = []
    for model in MODELS:
        cases = build_cases(model, args.abstain_source, args.qa_pairs)
        all_cases.extend(cases)
        print(f"  {model}: {len(cases)}건")
    print(f"전체 pooled: {len(all_cases)}건 (abstain-source={args.abstain_source})")

    # sonnet의 parser_fp(기권 재분류) 케이스는 audit 대상 자체가 아니므로 pool에서 제외
    parser_fp_qids = {qid for qid, v in sonnet_reviewed.items() if v["label"] == "parser_fp"}
    if args.include_draft_parser_fp:
        print(f"parser_fp 제외 안 함 (--include-draft-parser-fp): {len(parser_fp_qids)}건 유지")
    else:
        before = len(all_cases)
        all_cases = [c for c in all_cases if not (c["model"] == "sonnet" and c["qid"] in parser_fp_qids)]
        print(f"parser_fp 제외: {before} → {len(all_cases)} ({len(parser_fp_qids)}건 제외)")

    # 이미 라벨링된 시트 차감 → 보충 감사분만 남는다
    if args.exclude_sheet:
        done = set()
        for path in args.exclude_sheet:
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    done.add((r["model"], r["qid"]))
        before = len(all_cases)
        all_cases = [c for c in all_cases if (c["model"], c["qid"]) not in done]
        print(f"기존 라벨분 차감: {before} → {len(all_cases)} (기존 시트 {len(done)}건)")

    # ── calibration: sonnet 기존 판정에서 라벨별로 추출 ──────────────────────
    by_label = defaultdict(list)
    for qid, v in sonnet_reviewed.items():
        if v["label"] in CALIB_SPEC:
            by_label[v["label"]].append(qid)
    for v in by_label.values():
        rng.shuffle(v)

    calib_qids = set()
    for label, need_n in CALIB_SPEC.items():
        calib_qids.update(by_label[label][:need_n])

    calib_cases = [c for c in all_cases if c["model"] == "sonnet" and c["qid"] in calib_qids]
    main_cases = [c for c in all_cases if not (c["model"] == "sonnet" and c["qid"] in calib_qids)]

    calib_dist = Counter(sonnet_reviewed[c["qid"]]["label"] for c in calib_cases)
    print(f"\ncalibration: {len(calib_cases)}건 (라벨 분포: {dict(calib_dist)})")

    main_model_dist = Counter(c["model"] for c in main_cases)
    print(f"main: {len(main_cases)}건 (모델 분포: {dict(main_model_dist)})")

    # ── 저장 ───────────────────────────────────────────────────────────────
    with open(out_dir / f"{args.prefix}_main.jsonl", "w", encoding="utf-8") as f:
        for c in main_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(out_dir / f"{args.prefix}_calibration.jsonl", "w", encoding="utf-8") as f:
        for c in calib_cases:
            row = dict(c)
            row["draft_label"] = sonnet_reviewed[c["qid"]]["label"]
            row["draft_reason"] = sonnet_reviewed[c["qid"]]["reason"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    main_rows = [to_sheet_row(c, draft_label=sonnet_reviewed.get(c["qid"], {}).get("label_ko", ""))
                 for c in main_cases]
    calib_rows = [to_sheet_row(c, draft_label=sonnet_reviewed[c["qid"]]["label_ko"], calib_type="calibration")
                  for c in calib_cases]

    write_tsv(main_rows, out_dir / f"{args.prefix}_main_sheet.tsv")
    write_tsv(calib_rows, out_dir / f"{args.prefix}_calibration_sheet.tsv")

    print(f"\n저장:")
    print(f"  {out_dir}/{args.prefix}_main_sheet.tsv ({len(main_rows)} rows)")
    print(f"  {out_dir}/{args.prefix}_calibration_sheet.tsv ({len(calib_rows)} rows)")


if __name__ == "__main__":
    main()
