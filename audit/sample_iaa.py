#!/usr/bin/env python3
"""
audit/sample_iaa.py

Human IAA 샘플링 (6모델 pooling).

이전 단일모델 샘플러 대비 변경:
  - 단일 모델 → 6모델(aya/exaone/llama/qwen/sonnet/gemini) pooling
  - generated_text/retrieved_passages를 annotated hop이 아니라 trajectory에서
    가져오도록 수정 (이전 스크립트의 버그: 두 필드 모두 annotated hop에 애초에
    존재하지 않아 calibration 시트는 passages/generated_text가 전부 공백이었고,
    main 시트도 generated_text가 전부 공백이었음)
  - 분석 대상 QA로 상시 제한 (8B/frontier 공통 qid 공간)
  - 출력 행에 model 열 추가

출력:
  data/human_audit/iaa/iaa_main_sample.jsonl
  data/human_audit/iaa/iaa_calibration.jsonl
  data/human_audit/iaa/iaa_main_sheet.tsv
  data/human_audit/iaa/iaa_calibration_sheet.tsv
"""

import argparse
import json
import random
from collections import defaultdict, Counter
from pathlib import Path


# ── 설정 ──────────────────────────────────────────────────────────────────
MAIN_N         = 50   # 본 샘플 trajectory 수
MAIN_HALLU_R   = 0.40 # hallu item 비율 목표
CALIB_N        = 20   # calibration 수
SEED           = 42

CALIB_SPEC = {
    "supported_clear":    8,
    "unsupported_clear":  6,
    "contradicted_clear": 3,
    "borderline":         3,
}
CALIB_CROSS_MIN = 5

# 정본 6모델: (label, annotated_path, trajectory_path)
MODELS = [
    ("aya",    "data/annotated/aya_ecological.jsonl",
               "data/trajectories/aya_ecological.jsonl"),
    ("exaone", "data/annotated/exaone_ecological.jsonl",
               "data/trajectories/exaone_ecological.jsonl"),
    ("llama",  "data/annotated/llama_ecological.jsonl",
               "data/trajectories/llama_ecological.jsonl"),
    ("qwen",   "data/annotated/qwen_ecological.jsonl",
               "data/trajectories/qwen_ecological.jsonl"),
    ("sonnet", "data/annotated/sonnet_ecological.jsonl",
               "data/trajectories/sonnet_ecological.jsonl"),
    ("gemini", "data/annotated/gemini_ecological.jsonl",
               "data/trajectories/gemini_ecological.jsonl"),
]


def load_eligible(model: str, ann_path: str, traj_path: str, qa_qids: set) -> list[dict]:
    trajs = {}
    for line in open(traj_path):
        it = json.loads(line)
        trajs[it["qid"]] = it

    eligible = []
    for line in open(ann_path):
        it = json.loads(line)
        if it["qid"] not in qa_qids:
            continue
        hops = it.get("hops", [])
        if len(hops) < 2:
            continue
        if any(h.get("abstained") for h in hops):
            continue
        if any(not h.get("claims") for h in hops):
            continue
        has_hallu = any(h.get("has_hallucination") for h in hops)
        has_cross = any(h.get("gold_lang") == "en" for h in hops)
        eligible.append({
            "model":     model,
            "item":      it,
            "traj":      trajs.get(it["qid"], {}),
            "has_hallu": has_hallu,
            "has_cross": has_cross,
        })
    return eligible


def traj_hop_map(traj: dict) -> dict:
    return {h.get("hop"): h for h in traj.get("trajectory", [])}


def load_sub_q_map(qa_records: list[dict]) -> dict:
    """(qid, hop) -> qa_pairs 정본 sub_q (gold bridge entity로 고정된 질문).

    dynamic_subq 프로토콜의 hop1 실제 검색어(trajectory retrieval_query)는 hop0의
    '실제' 답(오답이어도)을 템플릿에 끼워 넣어 만들어지므로, judge가 채점에 쓰는
    정본 sub_q(pipeline/06_annotate.py의 sub_q_map, qa_pairs 기준)와 달라질 수 있다 —
    hop1의 73%(calib 89%/main 71%)에서 실측 상이. annotator에게는 정본 sub_q를
    주 질문으로 보여주고, 실제 검색어는 다를 때만 참고용으로 별도 표시한다.
    """
    sub_q_map = {}
    for qa in qa_records:
        for h in qa.get("hops", []):
            sq = h.get("sub_q", "")
            if sq:
                sub_q_map[(qa["qid"], h["hop"])] = sq
    return sub_q_map


def sample_main(eligible: list[dict], n: int, hallu_ratio: float, rng: random.Random) -> list[dict]:
    by_pat = defaultdict(lambda: {"hallu": [], "clean": []})
    for e in eligible:
        p = e["item"]["pattern"]
        key = "hallu" if e["has_hallu"] else "clean"
        by_pat[p][key].append(e)

    patterns = list("abcd")
    base, rem = divmod(n, len(patterns))
    pat_n = {p: base + (1 if i < rem else 0) for i, p in enumerate(patterns)}

    selected = []
    for p in patterns:
        total_p   = pat_n[p]
        hallu_n   = round(total_p * hallu_ratio)
        clean_n   = total_p - hallu_n

        pool_h = by_pat[p]["hallu"].copy()
        pool_c = by_pat[p]["clean"].copy()
        rng.shuffle(pool_h)
        rng.shuffle(pool_c)

        if len(pool_h) < hallu_n:
            hallu_n = len(pool_h)
            clean_n = total_p - hallu_n

        selected.extend(pool_h[:hallu_n])
        selected.extend(pool_c[:clean_n])

    rng.shuffle(selected)
    return selected


def is_self_contradictory(label: str, reason: str) -> bool:
    """judge가 남긴 reason의 결론과 기록된 label의 극성이 어긋나는 경우 True.

    pipeline/06_annotate.py judge 출력에서 reason이 chain-of-thought 도중의 자기수정
    ("Wait... so it is supported")을 그대로 담고 최종 label과 불일치하는 사례가
    실측 0.3%(aya+exaone, reason 존재 5189건 중 14건) 발견됨 — calibration
    "연구자 참고 판정"으로 이런 항목이 뽑히면 annotator에게 claim과 무관해
    보이는 판정이 노출된다. calibration 후보 pool에서만 제외한다(본 판정은
    label을 아예 보여주지 않으므로 영향 없음).
    """
    if not reason:
        return False
    low = reason.lower()
    if label == "contradicted" and ("not contradicted" in low or "is supported" in low or "so this answer is supported" in low):
        return True
    if label == "supported" and ("not supported" in low or "is unsupported" in low or "contradicts" in low or "is contradicted" in low):
        return True
    if label == "unsupported" and ("is supported" in low or "is contradicted" in low):
        return True
    return False


def sample_calibration(eligible: list[dict], main_keys: set, spec: dict,
                        cross_min: int, rng: random.Random, sub_q_map: dict) -> list[dict]:
    pool_by_label = defaultdict(list)

    for e in eligible:
        if (e["model"], e["item"]["qid"]) in main_keys:
            continue
        traj_hops = traj_hop_map(e["traj"])
        for h in e["item"].get("hops", []):
            t_hop = traj_hops.get(h["hop"], {})
            static_sq  = sub_q_map.get((e["item"]["qid"], h["hop"]), "")
            dynamic_sq = t_hop.get("retrieval_query", "")
            for ci, c in enumerate(h.get("claims", [])):
                label = c.get("label", "?")
                reason = c.get("reason", "")
                if is_self_contradictory(label, reason):
                    continue
                pool_by_label[label].append({
                    "model":          e["model"],
                    "qid":            e["item"]["qid"],
                    "pattern":        e["item"]["pattern"],
                    "hop":            h["hop"],
                    "hop_idx":        h["hop"],
                    "claim_idx":      ci,
                    "claim_text":     c.get("text", ""),
                    "label":          label,
                    "gold_lang":      h.get("gold_lang"),
                    "sub_q":          static_sq or dynamic_sq,
                    "sub_q_dynamic":  dynamic_sq if (static_sq and dynamic_sq and static_sq != dynamic_sq) else "",
                    "generated_text": t_hop.get("generated_text", ""),
                    "passages":       t_hop.get("retrieved_passages", []),
                    "reason":         reason,
                    "is_cross":       h.get("gold_lang") == "en",
                })

    label_map = {
        "supported_clear":    "supported",
        "unsupported_clear":  "unsupported",
        "contradicted_clear": "contradicted",
    }

    selected_claims = []
    cross_count = 0

    for spec_key, need_n in spec.items():
        if spec_key == "borderline":
            continue
        lbl = label_map[spec_key]
        pool = pool_by_label.get(lbl, []).copy()
        rng.shuffle(pool)

        picked = 0
        for c in pool:
            if picked >= need_n:
                break
            selected_claims.append(c)
            if c["is_cross"]:
                cross_count += 1
            picked += 1

    border_pool = [c for c in pool_by_label.get("contradicted", [])
                   if len(c["reason"].split()) > 15
                   and not any(sc["model"] == c["model"] and sc["qid"] == c["qid"] and sc["hop_idx"] == c["hop_idx"]
                               and sc["claim_idx"] == c["claim_idx"] for sc in selected_claims)]
    rng.shuffle(border_pool)
    selected_claims.extend(border_pool[:spec.get("borderline", 0)])

    if cross_count < cross_min:
        extra_cross = [c for c in pool_by_label.get("supported", []) + pool_by_label.get("unsupported", [])
                       if c["is_cross"] and c not in selected_claims]
        rng.shuffle(extra_cross)
        need = cross_min - cross_count
        selected_claims.extend(extra_cross[:need])

    rng.shuffle(selected_claims)
    return selected_claims


def make_sheet_rows(sample: list[dict], sub_q_map: dict) -> list[dict]:
    rows = []
    for e in sample:
        it   = e["item"]
        traj_hops = traj_hop_map(e["traj"])
        for h in it.get("hops", []):
            hop_idx = h["hop"]
            t_hop   = traj_hops.get(hop_idx, {})
            passages = t_hop.get("retrieved_passages", [])
            psg_text = "\n---\n".join(passages[:5]) if passages else ""
            gen_text = t_hop.get("generated_text", "")
            static_sq  = sub_q_map.get((it["qid"], hop_idx), "")
            dynamic_sq = t_hop.get("retrieval_query", "")
            for ci, c in enumerate(h.get("claims", [])):
                rows.append({
                    "model":          e["model"],
                    "qid":            it["qid"],
                    "pattern":        it["pattern"],
                    "hop":            hop_idx,
                    "gold_lang":      h.get("gold_lang", ""),
                    "sub_q":          static_sq or dynamic_sq,
                    "sub_q_dynamic":  dynamic_sq if (static_sq and dynamic_sq and static_sq != dynamic_sq) else "",
                    "generated_text": gen_text,
                    "claim_idx":      ci + 1,
                    "claim":          c.get("text", ""),
                    "passages":       psg_text,
                    "label_A":        "",
                    "reason_A":       "",
                    "label_B":        "",
                    "reason_B":       "",
                    "label_C":        "",
                    "reason_C":       "",
                    "judge_label":    c.get("label", ""),
                })
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-pairs",   default="data/dataset/qa_pairs.jsonl",
                         help="분석 대상 QA JSONL (이 파일의 qid 로 샘플링 모집단 제한)")
    parser.add_argument("--output-dir", default="data/human_audit/iaa/")
    parser.add_argument("--seed",       type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_records = [json.loads(l) for l in Path(args.qa_pairs).read_text().splitlines() if l.strip()]
    qa_qids = {r["qid"] for r in qa_records}
    sub_q_map = load_sub_q_map(qa_records)
    print(f"분석 대상 qid: {len(qa_qids)}개 ({args.qa_pairs})")

    eligible = []
    for model, ann_path, traj_path in MODELS:
        pool = load_eligible(model, ann_path, traj_path, qa_qids)
        print(f"  {model}: eligible {len(pool)}개")
        eligible.extend(pool)
    print(f"eligible pool 총합: {len(eligible)}개")

    # ── 본 샘플 ────────────────────────────────────────────────────────────
    main_sample = sample_main(eligible, MAIN_N, MAIN_HALLU_R, rng)
    main_keys   = {(e["model"], e["item"]["qid"]) for e in main_sample}

    hallu_n = sum(1 for e in main_sample if e["has_hallu"])
    cross_n = sum(1 for e in main_sample if e["has_cross"])
    pat_dist   = Counter(e["item"]["pattern"] for e in main_sample)
    model_dist = Counter(e["model"] for e in main_sample)
    print(f"\n본 샘플: {len(main_sample)}개")
    print(f"  패턴 분포: {dict(pat_dist)}")
    print(f"  모델 분포: {dict(model_dist)}")
    print(f"  hallu 포함: {hallu_n} ({hallu_n/len(main_sample):.1%})")
    print(f"  cross-index 포함: {cross_n} ({cross_n/len(main_sample):.1%})")

    # ── Calibration ────────────────────────────────────────────────────────
    calib_claims = sample_calibration(eligible, main_keys, CALIB_SPEC, CALIB_CROSS_MIN, rng, sub_q_map)
    cross_calib  = sum(1 for c in calib_claims if c["is_cross"])
    lbl_dist     = Counter(c["label"] for c in calib_claims)
    calib_model_dist = Counter(c["model"] for c in calib_claims)
    print(f"\nCalibration: {len(calib_claims)}개 claims")
    print(f"  label 분포: {dict(lbl_dist)}")
    print(f"  모델 분포: {dict(calib_model_dist)}")
    print(f"  cross-lingual: {cross_calib}")

    # ── 저장 ───────────────────────────────────────────────────────────────
    with open(out_dir / "iaa_main_sample.jsonl", "w", encoding="utf-8") as f:
        for e in main_sample:
            row = {k: v for k, v in e["item"].items()}
            row["model"] = e["model"]
            row["traj_summary"] = {
                "qid": e["traj"].get("qid"),
                "model": e["traj"].get("model"),
                "final_answer": e["traj"].get("final_answer"),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(out_dir / "iaa_calibration.jsonl", "w", encoding="utf-8") as f:
        for c in calib_claims:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    main_rows  = make_sheet_rows(main_sample, sub_q_map)
    calib_rows = [
        {
            "model": c["model"], "qid": c["qid"], "pattern": c["pattern"], "hop": c["hop"],
            "gold_lang": c["gold_lang"], "sub_q": c["sub_q"], "sub_q_dynamic": c.get("sub_q_dynamic", ""),
            "generated_text": c["generated_text"],
            "claim_idx": c["claim_idx"] + 1,
            "claim": c["claim_text"],
            "passages": "\n---\n".join(c["passages"][:5]) if c["passages"] else "",
            "label_A": "", "reason_A": "",
            "label_B": "", "reason_B": "",
            "label_C": "", "reason_C": "",
            "judge_label": c["label"],
            "calib_type": "",
        }
        for c in calib_claims
    ]

    write_tsv(main_rows,  out_dir / "iaa_main_sheet.tsv")
    write_tsv(calib_rows, out_dir / "iaa_calibration_sheet.tsv")

    print(f"\n저장:")
    print(f"  {out_dir}/iaa_main_sample.jsonl ({len(main_sample)} items)")
    print(f"  {out_dir}/iaa_calibration.jsonl ({len(calib_claims)} claims)")
    print(f"  {out_dir}/iaa_main_sheet.tsv ({len(main_rows)} rows)")
    print(f"  {out_dir}/iaa_calibration_sheet.tsv ({len(calib_rows)} rows)")


if __name__ == "__main__":
    main()
