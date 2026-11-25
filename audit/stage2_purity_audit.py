#!/usr/bin/env python3
"""
audit/stage2_purity_audit.py
=====================================
Stage-2 (surface-normalized ablation = dynamic_ablation_normalized) purity audit.

목적: QMC stage-2 정규화가 "표면만 고치고 정체성은 절대 바꾸지 않는다"는
매뉴스크립트 §3.2/§7 주장을 norm trajectory 데이터로 검증하고,
§7 [PENDING] rate를 정량화한다. (정량화 전용 — 본문 문구 수정은 별도)

정규화 구현 (pipeline/05_pilot_runner.py:_normalize_bridge):
  hop0 raw 답이 gold bridge 의 KO/EN canonical 라벨과 정확 일치할 때만
  target-lang 라벨로 치환 → success_gold_bridge_alias
  아니면 raw 유지 → failed_unlinked / prev_answer 없음 → skipped_abstain

검증 2축:
  A. Identity purity — success hop 전수검사:
       prev_answer_raw ∈ {gold KO, gold EN} 이고 bridge_insert_text ∈ {gold KO, gold EN}
       (= 같은 gold 엔티티의 두 표면 사이 교체)인가? 위반=정체성 오염.
       oracle_gold_bridge_forced count == 0 인가?
  B. Rate 정량화 — 모델×same/cross 별 status 분포,
       linking-failure(failed_unlinked) / abstention(skipped_abstain, prev_answer_abstained) rate,
       success 중 실효 정규화(문자열 실제 변경) vs no-op.

입력:
  --qa-pairs   data/dataset/qa_pairs.jsonl  (분석 대상 1000 경로. hop1.sub_q_bridge_label / sub_q_en_bridge_label)
  --traj       data/trajectories/trajectories_{model}_norm*.jsonl  (반복 지정, full flag 보유)

실행:
  python3 audit/stage2_purity_audit.py \\
    --qa-pairs data/dataset/qa_pairs.jsonl \\
    --traj data/trajectories/aya_normalized.jsonl \\
    --traj data/trajectories/exaone_normalized.jsonl \\
    --traj data/trajectories/llama_normalized.jsonl \\
    --traj data/trajectories/qwen_normalized.jsonl \\
    --traj data/trajectories/sonnet_normalized.jsonl \\
    --traj data/trajectories/gemini_normalized.jsonl \\
    --out results/stage2_purity_audit.txt
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.qa_set import restrict  # noqa: E402  (위 경로 삽입 뒤에 import)


def load_jsonl(path: str):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def model_name(path: str) -> str:
    s = Path(path).stem
    s = s.replace("trajectories_", "")
    s = re.sub(r"_norm.*$", "", s)
    return s


def hop1_of(row: dict) -> dict | None:
    traj = row.get("trajectory") or row.get("hops") or []
    for h in traj:
        if h.get("hop") == 1:
            return h
    return None


def pct(n, d) -> str:
    return f"{(100.0*n/d):5.1f}%" if d else "  N/A "


def audit_model(rows, qa_labels):
    """rows: normalized trajectory rows (분석 대상 QA로 제한됨). qa_labels: qid -> (ko, en)."""
    stat = Counter()                       # normalization_status 분포
    stat_by_group = defaultdict(Counter)   # same/cross -> status
    success_same_cross = Counter()         # success 발동의 same/cross
    n_hop1 = 0
    n_prev_abstained = 0
    n_oracle_forced = 0
    succ_effective = 0   # success & bridge 문자열이 실제로 바뀜
    succ_noop = 0        # success & no-op (이미 target 표면)
    eff_by_group = Counter()   # 실효 정규화의 same/cross 분포
    violations = []      # identity 오염 (success인데 gold 라벨 아님)

    for r in rows:
        h = hop1_of(r)
        if h is None:
            continue
        n_hop1 += 1
        status = h.get("normalization_status", "missing")
        stat[status] += 1
        # QMC 분해의 same/cross는 조건 불변이어야 한다. ablation/norm 조건에서는
        # design_query_index_match가 template 매칭으로 항상 True이므로 사용 불가.
        # 홉 본연의 index 언어(gold_lang)로 분류: same=ko-index, cross=en-index.
        group = "cross" if h.get("gold_lang") == "en" else "same"
        stat_by_group[group][status] += 1
        if h.get("prev_answer_abstained"):
            n_prev_abstained += 1
        if status == "oracle_gold_bridge_forced":
            n_oracle_forced += 1

        if status == "success_gold_bridge_alias":
            success_same_cross[group] += 1
            ko, en = qa_labels.get(r["qid"], ("", ""))
            prev = (h.get("prev_answer_raw") or "").strip()
            bridge = h.get("bridge_insert_text")
            gold_set_lc = {x.lower() for x in (ko, en) if x}
            gold_set = {x for x in (ko, en) if x}
            input_is_gold = prev.lower() in gold_set_lc
            output_is_gold = bridge in gold_set
            if not (input_is_gold and output_is_gold):
                violations.append({
                    "qid": r["qid"], "prev": prev, "bridge": bridge,
                    "gold_ko": ko, "gold_en": en,
                    "input_is_gold": input_is_gold, "output_is_gold": output_is_gold,
                })
            if bridge is not None and bridge != prev:
                succ_effective += 1
                eff_by_group[group] += 1
            else:
                succ_noop += 1

    return {
        "n_hop1": n_hop1, "stat": stat, "stat_by_group": stat_by_group,
        "n_prev_abstained": n_prev_abstained, "n_oracle_forced": n_oracle_forced,
        "succ_effective": succ_effective, "succ_noop": succ_noop,
        "eff_by_group": eff_by_group,
        "violations": violations, "success_same_cross": success_same_cross,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--traj", action="append", required=True,
                    help="norm trajectory 파일 (반복 지정)")
    ap.add_argument("--out", default=None, help="콘솔 출력 tee 저장")
    ap.add_argument("--max-violations", type=int, default=10,
                    help="위반 케이스 상세 출력 최대 개수")
    args = ap.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    qa = load_jsonl(args.qa_pairs)
    qa_labels = {
        r["qid"]: (r["hops"][1].get("sub_q_bridge_label", "") or "",
                   r["hops"][1].get("sub_q_en_bridge_label", "") or "")
        for r in qa if len(r.get("hops", [])) > 1
    }

    print("=" * 70)
    print("Stage-2 Purity Audit — dynamic_ablation_normalized (QMC stage-2)")
    print("=" * 70)
    print(f"qa_pairs: {len(qa_labels)} qids (hop1 라벨)\n")

    summary = []
    STATUSES = ["success_gold_bridge_alias", "failed_unlinked",
                "skipped_abstain", "oracle_gold_bridge_forced", "not_applicable"]

    for traj_path in args.traj:
        rows = load_jsonl(traj_path)
        n_all = len(rows)
        rows = restrict(rows, args.qa_pairs, quiet=True)
        m = model_name(traj_path)
        a = audit_model(rows, qa_labels)
        n = a["n_hop1"]

        print("-" * 70)
        print(f"[{m}]  rows={n_all} → 분석대상 {len(rows)} → hop1 {n}")
        print("  normalization_status 분포 (hop1):")
        for s in STATUSES:
            c = a["stat"].get(s, 0)
            if c or s in ("success_gold_bridge_alias", "failed_unlinked", "skipped_abstain"):
                print(f"    {s:28} {c:5d}  {pct(c, n)}")
        other = {k: v for k, v in a["stat"].items() if k not in STATUSES}
        for k, v in other.items():
            print(f"    {k:28} {v:5d}  {pct(v, n)}   [예상밖]")

        succ = a["stat"].get("success_gold_bridge_alias", 0)
        print(f"  prev_answer_abstained (hop0 기권):    "
              f"{a['n_prev_abstained']:5d}  {pct(a['n_prev_abstained'], n)}")
        print(f"  success 중 실효 정규화(문자열 변경):  "
              f"{a['succ_effective']:5d}  {pct(a['succ_effective'], succ)} of success")
        print(f"  success 중 no-op(이미 target 표면):   "
              f"{a['succ_noop']:5d}  {pct(a['succ_noop'], succ)} of success")
        print(f"  success 발동 same/cross (gold_lang):   "
              f"same(ko)={a['success_same_cross'].get('same',0)} "
              f"cross(en)={a['success_same_cross'].get('cross',0)}")
        print(f"  실효 정규화 same/cross (gold_lang):    "
              f"same(ko)={a['eff_by_group'].get('same',0)} "
              f"cross(en)={a['eff_by_group'].get('cross',0)}")

        # Axis A 판정
        nv = len(a["violations"])
        print(f"  >> Identity purity: oracle_gold_bridge_forced={a['n_oracle_forced']}  "
              f"| identity 위반(success 비-gold)={nv}")
        if nv == 0 and a["n_oracle_forced"] == 0:
            print(f"  >> 판정: PURE — 정규화가 gold 엔티티의 KO↔EN 표면만 교체, 정체성 변경 없음")
        else:
            print(f"  >> 판정: IMPURE — 아래 위반 케이스 확인 필요")
            for v in a["violations"][:args.max_violations]:
                print(f"       {v}")

        summary.append((m, n, succ, a["stat"].get("failed_unlinked", 0),
                        a["stat"].get("skipped_abstain", 0), a["n_prev_abstained"],
                        a["succ_effective"], nv, a["n_oracle_forced"]))
        print()

    # ── 집계표 ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("집계 (hop1 기준)")
    print("=" * 70)
    hdr = (f"{'model':12} {'n':>5} {'success':>9} {'failed_unl':>11} "
           f"{'skip_abs':>9} {'prev_abst':>10} {'eff_norm':>9} {'viol':>5} {'oracle':>7}")
    print(hdr)
    print("-" * len(hdr))
    for (m, n, succ, fu, sa, pa, eff, nv, orc) in summary:
        print(f"{m:12} {n:5d} "
              f"{succ:5d}/{pct(succ,n):>5} {fu:5d}/{pct(fu,n):>5} "
              f"{sa:4d}/{pct(sa,n):>5} {pa:4d}/{pct(pa,n):>5} "
              f"{eff:5d}{'':4} {nv:5d} {orc:7d}")
    print()
    print("열 설명:")
    print("  success     = success_gold_bridge_alias (정규화 발동)")
    print("  failed_unl  = failed_unlinked = linking-failure rate (§7 PENDING)")
    print("  skip_abs    = skipped_abstain (prev_answer 비어있음)")
    print("  prev_abst   = prev_answer_abstained=True (hop0 기권, abstention-without-useful-bridge)")
    print("  eff_norm    = success 중 bridge 문자열 실제 변경(실효 정규화) 건수")
    print("  viol        = identity 위반(success인데 gold 라벨 아님) — 0이어야 PURE")
    print("  oracle      = oracle_gold_bridge_forced — 0이어야 함")


if __name__ == "__main__":
    main()
