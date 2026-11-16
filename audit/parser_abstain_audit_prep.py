#!/usr/bin/env python3
"""
audit/parser_abstain_audit_prep.py

기권(abstain) parser 품질 수동 감사 — 시트 생성/채점.

기권 분류기를 사람 gold와 대조해 Precision/Recall/F1을 측정한다. 두 방향을 모두 본다:
  - parser=abstain  표본(positive) → FP(과기권) 측정
  - parser=답변(non-abstain) 표본(negative) → FN(놓친 기권, FaithRate 오염원) 측정

각 행에 disjoint `stratum`과 그 층의 모집단 크기 `stratum_pop`를 임베드해, 채점 시
표본 비율을 모집단으로 재가중한다(층화 추정). 6모델을 한 시트로 풀링해도 (config,stratum)
단위로 합산되어 정확하다.

사용:
  # (1) abstain 시트 (FP 측정) — parser가 기권으로 부른 홉
  python audit/parser_abstain_audit_prep.py prep --class abstain \
    --annotated <ann.jsonl> --trajectory <traj.jsonl> --config <name> \
    --qa-pairs data/dataset/qa_pairs.jsonl \
    --sample 0 --out-csv reports/parser_audit/abstain_<name>.csv

  # (2) nonabstain 시트 (FN 측정) — parser가 답변으로 본 홉, 단문 오버샘플
  python audit/parser_abstain_audit_prep.py prep --class nonabstain \
    --annotated <ann.jsonl> --trajectory <traj.jsonl> --config <name> \
    --qa-pairs data/dataset/qa_pairs.jsonl \
    --sample 150 --out-csv reports/parser_audit/nonabstain_<name>.csv

  # (3) 채점 — 채운 CSV 여러 개 한꺼번에
  python audit/parser_abstain_audit_prep.py score \
    --filled reports/parser_audit/*.csv

human_label 값 (RUBRIC.md 참조): abstain | answer | unclear
  - partial(일부 엔티티만, 물은 대상 미확인)은 rubric상 abstain으로 귀속해 라벨링.
  - unclear는 P/R 계산에서 제외(별도 보고).
"""
import argparse
import csv
import glob
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from utils.answer_parser import normalize_entity, _check_abstain
from utils.qa_set import load_qids

CSV_COLS = ["idx", "parser_class", "stratum", "stratum_pop", "config", "qid", "hop",
            "pattern", "gold_lang", "gold_entity", "gold_in_passages",
            "parser_source", "parse_issue", "marker_count", "generated_text",
            "human_label", "notes"]

LONG_CHARS = 40  # 이보다 길면 'long'(abstain)/'normal'(nonabstain)


def load_by_qid(path: str) -> dict:
    return {r["qid"]: r for r in (json.loads(l) for l in open(path, encoding="utf-8") if l.strip())}


def gold_in_passages(gold: str, passages: list[str]) -> bool:
    if not gold:
        return False
    ng = normalize_entity(gold)
    return any(ng and ng in normalize_entity(p) for p in passages if p)


def abstain_stratum(text: str, parse_issue, gold_hit: bool) -> str:
    """parser=abstain 홉의 disjoint 층 (우선순위: FP가 숨기 쉬운 순)."""
    if not _check_abstain(text or ""):
        return "no_abstain_kw"      # 기권 키워드 없이 abstained → FP 최우선
    if gold_hit:
        return "gold_in_passage"    # 근거 있는데 기권 → 과기권 or 우연매칭
    if len((text or "").strip()) > LONG_CHARS:
        return "long"               # 장문 hedge → 부분답변 경계
    return "clean"                  # 명확한 키워드 단독 기권 → FP≈0 기대


def nonabstain_stratum(text: str) -> str:
    """parser=답변 홉의 disjoint 층. 단문에 markerless 기권(FN)이 숨는다."""
    return "short" if len((text or "").strip()) <= LONG_CHARS else "normal"


def collect(args, keep: set | None):
    """모든 대상 홉을 (row, stratum) 으로 수집 (표집 전 전수)."""
    ann = load_by_qid(args.annotated)
    traj = load_by_qid(args.trajectory)
    rows = []
    for qid, a in ann.items():
        if keep is not None and qid not in keep:
            continue
        t = traj.get(qid)
        if not t:
            continue
        steps = {s["hop"]: s for s in t["trajectory"]}
        for h in a["hops"]:
            abst = bool(h.get("abstained"))
            if args.parser_class == "abstain" and not abst:
                continue
            if args.parser_class == "nonabstain" and (abst or h.get("ambiguous_answer_marker")):
                continue
            s = steps.get(h["hop"], {})
            gold = s.get("gold_entity", "") or ""
            text = (s.get("generated_text", "") or "").strip()
            gp = gold_in_passages(gold, s.get("retrieved_passages", []))
            stratum = (abstain_stratum(text, h.get("parse_issue"), gp)
                       if args.parser_class == "abstain" else nonabstain_stratum(text))
            rows.append({
                "parser_class": args.parser_class, "stratum": stratum,
                "config": args.config, "qid": qid, "hop": h["hop"],
                "pattern": a.get("pattern", ""), "gold_lang": h.get("gold_lang", ""),
                "gold_entity": gold, "gold_in_passages": gp,
                "parser_source": h.get("claim_extraction_source", ""),
                "parse_issue": h.get("parse_issue") or "",
                "marker_count": h.get("answer_marker_count", ""),
                "generated_text": text,
            })
    return rows


def prep(args):
    keep = load_qids(args.qa_pairs) if args.qa_pairs else None
    rows = collect(args, keep)
    pop = Counter(r["stratum"] for r in rows)
    for r in rows:
        r["stratum_pop"] = pop[r["stratum"]]

    by_str = defaultdict(list)
    for r in rows:
        by_str[r["stratum"]].append(r)
    rng = random.Random(args.seed)

    # 표집 정책
    if args.parser_class == "abstain":
        # FP가 숨는 층(clean 외) 전수 + clean은 control 표본만
        priority = ["no_abstain_kw", "gold_in_passage", "long"]
        sample = [r for s in priority for r in by_str.get(s, [])]
        clean = by_str.get("clean", [])[:]
        rng.shuffle(clean)
        cap_clean = len(clean) if args.sample <= 0 else max(0, min(len(clean), args.sample - len(sample)))
        sample += clean[:cap_clean]
    else:
        # FN: short 오버샘플(절반↑) + normal 무작위
        short = by_str.get("short", [])[:]; normal = by_str.get("normal", [])[:]
        rng.shuffle(short); rng.shuffle(normal)
        if args.sample <= 0:
            sample = short + normal
        else:
            n_short = min(len(short), max(args.sample // 2, args.sample - len(normal)))
            sample = short[:n_short] + normal[:args.sample - n_short]

    rng.shuffle(sample)
    for i, r in enumerate(sample, 1):
        r["idx"] = i
        r["human_label"] = ""
        r["notes"] = ""

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sample)

    print(f"[{args.config}] class={args.parser_class}  대상 {len(rows)}홉 → 표집 {len(sample)} → {args.out_csv}")
    print("  층(모집단/표집):")
    samp_cnt = Counter(r["stratum"] for r in sample)
    for s in sorted(pop, key=lambda k: -pop[k]):
        print(f"    {s:<16} pop={pop[s]:>5}  sampled={samp_cnt.get(s,0)}")


# ─── 채점 ────────────────────────────────────────────────────────────────────

def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return p, c - hw, c + hw


def _reweighted_cells(labeled: list[dict]):
    """(config,stratum)별 표본 비율 → 모집단 재가중. parser×human population 셀 추정."""
    by_cell = defaultdict(list)
    for r in labeled:
        by_cell[(r["parser_class"], r["config"], r["stratum"], int(float(r["stratum_pop"])))].append(
            r["human_label"].strip())
    TP = FP = FN = TN = 0.0
    for (pc, cfg, st, npop), labs in by_cell.items():
        labs = [l for l in labs if l in ("abstain", "answer")]
        if not labs:
            continue
        n = len(labs)
        f_abstain = sum(1 for l in labs if l == "abstain") / n
        f_answer = 1 - f_abstain
        if pc == "abstain":          # parser=abstain → TP(human abstain) / FP(human answer)
            TP += f_abstain * npop
            FP += f_answer * npop
        else:                        # parser=답변 → FN(human abstain) / TN(human answer)
            FN += f_abstain * npop
            TN += f_answer * npop
    return TP, FP, FN, TN


def _prf(TP, FP, FN):
    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = 2*prec*rec/(prec+rec) if prec and rec and not (math.isnan(prec) or math.isnan(rec)) else float("nan")
    return prec, rec, f1


def score(args):
    files = [p for pat in args.filled for p in glob.glob(pat)]
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f, encoding="utf-8")))
    labeled = [r for r in rows if (r.get("human_label") or "").strip()]
    valid = [r for r in labeled if r["human_label"].strip() in ("abstain", "answer")]
    unclear = sum(1 for r in labeled if r["human_label"].strip() == "unclear")

    print(f"파일 {len(files)}개 | 행 {len(rows)} | 라벨 {len(labeled)} (valid {len(valid)}, unclear {unclear})")
    if not valid:
        print("  라벨된 abstain/answer 행이 없다."); return

    # 원시 confusion (표본 기준)
    print("\n[표본 confusion]")
    for pc in ("abstain", "nonabstain"):
        sub = [r for r in valid if r["parser_class"] == pc]
        c = Counter(r["human_label"].strip() for r in sub)
        print(f"  parser={pc:<10} n={len(sub):<4} human: abstain={c.get('abstain',0)} answer={c.get('answer',0)}")

    # 모집단 재가중 P/R/F1
    TP, FP, FN, TN = _reweighted_cells(valid)
    prec, rec, f1 = _prf(TP, FP, FN)
    print("\n[모집단 재가중 추정]")
    print(f"  N̂: TP={TP:.0f} FP={FP:.0f} FN={FN:.0f} TN={TN:.0f}")
    print(f"  Precision(abstain) = {prec:.3f}   Recall(abstain) = {rec:.3f}   F1 = {f1:.3f}")

    # 층화 부트스트랩 CI (각 (class,stratum) 내 재표집 → 재가중)
    rng = random.Random(args.seed)
    by_cell = defaultdict(list)
    for r in valid:
        by_cell[(r["parser_class"], r["config"], r["stratum"], int(float(r["stratum_pop"])))].append(r)
    precs, recs, f1s = [], [], []
    for _ in range(args.n_boot):
        boot = []
        for cell, rs in by_cell.items():
            boot += [rng.choice(rs) for _ in rs]
        tp, fp, fn, tn = _reweighted_cells(boot)
        p, rc, fr = _prf(tp, fp, fn)
        if not math.isnan(p): precs.append(p)
        if not math.isnan(rc): recs.append(rc)
        if not math.isnan(fr): f1s.append(fr)

    def ci(v):
        if len(v) < args.n_boot * 0.5:
            return "  [CI N/A]"
        s = sorted(v)
        return f"  [{s[int(.025*len(s))]:.3f}, {s[int(.975*len(s))-1]:.3f}]"
    print(f"  Precision 95%CI{ci(precs)}   Recall 95%CI{ci(recs)}   F1 95%CI{ci(f1s)}")

    # 층별 분해
    print("\n[층별 sampled human-비율]")
    by_str = defaultdict(lambda: Counter())
    pops = {}
    for r in valid:
        k = (r["parser_class"], r["stratum"])
        by_str[k][r["human_label"].strip()] += 1
        pops[k] = pops.get(k, 0)
    # pop는 (class,stratum) 합산 (config 합)
    pop_sum = defaultdict(int)
    seen = set()
    for r in valid:
        key = (r["parser_class"], r["config"], r["stratum"])
        if key in seen:
            continue
        seen.add(key)
        pop_sum[(r["parser_class"], r["stratum"])] += int(float(r["stratum_pop"]))
    for k in sorted(by_str):
        c = by_str[k]; n = sum(c.values())
        minor = "answer" if k[0] == "abstain" else "abstain"   # 오류 방향
        err = c.get(minor, 0)
        p, lo, hi = wilson(err, n)
        tag = "FP" if k[0] == "abstain" else "FN"
        print(f"  {k[0]:<10} {k[1]:<16} pop={pop_sum[k]:>5} n={n:>3}  {tag}={err}/{n}={p:.2f} [{lo:.2f},{hi:.2f}]")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--class", dest="parser_class", choices=["abstain", "nonabstain"], required=True)
    p.add_argument("--annotated", required=True)
    p.add_argument("--trajectory", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--sample", type=int, default=0, help="0=전수(abstain non-clean+clean control), nonabstain은 총 표본수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--qa-pairs", default=None)
    p.set_defaults(func=prep)
    s = sub.add_parser("score")
    s.add_argument("--filled", nargs="+", required=True, help="채운 CSV(글롭 가능)")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=score)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
