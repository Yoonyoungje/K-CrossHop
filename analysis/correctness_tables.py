#!/usr/bin/env python3
"""
analysis/correctness_tables.py

correctness_axis.py 출력 CSV → 보고서용 마크다운 표 (stdout). 계산 없음, 표 조립만.

사용: python3 analysis/correctness_tables.py data/correctness/all
"""
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

d = Path(sys.argv[1])
YIELD_CHECK = d.parent / "yield_check"
MODELS = ["aya", "exaone", "llama", "qwen", "sonnet", "gemini"]
NAME = {"aya": "Aya Expanse 8B", "exaone": "EXAONE 3.5 8B", "llama": "Llama 3.1 8B",
        "qwen": "Qwen3 8B", "sonnet": "Claude Sonnet 4.6", "gemini": "Gemini 3.5 Flash"}
CONDS = ["controlled", "template_matched", "surface_normalized"]
# 논문 yield QMC 표(tab:qmc-yield)의 값
PAPER_YIELD = {
    "aya": (0.061, 0.017, -0.005), "exaone": (0.098, 0.020, -0.033), "llama": (0.086, 0.031, -0.002),
    "qwen": (0.062, 0.030, 0.018), "sonnet": (0.075, 0.027, -0.009), "gemini": (0.069, 0.036, 0.012),
}

S = defaultdict(dict)
for r in csv.DictReader((d / "stats_bootstrap.csv").open(encoding="utf-8")):
    S[r["model"]][r["stat"]] = (float(r["point"]), float(r["ci_lo"]), float(r["ci_hi"]), r["identity_max_abs_err"])


def f(model, stat, pct=False):
    p, lo, hi, _ = S[model][stat]
    if pct:
        return f"{p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"
    sig = "*" if (lo > 0 or hi < 0) else ""
    return f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]{sig}"


def rows_csv(name):
    return list(csv.DictReader((d / name).open(encoding="utf-8")))


print("### T1. same/cross hop 정확도 (주 매처)\n")
print("| 모델 | 조건 | acc same | acc cross | acc gap | yield gap (재계산) |")
print("|-|-|-|-|-|-|")
for m in MODELS:
    for c in CONDS:
        ps, _, _, _ = S[m][f"{c}|acc_same"]
        pc, _, _, _ = S[m][f"{c}|acc_cross"]
        print(f"| {NAME[m]} | {c} | {ps:.3f} | {pc:.3f} | {f(m, f'{c}|acc_gap')} | {f(m, f'{c}|yield_gap')} |")

print("\n### T2. QMC 분해 — 정확도 축 vs yield 축 (joint bootstrap, * = CI가 0 제외)\n")
print("| 모델 | 축 | gap(ctrl) | template-language | bridge-surface | residual | 합−gap 최대오차(전 resample) |")
print("|-|-|-|-|-|-|-|")
for m in MODELS:
    for ax, lab in (("acc", "정확도(주 매처)"), ("accalt", "정확도(hit@5 매처)"), ("yield", "yield(재계산)")):
        ident = S[m]["qmc|acc_gap_ctrl"][3] if ax != "accalt" else "—"
        print(f"| {NAME[m]} | {lab} | {f(m, f'qmc|{ax}_gap_ctrl')} | {f(m, f'qmc|{ax}_template')} | "
              f"{f(m, f'qmc|{ax}_bridge')} | {f(m, f'qmc|{ax}_residual')} | {ident} |")
    t, b, r = PAPER_YIELD[m]
    print(f"| {NAME[m]} | yield(논문 표) | {t+b+r:+.3f} (합) | {t:+.3f} | {b:+.3f} | {r:+.3f} | — |")

print("\n### T3. yield 재계산 vs 기존 analysis/yield_qmc_joint_ci.py 출력\n")
print("| 모델 | 지표 | 이 스크립트 | 기존 스크립트 | 일치 |")
print("|-|-|-|-|-|")
KEYMAP = {"YL_ctrl": "yield_gap_ctrl", "stage1": "yield_template", "stage2": "yield_bridge", "residual": "yield_residual"}
for m in MODELS:
    p = YIELD_CHECK / f"yield_qmc_joint_ci_{m}.txt"
    if not p.exists():
        continue
    for line in p.read_text(encoding="utf-8").splitlines():
        mm = re.match(r"\s+(YL_ctrl|stage1|stage2|residual)\s+([+-][\d.]+)\s+\[([+-][\d.]+), ([+-][\d.]+)\]", line)
        if not mm:
            continue
        k, pt, lo, hi = mm.group(1), *map(float, mm.groups()[1:])
        mp, mlo, mhi, _ = S[m][f"qmc|{KEYMAP[k]}"]
        mine = f"{mp:+.3f} [{mlo:+.3f}, {mhi:+.3f}]"
        old = f"{pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
        print(f"| {NAME[m]} | {k} | {mine} | {old} | {'예' if mine == old else '아니오'} |")

print("\n### T4. 패턴별 궤적 성공률 (hop0 correct ∧ hop1 correct)\n")
print("| 모델 | 조건 | 전체 | a KO→KO | b EN→KO | c KO→EN | d EN→EN |")
print("|-|-|-|-|-|-|-|")
for m in MODELS:
    for c in CONDS:
        cells = " | ".join(f(m, f"{c}|traj_{p}", pct=True) for p in ("all", "a", "b", "c", "d"))
        print(f"| {NAME[m]} | {c} | {cells} |")

if "controlled|final_all" in S[MODELS[0]]:  # 최종 답 정확도가 없는 이전 출력은 건너뛴다
    print("\n### T4b. 최종 답 정확도 (hop1 정답률, hop0 정오와 무관)\n")
    print("| 모델 | 조건 | 전체 | a KO→KO | b EN→KO | c KO→EN | d EN→EN |")
    print("|-|-|-|-|-|-|-|")
    for m in MODELS:
        for c in CONDS:
            cells = " | ".join(f(m, f"{c}|final_{p}", pct=True) for p in ("all", "a", "b", "c", "d"))
            print(f"| {NAME[m]} | {c} | {cells} |")

print("\n### T5. faithful × correct (ambiguous 제외, 기권은 별도 행)\n")
ct = defaultdict(lambda: defaultdict(int))
for r in rows_csv("faithful_x_correct.csv"):
    key = (r["model"], r["cond"])
    if r["row"] == "abstained":
        ct[key]["abst"] += int(r["n"])
        if r["abstained_gold_in_text_raw"] == "1":
            ct[key]["abst_gold"] += int(r["n"])
    else:
        ct[key][f"{r['row']}_{r['correct']}"] += int(r["n"])
print("| 모델 | 조건 | faithful∧correct | faithful∧incorrect | ¬faithful∧correct | ¬faithful∧incorrect | 기권 (그중 응답에 gold 문자열) | P(correct\\|faithful) | P(faithful\\|correct) |")
print("|-|-|-|-|-|-|-|-|-|")
for m in MODELS:
    for c in CONDS:
        x = ct[(m, c)]
        fc, fi, nc, ni = x["faithful_1"], x["faithful_0"], x["not_faithful_1"], x["not_faithful_0"]
        print(f"| {NAME[m]} | {c} | {fc} | {fi} | {nc} | {ni} | {x['abst']} ({x['abst_gold']}) | "
              f"{fc/(fc+fi):.3f} | {fc/(fc+nc):.3f} |")

print("\n### T6. 탈선했지만 faithful / hop0 오답·hop1 정답 / 매처 비교\n")
print("| 모델 | 조건 | ambiguous hop | 궤적 분모 | hop1 faithful | 탈선 faithful (hop0 오답/기권) | 비율 [95% CI] | 그중 hop1 correct | hop0✗·hop1✓ (hop0 답함/기권) | 주 매처 correct | hit@5 매처 correct | 불일치 (주만/hit@5만) |")
print("|-|-|-|-|-|-|-|-|-|-|-|-|")
for r in rows_csv("counts.csv"):
    g = lambda k: int(r.get(k) or 0)  # noqa: E731
    m, c = r["model"], r["cond"]
    print(f"| {NAME[m]} | {c} | {g('n_ambiguous')} | {g('traj_n')} | {g('hop1_faithful')} | "
          f"{g('derailed_faithful')} ({g('derailed_faithful_hop0_wrong_answer')}/{g('derailed_faithful_hop0_abstained')}) | "
          f"{f(m, f'{c}|derailed_faithful_share', pct=True)} | {g('derailed_faithful_and_hop1_correct')} | "
          f"{g('hop0_wrong_hop1_right')} ({g('hop0_wrong_hop1_right_hop0_answered')}/{g('hop0_wrong_hop1_right_hop0_abstained')}) | "
          f"{g('n_correct')} | {g('n_correct_hit5matcher')} | {g('matcher_disagree')} ({g('pdr_only')}/{g('hit5_only')}) |")

print("\n### T7. 매칭 부풀림 위험 플래그 (주 매처로 correct=1 인 hop 중)\n")
fl = rows_csv("match_risk_flags.csv")
keys = [k for k in fl[0] if k.startswith("flag_")] + ["any_flag"]
print("| 모델 | 조건 | correct hop | " + " | ".join(keys) + " |")
print("|-|-|-|" + "-|" * len(keys))
tot = defaultdict(int)
for r in fl:
    print(f"| {NAME[r['model']]} | {r['cond']} | {r['n_correct']} | " + " | ".join(r[k] for k in keys) + " |")
    tot["n"] += int(r["n_correct"])
    for k in keys:
        tot[k] += int(r[k])
print(f"| **합계** | | {tot['n']} | " + " | ".join(str(tot[k]) for k in keys) + " |")
