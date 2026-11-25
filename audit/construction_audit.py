#!/usr/bin/env python3
"""
audit/construction_audit.py

데이터셋 구축 감사: 원고 구축 절·부록의 수치와 중간 단계 ID 목록을 낸다. 읽기 전용, API 호출 없음.

  python3 audit/construction_audit.py --out-dir results/camera_ready/construction

입력 (공개 데이터에 그대로 싣는다):
  게이트 판정    693 후보 × 네 evidence 언어 패턴 (dataset/apply_caps.GATE_PATH)
  410 ID 목록    네 패턴 통과 450 에서 단일 정답 정리 후 (dataset/apply_caps.FULL_IDS_PATH)
  cap 기록       최종 250 · cap 탈락 · 수동 제거 14 · 대체 1
  62 표본 원문   수동 검수 표본(위험 32 + 무작위 30)
  answer_type   cap 그룹 규칙(location 정답)용 (dataset/apply_caps.ANSWER_TYPE_PATH)

출력:
  funnel.txt                 게이트 패턴별 통과율(원고 §3.1), 수집 회차별 단계 수
  sample62.txt               62 표본 분해와 무작위 표본 집계("24 중 7")
  manual_audit_sample62.csv  62 표본 항목별 표
  ids_410.txt, ids_263.txt   중간 단계 ID 목록. 263 은 cap 기록에서 복원한다(최종 250 − 대체 1 + 수동 제거 14)
  cap_check.txt              기록된 263·250 이 cap 규칙을 지키는지, 순서 무관 cap 과의 비교

triple ID 의 접두사가 수집 회차다. 회차 구분은 아래 ROUND 의 접두사로만 한다.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
import apply_caps as capmod  # noqa: E402  cap 규칙·그룹 정의와 입력 경로를 함께 쓴다

CAP_RECORD = ROOT / "data/construction/cap_record.json"
SAMPLE62 = ROOT / "data/construction/manual_audit_62items.txt"

C4 = ["a_ko_ko", "b_en_ko", "c_ko_en", "d_en_en"]
ROUND = {"p1_": "1차", "p2_": "2차", "p3_": "3차", "p4_": "4차"}
LAST_ROUND = "p4_"   # 4차 수집: 1차 cap·다중정답 점검·62 표본보다 늦게 합류했다
REUSED = "REUSED_FROM_EARLIER_RUN"   # 1차 수집 경로의 KO/KO 판정을 이전 실행에서 재사용한 행
TAG = re.compile(r"\[(RISK|RAND)\]\[(\d+)\] (\S+)")

# 수동 제거 13건(62 표본 판정)의 분류. 검수자(저자)가 2026-06-17 에 남긴 기록이다.
MANUAL13 = [
    ("11", "RISK", "p1_Q1124867_Q50764_Q209459", "granularity", "award Career/honorary 계열"),
    ("32", "RISK", "p1_Q18209661_Q152864_Q1867", "granularity", "city/district"),
    ("37", "RAND", "p1_Q14389650_Q20718_Q20714", "granularity", "city/district"),
    ("52", "RAND", "p1_Q11663968_Q56143_Q41283", "granularity", "city/district"),
    ("56", "RAND", "p1_Q1139458_Q740758_Q1490", "granularity", "birthplace"),
    ("58", "RAND", "p3_Q483070_Q1061696_Q1140581", "granularity", "broad/specific league"),
    ("57", "RAND", "p1_Q1048495_Q612_Q613", "ambiguity", "city/emirate"),
    ("62", "RAND", "p1_Q105557_Q191642_Q175", "ambiguity", "state/city"),
    ("28", "RISK", "p2_Q1057784_Q686915_Q241257", "B-A중복+subtype", "award subtype 표현 불안정"),
    ("23", "RISK", "p1_Q111918059_Q313039_Q49114", "B-A중복", "[07]과 반복"),
    ("18", "RISK", "p1_Q25338_Q2908_Q572316", "B-A중복", "[06]과 반복"),
    ("59", "RAND", "p1_Q14622847_Q2617855_Q38283", "B-A중복", "[35]와 반복"),
    ("41", "RAND", "p1_Q16970498_Q3566785_Q1319893", "라벨불안정", "answer_type other"),
]
# 항목별 표의 사유 코드와 영문 설명
SAMPLE62_REASON = {"granularity": "granularity", "ambiguity": "ambiguity", "라벨불안정": "label_instability",
                   "B-A중복": "duplicate_bridge_answer", "B-A중복+subtype": "duplicate_bridge_answer"}
SAMPLE62_NOTE = {
    "11": "Golden Lion vs career/honorary Golden Lion",
    "32": "city vs district (gate predicted Beitou District)",
    "37": "city vs district (gate predicted Yeongtong District)",
    "52": "city vs district (gate predicted Dong-gu)",
    "56": "city vs ward (gate predicted Nerima)",
    "58": "J2 League vs J.League (Korean gate answer broader)",
    "57": "Dubai city vs Emirate of Dubai",
    "62": "Sao Paulo city vs state",
    "28": "same bridge-answer pair as sample 05; award subtype wording also unstable",
    "23": "same bridge-answer pair as sample 07",
    "18": "same bridge-answer pair as sample 06",
    "59": "same bridge-answer pair as sample 35",
    "41": "answer type labelled other (Valencia, California)",
}


def load():
    by, meta = defaultdict(dict), {}
    for line in capmod.GATE_PATH.open(encoding="utf-8"):
        r = json.loads(line)
        by[r["triple_id"]][r["condition"]] = r
        meta[r["triple_id"]] = r
    p450 = {t for t, d in by.items() if len(d) == 4 and all(x["oracle_verdict"] == "PASS" for x in d.values())}
    f410 = set(json.loads(capmod.FULL_IDS_PATH.read_text(encoding="utf-8")))
    cap = json.loads(CAP_RECORD.read_text(encoding="utf-8"))
    return by, meta, p450, f410, cap


def parts(meta, p450, f410):
    """450 → 410 에서 빠진 40 = subject 중복 6 + hop1 다중정답 34."""
    miss = p450 - f410
    subj = Counter(meta[t]["subj_qid"] for t in p450)
    six = {t for t in miss if subj[meta[t]["subj_qid"]] > 1}
    return miss, six, miss - six


def stage_sets(cap):
    kept = set(cap["kept_triple_ids"])
    rep = {r["triple_id"] for r in cap["replacement_added"]}
    man = set(cap["manual_audit_removed"])
    return kept, rep, man, (kept - rep) | man   # 마지막이 cap 출력 263


def section_gate_patterns(by):
    # 1차 수집 경로의 KO/KO 판정은 이전 실행(좁은 evidence 창) 결과를 재사용했으므로,
    # 패턴별 통과율은 네 패턴을 같은 실행에서 판정한 경로만으로 낸다.
    print("## 게이트 패턴별 통과 (원고 §3.1)")
    reused = {t for t, d in by.items() if d["a_ko_ko"]["oracle_model"] == REUSED}
    assert all(sorted(d) == C4 for d in by.values())
    ok = lambda t, c: by[t][c]["oracle_verdict"] == "PASS"
    for name, S in (("게이트 입력 전체", set(by)), ("네 패턴을 같은 실행에서 판정", set(by) - reused),
                    ("KO/KO 판정 재사용(1차 수집)", reused)):
        n = len(S)
        print(f"  {name} n={n}: " + " | ".join(f"{c} {sum(ok(t, c) for t in S)} ({100 * sum(ok(t, c) for t in S) / n:.1f}%)"
                                                for c in C4)
              + f" | 네 패턴 모두 통과 {sum(all(ok(t, c) for c in C4) for t in S)}")
    one = Counter(c for t in set(by) - reused for c in C4
                  if not ok(t, c) and all(ok(t, x) for x in C4 if x != c))
    print("  한 패턴만 FAIL (같은 실행 판정 경로): " + " | ".join(f"{c} {one[c]}" for c in C4))


def section_round_funnel(by, meta, p450, f410, cap):
    print("\n## 수집 회차별 단계 수")
    miss, six, t34 = parts(meta, p450, f410)
    kept, rep, man, after_cap = stage_sets(cap)
    stages = [("게이트 입력", set(by)), ("네 패턴 통과", p450), ("단일 정답 정리 후", f410),
              ("2차 cap 후", after_cap), ("최종", kept)]
    print("  " + " | ".join(f"{s} {len(S)}" for s, S in stages))
    for ph, lab in ROUND.items():
        print(f"  {lab} 수집: " + " | ".join(f"{s} {sum(t.startswith(ph) for t in S)}" for s, S in stages))
    p4 = {t for t in f410 if t.startswith(LAST_ROUND)}
    early = {t for t in p450 if not t.startswith(LAST_ROUND)}
    print(f"  단일 정답 정리: 1~3차 수집 {len(early)} 에서 subject 중복 {len(six)} + hop1 다중정답 {len(t34)} 제거, "
          f"4차 수집 {len(p4)} 은 이 점검 없이 합류 → {len(early)} − {len(miss)} + {len(p4)} = {len(early - miss) + len(p4)}")
    print(f"  4차 수집 줄어든 수 {len(p4 - kept)}: 2차 cap 탈락 {len(p4 - after_cap)}, 수동 제거 {len(p4 & man)}")
    print(f"  2차 cap 뒤 수동 제거 {len(man)} (62 표본 판정 {len(man) - len(rep)} + 다른 bridge 로 같은 subject·정답을 "
          f"반복한 경로 {len(rep)}), 그 자리에 cap 탈락 경로 {len(rep)} 보충")


def section_sample62(meta, p450, f410, cap, csv_out):
    print("\n## §14 62 표본 분해")
    miss, six, t34 = parts(meta, p450, f410)
    kept250, man = set(cap["kept_triple_ids"]), set(cap["manual_audit_removed"])
    items = TAG.findall(SAMPLE62.read_text(encoding="utf-8"))
    s62 = {t for _, _, t in items}
    print(f"  태그: {dict(Counter(g for g, _, _ in items))} | 62⊂429={all(not t.startswith(LAST_ROUND) for t in s62)}")
    print(f"  62∩40={len(s62 & miss)} (그중 six {len(s62 & six)} / 34 {len(s62 & t34)}) | "
          f"62∩410={len(s62 & f410)} | 62∩수동제거={len(s62 & man)}")
    fate = lambda t: "40제거" if t in miss else "수동13" if t in man else "최종250" if t in kept250 else "2차cap탈락"
    rows = Counter((g, fate(t)) for g, _, t in items)
    for k, v in sorted(rows.items()):
        print(f"    {k}: {v}")

    # 원고 구축 절의 "무작위 표본 24 중 7건": 수동 제거 판단까지 남은 RAND(최종 250 + 수동 제거)가 분모,
    # 수동 제거 사유가 B-A 중복이 아닌 것(granularity·ambiguity·라벨 불안정)이 품질 결함이다.
    why = {n: SAMPLE62_REASON[r] for n, _, _, r, _ in MANUAL13}
    reached = [n for g, n, t in items if g == "RAND" and fate(t) in ("수동13", "최종250")]
    removed = [n for n in reached if n in why]
    defect = Counter(why[n] for n in removed if why[n] != "duplicate_bridge_answer")
    print(f"  RAND 수동 제거 판단까지 남음 {len(reached)} (최종 {len(reached) - len(removed)} + 수동 제거 {len(removed)})"
          f" | 품질 결함 {sum(defect.values())} {dict(defect.most_common())}"
          f" | B-A 중복 {len(removed) - sum(defect.values())}")

    out = {"40제거": "removed_multi_answer", "수동13": "removed_manual",
           "최종250": "kept_final", "2차cap탈락": "dropped_second_cap"}
    assert not s62 & six   # 표본의 40제거는 전부 다중정답 34 쪽이다
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_no", "stratum", "path_id", "schema", "outcome", "removal_reason", "note"])
        for g, n, t in items:
            w.writerow([n, {"RISK": "risk", "RAND": "random"}[g], t.split("_", 1)[1],
                        re.sub(r"_[abcd]$", "", meta[t]["query_type"]), out[fate(t)],
                        why.get(n, ""), SAMPLE62_NOTE.get(n, "")])
    print(f"  항목별 표 {len(items)}행 저장")


def check_caps(rows, ids, label, answer_types):
    t = [rows[i] for i in ids]
    print(f"\n[{label}] n = {len(t)}")
    for key, lim in (("answer_qid", capmod.ANSWER_QID_CAP), ("subj_qid", capmod.SUBJECT_QID_CAP),
                     ("bridge_qid", capmod.BRIDGE_QID_CAP)):
        c = Counter(r[key] for r in t)
        over = {k: v for k, v in c.items() if v > lim}
        print(f"  {key:<11} ≤ {lim}: 최대 {max(c.values())} · 위반 {len(over)}{' ' + str(over) if over else ''}")
    g = Counter(x for r in t for x in capmod.triple_groups(r, answer_types))
    for name, p in capmod.GROUP_PCT_CAPS.items():
        share = g[name] / len(t)
        print(f"  {name:<22} ≤ {p:.0%}: {g[name]}/{len(t)} = {share:.1%} · {'OK' if share <= p else '위반'}")
    k, v = Counter(r["query_type"] for r in t).most_common(1)[0]
    print(f"  (보고용) query_type 최다 {k} {v}/{len(t)} = {v / len(t):.1%} · 기준 {capmod.QUERY_TYPE_PCT_CAP:.0%}")
    k, v = Counter(answer_types.get(r["answer_qid"], {}).get("answer_type", "other") for r in t).most_common(1)[0]
    print(f"  (보고용) answer_type 최다 {k} {v}/{len(t)} = {v / len(t):.1%} · 기준 {capmod.ANSWER_TYPE_PCT_CAP:.0%}")


def section_cap_check(meta, f410, cap):
    # 기록된 cap 결과는 집합 순회 순서에 의존한 원래 실행에서 나와 다시 만들 수 없다. 대신 규칙을 지켰는지 확인한다.
    #   개체 cap: 최종 비율이 아니라 건수라 cap 출력에서 바로 확인된다.
    #   그룹 cap: 코드는 추가 시점의 누적 비율 (g+1)/(k+1) ≤ p 로 판정하므로 최종 비율 ≤ p 는 필요조건이다.
    #   탈락: 개체 카운트는 단조 증가하므로, 개체 cap 으로 탈락한 경로는 cap 출력에서도 개체 cap 에 막혀 있어야 한다.
    #         막혀 있지 않다면 그룹 cap 탈락이어야 하고, 그러면 그룹 소속이어야 한다.
    print("## 기록된 cap 결과 점검 (규칙: dataset/apply_caps.py)")
    answer_types = json.loads(capmod.ANSWER_TYPE_PATH.read_text(encoding="utf-8"))
    kept, rep, man, capped = stage_sets(cap)
    dropped = f410 - capped
    man13 = {t for _, _, t, _, _ in MANUAL13}
    print(f"  입력 {len(f410)} · cap 출력(복원) {len(capped)} · 탈락 {len(dropped)} · 최종 {len(kept)}")
    print(f"  대체 {len(rep)}건은 cap 탈락 쪽에 있었나: {rep <= dropped} · 수동 제거 {len(man)}건 ⊂ cap 출력: {man <= capped}"
          f" · cap 출력 ⊂ 입력: {capped <= f410}")
    # 기록의 탈락 목록은 수리 단계(같은 subject·정답을 반복한 경로 제거 1, 대체 1) 전에 저장됐다
    print(f"  기록의 탈락 목록 {len(cap['dropped_triple_ids'])} = cap 탈락 {len(dropped)} + 62 표본 수동 제거 {len(man13)}: "
          f"{set(cap['dropped_triple_ids']) == dropped | man13} (수리 단계의 제거 {len(man - man13)}·대체 {len(rep)} 전 상태)")
    check_caps(meta, capped, "cap 출력 263", answer_types)
    check_caps(meta, kept, "최종 250", answer_types)

    cnt = {k: Counter(meta[i][k] for i in capped) for k in ("answer_qid", "subj_qid", "bridge_qid")}
    lim = {"answer_qid": capmod.ANSWER_QID_CAP, "subj_qid": capmod.SUBJECT_QID_CAP, "bridge_qid": capmod.BRIDGE_QID_CAP}
    entity, group, unexplained = [], [], []
    for i in sorted(dropped):
        r = meta[i]
        if any(cnt[k][r[k]] >= lim[k] for k in lim):
            entity.append(i)
        elif capmod.triple_groups(r, answer_types):
            group.append(i)
        else:
            unexplained.append(i)
    print(f"\n[탈락 {len(dropped)}] cap 출력에서 개체 cap 에 막힘 {len(entity)} · "
          f"개체 cap 은 아니고 그룹 소속(그룹 cap 탈락 가능) {len(group)} · 설명 불가 {len(unexplained)}")
    print(f"  그룹 cap 후보의 그룹: {dict(Counter(x for i in group for x in capmod.triple_groups(meta[i], answer_types)))}")

    print("\n## 순서 무관 cap (dataset/apply_caps.py — triple ID 정렬 후 seed 42)")
    sorted_kept, _ = capmod.apply_entity_caps(capmod.load_triples(), answer_types)
    s = {t["triple_id"] for t in sorted_kept}
    print(f"  {len(f410)} → {len(s)} · 기록된 cap 출력 {len(capped)} 과 공통 {len(s & capped)} · "
          f"순서 무관 쪽에만 {len(s - capped)} · 기록 쪽에만 {len(capped - s)}")
    check_caps(meta, s, f"순서 무관 cap 출력 {len(s)}", answer_types)


def write_ids(path, ids):
    path.write_text("".join(t + "\n" for t in sorted(ids)), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="데이터셋 구축 감사 수치와 중간 단계 ID 목록을 낸다")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by, meta, p450, f410, cap = load()
    with open(out / "funnel.txt", "w", encoding="utf-8") as f, redirect_stdout(f):
        section_gate_patterns(by)
        section_round_funnel(by, meta, p450, f410, cap)
    with open(out / "sample62.txt", "w", encoding="utf-8") as f, redirect_stdout(f):
        section_sample62(meta, p450, f410, cap, out / "manual_audit_sample62.csv")
    with open(out / "cap_check.txt", "w", encoding="utf-8") as f, redirect_stdout(f):
        section_cap_check(meta, f410, cap)
    write_ids(out / "ids_410.txt", f410)
    write_ids(out / "ids_263.txt", stage_sets(cap)[3])
    print(f"저장: {out}/{{funnel,sample62,cap_check}}.txt, manual_audit_sample62.csv, ids_410.txt, ids_263.txt")


if __name__ == "__main__":
    main()
