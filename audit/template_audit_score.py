#!/usr/bin/env python3
"""
audit/template_audit_score.py

템플릿 쌍 검수 채점. `audit/score_human_audit.py`의 통계 함수를 그대로 import해 쓰고
(정의 분기 방지) task별 분기만 여기서 처리한다. 기존 파일은 수정하지 않는다.

어노테이터가 보내온 결과 JSON을 `reports/rebuttal/template_audit/results/`에 넣고 인자 없이 돌린다.

  python3 audit/template_audit_score.py

공개본만으로 다시 계산할 때는 실명 결과 JSON 대신 익명 라벨 표를 준다 (scripts/camera_ready/run_template_audit.sh).
--revert-relabels 를 같이 주면 재라벨 기록의 수정을 되돌린 수정 전 값이 나온다.

  python3 audit/template_audit_score.py \\
      --labels-csv human_audit/template_audit_labels.csv \\
      [--revert-relabels human_audit/template_audit_relabels.csv]

출력 (reports/template_audit/ — 기존 iaa / proxy_audit / parser_audit과 같은 관례):
  score.json          Fleiss κ · Gwet AC1 · 소수 라벨 특정 일치율 · 쌍별 Cohen κ · 다수결 · 초안 대조.
                      **모든 계수에 p_o·p_e를 같이 적는다** — 이 과제는 대부분이 equivalent라
                      p_e가 높고(kappa paradox), κ 하나로는 일치도 저하와 기저율 쏠림을 구분할 수 없다.
  disagreements.csv   id,section,reference,human_majority,votes   (기존 3갈래와 같은 5열)
  gold_majority.json  --out-gold를 주면 앱 결과와 같은 형식으로 다수결 라벨 저장
"""
import argparse
import csv
import glob
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = "template"
LABELS = ["equivalent", "relation_mismatch", "scope_mismatch", "unnatural_ko", "unnatural_en"]
GATE_SECTION = "main"
KAPPA_GATE = 0.60          # 가이드라인 목표는 0.75, 재작업 판단선은 0.60

# 소수 라벨 정의 — 3번째 어노테이터 결과를 보기 전에(2026-09-18) 고정했다. 결과를 보고 바꾸지 않는다.
MINORITY = {
    "flag_any":     ("equivalent 이외 전부 (어색함 포함)", lambda x: x != "equivalent"),
    "flag_meaning": ("relation_mismatch 또는 scope_mismatch (의미 불일치만)",
                     lambda x: x in ("relation_mismatch", "scope_mismatch")),
}


def load_scorer():
    spec = importlib.util.spec_from_file_location("sha", ROOT / "audit/score_human_audit.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["sha"] = m
    spec.loader.exec_module(m)
    return m


def ac1_parts(items, labels):
    """Gwet AC1 (완전 자료, 다평가자). → (AC1, p_a, p_e)

    p_a는 Fleiss의 P̄(항목별 쌍 일치 비율 평균)와 같다. p_e = Σ_k π_k(1−π_k) / (Q−1)이고
    Q는 **사전에 정한 라벨 수(5)**다 — 관측된 라벨 수가 아니다. 기저율이 한쪽으로 쏠려도
    p_e가 커지지 않아 kappa paradox에 덜 민감하다 (Gwet 2008).
    """
    n, N, Q = len(items[0]), len(items), len(labels)
    if n < 2 or N == 0:
        return None, None, None
    counts = [[row.count(l) for l in labels] for row in items]
    pa = sum((sum(c * c for c in cs) - n) / (n * (n - 1)) for cs in counts) / N
    pi = [sum(cs[k] for cs in counts) / (N * n) for k in range(Q)]
    pe = sum(x * (1 - x) for x in pi) / (Q - 1)
    return (pa - pe) / (1 - pe), pa, pe


def specific_agreement(items, is_pos):
    """한 범주의 특정 일치율 (positive specific agreement, 다평가자 일반형).

    한 평가자가 이 범주를 붙였을 때 다른 평가자도 붙였을 확률 = Σ r_i(r_i−1) / Σ r_i(n−1).
    2인이면 2a/(2a+b+c). 아무도 붙이지 않았으면 None.
    """
    n = len(items[0])
    r = [sum(1 for x in row if is_pos(x)) for row in items]
    den = sum(ri * (n - 1) for ri in r)
    return sum(ri * (ri - 1) for ri in r) / den if den else None


def boot_ci(items, stat, n_boot, seed):
    """항목 단위 bootstrap 95% CI. score_human_audit.fleiss_boot_ci와 같은 방식·같은 seed라
    Fleiss κ와 **같은 리샘플**에서 계산된다. 계산 불가 리샘플은 버린다."""
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    N = len(items)
    draws = []
    for _ in range(n_boot):
        v = stat([items[rng.randrange(N)] for _ in range(N)])
        if v is not None:
            draws.append(v)
    if not draws:
        return None
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def select_files(patterns, keep_partial=False):
    """어노테이터별로 채점할 결과 파일을 고른다.

    앱의 '진행상황 백업'은 `..._partial.json`으로 떨어진다. 최종본과 같이 넣으면 백업 뒤
    라벨을 고친 항목에서 load_results가 충돌로 중단하므로, 최종본이 있는 사람은 최종본만 쓴다.
    최종본 없이 백업만 보낸 사람은(다 끝내고 백업 버튼을 누른 경우가 실제로 있었다)
    가장 최근 백업을 쓰고 완료 건수를 찍는다. --keep-partial이면 전부 넘겨 병합한다.
    """
    # 다른 인자와 마찬가지로 ROOT 기준으로 푼다 (절대경로를 주면 그대로 쓴다).
    paths = sorted({p for pat in patterns for p in glob.glob(str(ROOT / pat))})
    if not paths:
        raise SystemExit(f"결과 파일을 못 찾았다: {patterns}")
    if keep_partial:
        return paths
    by = defaultdict(lambda: {"final": [], "partial": []})
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        kind = "partial" if "_partial" in Path(p).name else "final"
        by[d.get("annotator") or Path(p).stem][kind].append((d.get("exported_at", ""), p, d))
    out = []
    for name, g in sorted(by.items()):
        if g["final"]:
            out += [p for _, p, _ in g["final"]]
            if g["partial"]:
                print(f"  {name}: 최종본이 있어 중간 백업 {len(g['partial'])}개 제외")
        else:
            _, p, d = max(g["partial"])
            done = sum(1 for a in d.get("answers", [])
                       if a.get("label") and (a.get("reason") or "").strip())
            print(f"  {name}: 최종본 없음 → 가장 최근 중간 백업 사용 "
                  f"({Path(p).name}, 라벨+이유 {done}/{d.get('total_items')})")
            out.append(p)
    return out


def load_labels_csv(path):
    """공개 라벨 표(template_audit_export.py 출력)를 load_results와 같은 모양으로 읽는다.
    → ({"Annotator A": {id: (section, label)}}, {id: 작성자 초안})"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = [c for c in rows[0] if c.startswith("label_")]
    res = {f"Annotator {c[len('label_'):]}": {r["id"]: (r["section"], r[c]) for r in rows if r[c]}
           for c in cols}
    return res, {r["id"]: r["author_draft"] for r in rows if r["author_draft"]}


def revert_relabels(res, path):
    """재라벨 기록의 수정을 되돌린다 (수정 전 값 재구성). 현재 라벨이 기록의 '수정 후'와 다르면 중단."""
    for r in csv.DictReader(open(path, encoding="utf-8")):
        sec, cur = res[r["annotator"]][r["id"]]
        if cur != r["label_after"]:
            raise SystemExit(f"{r['annotator']} {r['id']}: 현재 {cur} ≠ 기록의 수정 후 {r['label_after']}")
        res[r["annotator"]][r["id"]] = (sec, r["label_before"])
        print(f"  되돌림: {r['annotator']} {r['id']} {cur} → {r['label_before']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+",
                    default=["reports/rebuttal/template_audit/results/template_results_*.json"])
    ap.add_argument("--labels-csv", default=None,
                    help="실명 결과 JSON 대신 공개 익명 라벨 표로 채점한다 (초안 라벨도 이 표에서 읽는다)")
    ap.add_argument("--revert-relabels", default=None,
                    help="재라벨 기록 CSV — 기록된 수정을 되돌려 수정 전 값을 계산한다")
    ap.add_argument("--calibration", default="reports/rebuttal/template_audit/template_calibration.jsonl")
    ap.add_argument("--out-json", default="reports/template_audit/score.json")
    ap.add_argument("--out-disagreements", default="reports/template_audit/disagreements.csv")
    ap.add_argument("--out-gold", default=None)
    ap.add_argument("--keep-partial", action="store_true",
                    help="중간 백업을 전부 병합한다 (기본: 사람별로 최종본, 없으면 최근 백업 1개)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    S = load_scorer()
    ref = {}
    if args.labels_csv:
        res, ref = load_labels_csv(ROOT / args.labels_csv)
    else:
        res = S.load_results(select_files(args.results, args.keep_partial), TASK)
        cal = Path(ROOT / args.calibration)
        if cal.exists():
            for line in cal.open(encoding="utf-8"):
                r = json.loads(line)
                ref[r["id"]] = r.get("ref_label")
    if args.revert_relabels:
        revert_relabels(res, ROOT / args.revert_relabels)
    names = sorted(res)
    print(f"어노테이터 {len(names)}명: {names}")

    # 전원이 라벨한 항목만 일치도 계산에 쓴다 (기존 3갈래와 같은 규칙)
    common = set.intersection(*(set(v) for v in res.values()))
    sec_of = {i: res[names[0]][i][0] for i in common}
    by_sec = defaultdict(list)
    for i in sorted(common):
        by_sec[sec_of[i]].append(i)
    n_partial = sum(len(v) for v in res.values()) - len(common) * len(names)
    print(f"  전원 라벨 항목 {len(common)} "
          f"({ {k: len(v) for k, v in by_sec.items()} }), 부분 라벨 {n_partial}건 제외")

    out = {"task": TASK, "annotators": names, "labels": LABELS,
           "n_complete": len(common), "n_dropped_partial": n_partial}

    # 한 라벨도 안 나온 경우 등 축퇴 상황에서 fleiss_kappa는 None을 돌려준다 → 그대로 둔다.
    rnd = lambda x: round(x, 4) if isinstance(x, (int, float)) else None
    rci = lambda ci: [rnd(x) for x in ci] if ci else None
    for sec, ids in by_sec.items():
        items = [[res[n][i][1] for n in names] for i in ids]
        out[f"fleiss_{sec}"] = rnd(S.fleiss_kappa(items, LABELS))
        _, po, pe = S.fleiss_parts(items, LABELS)   # (κ, p_o, p_e) 순서다
        ci = S.fleiss_boot_ci(items, LABELS, args.n_boot, args.seed)
        out[f"fleiss_{sec}_parts"] = {"p_o": rnd(po), "p_e": rnd(pe), "n": len(ids), "ci": rci(ci)}
        for lab in LABELS:
            bi = [[lab if x == lab else "other" for x in row] for row in items]
            k, bpo, bpe = S.fleiss_parts(bi, [lab, "other"])
            out[f"fleiss_{sec}_{lab}"] = rnd(k)
            out[f"fleiss_{sec}_{lab}_n"] = sum(row.count(lab) for row in items)
            out[f"fleiss_{sec}_{lab}_parts"] = {"p_o": rnd(bpo), "p_e": rnd(bpe)}

        a, pa, ape = ac1_parts(items, LABELS)
        out[f"ac1_{sec}"] = {"ac1": rnd(a), "p_a": rnd(pa), "p_e": rnd(ape), "q": len(LABELS),
                             "n": len(ids),
                             "ci": rci(boot_ci(items, lambda s: ac1_parts(s, LABELS)[0],
                                               args.n_boot, args.seed))}

        sa = {}
        for key, (desc, f) in MINORITY.items():
            flags = [sum(1 for x in row if f(x)) for row in items]
            sa[key] = {"def": desc,
                       "psa": rnd(specific_agreement(items, f)),
                       "ci": rci(boot_ci(items, lambda s, f=f: specific_agreement(s, f),
                                         args.n_boot, args.seed)),
                       "n_flags": sum(flags),
                       "n_items_any": sum(1 for r in flags if r >= 1),
                       "n_items_all": sum(1 for r in flags if r == len(names))}
        sa["per_label"] = {lab: {"psa": rnd(specific_agreement(items, lambda x, l=lab: x == l)),
                                 "n_flags": sum(row.count(lab) for row in items)} for lab in LABELS}
        out[f"specific_agreement_{sec}"] = sa

    gate = out.get(f"fleiss_{GATE_SECTION}")
    out["kappa_gate"] = {"kappa": gate, "gate": KAPPA_GATE,
                         "pass": bool(gate is not None and gate >= KAPPA_GATE)}

    out["cohen_pairwise"], out["cohen_pairwise_parts"] = {}, {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            ids = by_sec.get(GATE_SECTION, sorted(common))
            la = [res[names[a]][i][1] for i in ids]
            lb = [res[names[b]][i][1] for i in ids]
            key = f"{names[a]}|{names[b]}"
            out["cohen_pairwise"][key] = rnd(S.cohen_kappa(la, lb, LABELS))
            # κ는 기존 함수 값을 쓰고, 세부는 같은 정의(score_human_audit.cohen_kappa)로 따로 적는다
            ca, cb = Counter(la), Counter(lb)
            out["cohen_pairwise_parts"][key] = {
                "p_o": rnd(sum(x == y for x, y in zip(la, lb)) / len(ids)),
                "p_e": rnd(sum(ca[l] * cb[l] for l in LABELS) / len(ids) ** 2), "n": len(ids)}

    # 다수결 + 불일치
    gold, ties, rows = {}, 0, []
    for i in sorted(common):
        votes = [res[n][i][1] for n in names]
        mj, _ = S.majority(votes)   # (다수 라벨 또는 None, 표 분포)
        if mj is None:
            ties += 1
            rows.append([i, sec_of[i], ref.get(i, ""), "TIE", "|".join(votes)])
        else:
            gold[i] = mj
            if ref.get(i) and ref[i] != mj:
                rows.append([i, sec_of[i], ref[i], mj, "|".join(votes)])
    out["n_gold"], out["n_tie"] = len(gold), ties
    out["label_dist_main"] = dict(Counter(gold[i] for i in by_sec.get("main", []) if i in gold).most_common())
    agree = [(i, ref[i] == gold[i]) for i in gold if ref.get(i)]
    if agree:
        k = sum(x for _, x in agree)
        out["draft_agreement"] = {"n": len(agree), "agree": k, "rate": round(k / len(agree), 4),
                                  "note": "초안 라벨은 작성자 판단이며 정답이 아니다. 참고용."}

    for p, data in ((args.out_json, out), (args.out_disagreements, rows)):
        Path(ROOT / p).parent.mkdir(parents=True, exist_ok=True)
    Path(ROOT / args.out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    with (ROOT / args.out_disagreements).open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "section", "reference", "human_majority", "votes"])
        w.writerows(rows)
    if args.out_gold:
        Path(ROOT / args.out_gold).write_text(json.dumps(
            {"task": TASK, "annotator": "majority",
             "answers": [{"id": i, "section": sec_of[i], "label": gold[i]} for i in sorted(gold)]},
            ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(out, by_sec, names)
    print(f"\n저장 -> {args.out_json}, {args.out_disagreements}")


def print_summary(out, by_sec, names):
    f = lambda x: "  —  " if x is None else f"{x:6.3f}"
    fc = lambda ci: "" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
    for sec in by_sec:
        fp, ac = out[f"fleiss_{sec}_parts"], out[f"ac1_{sec}"]
        print(f"\n[{sec}] 항목 {fp['n']} · 어노테이터 {len(names)}명")
        print(f"  {'지표':<30}{'값':>6}  {'p_o':>6}  {'p_e':>6}   95% CI")
        print(f"  {'Fleiss κ':<30}{f(out[f'fleiss_{sec}'])}  {f(fp['p_o'])}  {f(fp['p_e'])}   {fc(fp['ci'])}")
        print(f"  {'Gwet AC1':<30}{f(ac['ac1'])}  {f(ac['p_a'])}  {f(ac['p_e'])}   {fc(ac['ci'])}")
        print(f"  라벨별 κ (그 라벨 vs 나머지)")
        for lab in LABELS:
            pp = out[f"fleiss_{sec}_{lab}_parts"]
            print(f"    {lab:<28}{f(out[f'fleiss_{sec}_{lab}'])}  "
                  f"{f(pp['p_o'])}  {f(pp['p_e'])}   (표 {out[f'fleiss_{sec}_{lab}_n']})")
        sa = out[f"specific_agreement_{sec}"]
        print(f"  소수 라벨 특정 일치율")
        for key in MINORITY:
            s = sa[key]
            print(f"    {key:<28}{f(s['psa'])}   {fc(s['ci']):<18} 표 {s['n_flags']} · "
                  f"1명 이상 {s['n_items_any']}항목 · 전원 {s['n_items_all']}항목  — {s['def']}")
    if out["cohen_pairwise"]:
        print(f"\n쌍별 Cohen κ ({GATE_SECTION})")
        for k, v in out["cohen_pairwise"].items():
            pp = out["cohen_pairwise_parts"][k]
            print(f"  {k:<30}{f(v)}  {f(pp['p_o'])}  {f(pp['p_e'])}   (κ · p_o · p_e)")
    print(f"\n다수결 {out['n_gold']} · TIE {out['n_tie']} · main 다수결 분포 {out['label_dist_main']}")


if __name__ == "__main__":
    main()
