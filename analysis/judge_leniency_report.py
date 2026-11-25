#!/usr/bin/env python3
"""
analysis/judge_leniency_report.py

원 실험 judge 관대함 규모 집계 → reports/rebuttal/judge_leniency.md. API·GPU 미사용.
`judge_leniency_check.py` 의 판정 결과와 기존 정본 결과만 읽는다. `response_numbers.md` 에는 넣지 않는다.
원고 Limitations 의 비율(7.6–14.5%)은 §2 다. `--skip-posthoc` 은 §5 를 건너뛴다(§5 만 쓰는 모듈 없이 돈다).

내는 것
  1. 모델 × 조건 × same/cross: 대상 수(yield=1 ∧ 주 매처 오답), same_entity 비율(CI), no 중 passage 내 존재 비율
  2. 모델별: 전체 비기권 hop 대비 "judge faithful ∧ 다른 엔티티" 비율 (= same_entity no 인 hop / 비기권 hop)
  3. 표기 차이 유형 분포, 사례
  5. 사후 정의 매처(same_entity=yes 를 정답 인정)로 정확도 gap·QMC 재계산, same_entity=yes 비율의 same−cross
     — 재표집은 `matcher_bounds.py` / `r2w3_analysis.py` 와 **같은 numpy 흐름**(모델마다 `default_rng(42)`,
       한 draw 에서 두 매처·세 조건을 함께 평가 = joint paired). 주 매처 칸은 `matcher_bounds.csv` 와 대조한다.
passage 내 존재: 그 hop 의 top-5 passage(궤적 `retrieved_passages`) 안에 응답 claim 문자열이 있는지 —
  원고 주 매처 판정 함수(`16_pdr_analysis.entity_in_text`)에 claim 을 넣어 본다.
CI: triple 250 클러스터 재표집, numpy `default_rng(42)`, B = 10,000, 95% percentile.

사용:
  python3 analysis/judge_leniency_report.py
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

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import matched_subset as ms          # noqa: E402  (모델 표시명)
import correctness_axis as ca        # noqa: E402  (paths — 궤적 정본 입력)
mb = r2 = None                       # §5 에서만 불러온다 (_load_posthoc)


def _load_posthoc():
    global mb, r2
    import matcher_bounds as mb      # noqa: E402  (variant_values — 주 매처 값)
    import r2w3_analysis as r2       # noqa: E402  (Reg / V / qmc4 / status — 재표집 흐름)

_spec = importlib.util.spec_from_file_location("pdr16", ROOT / "analysis/16_pdr_analysis.py")
pdr16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pdr16)

csv.field_size_limit(10 ** 9)
HOPS = ROOT / "data/correctness/all/hops.csv"
CHECK = ROOT / "data/correctness/entity_check.jsonl"
QA = ROOT / "data/dataset/qa_pairs.jsonl"
OUT = ROOT / "reports/rebuttal/judge_leniency.md"
CONDS = ["controlled", "template_matched", "surface_normalized"]
CL = {"controlled": "controlled", "template_matched": "template-matched",
      "surface_normalized": "surface-normalized"}
B, SEED = 10000, 42


def ci(point, boot):
    b = np.sort(boot[~np.isnan(boot)])
    if not len(b):
        return "N/A"
    lo, hi = b[int(0.025 * len(b))], b[int(0.975 * len(b)) - 1]
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


# ── §5 사후 정의 매처 ─────────────────────────────────────────────────────────

M2L = {"pdr": "주 매처", "posthoc": "사후 매처"}
S3L = {"template": "template", "bridge": "bridge", "residual": "residual"}


def posthoc_value(r, verdict):
    """(주 매처, 사후 매처) 값. ambiguous → (None, None). same_entity=yes 면 정답으로 올린다."""
    c = mb.variant_values(r)["pdr"]
    if c is None:
        return None, None
    v = verdict.get((r["model"], r["cond"], r["qid"], int(r["hop"])))
    return c, (1 if (c == 1 or (v and v["same"] == "yes")) else 0)


def boot_model(model, rows, triple_of, verdict):
    """r2w3_analysis.run_model 과 같은 재표집 흐름 — 모델마다 default_rng(42), triple 250 클러스터 재표집,
    한 draw 에서 두 매처·세 조건·yes 비율을 함께 평가한다(joint paired)."""
    qids = sorted({r["qid"] for r in rows[(model, CONDS[0])]})
    qi = {q: i for i, q in enumerate(qids)}
    reg, cnt = r2.Reg(len(qids)), Counter()
    for cond in CONDS:
        for r in rows[(model, cond)]:
            i, side = qi[r["qid"]], r["index_class"]
            c, ph = posthoc_value(r, verdict)
            if c is None:
                continue
            reg.add(f"acc|pdr|{cond}|{side}", i, float(c))
            reg.add(f"acc|posthoc|{cond}|{side}", i, float(ph))
            cnt[(cond, side, "up")] += int(ph > c)
            v = verdict.get((model, cond, r["qid"], int(r["hop"])))
            if v:
                reg.add(f"yes|{cond}|{side}", i, float(v["same"] == "yes"))
                cnt[(cond, side, "n")] += 1
    NUM, DEN = np.array(reg.num), np.array(reg.den)
    by_t = defaultdict(list)
    for q in qids:
        by_t[triple_of[q]].append(qi[q])
    tl = sorted(by_t)
    tidx = np.array([by_t[t] for t in tl])
    rng = np.random.default_rng(SEED)
    C = np.empty((B, len(qids)))
    for b in range(B):
        C[b] = np.bincount(tidx[rng.integers(0, len(tl), len(tl))].ravel(), minlength=len(qids))
    S, D = C @ NUM.T, C @ DEN.T
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.where(D > 0, S / D, np.nan)
        P = np.where(DEN.sum(1) > 0, NUM.sum(1) / DEN.sum(1), np.nan)

    def get(key):
        k = reg.idx[key]
        return r2.V(P[k], R[:, k])

    return get, cnt, len(tl)


def incl0(v) -> bool:
    """CI 가 0 을 포함하는가 (경계도 포함으로 센다)."""
    lo, hi = v.ci()
    return not (lo > 0 or hi < 0)


def section5(w, rows, verdict, triple_of, n_judged, n_target):
    G, CNT = {}, {}
    for model in ms.MODELS:
        G[model], CNT[model], nt = boot_model(model, rows, triple_of, verdict)
        print(f"[§5 {model}] triple {nt}개, B={B:,}, 재표집 완료")

    # 대조: 주 매처 칸은 matcher_bounds.csv 와 같아야 한다
    ref = {}
    with r2.MB_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ref[(r["model"], r["variant"], r["part"], r["stat"])] = float(r["point"])
    dmax, n_chk = 0.0, 0
    for model in ms.MODELS:
        q = r2.qmc4(G[model], "acc|pdr")
        pairs = [(("qmc", s), q[s]) for s in S3L] + \
                [((mb.SHORT[c], "gap"), r2.gap(G[model], "acc|pdr", c)) for c in CONDS]
        for (part, stat), v in pairs:
            dmax = max(dmax, abs(v.p - ref[(model, "pdr", part, stat)]))
            n_chk += 1
    print(f"[§5 대조] matcher_bounds.csv 주 매처 gap·QMC {n_chk}개 점추정 최대 차 {dmax:.2e}")

    w("\n## 5. 사후 정의 매처 — same_entity=yes 를 정답 인정\n")
    w("> **사후 정의 매처 — 결과를 본 뒤 정의, 주 분석 아님.** 논문·응답의 정확도 수치는 주 매처 값이다.\n")
    miss = Counter()
    for model in ms.MODELS:
        for cond in CONDS:
            for r in rows[(model, cond)]:
                if (r["ambiguous"] == "True" or r["abstained"] == "True"
                        or not (r["faithful"] == "1.0" and r["correct"] == "0")):
                    continue
                if (model, cond, r["qid"], int(r["hop"])) not in verdict:
                    miss[(model, cond, r["index_class"])] += 1
    w("정의: `사후 = 1` ⟺ 주 매처 정답 **또는** §1 판정에서 `same_entity=yes`. 판정 대상이 "
      "**yield=1 ∧ 주 매처 오답** 인 hop 뿐이므로 두 매처는 그 hop 들에서만 다르다. "
      + (f"미판정 {n_target - n_judged}건은 사후 매처에서도 오답으로 둔다(아래 값은 그만큼 보수적) — "
         + ", ".join(f"{ms.NAMES[m]} {CL[c]} {s} {n}건"
                     for (m, c, s), n in sorted(miss.items(), key=lambda x: -x[1])) + ". "
         if miss else "")
      + "ambiguous hop 은 두 매처 모두 분모에서 빠진다.\n")
    w(f"재표집: `judge_leniency_report.py` · numpy `default_rng({SEED})` — `matcher_bounds.py` / "
      f"`r2w3_analysis.py` 와 같은 흐름. triple 250 클러스터, B = {B:,}, 한 draw 에서 두 매처·세 조건을 "
      "함께 평가(joint paired). 주 매처 칸은 `matcher_bounds.csv` 와 소수점 끝까지 같다(스크립트가 대조).\n")

    w("### 5-1. 정확도 gap (same − cross)\n")
    w("| 모델 | 조건 | 주 매처 gap [95% CI] | 사후 매처 gap [95% CI] | 사후 − 주 [95% CI] | 사후에서 정답으로 올라간 hop (same / cross) |")
    w("|-|-|-|-|-|-|")
    ch1 = []
    for model in ms.MODELS:
        for cond in CONDS:
            a = r2.gap(G[model], "acc|pdr", cond)
            b = r2.gap(G[model], "acc|posthoc", cond)
            if incl0(a) != incl0(b):
                ch1.append((ms.NAMES[model], CL[cond], r2.status(*a.ci()), r2.status(*b.ci())))
            u = CNT[model]
            w(f"| {ms.NAMES[model]} | {CL[cond]} | {r2.fs(a)} | {r2.fs(b)} | {r2.fs(b - a)} | "
              f"{u[(cond, 'same', 'up')]} / {u[(cond, 'cross', 'up')]} |")
    w("\n0 포함 여부가 달라진 칸: " + ("없음" if not ch1 else
      "; ".join(f"{m} {c} (주 {x} → 사후 {y})" for m, c, x, y in ch1)) + ".\n")

    w("### 5-2. 정확도 축 QMC 세 성분\n")
    w("`template` = controlled gap − template-matched gap, `bridge` = template-matched gap − "
      "surface-normalized gap, `residual` = surface-normalized gap.\n")
    w("| 모델 | 성분 | 주 매처 [95% CI] | 사후 매처 [95% CI] | 사후 − 주 [95% CI] |")
    w("|-|-|-|-|-|")
    ch2 = []
    for model in ms.MODELS:
        qa_, qb = r2.qmc4(G[model], "acc|pdr"), r2.qmc4(G[model], "acc|posthoc")
        for s in S3L:
            if incl0(qa_[s]) != incl0(qb[s]):
                ch2.append((ms.NAMES[model], S3L[s], r2.status(*qa_[s].ci()), r2.status(*qb[s].ci())))
            w(f"| {ms.NAMES[model]} | {S3L[s]} | {r2.fs(qa_[s])} | {r2.fs(qb[s])} | {r2.fs(qb[s] - qa_[s])} |")
    w("\n**0 포함 여부가 달라진 칸**" + (" — 없음.\n" if not ch2 else ":\n"))
    if ch2:
        w("| 모델 | 성분 | 주 매처 판정 | 사후 매처 판정 |")
        w("|-|-|-|-|")
        for m, s, x, y in ch2:
            w(f"| {m} | {s} | {x} | {y} |")
        w("")

    w("### 5-3. same_entity=yes 비율의 same − cross\n")
    w("분모는 그 칸의 **판정을 받은 대상 hop**(yield=1 ∧ 주 매처 오답)이다. 같은 비율의 §1 CI 와 "
      "끝자리가 다를 수 있다 — §1 은 칸마다 `default_rng(42)` 를 새로 만들고, 여기서는 모델 하나에 "
      "draw 를 한 벌만 만들어 두 side·세 조건을 함께 평가한다(same − cross 를 같은 draw 에서 짝지으려면 "
      "이렇게 해야 한다).\n")
    w("| 모델 | 조건 | same [95% CI] | cross [95% CI] | same − cross [95% CI] | n same / n cross |")
    w("|-|-|-|-|-|-|")
    for model in ms.MODELS:
        get, u = G[model], CNT[model]
        for cond in CONDS:
            s, c = get(f"yes|{cond}|same"), get(f"yes|{cond}|cross")
            w(f"| {ms.NAMES[model]} | {CL[cond]} | {r2.fp(s)} | {r2.fp(c)} | {r2.fs(s - c)} | "
              f"{u[(cond, 'same', 'n')]} / {u[(cond, 'cross', 'n')]} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--skip-posthoc", action="store_true",
                    help="§5(사후 정의 매처로 정확도 gap·QMC 재표집)를 건너뛴다")
    args = ap.parse_args()
    if not CHECK.exists():
        sys.exit(f"판정 결과 없음: {CHECK} — judge_leniency_check.py 를 먼저 돌린다")
    qa = {r["qid"]: r for r in ca.load_jsonl(QA)}
    triple_of = {q: r["source_path_id"].rsplit("_pat_", 1)[0] for q, r in qa.items()}
    verdict = {}
    for r in ca.load_jsonl(CHECK):
        pr = r.get("parsed") or {}
        verdict[(r["model"], r["cond"], r["qid"], r["hop"])] = {
            "same": str(pr.get("same_entity", "")).lower(), "type": pr.get("surface_type", ""),
            "reason": pr.get("reason", ""), "claims": r.get("claims", []),
            "gold_ko": r.get("gold_ko", ""), "question": r.get("question", ""),
        }
    rows = defaultdict(list)
    side_of = {}
    with HOPS.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[(r["model"], r["cond"])].append(r)
            side_of[(r["model"], r["cond"], r["qid"], int(r["hop"]))] = r["index_class"]

    # 궤적에서 top-5 passage (조건·모델별로 읽고 바로 버린다)
    in_passage = {}
    for model in ms.MODELS:
        for cond in CONDS:
            _ann, trj = ca.paths(model, cond)
            psg = {}
            for t in ca.load_jsonl(trj):
                for st in t["trajectory"]:
                    psg[(t["qid"], st["hop"])] = st.get("retrieved_passages", [])
            for r in rows[(model, cond)]:
                k = (model, cond, r["qid"], int(r["hop"]))
                v = verdict.get(k)
                if not v or v["same"] != "no":
                    continue
                claims = v["claims"] or [r["generated_text"]]
                surf = {pdr16.normalize(c) for c in claims if c}
                joined = "\n".join(psg.get((r["qid"], int(r["hop"])), []))
                in_passage[k] = bool(surf) and pdr16.entity_in_text(surf, joined)

    L, w = [], None
    L = []
    w = L.append
    w("# 원 실험 judge 관대함 규모\n")
    w(f"대상: 6모델 × 3조건의 **yield=1 ∧ 주 매처 오답 ∧ 비기권 ∧ 비-ambiguous** hop {len(verdict)}건. "
      "판정은 응답이 gold 와 같은 엔티티를 가리키는지를 `gpt-5.4-2026-03-05`로 물었다(원고 judge 와 같은 요청 본문, "
      "`temperature 0`·`seed 42`, 입력은 질문·gold KO·EN 표기·응답, **passage 미제공**). "
      "기록 `entity_check.jsonl`. 집계 `judge_leniency_report.py` · numpy `default_rng(42)`, "
      f"triple 250 클러스터 재표집 B = {B:,}.\n")
    st = Counter(v["same"] for v in verdict.values())
    n_tg = sum(1 for k in rows for r in rows[k]
               if r["ambiguous"] != "True" and r["abstained"] != "True"
               and r["faithful"] == "1.0" and r["correct"] == "0")
    w(f"전체 판정: yes {st['yes']} / no {st['no']} / unclear {st['unclear']}.")
    if len(verdict) < n_tg:
        w(f"\n⚠ **커버리지 {len(verdict)}/{n_tg} ({len(verdict) / n_tg:.1%})** — 나머지 {n_tg - len(verdict)}건은 "
          f"판정을 받지 못했다(OpenAI 크레딧 소진으로 중단). 아래 비율의 분모는 **판정을 받은 hop** 이고, "
          f"§2 의 '전체 비기권 hop 대비' 비율은 미판정분만큼 과소 추정이다(최대 {n_tg - len(verdict)}건).\n")
    else:
        w("")

    w("## 1. 모델 × 조건 × same/cross\n")
    w("| 모델 | 조건 | side | 비기권 hop | 대상 | 대상 비율 | same_entity=yes [95% CI] | no | unclear | no 중 passage 내 존재 |")
    w("|-|-|-|-|-|-|-|-|-|-|")
    for model in ms.MODELS:
        for cond in CONDS:
            for side in ("same", "cross"):
                R = [r for r in rows[(model, cond)]
                     if r["index_class"] == side and r["ambiguous"] != "True" and r["abstained"] != "True"]
                tg = [r for r in R if r["faithful"] == "1.0" and r["correct"] == "0"]
                vs = [verdict[(model, cond, r["qid"], int(r["hop"]))] for r in tg
                      if (model, cond, r["qid"], int(r["hop"])) in verdict]
                yes = sum(1 for v in vs if v["same"] == "yes")
                no = [r for r in tg if verdict.get((model, cond, r["qid"], int(r["hop"])), {}).get("same") == "no"]
                unc = sum(1 for v in vs if v["same"] == "unclear")
                # triple 클러스터 재표집: hop 단위 num/den 을 triple 로 묶는다
                tl = sorted({triple_of[r["qid"]] for r in R})
                ti = {t: i for i, t in enumerate(tl)}
                num = np.zeros(len(tl)); den = np.zeros(len(tl))
                for r in tg:
                    v = verdict.get((model, cond, r["qid"], int(r["hop"])))
                    if not v:
                        continue
                    i = ti[triple_of[r["qid"]]]
                    den[i] += 1
                    num[i] += v["same"] == "yes"
                rng = np.random.default_rng(SEED)
                boot = np.empty(B)
                for b in range(B):
                    pick = rng.integers(0, len(tl), len(tl))
                    c = np.bincount(pick, minlength=len(tl)).astype(float)
                    d = c @ den
                    boot[b] = (c @ num) / d if d else np.nan
                p = num.sum() / den.sum() if den.sum() else np.nan
                inp = [in_passage.get((model, cond, r["qid"], int(r["hop"]))) for r in no]
                inp_s = f"{sum(1 for x in inp if x)}/{len(inp)} ({sum(1 for x in inp if x) / len(inp):.3f})" if inp else "—"
                w(f"| {ms.NAMES[model]} | {CL[cond]} | {side} | {len(R)} | {len(tg)} | {len(tg) / len(R):.3f} | "
                  f"{ci(p, boot)} | {len(no)} | {unc} | {inp_s} |")

    w("\n## 2. 모델별 — 전체 비기권 hop 대비\n")
    w('"judge faithful ∧ 다른 엔티티" = yield=1 ∧ 주 매처 오답 ∧ same_entity=no. 분모는 비기권·비-ambiguous hop 전체(3조건 합).\n')
    w("| 모델 | 비기권 hop | 대상 (yield=1 ∧ 오답) | 대상 비율 | same_entity=no | **judge faithful ∧ 다른 엔티티 비율** | 그중 passage 내 존재 |")
    w("|-|-|-|-|-|-|-|")
    for model in ms.MODELS:
        R = [r for c in CONDS for r in rows[(model, c)]
             if r["ambiguous"] != "True" and r["abstained"] != "True"]
        tg = [r for r in R if r["faithful"] == "1.0" and r["correct"] == "0"]
        no = [(r["cond"], r) for r in tg
              if verdict.get((model, r["cond"], r["qid"], int(r["hop"])), {}).get("same") == "no"]
        inp = [in_passage.get((model, c, r["qid"], int(r["hop"]))) for c, r in no]
        w(f"| {ms.NAMES[model]} | {len(R)} | {len(tg)} | {len(tg) / len(R):.3f} | {len(no)} | "
          f"**{len(no) / len(R):.3f}** | {sum(1 for x in inp if x)}/{len(inp)}"
          + (f" ({sum(1 for x in inp if x) / len(inp):.3f})" if inp else "") + " |")

    w("\n## 3. 표기 차이 유형 (same_entity=yes)\n")
    types = ("어순", "음역 차이", "띄어쓰기·구두점", "상위·하위 개념", "기타")
    w("| 모델 | side | yes | " + " | ".join(types) + " |")
    w("|-|-|-|" + "-|" * len(types))
    for model in ms.MODELS:
        for side in ("same", "cross"):
            vs = [v for k, v in verdict.items()
                  if k[0] == model and v["same"] == "yes" and side_of.get(k) == side]
            ty = Counter(v["type"] for v in vs)
            w(f"| {ms.NAMES[model]} | {side} | {len(vs)} | " + " | ".join(str(ty.get(x, 0)) for x in types) + " |")

    w("\n## 4. 사례 — same_entity=no (무작위 15건, seed 42)\n")
    pool = [(k, v) for k, v in verdict.items() if v["same"] == "no"]
    for (m, c, q, h), v in random.Random(SEED).sample(pool, min(15, len(pool))):
        w(f"- **{ms.NAMES[m]}** · {CL[c]} · `{q}` hop{h} · passage 내 존재 "
          f"{'예' if in_passage.get((m, c, q, h)) else '아니오'}")
        w(f"  - 질문: {v['question']} / gold KO 「{v['gold_ko']}」")
        w(f"  - 응답 claim: {v['claims']}")
        w(f"  - 판정 근거: {re.sub(chr(10), ' ', v['reason'])[:200]}")

    if not args.skip_posthoc:
        _load_posthoc()
        section5(w, rows, verdict, triple_of, len(verdict), n_tg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"저장: {out.name}")


if __name__ == "__main__":
    main()
