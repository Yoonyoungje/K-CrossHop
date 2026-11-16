#!/usr/bin/env python3
"""
audit/score_human_audit.py

Human IAA / proxy audit 채점 — 어노테이션 앱이 내보낸 결과 JSON(어노테이터별 1개)을
받아 일치도와 최종 판정을 산출한다.

  1. Fleiss' κ (section별: calibration / main) + 라벨별 one-vs-rest κ
  2. 어노테이터 쌍별 Cohen's κ
  3. 다수결 판정 (동수 = tie → adjudication 대상으로 출력)
  4. task별 검증:
     - iaa   : judge label vs human 다수결 (일치율 / Cohen κ / 혼동행렬)
              → LLM judge 신뢰도 근거
     - proxy : 다수결 라벨 분포 → answer_bearing_hit 오탐률(proxy-FP) Wilson CI
              → proxy 타당성 근거

사용:
  python audit/score_human_audit.py --task iaa \
    --results data/human_audit/iaa/results/*.json

  python audit/score_human_audit.py --task proxy \
    --results data/human_audit/proxy/results/*.json --out-json reports/proxy_audit_score.json

입력 JSON 스키마(앱 downloadResults()): {task, annotator, answers:[{id, section, label, ...}]}
"""
import argparse
import csv
import glob
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.qa_set import load_qids

LABELS = {
    "iaa":    ["supported", "unsupported", "contradicted"],
    "proxy":  ["proxy_fp", "overabstain", "ambiguous", "parser_fp"],
    "parser": ["abstain", "answer", "unclear"],
}

# 일치도·검증을 볼 섹션. parser 감사는 main이 라벨러별 disjoint shard라 겹치는 항목이 없다 —
# 합의 검증은 전원 공통인 calibration(=overlap)에서만 가능하다.
GATE_SECTION = {"iaa": "main", "proxy": "main", "parser": "calibration"}
KAPPA_GATE   = {"iaa": 0.60, "proxy": 0.70, "parser": 0.60}   # RUBRIC.md / 가이드라인 기준

IAA_MAIN   = "data/human_audit/iaa/iaa_main_sample.jsonl"
IAA_CALIB  = "data/human_audit/iaa/iaa_calibration.jsonl"
PARSER_OVERLAP = "data/human_audit/parser/parser_overlap_ids.json"
PARSER_ID_MAP = "data/human_audit/parser/parser_id_map.json"
# 앱 결과 JSON의 annotator가 회차마다 약칭으로 나오는 경우가 있다(예: 보충 14건은 "Annotator A").
# 같은 사람이 다른 이름으로 잡히면 어노테이터 수가 늘어 일치도가 깨지므로 정규화한다.
ANNOTATOR_MAP = "data/human_audit/annotator_map.json"



# ── 일치도 ────────────────────────────────────────────────────────────────────
def fleiss_parts(items: list[list[str]], labels: list[str]):
    """(κ, p_o, p_e). p_o = 항목별 쌍 일치 비율 평균(P̄), p_e = 기저율로 기대되는 우연일치.

    κ 하나만 보면 일치도 저하와 기저율 쏠림(kappa paradox)을 구분할 수 없다 → 셋을 함께 낸다."""
    n = len(items[0])
    N = len(items)
    if n < 2 or N == 0:
        return None, None, None
    counts = [[row.count(l) for l in labels] for row in items]
    p_j = [sum(c[j] for c in counts) / (N * n) for j in range(len(labels))]
    P_i = [(sum(c[j] ** 2 for j in range(len(labels))) - n) / (n * (n - 1)) for c in counts]
    P_bar = sum(P_i) / N
    P_e = sum(p ** 2 for p in p_j)
    if abs(1 - P_e) < 1e-12:
        return None, P_bar, P_e
    return (P_bar - P_e) / (1 - P_e), P_bar, P_e


def fleiss_kappa(items: list[list[str]], labels: list[str]):
    return fleiss_parts(items, labels)[0]


def fleiss_boot_ci(items: list[list[str]], labels: list[str], n_boot: int = 10000, seed: int = 42):
    """항목 단위 bootstrap 95% CI. (lo, hi) — 계산 불가 리샘플은 버린다."""
    if len(items) < 2:
        return None
    rng = random.Random(seed)
    N = len(items)
    draws = []
    for _ in range(n_boot):
        s = [items[rng.randrange(N)] for _ in range(N)]
        k = fleiss_parts(s, labels)[0]
        if k is not None:
            draws.append(k)
    if not draws:
        return None
    draws.sort()
    return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]


def cohen_kappa(a: list[str], b: list[str], labels: list[str]):
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1 - pe) < 1e-12:
        return None
    return (po - pe) / (1 - pe)


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


# ── 입력 ──────────────────────────────────────────────────────────────────────
def load_results(paths: list[str], task: str):
    """→ {annotator: {item_id: (section, label)}}"""
    # parser 앱 초기 빌드는 결과 JSON에 section을 넣지 않았다 → overlap id로 복원.
    overlap = (set(json.load(open(PARSER_OVERLAP, encoding="utf-8"))["overlap_ids"])
               if task == "parser" and Path(PARSER_OVERLAP).exists() else set())
    aliases = (json.load(open(ANNOTATOR_MAP, encoding="utf-8")).get("aliases", {})
               if Path(ANNOTATOR_MAP).exists() else {})
    out = {}
    for path in sorted(p for pat in paths for p in glob.glob(pat)):
        d = json.load(open(path, encoding="utf-8"))
        if d.get("task") != task:
            raise SystemExit(f"{path}: task={d.get('task')} 인데 --task {task} 로 채점 시도")
        name = d.get("annotator") or Path(path).stem
        if name in aliases:
            print(f"  약칭 정규화: {name} → {aliases[name]}")
            name = aliases[name]
        # 앱이 calibration/main을 따로 내보낸 경우 같은 어노테이터의 파일이 여러 개다 → 병합.
        # 같은 항목에 서로 다른 라벨이 오면 어느 쪽이 최신인지 알 수 없으므로 중단한다.
        ans = out.setdefault(name, {})
        n_new = 0
        for a in d.get("answers", []):
            if not a.get("label"):
                continue
            sec = a.get("section") or ("calibration" if a["id"] in overlap else "main")
            prev = ans.get(a["id"])
            if prev is not None and prev[1] != a["label"]:
                raise SystemExit(f"{name}: {a['id']} 라벨 충돌 ({prev[1]} vs {a['label']}) — {path}")
            if prev is None:
                n_new += 1
            ans[a["id"]] = (sec, a["label"])
        print(f"  {path}  annotator={name}  labeled=+{n_new} (누적 {len(ans)})"
              f"/{len(d.get('answers', []))}")
    if len(out) < 2:
        raise SystemExit("어노테이터가 2명 미만이면 일치도를 계산할 수 없습니다.")
    return out


def parser_item_qid() -> dict:
    """parser 항목 id(p0001…)는 qid를 담고 있지 않다 → id map으로 원본 CSV 행에서 가져온다."""
    id_map = json.load(open(PARSER_ID_MAP, encoding="utf-8"))
    cache, out = {}, {}
    for item_id, loc in id_map.items():
        if loc["csv"] not in cache:
            with open(loc["csv"], encoding="utf-8") as f:
                cache[loc["csv"]] = {r["idx"]: r["qid"] for r in csv.DictReader(f)}
        out[item_id] = cache[loc["csv"]].get(loc["idx"])
    return out


def restrict_to_qa(res: dict, task: str, qa_pairs: str) -> dict:
    """감사 항목을 분석 대상 QA 의 qid 로 제한.

    proxy/iaa 는 항목 id 의 2번째 토큰이 qid (model__qid[__hN__cM]), parser 는 id map 경유.
    """
    keep = load_qids(qa_pairs)
    qid_of = ((lambda i: parser_qid.get(i)) if task == "parser"
              else (lambda i: i.split("__")[1]))
    parser_qid = parser_item_qid() if task == "parser" else {}
    out = {n: {i: v for i, v in items.items() if qid_of(i) in keep}
           for n, items in res.items()}
    n_before = len(set().union(*(set(v) for v in res.values())))
    n_after = len(set().union(*(set(v) for v in out.values())))
    print(f"  분석 대상 QA 로 제한: {n_before} → {n_after} 항목")
    return out


def proxy_rates(item_ids: list[str], gold: dict, n_boot: int = 10000, seed: int = 42) -> dict:
    """보고서 §5 비율 — main+calibration 전체, tie는 분모에만 든다.

    CI는 **qid 클러스터** bootstrap이다. pattern b/d 쌍은 hop0 sub_q가 같아 같은 passage를
    보므로 항목을 독립 표본으로 볼 수 없다 (§7.3). 클러스터 = qid.
    """
    lab = {i: gold.get(i, "tie") for i in item_ids}
    by_qid = defaultdict(list)
    for i in lab:
        by_qid[i.split("__")[1]].append(i)
    qids = sorted(by_qid)

    def rates(items):
        n = len(items)
        c = Counter(lab[i] for i in items)
        real = n - c["parser_fp"]
        return {"parser_fp": c["parser_fp"] / n if n else 0.0,
                "proxy_fp": c["proxy_fp"] / real if real else 0.0,
                "overabstain": c["overabstain"] / real if real else 0.0}

    point = rates(list(lab))
    rng = random.Random(seed)
    draws = defaultdict(list)
    for _ in range(n_boot):
        items = [i for q in (rng.choice(qids) for _ in qids) for i in by_qid[q]]
        for k, v in rates(items).items():
            draws[k].append(v)
    out = {}
    c = Counter(lab.values())
    n = len(lab)
    for k, v in point.items():
        d = sorted(draws[k])
        out[k] = {"k": c[k], "n": n if k == "parser_fp" else n - c["parser_fp"], "rate": v,
                  "ci": [d[int(0.025 * n_boot)], d[int(0.975 * n_boot)]]}
    out["dist"] = dict(c)
    out["n_clusters"] = len(qids)
    return out


def judge_labels() -> dict:
    """IAA 항목 id → LLM judge label. 24_export_annotation_json.py의 id 규칙과 동일."""
    jl = {}
    for r in map(json.loads, open(IAA_CALIB, encoding="utf-8")):
        jl[f"{r['model']}__{r['qid']}__h{r['hop']}__c{r['claim_idx']}"] = r["label"]
    for r in map(json.loads, open(IAA_MAIN, encoding="utf-8")):
        for h in r.get("hops", []):
            for ci, c in enumerate(h.get("claims", []), start=1):
                jl[f"{r['model']}__{r['qid']}__h{h['hop']}__c{ci}"] = c["label"]
    return jl


# ── 채점 ──────────────────────────────────────────────────────────────────────
def majority(votes: list[str]):
    c = Counter(votes).most_common()
    if len(c) > 1 and c[0][1] == c[1][1]:
        return None, c          # tie → adjudication
    return c[0][0], c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["iaa", "proxy", "parser"], required=True)
    ap.add_argument("--results", nargs="+", required=True, help="앱 결과 JSON (글롭 가능)")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-gold", default=None,
                    help="다수결 확정 라벨을 앱 결과 형식으로 저장 (27 merge / score 투입용, tie 제외)")
    ap.add_argument("--out-disagreements", default=None, help="tie/불일치 항목 CSV")
    ap.add_argument("--qa-pairs", default=None,
                    help="분석 대상 QA JSONL (지정 시 이 파일의 qid 로 항목 제한)")
    ap.add_argument("--n-boot", type=int, default=10000, help="bootstrap 반복 수")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels = LABELS[args.task]
    gate_sec = GATE_SECTION[args.task]
    print(f"[{args.task}] 결과 파일 로드")
    res = load_results(args.results, args.task)
    if args.qa_pairs:
        res = restrict_to_qa(res, args.task, args.qa_pairs)
    annotators = sorted(res)
    n_ann = len(annotators)

    # 전원이 라벨한 항목만 Fleiss 대상 (Fleiss는 항목별 평정자 수 동일 가정)
    all_ids = set().union(*(set(a) for a in res.values()))
    complete = [i for i in sorted(all_ids) if all(i in res[a] for a in annotators)]
    dropped = len(all_ids) - len(complete)
    section = {i: next(res[a][i][0] for a in annotators if i in res[a]) for i in all_ids}
    print(f"\n어노테이터 {n_ann}명 | 공통 라벨 항목 {len(complete)} (부분 라벨 {dropped}개 제외)")

    report = {"task": args.task, "annotators": annotators,
              "n_complete": len(complete), "n_dropped_partial": dropped}

    # 1) Fleiss κ (section별)
    print("\n── Fleiss' κ ──")
    for sec in ("calibration", "main"):
        ids = [i for i in complete if section[i] == sec]
        if not ids:
            continue
        rows = [[res[a][i][1] for a in annotators] for i in ids]
        k, p_o, p_e = fleiss_parts(rows, labels)
        ci = fleiss_boot_ci(rows, labels, args.n_boot, args.seed) if sec == gate_sec else None
        cistr = f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else ""
        print(f"  {sec:<12} n={len(ids):>4}  κ = {k:.3f}{cistr}   p_o = {p_o:.3f}  p_e = {p_e:.3f}"
              if k is not None else f"  {sec}: κ 계산 불가")
        report[f"fleiss_{sec}"] = k
        report[f"fleiss_{sec}_parts"] = {"p_o": p_o, "p_e": p_e, "n": len(ids), "ci": ci}
        if sec == gate_sec:
            for l in labels:                      # 라벨별 one-vs-rest
                bin_rows = [[(x if x == l else "_other") for x in r] for r in rows]
                kb, bo, be = fleiss_parts(bin_rows, [l, "_other"])
                bci = fleiss_boot_ci(bin_rows, [l, "_other"], args.n_boot, args.seed)
                obs = sum(r.count(l) for r in rows)
                bcistr = f" [{bci[0]:.3f}, {bci[1]:.3f}]" if bci else ""
                print(f"      · {l:<13} κ={kb:.3f}{bcistr}" if kb is not None
                      else f"      · {l:<13} κ=n/a",
                      f" p_o={bo:.3f} p_e={be:.3f} (표결 {obs}/{len(ids)*n_ann})")
                report[f"fleiss_{gate_sec}_{l}"] = kb
                report[f"fleiss_{gate_sec}_{l}_parts"] = {"p_o": bo, "p_e": be, "ci": bci}

    # 2) pairwise Cohen κ
    print(f"\n── 쌍별 Cohen's κ ({gate_sec}) ──")
    gate_ids = [i for i in complete if section[i] == gate_sec]
    report["cohen_pairwise"] = {}
    for x in range(n_ann):
        for y in range(x + 1, n_ann):
            a, b = annotators[x], annotators[y]
            ka = cohen_kappa([res[a][i][1] for i in gate_ids],
                             [res[b][i][1] for i in gate_ids], labels)
            print(f"  {a} × {b}: κ = {ka:.3f}" if ka is not None else f"  {a} × {b}: n/a")
            report["cohen_pairwise"][f"{a}|{b}"] = ka

    # 3) 다수결
    gold, ties = {}, []
    for i in complete:
        votes = [res[a][i][1] for a in annotators]
        m, c = majority(votes)
        if m is None:
            ties.append((i, section[i], votes))
        else:
            gold[i] = m
    # main shard는 라벨러별 disjoint라 단일 라벨이다 → gold 출력에는 포함(일치도 계산에는 미포함)
    single = {i: next(res[a][i][1] for a in annotators if i in res[a])
              for i in all_ids if i not in complete
              and sum(i in res[a] for a in annotators) == 1}
    print(f"\n── 다수결 ── 확정 {len(gold)} | tie(adjudication 필요) {len(ties)}"
          f" | 단일 라벨(disjoint shard) {len(single)}")
    for i, sec, v in ties[:10]:
        print(f"  TIE  {i} ({sec}): {v}")
    if len(ties) > 10:
        print(f"  ... 외 {len(ties)-10}건")
    report["n_gold"] = len(gold)
    report["n_tie"] = len(ties)

    # 4) task별 검증
    disagreements = []
    if args.task == "iaa":
        jl = judge_labels()
        ids = [i for i in gate_ids if i in gold and i in jl]
        hum = [gold[i] for i in ids]
        jud = [jl[i] for i in ids]
        agree = sum(h == j for h, j in zip(hum, jud))
        k = cohen_kappa(hum, jud, labels)
        p, lo, hi = wilson(agree, len(ids))
        print(f"\n── LLM judge 검증 (main, n={len(ids)}) ──")
        print(f"  일치율 {agree}/{len(ids)} = {p:.3f} [{lo:.3f},{hi:.3f}]")
        print(f"  Cohen's κ (judge × human 다수결) = {k:.3f}" if k is not None else "  κ n/a")
        print("  혼동행렬 (행=judge, 열=human)")
        print("           " + "".join(f"{l:>14}" for l in labels))
        for jlab in labels:
            row = [sum(1 for h, j in zip(hum, jud) if j == jlab and h == hlab) for hlab in labels]
            print(f"  {jlab:<9}" + "".join(f"{v:>14}" for v in row))
        report["judge_agreement"] = {"n": len(ids), "agree": agree, "rate": p,
                                     "ci": [lo, hi], "cohen_kappa": k}
        # FaithRate가 실제로 쓰는 구분은 supported vs 나머지뿐이다. unsupported↔contradicted
        # 경계(=PDR/ContraRate)는 인간 일치도가 낮아 논문 지표로 쓰지 않는다.
        # 이진화 후 다수결. 3분류 tie(sup/unsup/contra 3파전)도 이진에서는 2:1로 확정되므로
        # 3분류 gold를 축약하지 않고 원표를 다시 집계한다.
        bin_ids = [i for i in gate_ids if i in jl
                   and all(i in res[a] for a in annotators)]
        bz = lambda x: "sup" if x == "supported" else "not"
        bh = [Counter(bz(res[a][i][1]) for a in annotators).most_common(1)[0][0] for i in bin_ids]
        bj = [bz(jl[i]) for i in bin_ids]
        nb = len(bin_ids)
        bagree = sum(x == y for x, y in zip(bh, bj))
        bk = cohen_kappa(bh, bj, ["sup", "not"])
        bp, blo, bhi = wilson(bagree, nb)
        print(f"\n  [이진: supported vs 나머지 — FaithRate 기준, n={nb}]")
        print(f"  일치율 {bagree}/{nb} = {bp:.3f} [{blo:.3f},{bhi:.3f}]"
              + (f"   Cohen's κ = {bk:.3f}" if bk is not None else ""))
        print(f"  FaithRate  judge {sum(x=='sup' for x in bj)/nb:.3f}"
              f"  human {sum(x=='sup' for x in bh)/nb:.3f}")
        report["judge_agreement_binary"] = {"n": nb, "agree": bagree, "rate": bp,
                                            "ci": [blo, bhi], "cohen_kappa": bk}
        disagreements = [(i, jl[i], gold[i]) for i in ids if jl[i] != gold[i]]
        print(f"  3분류 불일치 {len(disagreements)}건 / 이진 불일치 {nb-bagree}건")
    elif args.task == "proxy":
        ids = [i for i in gate_ids if i in gold]
        dist = Counter(gold[i] for i in ids)
        print(f"\n── proxy 판정 분포 (main, n={len(ids)}) ──")
        for l in labels:
            print(f"  {l:<13} {dist.get(l,0):>4}  ({dist.get(l,0)/max(len(ids),1):.1%})")
        # proxy-FP rate: proxy가 retrieval 성공이라 부른 케이스 중 실제 오탐 비율
        denom = dist.get("proxy_fp", 0) + dist.get("overabstain", 0)
        p, lo, hi = wilson(dist.get("proxy_fp", 0), denom)
        print(f"\n  proxy-FP rate (ambiguous/parser_fp 제외, n={denom})"
              f" = {p:.3f} [{lo:.3f},{hi:.3f}]")
        # 보고 기준은 gate 섹션(main)뿐이다 — calibration은 draft 라벨별 층화 추출한
        # 경계 사례 집합이라 비율 분모에 넣으면 borderline/split을 과대 대표한다.
        # (IAA judge 검증이 main 한정인 것과 같은 논리)
        rate_ids = [i for i in complete if section[i] == gate_sec]
        pr = proxy_rates(rate_ids, gold)
        print(f"\n── 판정 비율 ({gate_sec} n={len(rate_ids)}, tie는 분모만) ──"
              f"  qid 클러스터 bootstrap B=10,000 seed=42, 클러스터 {pr['n_clusters']}개")
        for key, name in [("parser_fp", "파서FP (기권 아님)"), ("proxy_fp", "proxy-FP"),
                          ("overabstain", "과기권")]:
            r = pr[key]
            print(f"  {name:<18} {r['k']:>3}/{r['n']:<3} = {r['rate']:.3f}"
                  f" [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]")
        report["proxy_dist"] = dict(dist)
        report["proxy_fp_rate"] = {"k": dist.get("proxy_fp", 0), "n": denom, "rate": p, "ci": [lo, hi]}
        report["proxy_rates"] = pr

        lab = {i: gold.get(i, "split") for i in rate_ids}
        votes = {i: [res[a][i][1] for a in annotators] for i in rate_ids}
        cats = ["parser_fp", "proxy_fp", "ambiguous", "overabstain", "split"]
        calib = Counter(i.split("__")[0] for i in complete if section[i] != gate_sec)
        print(f"\n── 보고 기준: {gate_sec} n={len(rate_ids)} "
              f"(calibration {sum(calib.values())}건 제외: {dict(calib)}) ──")

        # 적대적 상한: borderline·split을 전부 모델에 불리하게(=과기권) 해소
        real = len(lab) - sum(1 for v in lab.values() if v == "parser_fp")
        n_oa = sum(1 for v in lab.values() if v == "overabstain")
        n_adv = n_oa + sum(1 for v in lab.values() if v in ("ambiguous", "split"))
        n_unan = sum(1 for i, v in votes.items() if len(set(v)) == 1 and v[0] == "overabstain")
        print(f"\n── 과기권 판정 규칙별 (분모 = 진짜 기권 {real}) ──")
        print(f"  3인 만장일치      {n_unan}/{real} = {n_unan/real:.1%}")
        print(f"  다수결            {n_oa}/{real} = {n_oa/real:.1%}")
        print(f"  1표 이상          {sum(1 for v in votes.values() if 'overabstain' in v)}/{real}")
        print(f"  적대적 상한       {n_adv}/{real} = {n_adv/real:.1%}  (borderline·split 전부 과기권 해소)")
        report["overabstain_rules"] = {"unanimous": n_unan, "majority": n_oa,
                                       "any_vote": sum(1 for v in votes.values() if "overabstain" in v),
                                       "adversarial_upper": n_adv, "denom": real}

        # 모델별 (Table 9)
        print(f"\n── 모델별 판정 (다수결) ──")
        print(f"  {'model':<8}{'n':>5}" + "".join(f"{c:>13}" for c in cats) + f"{'적대적상한':>10}")
        by_model = defaultdict(Counter)
        for i, v in lab.items():
            by_model[i.split("__")[0]][v] += 1
        for m_ in by_model:                      # 적대적 상한 = borderline·split도 과기권으로
            c_ = by_model[m_]
            by_model[m_]["_adv"] = c_["overabstain"] + c_["ambiguous"] + c_["split"]
            by_model[m_]["_real"] = sum(c_[x] for x in cats) - c_["parser_fp"]
        for m in sorted(by_model):
            c = by_model[m]
            print(f"  {m:<8}{sum(c[x] for x in cats):>5}" + "".join(f"{c[x]:>13}" for x in cats)
                  + f"{c['_adv']:>8}/{c['_real']:<4}")
        tot = Counter(lab.values())
        print(f"  {'계':<8}{len(lab):>5}" + "".join(f"{tot[x]:>13}" for x in cats)
              + f"{sum(by_model[m]['_adv'] for m in by_model):>8}/"
                f"{sum(by_model[m]['_real'] for m in by_model):<4}")
        report["proxy_by_model"] = {m: dict(c) for m, c in by_model.items()}

        # 불일치 방향 — 어노테이터 쌍별 교차표
        print(f"\n── 불일치 방향 (행 = 앞 어노테이터) ──")
        for x, y in [(annotators[i], annotators[j])
                     for i in range(n_ann) for j in range(i + 1, n_ann)]:
            print(f"  [{x} \\ {y}]  " + "".join(f"{c:>13}" for c in labels))
            for rl in labels:
                row = [sum(1 for i in rate_ids if res[x][i][1] == rl and res[y][i][1] == cl)
                       for cl in labels]
                print(f"  {rl:<13}" + "".join(f"{v:>13}" for v in row))
        report["pair_confusion"] = {
            f"{x}|{y}": {rl: {cl: sum(1 for i in rate_ids
                                      if res[x][i][1] == rl and res[y][i][1] == cl)
                              for cl in labels} for rl in labels}
            for i_ in range(n_ann) for j_ in range(i_ + 1, n_ann)
            for x, y in [(annotators[i_], annotators[j_])]}
        disagreements = [(i, "-", gold[i]) for i in ids if gold[i] == "ambiguous"]
    else:  # parser — calibration(overlap) κ 게이트. 통과해야 main shard 835홉을 시작한다.
        ids = [i for i in gate_ids if i in gold]
        dist = Counter(gold[i] for i in ids)
        print(f"\n── parser calibration 판정 분포 (다수결 확정 n={len(ids)}) ──")
        for l in labels:
            print(f"  {l:<9} {dist.get(l,0):>4}  ({dist.get(l,0)/max(len(ids),1):.1%})")
        print(f"  unclear는 P/R 계산에서 제외된다 (parser_abstain_audit_prep.py score)")
        k = report.get("fleiss_calibration")
        gate = KAPPA_GATE["parser"]
        print(f"\n── κ 게이트 (RUBRIC.md 절차 2) ──")
        if k is None:
            print("  κ 계산 불가 — 라벨이 한 종류뿐이거나 공통 항목이 없다")
        elif k >= gate:
            print(f"  Fleiss κ = {k:.3f} ≥ {gate} → 통과. tie/불일치 adjudication 후 main shard 진행")
        else:
            print(f"  Fleiss κ = {k:.3f} < {gate} → 미달. rubric 보강 후 calibration 재라벨 "
                  f"(main 835홉 시작하지 말 것)")
        report["parser_dist"] = dict(dist)
        report["kappa_gate"] = {"kappa": k, "gate": gate, "pass": (k is not None and k >= gate)}
        disagreements = [(i, "-", gold[i]) for i in ids if gold[i] == "unclear"]

    # 출력
    if args.out_gold:
        Path(args.out_gold).parent.mkdir(parents=True, exist_ok=True)
        out_gold = {**single, **gold}
        json.dump({"task": args.task, "annotator": "majority",
                   "answers": [{"id": i, "section": section[i], "label": out_gold[i]}
                               for i in sorted(out_gold)]},
                  open(args.out_gold, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n→ {args.out_gold}  (다수결 {len(gold)} + 단일 {len(single)} = {len(out_gold)}건,"
              f" tie {len(ties)}건 제외)")
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(report, open(args.out_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\n→ {args.out_json}")
    if args.out_disagreements:
        Path(args.out_disagreements).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_disagreements, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "section", "reference", "human_majority", "votes"])
            for i, sec, v in ties:
                w.writerow([i, sec, "", "TIE", "|".join(v)])
            for i, ref, hm in disagreements:
                w.writerow([i, "main", ref, hm,
                            "|".join(res[a][i][1] for a in annotators)])
        print(f"→ {args.out_disagreements}")


if __name__ == "__main__":
    main()
