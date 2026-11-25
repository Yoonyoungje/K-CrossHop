#!/usr/bin/env python3
"""
analysis/qualitative_cases.py — 모델별 정성 오류 분석 자료 (camera-ready, R3 "qualitative error analysis").

홉마다 (index class same/cross) × (answer-bearing 검색 R=1/0) × (결과)로 나눈다.
  결과: 지지(supported) = hop_yield 1 / 기권(abstain) = abstained / 근거 없는 답(unsupported) = 비기권이면서 yield 0.
        ambiguous 홉(hop_yield None)은 뺀다. 정의는 utils.metrics 공유 함수(index_class, hop_yield)를 따른다.
  R: retrieval_v3 의 answer_bearing_hit (§4.1 매처, 13·12 와 같은 입력).
hop1 은 앞 홉 답이 gold bridge 인지(prev_answer_is_gold_bridge)도 함께 센다.

--correctness (선택): 정답 정확도 축 hop 표(data/correctness/all/hops.csv, cond=controlled)의 correct(주 매처)를
붙인다 → judge 결과와 정답 여부가 어긋나는 홉(지지 안 됨∧정답, 지지됨∧오답)을 class 별로 센다.

출력(모델 하나): 유형별 비율 표 + cross-index 오류 유형별 사례 k 개(seed 고정 무작위) + hop1 잘못된 bridge 사례.

  python3 analysis/qualitative_cases.py --model llama --annotated A.jsonl --trajectories T.jsonl \\
      --retrieval-hits H.jsonl --qa-pairs data/dataset/qa_pairs.jsonl --k 3 > cases_llama.md
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import hop_yield, index_class  # noqa: E402

OUTCOMES = ("supported", "abstain", "unsupported")
ERROR_TYPES = {  # cross-index 홉에서 사례를 뽑는 오류 유형
    "R1-abstain": "답 근거를 검색했는데 기권 (missed evidence)",
    "R1-unsupported": "답 근거를 검색했는데 지지되지 않는 답 (grounding failure)",
    "R0-unsupported": "답 근거 없이 답함 (ungrounded answer)",
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def outcome(h):
    y = hop_yield(h)
    if y is None:
        return None
    if h.get("abstained"):
        return "abstain"
    return "supported" if y == 1.0 else "unsupported"


def clip(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + "…"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--annotated", required=True)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--retrieval-hits", required=True)
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--correctness", default=None, help="정답 정확도 축 hops.csv (선택)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    qa = {q["qid"]: q for q in load(args.qa_pairs)}
    ann = {r["qid"]: r for r in load(args.annotated) if r["qid"] in qa}
    traj = {t["qid"]: t for t in load(args.trajectories) if t["qid"] in qa}
    hit = {(r["qid"], r["hop"]): bool(r["answer_bearing_hit"]) for r in load(args.retrieval_hits) if r["qid"] in qa}

    corr = {}
    if args.correctness:
        with open(args.correctness, encoding="utf-8") as f:
            corr = {(r["qid"], int(r["hop"])): r["correct"] == "1" for r in csv.DictReader(f)
                    if r["model"] == args.model and r["cond"] == "controlled"}

    rows, missing = [], Counter()
    for qid, r in ann.items():
        for h in r["hops"]:
            o = outcome(h)
            if o is None:
                missing["ambiguous"] += 1
                continue
            if (qid, h["hop"]) not in hit:
                missing["R 없음"] += 1
                continue
            rows.append({"qid": qid, "hop": h["hop"], "cls": index_class(h), "R": hit[(qid, h["hop"])],
                         "out": o, "h": h, "correct": corr.get((qid, h["hop"]))})

    print(f"# {args.model} — 정성 오류 분석 자료\n")
    print(f"홉 {len(rows)}개 (제외: {dict(missing) or '없음'}). R = §4.1 answer-bearing hit@5.\n")
    print("## 유형별 비율 (index class 안에서 %)\n")
    print("| class | n | R1 지지 | R1 기권 | R1 근거 없는 답 | R0 지지 | R0 기권 | R0 근거 없는 답 |")
    print("|-|-|-|-|-|-|-|-|")
    for cls in ("same", "cross"):
        sub = [x for x in rows if x["cls"] == cls]
        c = Counter((x["R"], x["out"]) for x in sub)
        cells = [f"{100 * c[(R, o)] / len(sub):.1f}" for R in (True, False) for o in OUTCOMES]
        print(f"| {cls} | {len(sub)} | " + " | ".join(cells) + " |")

    if corr:
        print("\njudge 결과와 정답(주 매처)이 어긋나는 홉:")
        for cls in ("same", "cross"):
            sub = [x for x in rows if x["cls"] == cls and x["correct"] is not None]
            uc = sum(x["out"] == "unsupported" and x["correct"] for x in sub)
            si = sum(x["out"] == "supported" and not x["correct"] for x in sub)
            print(f"- {cls}: 지지 안 됨∧정답 {uc} / 지지 안 됨 {sum(x['out'] == 'unsupported' for x in sub)}, "
                  f"지지됨∧오답 {si} / 지지됨 {sum(x['out'] == 'supported' for x in sub)}")

    h1 = [x for x in rows if x["hop"] == 1 and not x["h"].get("prev_answer_abstained")]
    unmatched = [x for x in h1 if x["h"].get("prev_answer_is_gold_bridge") is False]
    print(f"\nhop1 (앞 홉 비기권 {len(h1)}개) 중 bridge 가 gold 정본 라벨과 다른 홉 {len(unmatched)}개 "
          f"({100 * len(unmatched) / max(len(h1), 1):.1f}%) — 표기 변형(정답 엔티티)·오답이 섞여 있다(Table 8 unmatched)")
    # cascade 사례: 앞 홉이 실제로 틀린 경우(정답 축). 정답 축이 없으면 정본 라벨 불일치로 대신한다.
    wrong = ([x for x in h1 if corr.get((x["qid"], 0)) is False] if corr else unmatched)
    cw = Counter(x["out"] for x in wrong)
    print(f"그중 앞 홉 답이 오답인 hop1 {len(wrong)}개 → 결과: " + ", ".join(f"{o} {cw[o]}" for o in OUTCOMES))

    rng = random.Random(args.seed)
    groups = [(t, d, [x for x in rows if x["cls"] == "cross" and f"R{int(x['R'])}-{x['out']}" == t])
              for t, d in ERROR_TYPES.items()]
    groups.append(("hop1-wrong-bridge", "앞 홉 오답이 bridge 로 넘어감 (cascade)", wrong))
    for t, desc, pool in groups:
        print(f"\n## {t} — {desc} (cross-index 후보 {len(pool)}개)" if t != "hop1-wrong-bridge"
              else f"\n## {t} — {desc} (후보 {len(pool)}개)")
        for x in rng.sample(sorted(pool, key=lambda x: (x["qid"], x["hop"])), min(args.k, len(pool))):
            q = qa[x["qid"]]
            th = next(t for t in traj[x["qid"]]["trajectory"] if t["hop"] == x["hop"])
            hq = q["hops"][x["hop"]]
            print(f"\n### {x['qid']} hop{x['hop']} (pattern {q['pattern']}, {x['cls']}, gold_lang {hq['gold_lang']})")
            print(f"- sub_q: {hq['sub_q']}")
            print(f"- 검색 쿼리: {th.get('retrieval_query')}")
            ok = {True: " · 정답", False: " · 오답", None: ""}[x["correct"]]
            print(f"- gold: {hq['gold_entity']} · R={int(x['R'])} · 결과={x['out']}{ok}")
            if x["hop"] == 1:
                print(f"- 앞 홉 답(bridge): {x['h'].get('prev_answer_raw')} "
                      f"(gold bridge {'맞음' if x['h'].get('prev_answer_is_gold_bridge') else '아님'})")
            print(f"- 생성: {clip(th.get('generated_text'), 300)}")
            labels = [c.get('label') for c in x['h'].get('claims', [])]
            print(f"- claim 라벨: {labels or '(claim 없음)'}")
            for i, p in enumerate((th.get("retrieved_passages") or [])[:3]):
                print(f"- 패시지 {i + 1}: {clip(p, 160)}")


if __name__ == "__main__":
    main()
