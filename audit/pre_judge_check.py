#!/usr/bin/env python3
"""
audit/pre_judge_check.py

생성 후 judge 전 사전 점검 — 프로젝트 문서 「생성 후 judge 전 사전 점검 (필수)」의 1단계 자동 스캔.

judge(06)는 과금이고 생성 산출물의 결함은 판정을 통과해도 사라지지 않는다. 판정 뒤에 발견하면
비용이 전액 재지출되므로, 06을 부르기 전에 반드시 이 스캔을 통과시킨다. 모델 호출 0회.

스캔은 결함을 **판정하지 않는다.** 축별로 세어 보여줄 뿐이고, 각 축이 내 구현 탓인지 모델
특성인지는 2단계(표본 육안)에서 사람이 가른다 — 전자는 고치고 재실행, 후자는 플래그로 남긴다.

  python3 audit/pre_judge_check.py --trajectories paper2/data/trajectories/trajectories_aya_r1.jsonl
  python3 audit/pre_judge_check.py --trajectories A.jsonl B.jsonl \\
      --anchors data/trajectories/aya_query_matched.jsonl   # gold 누출 대조군
  python3 audit/pre_judge_check.py --trajectories A.jsonl --show-samples 5 --axis truncated
"""

import argparse
import json
import re
from pathlib import Path

HANGUL = re.compile(r"[가-힣]")
TERMINAL = ("?", ".", "!", '"', ")", "]", ":", ";")
CAP_SUSPECT_CHARS = 200      # 무종결 + 이 길이 이상이면 max_tokens 도달로 본다


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def hops_of(rows: list[dict]) -> list[dict]:
    return [h for r in rows for h in r.get("trajectory", [])]


def scan(hops: list[dict]) -> dict:
    """축별 카운트. 각 축은 (해당 홉 수, 분모)."""
    q = lambda h: h.get("retrieval_query") or ""           # noqa: E731
    en = [h for h in hops if h.get("gold_lang") == "en"]
    h1 = [h for h in hops if h.get("hop") == 1]
    linked = [h for h in h1 if h.get("bridge_link_status") == "linked_success"]
    return {
        "빈 쿼리":                    (sum(1 for h in hops if not q(h).strip()), len(hops)),
        "쿼리 >200자":                (sum(1 for h in hops if len(q(h)) > 200), len(hops)),
        "무종결로 끝남": (sum(1 for h in hops if q(h).strip()
                                          and not q(h).rstrip().endswith(TERMINAL)), len(hops)),
        "└ 그중 상한 도달 의심(장문)": (sum(1 for h in hops if q(h).strip()
                                          and not q(h).rstrip().endswith(TERMINAL)
                                          and len(q(h)) >= CAP_SUSPECT_CHARS), len(hops)),
        "api_error":                 (sum(1 for h in hops if h.get("api_error")), len(hops)),
        "검색 패시지 0건":             (sum(1 for h in hops if not h.get("retrieved_passages")), len(hops)),
        "생성 텍스트 빈 문자열":         (sum(1 for h in hops if not (h.get("generated_text") or "").strip()), len(hops)),
        "실현 index ≠ gold_lang":     (sum(1 for h in hops
                                          if h.get("retrieval_index_used") not in (None, h.get("gold_lang"))), len(hops)),
        "패시지 언어 ≠ 실현 index":     (sum(1 for h in hops if h.get("retrieval_index_used") not in (None, "unified")
                                          and any(l != h["retrieval_index_used"] for l in h.get("retrieved_langs", []))), len(hops)),
        "EN 홉 쿼리에 한글 잔존":        (sum(1 for h in en if HANGUL.search(q(h))), len(en)),
        "hop1 anchor 치환 실패":       (sum(1 for h in h1 if h.get("anchor_replacement_success") is False), len(h1)),
        "링크 라벨이 최종 쿼리에 없음":    (sum(1 for h in linked if h.get("bridge_insert_text")
                                          and h["bridge_insert_text"] not in q(h)), len(linked)),
        "gold 정답이 쿼리에 포함(누출)":  (sum(1 for h in hops if h.get("query_contains_gold_answer")), len(hops)),
    }


AXES = {
    "truncated": lambda h: (h.get("retrieval_query") or "").strip()
                 and not (h.get("retrieval_query") or "").rstrip().endswith(TERMINAL)
                 and len(h.get("retrieval_query") or "") >= CAP_SUSPECT_CHARS,
    "long":      lambda h: len(h.get("retrieval_query") or "") > 200,
    "hangul":    lambda h: h.get("gold_lang") == "en" and HANGUL.search(h.get("retrieval_query") or ""),
    "leak":      lambda h: h.get("query_contains_gold_answer"),
    "link_lost": lambda h: h.get("bridge_link_status") == "linked_success"
                 and h.get("bridge_insert_text") and h["bridge_insert_text"] not in (h.get("retrieval_query") or ""),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", nargs="+", required=True)
    ap.add_argument("--anchors", nargs="*", default=[],
                    help="gold 누출 등 대조군 (정본 앵커). 신규 arm 고유 결함인지 가른다")
    ap.add_argument("--qa-pairs", default=None,
                    help="이 파일의 qid 로 제한해 센다. 대조군과 분모를 맞추기 위해 거의 항상 지정한다 "
                         "(정본 앵커 파일에는 core 밖 qid 가 섞여 있다)")
    ap.add_argument("--show-samples", type=int, default=0)
    ap.add_argument("--axis", choices=sorted(AXES), default="truncated")
    args = ap.parse_args()

    keep = None
    if args.qa_pairs:
        keep = {json.loads(l)["qid"] for l in
                Path(args.qa_pairs).read_text(encoding="utf-8").splitlines() if l.strip()}

    def rows_of(path: Path) -> list[dict]:
        rows = load(path)
        return [r for r in rows if r["qid"] in keep] if keep else rows

    scans = {}
    for p in args.trajectories:
        path = Path(p)
        scans[path.stem.replace("trajectories_", "")] = scan(hops_of(rows_of(path)))

    names = list(scans)
    w = max(28, *(len(k) for k in next(iter(scans.values()))))
    print("생성 후 judge 전 사전 점검 (프로젝트 문서 1단계 자동 스캔 · 모델 호출 0회)")
    print("=" * (w + 14 * len(names)))
    print(f"{'검사':{w}} " + " ".join(f"{n[:12]:>13}" for n in names))
    print("-" * (w + 14 * len(names)))
    for axis in next(iter(scans.values())):
        cells = []
        for n in names:
            c, d = scans[n][axis]
            cells.append(f"{c:>5}/{d:<5}" if c else f"{'0':>5}/{d:<5}")
        print(f"{axis:{w}} " + " ".join(f"{c:>13}" for c in cells))

    if args.anchors:
        print("\n대조군 (정본 앵커) — gold 정답 누출률")
        for p in args.anchors:
            hs = hops_of(rows_of(Path(p)))
            n = sum(1 for h in hs if h.get("query_contains_gold_answer"))
            print(f"  {Path(p).stem:44} {n:>5}/{len(hs)} = {n/len(hs):>6.1%}")

    if args.show_samples:
        f = AXES[args.axis]
        print(f"\n표본 — 축 '{args.axis}' (2단계 육안 확인용)")
        for p in args.trajectories:
            hits = [h for h in hops_of(rows_of(Path(p))) if f(h)]
            if not hits:
                continue
            print(f"\n[{Path(p).stem}] {len(hits)}건 중 {min(args.show_samples, len(hits))}건")
            for h in hits[:args.show_samples]:
                if h.get("rewrite_source_query"):
                    print(f"  src: {h['rewrite_source_query']}")
                print(f"  →    {(h.get('retrieval_query') or '')[:220]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
