#!/usr/bin/env python3
"""
analysis/13_bootstrap_ci.py
==================
주요 지표 Bootstrap 95% CI 계산.

단위: QID (trajectory) 수준에서 resampling.

출력 지표:
  1. LSHD_end2end — gold_lang 기준 same-index FR − cross-index FR
  2. ΔCLEPR — pattern 기준 switch(b+c) − same(a+d)
  3. 패턴별 CLEPR (a/b/c/d)

실행:
  python analysis/13_bootstrap_ci.py \\
    --annotated data/annotated/annotated_{model}_dynamic.jsonl \\
    [--retrieval-hits  data/retrieval/retrieval_v3_{model}_dynamic_all_hops.jsonl] \\
    [--n-boot 10000] [--alpha 0.05]

  --retrieval-hits 생략 시 conditioned CI는 스킵된다.
"""

import argparse
import json
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.metrics import (clepr_by_pattern,
                           lshd, hop_faith, hop_yield, hop_claim_rates)
from utils.qa_set import restrict


def _gap_or_none(t: tuple) -> float | None:
    """utils.lshd 반환 튜플 → gap (nan이면 None, 부트스트랩 콜백 계약 유지)."""
    g = t[0]
    return None if math.isnan(g) else g


def _hops_at_index(sample: list[dict], hop_idx: int, hit_map: dict | None = None,
                   success: bool | None = None) -> list[dict]:
    """hop_idx 위치 홉만 단일 홉 레코드로 추출(+선택적 v3 조건). conditioned LSHD/yield용."""
    out = []
    for r in sample:
        if hit_map is not None and hit_map.get(r["qid"]) != success:
            continue
        if len(r["hops"]) <= hop_idx:
            continue
        out.append({"qid": r["qid"], "pattern": r.get("pattern"),
                    "hops": [r["hops"][hop_idx]]})
    return out

DEFAULT_N_BOOT = 10000
DEFAULT_ALPHA  = 0.05


def load_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def section(title: str):
    print(f"\n{'='*66}")
    print(title)
    print("=" * 66)


def fmt_ci(est: float, lo: float | None, hi: float | None, n: int = None) -> str:
    s = f"{est:+.3f}"
    if lo is not None and hi is not None:
        s += f"  [{lo:+.3f}, {hi:+.3f}]"
    else:
        s += "  [CI 계산 불가 — 샘플 부족]"
    if n is not None:
        s += f"  (n={n})"
    return s


# ─── 지표 함수 ────────────────────────────────────────────────────────────────

def lshd_end2end(sample: list[dict]) -> float | None:
    """LSHD_end2end = mean(same-index FR) − mean(cross-index FR). utils.lshd 단일 소스."""
    return _gap_or_none(lshd(sample, hop_faith))


def yield_lshd_end2end(sample: list[dict]) -> float | None:
    """Yield-LSHD = mean(same YR) − mean(cross YR), gold_lang 기준.

    기권·무claim→0, ambiguous→분모 제외 (utils.hop_yield). GATE-1 TOST용 보조지표.
    """
    return _gap_or_none(lshd(sample, hop_yield))


def yield_amplification(sample: list[dict]) -> float | None:
    """증폭 = Yield-LSHD − FR-LSHD (같은 resample에서 paired 차이).

    abstention이 cross-index에 더 집중될수록 양수. 같은 sample에서 두 LSHD를 동시에 계산해
    paired bootstrap → 두 지표의 상관을 반영한 차이 CI를 산출한다.
    """
    y = yield_lshd_end2end(sample)
    fr = lshd_end2end(sample)
    if y is None or fr is None:
        return None
    return y - fr


def delta_clepr_full(sample: list[dict]) -> float | None:
    """ΔCLEPR = CLEPR_switch(b+c) − CLEPR_same(a+d). utils.clepr_by_pattern 단일 소스."""
    c = clepr_by_pattern(sample)
    if c["switch"]["n"] == 0 or c["same"]["n"] == 0:
        return None
    return c["switch"]["rate"] - c["same"]["rate"]


def clepr_by_pat(sample: list[dict], pat: str) -> float | None:
    """단일 패턴 CLEPR = P(Hallu_1 | Hallu_0, pattern=pat). utils.clepr_by_pattern 단일 소스."""
    c = clepr_by_pattern(sample)
    return c[pat]["rate"] if c[pat]["n"] else None


def _hop_pdr(h: dict) -> float | None:
    m = hop_claim_rates(h)
    return m["pdr"] if m is not None else None


def dpdr_end2end(sample: list[dict]) -> float | None:
    """ΔPDR_end2end = mean(cross-index PDR) − mean(same-index PDR) = −LSHD(PDR)."""
    g = lshd(sample, _hop_pdr)[0]          # same − cross
    return None if math.isnan(g) else -g


# ─── PDR §10.10 헬퍼 ──────────────────────────────────────────────────────────

def _normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip().lower()


def _build_alias_set(qid: str, alias_cache: dict, gold_entity: str) -> set[str]:
    entry = alias_cache.get(qid)
    if not entry:
        return {_normalize(gold_entity)} if gold_entity else set()
    names = []
    for key in ("label_ko", "label_en", "wiki_title_ko", "wiki_title_en"):
        v = entry.get(key)
        if v:
            names.append(v)
    names.extend(entry.get("aliases_ko", []))
    names.extend(entry.get("aliases_en", []))
    return {_normalize(n) for n in names if n}


def build_pdr_data(traj_records: list[dict], hit_records: list[dict],
                   ann_records: list[dict], alias_cache: dict) -> dict[tuple, dict]:
    """(qid, hop) → {valid, correct, gold_lang} 사전 계산."""
    gen_text: dict[str, dict[int, str]] = defaultdict(dict)
    for r in traj_records:
        for step in r["trajectory"]:
            gen_text[r["qid"]][step["hop"]] = step.get("generated_text", "")

    abstained: dict[str, dict[int, bool]] = defaultdict(dict)
    for r in ann_records:
        for h in r["hops"]:
            abstained[r["qid"]][h["hop"]] = bool(
                h.get("abstained") or h.get("ambiguous_answer_marker")
            )

    data: dict[tuple, dict] = {}
    for r in hit_records:
        qid, hop = r["qid"], r["hop"]
        valid = (not r.get("answer_bearing_hit", True)
                 and not abstained.get(qid, {}).get(hop, False))
        correct = False
        if valid:
            gt = gen_text.get(qid, {}).get(hop, "")
            aliases = _build_alias_set(r.get("gold_entity_qid", ""), alias_cache,
                                       r.get("gold_entity", ""))
            t = _normalize(gt)
            correct = bool(gt and any(a in t for a in aliases if a))
        data[(qid, hop)] = {
            "valid": valid,
            "correct": correct,
            "gold_lang": r.get("gold_lang", "ko"),
        }
    return data


def _pdr_counts(sample, pdr_data, hop_idx):
    same_c = same_t = cross_c = cross_t = 0
    for r in sample:
        for h in r["hops"]:
            if hop_idx is not None and h["hop"] != hop_idx:
                continue
            entry = pdr_data.get((r["qid"], h["hop"]))
            if entry is None or not entry["valid"]:
                continue
            if entry["gold_lang"] == "en":
                cross_c += entry["correct"]; cross_t += 1
            else:
                same_c += entry["correct"]; same_t += 1
    return same_c, same_t, cross_c, cross_t


def same_pdr(sample: list[dict], pdr_data: dict, hop_idx: int | None = None) -> float | None:
    sc, st, _, _ = _pdr_counts(sample, pdr_data, hop_idx)
    return sc / st if st else None


def cross_pdr(sample: list[dict], pdr_data: dict, hop_idx: int | None = None) -> float | None:
    _, _, cc, ct = _pdr_counts(sample, pdr_data, hop_idx)
    return cc / ct if ct else None


def delta_pdr(sample: list[dict], pdr_data: dict, hop_idx: int | None = None) -> float | None:
    """ΔPDR = PDR(cross) − PDR(same), hop_idx=None이면 전체 홉."""
    sc, st, cc, ct = _pdr_counts(sample, pdr_data, hop_idx)
    if st == 0 or ct == 0:
        return None
    return cc / ct - sc / st


def lshd_conditioned_hop(sample: list[dict], hit_map: dict, hop_idx: int, success: bool) -> float | None:
    """LSHD_conditioned = FR(same) − FR(cross) at hop_idx, conditioned on hit_map. utils.lshd 단일 소스."""
    return _gap_or_none(lshd(_hops_at_index(sample, hop_idx, hit_map, success), hop_faith))


def yield_lshd_conditioned_hop(sample: list[dict], hit_map: dict, hop_idx: int, success: bool) -> float | None:
    """Yield-LSHD_conditioned = Yield(same) − Yield(cross) at hop_idx, conditioned on hit_map.
    utils.hop_yield 단일 소스 (FaithRate-conditioned 착시 보정용 yield 축)."""
    return _gap_or_none(lshd(_hops_at_index(sample, hop_idx, hit_map, success), hop_yield))


def delta_clepr_conditioned(sample: list[dict], hit_map: dict, success: bool) -> float | None:
    """ΔCLEPR conditioned on answer_bearing_hit. v3 필터 후 delta_clepr_full 위임."""
    sub = [r for r in sample if hit_map.get(r["qid"]) == success]
    return delta_clepr_full(sub)


# ─── Bootstrap runner ─────────────────────────────────────────────────────────

def ci(values: list[float], alpha: float) -> tuple[float, float]:
    lo = alpha / 2
    hi = 1 - lo
    s = sorted(values)
    n = len(s)
    return s[int(lo * n)], s[int(hi * n) - 1]


def bootstrap(records: list[dict], stat_fn, n_boot: int, alpha: float,
              cluster_map: dict | None = None,
              **kwargs) -> tuple[float | None, float | None, float | None, int]:
    """QID-level (기본) 또는 cluster-level resampling.

    cluster_map: bridge_cluster_id → [record, ...] 딕셔너리가 주어지면
                 클러스터 단위로 resample한다 (bridge 공유 상관 보정).
    """
    point = stat_fn(records, **kwargs)
    if point is None:
        return None, None, None, 0

    boot_stats = []
    if cluster_map is not None:
        cluster_ids = list(cluster_map.keys())
        n_clusters = len(cluster_ids)
        for _ in range(n_boot):
            sampled = random.choices(cluster_ids, k=n_clusters)
            resample = [r for cid in sampled for r in cluster_map[cid]]
            val = stat_fn(resample, **kwargs)
            if val is not None:
                boot_stats.append(val)
    else:
        for _ in range(n_boot):
            resample = random.choices(records, k=len(records))
            val = stat_fn(resample, **kwargs)
            if val is not None:
                boot_stats.append(val)

    if len(boot_stats) < n_boot * 0.9:
        return point, None, None, len(records)

    lo, hi = ci(boot_stats, alpha)
    return point, lo, hi, len(records)


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_all(records: list[dict], hit_map: dict, n_boot: int, alpha: float, model: str,
            pdr_data: dict | None = None, cluster_map: dict | None = None):
    mode = f"cluster n_cluster={len(cluster_map)}" if cluster_map else "QID-level"
    print(f"\n모델: {model}  (n={len(records)}, boot={n_boot:,}, α={alpha}, resample={mode})")

    # 1. LSHD_end2end
    section("1. LSHD_end2end  (same-index FR − cross-index FR, gold_lang 기준)")
    est, lo, hi, _ = bootstrap(records, lshd_end2end, n_boot, alpha, cluster_map=cluster_map)
    if est is not None:
        sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
        print(f"  LSHD_end2end = {fmt_ci(est, lo, hi)}  {sig}")
    else:
        print("  데이터 부족")

    # 1-B. Yield-LSHD (기권 포함, GATE-1 TOST용)
    section("1-B. Yield-LSHD  (same-index YR − cross-index YR, 기권→0 분모 포함)")
    est, lo, hi, _ = bootstrap(records, yield_lshd_end2end, n_boot, alpha, cluster_map=cluster_map)
    if est is not None:
        sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
        print(f"  Yield-LSHD = {fmt_ci(est, lo, hi)}  {sig}")
    else:
        print("  데이터 부족")

    # 1-C. 증폭 = Yield-LSHD − FR-LSHD (paired bootstrap)
    section("1-C. 증폭  (Yield-LSHD − FR-LSHD, paired)")
    est, lo, hi, _ = bootstrap(records, yield_amplification, n_boot, alpha, cluster_map=cluster_map)
    if est is not None:
        sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
        print(f"  증폭 = {fmt_ci(est, lo, hi)}  {sig}")
    else:
        print("  데이터 부족")

    # 2. ΔCLEPR (전체)
    section("2. ΔCLEPR = CLEPR_switch(b+c) − CLEPR_same(a+d)  [pattern 기준]")
    est, lo, hi, _ = bootstrap(records, delta_clepr_full, n_boot, alpha, cluster_map=cluster_map)
    if est is not None:
        sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
        print(f"  ΔCLEPR (전체) = {fmt_ci(est, lo, hi)}  {sig}")
    else:
        print("  데이터 부족")

    # 3. 패턴별 CLEPR
    section("3. 패턴별 CLEPR")
    LABELS = {"a": "KO→KO", "b": "EN→KO", "c": "KO→EN", "d": "EN→EN"}
    print(f"  {'패턴':<10} {'방향':<10} {'점추정':>8}  {'95% CI':^22}  sig")
    print("  " + "-" * 58)
    for pat in "abcd":
        est, lo, hi, n = bootstrap(records, clepr_by_pat, n_boot, alpha, cluster_map=cluster_map, pat=pat)
        if est is not None:
            sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
            print(f"  {pat:<10} {LABELS[pat]:<10} {fmt_ci(est, lo, hi, n)}  {sig}")
        else:
            print(f"  {pat:<10} {LABELS[pat]:<10}  N/A (샘플 부족)")

    # 4. ΔPDR_end2end
    section("4. ΔPDR_end2end  (cross-index PDR − same-index PDR, gold_lang 기준)")
    est, lo, hi, _ = bootstrap(records, dpdr_end2end, n_boot, alpha, cluster_map=cluster_map)
    if est is not None:
        sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
        print(f"  ΔPDR_end2end = {fmt_ci(est, lo, hi)}  {sig}")
    else:
        print("  데이터 부족")

    # 5. LSHD_conditioned (v3 있을 때만)
    if hit_map:
        section("5. LSHD_conditioned hop0  (v3=True 조건부, §9.4)")
        print(f"  {'조건':<20} {'점추정':>8}  {'95% CI':^22}  sig")
        print("  " + "-" * 58)
        est, lo, hi, _ = bootstrap(
            records, lshd_conditioned_hop, n_boot, alpha,
            cluster_map=cluster_map, hit_map=hit_map["hop0"], hop_idx=0, success=True
        )
        if est is not None:
            sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
            print(f"  {'hop0 v3=True':<20} {fmt_ci(est, lo, hi)}  {sig}")
        else:
            print(f"  {'hop0 v3=True':<20}  N/A (샘플 부족)")

        section("6. LSHD_conditioned hop1  (v3=True 조건부)")
        print(f"  {'조건':<20} {'점추정':>8}  {'95% CI':^22}  sig")
        print("  " + "-" * 58)
        est, lo, hi, _ = bootstrap(
            records, lshd_conditioned_hop, n_boot, alpha,
            cluster_map=cluster_map, hit_map=hit_map["hop1"], hop_idx=1, success=True
        )
        if est is not None:
            sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
            print(f"  {'hop1 v3=True':<20} {fmt_ci(est, lo, hi)}  {sig}")
        else:
            print(f"  {'hop1 v3=True':<20}  N/A (샘플 부족)")

        section("6-B. Yield-LSHD_conditioned  (v3=True 조건부, yield 축 — 착시 보정)")
        print(f"  {'조건':<20} {'점추정':>8}  {'95% CI':^22}  sig")
        print("  " + "-" * 58)
        for label, vm, hop_idx in [("hop0 v3=True", hit_map["hop0"], 0),
                                    ("hop1 v3=True", hit_map["hop1"], 1)]:
            est, lo, hi, _ = bootstrap(
                records, yield_lshd_conditioned_hop, n_boot, alpha,
                cluster_map=cluster_map, hit_map=vm, hop_idx=hop_idx, success=True
            )
            if est is not None:
                sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
                print(f"  {label:<20} {fmt_ci(est, lo, hi)}  {sig}")
            else:
                print(f"  {label:<20}  N/A (샘플 부족)")

        section("7. ΔCLEPR conditioned on hop0 answer_bearing_hit")
        print(f"  {'조건':<20} {'점추정':>8}  {'95% CI':^22}  sig")
        print("  " + "-" * 58)
        for label, filt in [("success=True", True), ("success=False", False)]:
            est, lo, hi, _ = bootstrap(
                records, delta_clepr_conditioned, n_boot, alpha,
                cluster_map=cluster_map, hit_map=hit_map["hop0"], success=filt
            )
            if est is not None:
                sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
                print(f"  {label:<20} {fmt_ci(est, lo, hi)}  {sig}")
            else:
                print(f"  {label:<20}  N/A (샘플 부족)")
    else:
        print("\n  (--retrieval-hits 미지정 — conditioned CI 스킵)")

    # 8. PDR §10.10 ΔPDR (trajectories + alias-cache 있을 때만)
    if pdr_data:
        section("8. PDR §10.10 — ΔPDR = PDR(cross) − PDR(same)  (v3=False, non-abstained)")
        print("  PDR = P(gold entity in gen_text | v3=False, non-abstained)")

        print(f"  {'지표':<14} {'점추정':>8}  {'95% CI':^22}  sig  n")
        print("  " + "-" * 58)
        for label, hop_idx in [("hop0", 0), ("hop1", 1), ("전체", None)]:
            sc, st, cc, ct = _pdr_counts(records, pdr_data, hop_idx)
            for name, fn, val, n in [
                ("same PDR", same_pdr, sc / st if st else None, st),
                ("cross PDR", cross_pdr, cc / ct if ct else None, ct),
                ("ΔPDR", delta_pdr, (cc / ct - sc / st) if (st and ct) else None, min(st, ct)),
            ]:
                _, lo, hi, _ = bootstrap(records, fn, n_boot, alpha,
                                         cluster_map=cluster_map,
                                         pdr_data=pdr_data, hop_idx=hop_idx)
                if val is not None:
                    sig = "(*)" if lo is not None and (lo > 0 or hi < 0) else "   "
                    tag = f"[{label}] {name}"
                    print(f"  {tag:<20} {fmt_ci(val, lo, hi)}  {sig}  n={n}")
                else:
                    print(f"  [{label}] {name:<14}  N/A (샘플 부족)")
            print()
    else:
        print("\n  (--trajectories / --alias-cache 미지정 — PDR §10.10 CI 스킵)")

    print("\n  (*) = 95% CI가 0을 포함하지 않음 → 유의")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotated",         required=True)
    parser.add_argument("--retrieval-hits",           default=None, help="선택: retrieval_v3_{model}.jsonl")
    parser.add_argument("--trajectories",      default=None, help="선택: trajectories_{model}.jsonl (PDR §10.10용)")
    parser.add_argument("--alias-cache",       default=None, help="선택: alias_cache.json (PDR §10.10용)")
    parser.add_argument("--qa-pairs",          default=None,
                        help="분석 대상 QA JSONL (지정 시 이 파일의 qid 로 제한, cluster bootstrap 에도 사용)")
    parser.add_argument("--cluster-bootstrap", action="store_true",
                        help="cluster bootstrap (--cluster-key 단위, --qa-pairs 필요)")
    parser.add_argument("--cluster-key", default="bridge_cluster_id",
                        choices=["bridge_cluster_id", "path_cluster_id"],
                        help="재표집 단위. path_cluster_id = entity path(triple) 250개 — 한 경로의 네 패턴을 "
                             "함께 뽑는다 (camera-ready 정본 CI)")
    parser.add_argument("--n-boot",            type=int,   default=DEFAULT_N_BOOT)
    parser.add_argument("--alpha",             type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--out", default=None,
                        help="콘솔 출력을 이 파일로도 저장 (tee, 미지정 시 화면만)")
    args = parser.parse_args()

    from utils.report import start_tee
    start_tee(args.out)

    random.seed(args.seed)
    records = load_jsonl(args.annotated)
    model = Path(args.annotated).stem.replace("annotated_", "")

    hit_map: dict = {}
    hit_records_raw: list[dict] = []
    if args.retrieval_hits:
        hit_records_raw = load_jsonl(args.retrieval_hits)
        hit_map = {
            "hop0": {r["qid"]: r.get("answer_bearing_hit") for r in hit_records_raw if r.get("hop", 0) == 0},
            "hop1": {r["qid"]: r.get("answer_bearing_hit") for r in hit_records_raw if r.get("hop", 0) == 1},
        }

    pdr_data: dict | None = None
    if args.trajectories and args.alias_cache and hit_records_raw:
        alias_cache = json.load(open(args.alias_cache, encoding="utf-8"))
        traj_records = load_jsonl(args.trajectories)
        pdr_data = build_pdr_data(traj_records, hit_records_raw, records, alias_cache)

    # 분석 대상 QA 로 제한
    records = restrict(records, args.qa_pairs)

    # cluster bootstrap: bridge_cluster_id 기준으로 record 그룹화
    cluster_map: dict | None = None
    if args.cluster_bootstrap:
        if not args.qa_pairs:
            parser.error("--cluster-bootstrap 사용 시 --qa-pairs 필요")
        qa_records = load_jsonl(args.qa_pairs)
        qid_to_cluster = {r["qid"]: r[args.cluster_key] for r in qa_records}
        cluster_map = defaultdict(list)
        for r in records:
            cid = qid_to_cluster.get(r["qid"])
            if cid is not None:
                cluster_map[cid].append(r)
        cluster_map = dict(cluster_map)
        print(f"[Cluster bootstrap] {args.cluster_key} {len(cluster_map)}개 클러스터")

    run_all(records, hit_map, args.n_boot, args.alpha, model,
            pdr_data=pdr_data, cluster_map=cluster_map)


if __name__ == "__main__":
    main()
