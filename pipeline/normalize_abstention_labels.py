#!/usr/bin/env python3
"""
pipeline/normalize_abstention_labels.py
=============================================
일회성 데이터 수리 — answer_parser 게이트 제거(markerless 길이게이트 → 무게이트)로
claim→abstain 단방향 전환된 홉만 기존 annotated_*.jsonl에 반영한다.
대상 디렉터리는 --ann-dir / --traj-dir 로 지정한다.

배경: utils/answer_parser.py 의 markerless fallback 경로에서 abstain 길이게이트
(<60/<80자)를 제거해 marker==1 경로와 동일 기준으로 통일했다. 변경은 monotonic
(claim→abstain 단방향)이므로 06을 재실행(전 홉 재판정, LLM 비결정성)하는 대신
영향받은 홉만 외과적으로 수정한다.

annotated 홉에는 generated_text가 없으므로 trajectory 파일과 (qid, hop)으로 join해
새 파서를 재실행한다. 같은 궤적의 분할 파일이 있으면 뒤쪽 우선으로 병합한다.

원본은 절대 수정하지 않는다. --apply 시 --out-dir(기본 {ann-dir}_normalized)에
전 파일을 복사 출력하며, 전환이 있는 파일은 수정본을, 없는/unsafe 파일은 원본
그대로를 쓴다(out-dir이 downstream에 그대로 쓸 수 있는 완결 집합이 되도록).

안전장치: 저장된 answer_marker_count와 새로 파싱한 marker_count가 한 홉이라도
불일치하거나(join 오류) unjoined 홉이 있으면 그 파일은 unsafe로 보고 수정하지 않고
원본을 그대로 복사한다.

수정 필드(06의 abstained-홉 스키마와 동일):
  abstained=True, claims=[], hop_faithfulness=None, has_hallucination=False,
  claim_extraction_source=pr.source, parse_issue=pr.parse_issue,
  answer_marker_count=pr.answer_marker_count, ambiguous_answer_marker=False
전환이 발생한 궤적은 top-level abstained(마지막 홉 기준)와
cross_lingual_error_propagation(compute_clepr)을 재계산한다.

dynamic_flags 등 알 수 없는 홉 키는 dict in-place 수정으로 보존한다.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.answer_parser import parse_generated_answer  # noqa: E402
from utils.metrics import valid_faith                    # noqa: E402

# annotated stem(접두사 annotated_ 제거 후) → trajectory stem 후보(뒤쪽 우선).
# 규칙적인 이름은 [core, core+'2']로 자동 해석되므로 불규칙 이름만 등록한다.
TRAJ_OVERRIDES = {
    "aya_ecological": ["aya_ecological", "aya_ecological"],
    "exaone_ecological": ["exaone_ecological"],
}


def resolve_traj(ann_stem: str, traj_dir: Path) -> dict[tuple[str, int], str]:
    """annotated stem → (qid, hop)→generated_text. 분할 파일 병합(뒤 우선)."""
    core = ann_stem[len("annotated_"):] if ann_stem.startswith("annotated_") else ann_stem
    candidates = TRAJ_OVERRIDES.get(core, [core, core + "2"])
    gt: dict[tuple[str, int], str] = {}
    for cand in candidates:
        for suffix in ("", "_smoke", "_patch"):
            p = traj_dir / f"trajectories_{cand}{suffix}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for hop in rec.get("trajectory", []):
                    gt[(rec["qid"], hop["hop"])] = hop.get("generated_text", "")
    return gt


def clepr_from_hops(hops: list[dict]) -> bool:
    """06.compute_clepr 와 동일 로직(dict 버전)."""
    for i in range(len(hops) - 1):
        h, nxt = hops[i], hops[i + 1]
        if not valid_faith(h.get("hop_faithfulness")) or not valid_faith(nxt.get("hop_faithfulness")):
            continue
        if h.get("lang_switch") and h.get("has_hallucination") and nxt.get("has_hallucination"):
            return True
    return False


def patch_file(ann_path: Path, traj_dir: Path) -> dict:
    """원본을 읽어 in-memory로만 수정. 디스크 쓰기는 호출측(main)이 out-dir에 수행."""
    raw = ann_path.read_text()
    gt = resolve_traj(ann_path.stem, traj_dir)
    if not gt:
        return {"stem": ann_path.stem, "skipped": "no_trajectory", "flips": 0, "raw": raw}

    records = [json.loads(l) for l in raw.splitlines() if l.strip()]
    flips = unjoined = marker_mismatch = monotonic_violations = 0

    for rec in records:
        rec_flipped = False
        for hop in rec["hops"]:
            key = (rec["qid"], hop["hop"])
            if key not in gt:
                unjoined += 1
                continue
            pr = parse_generated_answer(gt[key])
            if pr.answer_marker_count != hop.get("answer_marker_count"):
                marker_mismatch += 1
            if pr.abstained and not hop.get("abstained"):
                hop["abstained"] = True
                hop["claims"] = []
                hop["hop_faithfulness"] = None
                hop["has_hallucination"] = False
                hop["claim_extraction_source"] = pr.source
                hop["parse_issue"] = pr.parse_issue
                hop["answer_marker_count"] = pr.answer_marker_count
                hop["ambiguous_answer_marker"] = False
                flips += 1
                rec_flipped = True
            elif not pr.abstained and hop.get("abstained"):
                monotonic_violations += 1
        if rec_flipped:
            rec["abstained"] = rec["hops"][-1]["abstained"]
            rec["cross_lingual_error_propagation"] = clepr_from_hops(rec["hops"])

    unsafe = bool(unjoined or marker_mismatch or monotonic_violations)
    # 수정본을 쓸지(전환 있고 safe), 아니면 원본 복사를 쓸지 결정해 본문을 돌려준다.
    if flips and not unsafe:
        out_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n"
    else:
        out_text = raw

    return {"stem": ann_path.stem, "flips": flips, "unjoined": unjoined,
            "marker_mismatch": marker_mismatch, "monotonic_violations": monotonic_violations,
            "records": len(records), "unsafe": unsafe, "out_text": out_text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="out-dir에 복사본 출력(기본은 dry-run; 원본은 어떤 경우에도 미수정)")
    ap.add_argument("--ann-dir", default="data/annotated")
    ap.add_argument("--traj-dir", default="data/trajectories")
    ap.add_argument("--out-dir", default=None,
                    help="복사본 출력 디렉터리(기본 {ann-dir}_normalized)")
    args = ap.parse_args()

    ann_dir, traj_dir = Path(args.ann_dir), Path(args.traj_dir)
    out_dir = Path(args.out_dir) if args.out_dir else ann_dir.parent / f"{ann_dir.name}_normalized"
    files = sorted(ann_dir.glob("*.jsonl"))
    mode = f"APPLY→{out_dir}" if args.apply else "DRY-RUN"
    print(f"{mode}  ann={ann_dir}  traj={traj_dir}  ({len(files)}개)\n")
    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    any_unsafe = False
    for f in files:
        r = patch_file(f, traj_dir)
        if args.apply:  # 전 파일을 out-dir에 복사(수정본 또는 원본). 원본은 미수정.
            (out_dir / f.name).write_text(r.get("out_text", r.get("raw", "")))
        if r.get("skipped"):
            print(f"  - {r['stem']:30s} SKIP ({r['skipped']}) → 원본 복사")
            continue
        total += r["flips"]
        warn = ""
        for k in ("unjoined", "marker_mismatch", "monotonic_violations"):
            if r[k]:
                warn += f"  ⚠{k}={r[k]}"
        if r["unsafe"]:
            any_unsafe = True
            warn += "  ⚠UNSAFE→원본복사"
        print(f"  - {r['stem']:30s} flips={r['flips']:3d}  (records={r['records']}){warn}")
    print(f"\n총 claim→abstain 전환: {total}")
    if any_unsafe:
        print("⚠ unsafe 파일은 join 불일치로 수정 제외(원본 복사). 매핑/패치 확인 필요.")
    if not args.apply and total:
        extra = "" if args.ann_dir == "data/annotated" else f" --ann-dir {args.ann_dir} --traj-dir {args.traj_dir}"
        print(f"적용하려면: python3 {Path(__file__).as_posix()} --apply{extra}")


if __name__ == "__main__":
    main()
