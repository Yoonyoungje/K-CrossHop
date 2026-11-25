#!/usr/bin/env python3
"""
analysis/bridge_type_subsets.py

bridge-surface QMC 항의 relation 구성 분석 (camera-ready) — 입력 분할기.

bridge 유형 = 스키마 hop0 관계가 가리키는 값의 유형 (Wikidata property 의 값 범위):
  person       P57 감독 · P161 출연 · P112 설립자 · P50 저자 · P84 건축가 · P61 발명자 · P175 연주자
  organization P176 제조사 · P178 개발사 · P54 소속팀 · P108 고용주 · P137 운영자
  location     P159 본사 소재지
  award        P166 수상

--qa-pairs 를 유형별 부분집합으로 쪼개 --out-dir 에 쓴다. 지표는 기존 스크립트로 계산한다:
  python3 analysis/yield_qmc_joint_ci.py --controlled C --query-matched Q --normalized N \\
      --qa-pairs <out-dir>/qa_bridge_person.jsonl --cluster-key path_cluster_id

--pair 로 (query-matched 궤적, normalized 궤적) 쌍을 주면, 유형별로 정규화가 hop1 bridge 문자열을
실제로 바꾼 비율도 보고한다 (두 조건은 hop0 을 공유하므로 hop1 bridge_insert_text 가 다르면 바뀐 것).

--instance-of 로 bridge 개체의 P31(instance of) 캐시를 주면 스키마 유형을 개체 단위로 확인한다:
bridge 개체가 Q5(human)인지로 human / nonhuman 부분집합을, human bridge 는 hop1 관계별 부분집합
(qa_bridge_human_<P..>.jsonl)과 educated_at(P69)을 뺀 부분집합(qa_bridge_human_noP69.jsonl)도 쓴다. 캐시 파일이 없을 때만 Wikidata API 로 조회해 만든다
(dataset/entity_cache.fetch_instance_of). 있으면 네트워크를 쓰지 않는다.

  python3 analysis/bridge_type_subsets.py --qa-pairs data/dataset/qa_pairs.jsonl \\
      --out-dir data/camera_ready/bridge_type \\
      --pair llama data/trajectories/llama_query_matched.jsonl data/trajectories/llama_normalized.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HUMAN = "Q5"

BRIDGE_TYPE = {
    "P57": "person", "P161": "person", "P112": "person", "P50": "person",
    "P84": "person", "P61": "person", "P175": "person",
    "P176": "organization", "P178": "organization", "P54": "organization",
    "P108": "organization", "P137": "organization",
    "P159": "location",
    "P166": "award",
}
TYPES = ("person", "organization", "location", "award")


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def hop1(traj):
    return next(h for h in traj["trajectory"] if int(h["hop"]) == 1)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def instance_of_report(qa, type_of, cache_path, out):
    """bridge 개체 P31 로 스키마 유형을 확인하고 human / nonhuman / human×hop1 관계 부분집합을 쓴다."""
    bridges = sorted({r["hops"][0]["gold_entity_qid"] for r in qa})
    if cache_path.exists():
        p31 = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        sys.path.insert(0, str(ROOT))
        from dataset.entity_cache import fetch_instance_of
        p31 = fetch_instance_of(bridges, cache_path)
    missing = [q for q in bridges if q not in p31]
    if missing:
        raise SystemExit(f"P31 캐시에 없는 bridge {len(missing)}개: {missing[:5]}")
    human = {r["qid"]: HUMAN in p31[r["hops"][0]["gold_entity_qid"]] for r in qa}

    paths = {}
    for r in qa:
        paths.setdefault(r["path_cluster_id"], r)
    print(f"\n개체 단위 확인: bridge 개체 P31 에 {HUMAN}(human)가 있는가 (entity path 수)")
    for t in TYPES:
        ps = [r for r in paths.values() if type_of[r["qid"]] == t]
        nh = sum(human[r["qid"]] for r in ps)
        print(f"  {t:<13} human {nh:>3}   nonhuman {len(ps) - nh:>3}")
    print("  스키마 유형과 어긋난 bridge (person 인데 human 아님 / person 이 아닌데 human):")
    for pid, r in sorted(paths.items()):
        if (type_of[r["qid"]] == "person") != human[r["qid"]]:
            h0 = r["hops"][0]
            print(f"    {pid:<30} {r['relations'][0]:<5} {h0['gold_entity_qid']:<10} P31={','.join(p31[h0['gold_entity_qid']])}")

    groups = {"human": [r for r in qa if human[r["qid"]]],
              "nonhuman": [r for r in qa if not human[r["qid"]]]}
    for rel in sorted({r["relations"][1] for r in groups["human"]}):
        groups[f"human_{rel}"] = [r for r in groups["human"] if r["relations"][1] == rel]
    # educated_at(P69) 경로는 템플릿 감사의 범위 불일치 템플릿과 겹친다 — 그 경로를 뺀 human 도 쓴다
    groups["human_noP69"] = [r for r in groups["human"] if r["relations"][1] != "P69"]
    print("\n개체 단위 부분집합 (궤적 / entity path)")
    for name, rows in groups.items():
        write_rows(out / f"qa_bridge_{name}.jsonl", rows)
        print(f"  {name:<13} {len(rows):>4} / {len({r['path_cluster_id'] for r in rows}):>3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-pairs", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pair", nargs=3, action="append", default=[],
                    metavar=("LABEL", "QMATCHED_TRAJ", "NORMALIZED_TRAJ"))
    ap.add_argument("--instance-of", help="bridge 개체 P31 캐시 JSON (없으면 Wikidata 에서 만든다)")
    args = ap.parse_args()

    qa = load(args.qa_pairs)
    type_of = {}
    for r in qa:
        rel0 = r["relations"][0]
        if rel0 not in BRIDGE_TYPE:
            raise SystemExit(f"bridge 유형 미정의 관계: {rel0} ({r['query_type']})")
        type_of[r["qid"]] = BRIDGE_TYPE[rel0]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print("유형별 부분집합 (궤적 / entity path)")
    for t in TYPES:
        rows = [r for r in qa if type_of[r["qid"]] == t]
        write_rows(out / f"qa_bridge_{t}.jsonl", rows)
        print(f"  {t:<13} {len(rows):>4} / {len({r['path_cluster_id'] for r in rows}):>3}")

    if args.pair:
        print("\nhop1 bridge 문자열이 정규화로 바뀐 비율 (유형별, hop1 index 언어별)")
        print(f"  {'model':<10} {'type':<13} {'KO-index':>14} {'EN-index':>14}")
    keep = set(type_of)
    for label, qm_path, nm_path in args.pair:
        qm = {r["qid"]: hop1(r) for r in load(qm_path) if r["qid"] in keep}
        nm = {r["qid"]: hop1(r) for r in load(nm_path) if r["qid"] in keep}
        cnt = defaultdict(Counter)
        for qid, h in nm.items():
            if qid not in qm:
                continue
            key = (type_of[qid], h["gold_lang"])
            cnt[key]["n"] += 1
            cnt[key]["changed"] += (h.get("bridge_insert_text") or "") != (qm[qid].get("bridge_insert_text") or "")
        for t in TYPES:
            cells = []
            for lang in ("ko", "en"):
                c = cnt[(t, lang)]
                cells.append(f"{c['changed']:>4}/{c['n']:<4} {c['changed']/c['n']:>5.1%}" if c["n"] else f"{'-':>14}")
            print(f"  {label:<10} {t:<13} {cells[0]:>14} {cells[1]:>14}")

    if args.instance_of:
        instance_of_report(qa, type_of, Path(args.instance_of), out)


if __name__ == "__main__":
    main()
