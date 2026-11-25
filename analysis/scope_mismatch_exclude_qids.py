"""scope_mismatch 3쌍 제외 민감도용 qid 목록 (dataset_gate_audit_followup §37).

§34.2의 3인 만장일치 scope_mismatch 3쌍(t07·t09·t21)을 hop0 또는 hop1에 쓰는 경로를 찾아
네 패턴 qid를 모두 제외 목록으로 낸다. 판정은 schema_id가 아니라 템플릿 문자열로 한다:
경로의 query_type → pipeline/03_generate_qa.py의 QA_TEMPLATES → (hop, sub_q, sub_q_en)이
세 (KO, EN) 쌍 중 하나와 같은지 본다. 대상 문자열은 검수 세트(template_main.jsonl)에서 읽는다.
교차 확인으로 qa_pairs_core의 KO 인스턴스 sub_q가 KO 템플릿 정규식과 맞는지도 본다.

읽기만 한다. 출력은 correctness_axis.py --exclude-qids 입력 형식(한 줄에 qid 하나).

  python3 analysis/scope_mismatch_exclude_qids.py
"""
import importlib.util
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QA = ROOT / "data/dataset/qa_pairs.jsonl"
MAIN = ROOT / "human_audit/template_main.jsonl"
OUT = ROOT / "data/correctness"
TARGETS = {"all3": ("t07", "t09", "t21"), "t21": ("t21",)}
EXPECTED = {"all3": 44, "t21": 39}

spec = importlib.util.spec_from_file_location("gq", ROOT / "pipeline/03_generate_qa.py")
gq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gq)

items = {d["id"]: d for d in map(json.loads, MAIN.open(encoding="utf-8"))}
pairs = {t: (items[t]["hop"], items[t]["ko_template"], items[t]["en_template"]) for t in ("t07", "t09", "t21")}


def ko_regex(tpl: str) -> re.Pattern:
    s = re.escape(tpl)
    for a, b in (("을\\(를\\)", "(?:을|를)"), ("이\\(가\\)", "(?:이|가)"), ("은\\(는\\)", "(?:은|는)")):
        s = s.replace(a, b)
    s = re.sub(r"\\\{\d\\\}", ".+", s)
    return re.compile(f"^{s}$")


def main():
    qa = [json.loads(l) for l in QA.open(encoding="utf-8")]
    by_path = defaultdict(list)
    for r in qa:
        by_path[r["path_cluster_id"]].append(r)

    # 같은 문자열을 쓰는 schema (전체 _SCHEMA_BASE_TEMPLATES 기준)
    print("대상 쌍과 같은 문자열을 쓰는 schema (전체 66 schema 기준)")
    for t, (hop, ko, en) in pairs.items():
        both = sorted(s for s, v in gq._SCHEMA_BASE_TEMPLATES.items() if v["sub_q"][hop] == ko and v["sub_q_en"][hop] == en)
        ko_only = sorted(s for s, v in gq._SCHEMA_BASE_TEMPLATES.items()
                         if any(v["sub_q"][h] == ko for h in (0, 1)) and s not in both)
        en_only = sorted(s for s, v in gq._SCHEMA_BASE_TEMPLATES.items()
                         if any(v["sub_q_en"][h] == en for h in (0, 1)) and s not in both)
        print(f"  {t} hop{hop} {ko!r} / {en!r}")
        print(f"    (KO, EN) 쌍 일치: {both}")
        print(f"    KO만 같음(다른 hop 포함): {ko_only or '없음'} · EN만 같음: {en_only or '없음'}")

    for tag, ts in TARGETS.items():
        hit_paths, hop_use, schema_use, regex_bad = set(), Counter(), Counter(), []
        for pc, rs in by_path.items():
            qts = {r["query_type"] for r in rs}
            for r in rs:
                tpl = gq.QA_TEMPLATES[r["query_type"]]
                for t in ts:
                    hop, ko, en = pairs[t]
                    if tpl["sub_q"][hop] == ko and tpl["sub_q_en"][hop] == en:
                        hit_paths.add(pc)
                        if not ko_regex(ko).match(r["hops"][hop]["sub_q"]):
                            regex_bad.append((r["qid"], hop, r["hops"][hop]["sub_q"]))
            for t in ts:
                hop, ko, en = pairs[t]
                if all(gq.QA_TEMPLATES[q]["sub_q"][hop] == ko and gq.QA_TEMPLATES[q]["sub_q_en"][hop] == en for q in qts):
                    hop_use[(t, hop)] += 1
                    schema_use[(t, re.sub(r"_[abcd]$", "", rs[0]["query_type"]))] += 1
        qids = sorted(r["qid"] for pc in hit_paths for r in by_path[pc])
        print(f"\n[{tag}] 대상 {ts} → 경로 {len(hit_paths)} (기대 {EXPECTED[tag]}) · qid {len(qids)}")
        for (t, hop), n in sorted(hop_use.items()):
            print(f"  {t}: hop{hop}에 쓰인 경로 {n}")
        hops = Counter(h for (_, h), n in hop_use.items() for _ in range(n))
        print(f"  hop별 합계: hop0 {hops[0]} · hop1 {hops[1]}")
        print(f"  schema별: {dict(sorted(schema_use.items()))}")
        print(f"  KO 인스턴스 sub_q ↔ KO 템플릿 정규식 불일치: {len(regex_bad)}{' ' + str(regex_bad[:3]) if regex_bad else ''}")
        if len(hit_paths) != EXPECTED[tag]:
            raise SystemExit(f"경로 수 {len(hit_paths)} ≠ 기대 {EXPECTED[tag]} — 멈춘다")
        out = OUT / f"scope_{tag}_exclude_qids.txt"
        out.write_text("\n".join(qids) + "\n", encoding="utf-8")
        print(f"  → {out.relative_to(ROOT)} · 남는 경로 {len(by_path) - len(hit_paths)}")


if __name__ == "__main__":
    main()
