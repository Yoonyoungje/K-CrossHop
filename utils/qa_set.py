"""분석 대상 QA 집합 로더.

분석 스크립트는 `--qa-pairs` 로 받은 파일에 실린 qid 로만 레코드를 제한한다.
**무엇을 분석할지는 코드가 아니라 넘긴 파일이 정한다** — 어떤 부분집합을 쓰려면
그 부분집합만 담은 QA 파일을 만들어 넘기면 되고, 스크립트에 조건 분기를 두지 않는다.

annotated / trajectory 파일이 QA 파일보다 넓은 집합을 담고 있어도 (예: 별도 실험 QA가
섞인 산출물) 분석은 항상 넘긴 QA 파일 기준으로 정렬된다.
"""

import json
from pathlib import Path


def load_qids(qa_pairs) -> set:
    """QA JSONL 의 qid 집합."""
    return {json.loads(line)["qid"]
            for line in Path(qa_pairs).read_text(encoding="utf-8").splitlines()
            if line.strip()}


def restrict(records, qa_pairs, label="레코드", quiet=False):
    """records 를 qa_pairs 에 실린 qid 로 제한한다. qa_pairs 가 None 이면 그대로 반환."""
    if not qa_pairs:
        return records
    qids = load_qids(qa_pairs)
    before = len(records)
    kept = [r for r in records if r["qid"] in qids]
    if not quiet and len(kept) != before:
        print(f"  [{Path(qa_pairs).name}] {label} {before} → {len(kept)}개")
    return kept
