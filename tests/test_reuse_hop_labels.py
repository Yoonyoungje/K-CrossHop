"""
seed 홉 judge 라벨 재사용 (`06 --reuse-hop-labels`) 단위 테스트. API 호출 0회.

  - 원본 trajectory 의 생성물과 같은 홉이면 batch 요청에서 빠지고 원본 라벨이 그대로 쓰인다
  - 생성물이 다르면 재사용하지 않고 정상 판정 대상으로 남는다
  - 로더가 (qid, 0) → 라벨 + 원본 홉을 만든다 (hop1 은 싣지 않는다)
  - hop1 은 생성물이 원본과 같아도 재채점 대상으로 남는다
  - 재사용 홉은 파서 판정(기권 등)도 원본을 따른다 (원본이 이전 파서로 만들어진 경우)
"""
import asyncio
import importlib.util
import json
import math
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location("ann06", os.path.join(ROOT, "pipeline/06_annotate.py"))
ann06 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ann06)

JUDGE = SimpleNamespace(model="stub-judge", _OPENAI_SEED=42, _SYSTEM="sys",
                        _PROMPT="{question}|{passages}|{claims}",
                        _PROMPT_NO_QUESTION="{passages}|{claims}")


def _traj(qid, gen0="답변: 베이징시", psg0=("p0",)):
    return {"qid": qid, "pattern": "d", "final_answer": "답변: 중국",
            "trajectory": [
                {"hop": 0, "gold_lang": "en", "generated_text": gen0,
                 "retrieved_passages": list(psg0), "retrieved_langs": ["en"],
                 "retrieval_success": True, "api_error": False},
                {"hop": 1, "gold_lang": "en", "generated_text": "답변: 중국",
                 "retrieved_passages": ["p1"], "retrieved_langs": ["en"],
                 "retrieval_success": True, "api_error": False},
            ]}


def _ann(qid, label="supported"):
    return {"qid": qid, "model": "m", "pattern": "d",
            "hops": [{"hop": 0, "claims": [{"text": "베이징시", "label": label}]},
                     {"hop": 1, "claims": [{"text": "중국", "label": label}]}]}


def _write(tmp_path, name, rows):
    p = tmp_path / name
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(p)


def _load(tmp_path, trajs, anns):
    return ann06.load_reused_hop_labels(_write(tmp_path, "a.jsonl", anns),
                                        _write(tmp_path, "t.jsonl", trajs))


def test_loader_keys_and_payload(tmp_path):
    reuse = _load(tmp_path, [_traj("q1")], [_ann("q1")])
    assert set(reuse) == {("q1", 0)}                               # hop0 만 — seed 범위
    assert reuse[("q1", 0)]["ann"]["claims"] == [{"text": "베이징시", "label": "supported"}]
    assert reuse[("q1", 0)]["source_hop"]["generated_text"] == "답변: 베이징시"


def test_reused_labels_requires_identical_generation(tmp_path):
    reuse = _load(tmp_path, [_traj("q1")], [_ann("q1")])
    same = _traj("q1")["trajectory"][0]
    assert [c.label for c in ann06._reused_labels(reuse, "q1", 0, same)] == ["supported"]
    assert ann06._reused_labels(reuse, "q1", 0, _traj("q1", gen0="답변: 다른 답")["trajectory"][0]) is None
    assert ann06._reused_labels(reuse, "q1", 0, _traj("q1", psg0=("다른 패시지",))["trajectory"][0]) is None
    assert ann06._reused_labels(None, "q1", 0, same) is None


def test_batch_skips_reused_hop_and_uses_its_labels(tmp_path):
    reuse = _load(tmp_path, [_traj("q1")], [_ann("q1", label="contradicted")])
    seeded = _traj("q1")
    seeded["trajectory"][1]["generated_text"] = "답변: 새 답"       # hop1 만 새로 판정 대상
    pending, parsed = ann06._collect_batch_requests([seeded], JUDGE, {}, reuse)
    assert [p["custom_id"] for p in pending] == ["q1__hop1"]        # hop0 은 요청에서 빠짐

    out = tmp_path / "out.jsonl"
    verdict = json.dumps([{"claim": "중국", "label": "supported"}])
    ann06._assemble_batch_output(parsed, {"q1__hop1": verdict}, {}, out, False, reuse)
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["hops"][0]["claims"] == [{"text": "베이징시", "label": "contradicted"}]
    assert got["hops"][0]["hop_faithfulness"] == 0.0               # 원본 라벨 그대로 반영
    assert got["hops"][1]["claims"][0]["label"] == "supported"     # hop1 은 batch 결과


def test_without_reuse_all_hops_are_judged(tmp_path):
    pending, _ = ann06._collect_batch_requests([_traj("q1")], JUDGE, {}, None)
    assert [p["custom_id"] for p in pending] == ["q1__hop0", "q1__hop1"]


def _ann_abstained(qid):
    """이전 파서가 기권으로 본 원본 hop0 (claim 없음) — 지금 파서는 같은 생성문에서 claim 을 뽑는다."""
    return {"qid": qid, "model": "m", "pattern": "d",
            "hops": [{"hop": 0, "abstained": True, "ambiguous_answer_marker": False,
                      "hop_faithfulness": None, "has_hallucination": False, "claims": [],
                      "claim_extraction_source": "fallback_abstain", "parse_issue": None,
                      "answer_marker_count": 0},
                     {"hop": 1, "claims": []}]}


def test_reused_hop_keeps_source_abstention_batch(tmp_path):
    """파서 판정이 원본과 갈려도 원본 기권을 그대로 쓴다 — 빈 라벨로 faith 0·환각을 만들지 않는다."""
    reuse = _load(tmp_path, [_traj("q1")], [_ann_abstained("q1")])
    pending, parsed = ann06._collect_batch_requests([_traj("q1")], JUDGE, {}, reuse)
    assert [p["custom_id"] for p in pending] == ["q1__hop1"]
    out = tmp_path / "out.jsonl"
    verdict = json.dumps([{"claim": "중국", "label": "supported"}])
    ann06._assemble_batch_output(parsed, {"q1__hop1": verdict}, {}, out, False, reuse)
    h0 = json.loads(out.read_text(encoding="utf-8"))["hops"][0]
    assert (h0["abstained"], h0["claims"], h0["hop_faithfulness"], h0["has_hallucination"]) == \
        (True, [], None, False)
    assert h0["claim_extraction_source"] == "fallback_abstain"


def test_reused_hop_keeps_source_abstention_sync(tmp_path):
    reuse = _load(tmp_path, [_traj("q1")], [_ann_abstained("q1")])
    calls = []

    async def judge_hop(claims, passages, question=None):
        calls.append(claims)
        return [ann06.ClaimLabel(c, "supported", "") for c in claims]

    ann = asyncio.run(ann06.annotate_trajectory(_traj("q1"), SimpleNamespace(judge_hop=judge_hop),
                                                {}, {}, reuse))
    h0 = ann.hops[0]
    assert h0.abstained is True and h0.claims == [] and not h0.has_hallucination
    assert math.isnan(h0.hop_faithfulness)
    assert calls == [["중국"]]                                       # hop1 만 판정


def test_identical_hop1_is_still_judged(tmp_path):
    """생성물·패시지가 원본과 같은 hop1 도 재사용하지 않는다 (승인 범위 = seed hop0)."""
    reuse = _load(tmp_path, [_traj("q1")], [_ann("q1")])
    same = _traj("q1")                                              # hop0·hop1 모두 원본과 동일
    assert ann06._reused_labels(reuse, "q1", 1, same["trajectory"][1]) is None
    pending, _ = ann06._collect_batch_requests([same], JUDGE, {}, reuse)
    assert [p["custom_id"] for p in pending] == ["q1__hop1"]
