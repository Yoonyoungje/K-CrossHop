"""
camera-ready bridge-only 조건 dynamic_subq_bridgenorm 단위 테스트.

  - hop0 는 controlled(dynamic_subq)와 같다 → controlled 궤적에서 hop0 를 seed 할 수 있다
  - hop1 쿼리 = KO 템플릿 + 정규화 bridge. 앞선 2×2(hit@5) 분석의 KO+gold 칸과 문자열이 같아야 한다
  - 생성 질문 = 검색 쿼리 (coupled, 기존 dynamic 세 조건과 같은 방식)
"""
import glob
import importlib.util
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

MODE = "dynamic_subq_bridgenorm"


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


runner = _load("pilot05_bn", "pipeline/05_pilot_runner.py")

QA_PATH = os.path.join(ROOT, "data/dataset/qa_pairs.jsonl")
CELLS = sorted(glob.glob(os.path.join(
    ROOT, "data/self_translate/template_bridge_2x2/retrieval_*.jsonl")))


def test_mode_meta():
    meta = runner._TRAJ_MODE_META[MODE]
    assert meta["template_policy"] == "ko_template"
    assert meta["bridge_insert_policy"] == "prev_answer_normalized"
    assert meta["trajectory_mode"] == "dynamic_bridge_only_normalized"


@pytest.mark.skipif(not (os.path.exists(QA_PATH) and CELLS), reason="로컬 데이터 없음")
def test_hop0_equals_controlled():
    for line in open(QA_PATH, encoding="utf-8"):
        qa = json.loads(line)
        h0 = qa["hops"][0]
        a = runner.resolve_hop_query(MODE, qa, 0, h0, "")
        b = runner.resolve_hop_query("dynamic_subq", qa, 0, h0, "")
        assert (a["query"], a["lang"], a["agent_question"]) == \
               (b["query"], b["lang"], b["agent_question"])


@pytest.mark.skipif(not (os.path.exists(QA_PATH) and CELLS), reason="로컬 데이터 없음")
def test_hop1_matches_2x2_ko_gold_cell():
    qa = {json.loads(l)["qid"]: json.loads(l) for l in open(QA_PATH, encoding="utf-8")}
    n = 0
    for path in CELLS:
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            q = qa[row["qid"]]
            r = runner.resolve_hop_query(MODE, q, 1, q["hops"][1], row["bridge_raw"])
            assert r["query"] == row["query::KO+gold"], (path, row["qid"])
            assert r["agent_question"] == r["query"]          # coupled
            assert r["lang"] == q["hops"][1]["gold_lang"]
            n += 1
    assert n > 1000
