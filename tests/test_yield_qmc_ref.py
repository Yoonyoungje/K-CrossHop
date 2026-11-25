"""
camera-ready analysis/yield_qmc_joint_ci.py --ref-* (같은 재표집 위 모델 간 차이) 테스트.

  - ref 를 자기 자신으로 주면 모든 Δ 가 0 이고 CI 도 [0, 0] 이다
  - Δ 점추정 = (이 모델 점추정) − (비교 모델 점추정)
  - ref 를 주지 않으면 차이 블록이 없다 (기존 출력 유지)
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
SCRIPT = os.path.join(ROOT, "analysis", "yield_qmc_joint_ci.py")


def _rec(qid, ko_yield, en_yield):
    hop = lambda lang, ok: {"gold_lang": lang, "abstained": False, "hop_faithfulness": 1.0 if ok else 0.0}  # noqa: E731
    return {"qid": qid, "hops": [hop("ko", ko_yield), hop("en", en_yield)]}


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def _model(tmp_path, name, seed):
    # qid 마다 (ctrl, qm, norm) 의 ko/en 정답 여부를 결정적으로 바꿔 서로 다른 분해를 만든다.
    out = []
    for cond, shift in (("ctrl", 0), ("qm", 1), ("norm", 2)):
        rows = [_rec(f"q{i}", (i + seed) % 3 != 0, (i + seed + shift) % 4 == 0) for i in range(40)]
        out.append(_write(tmp_path / f"{name}_{cond}.jsonl", rows))
    return out


def _run(args):
    r = subprocess.run([sys.executable, SCRIPT, *args], capture_output=True, text=True, cwd=ROOT, check=True)
    return r.stdout


def _point(out, key):
    return float(re.search(rf"^\s+{key}\s+([+-]\d\.\d+)", out, re.M).group(1))


def _setup(tmp_path):
    qa = _write(tmp_path / "qa.jsonl", [{"qid": f"q{i}", "path_cluster_id": f"p{i // 4}"} for i in range(40)])
    return qa, _model(tmp_path, "a", 0), _model(tmp_path, "b", 1)


def _args(a, qa, extra=()):
    return ["--controlled", a[0], "--query-matched", a[1], "--normalized", a[2], "--qa-pairs", qa,
            "--cluster-key", "path_cluster_id", "--n-boot", "300", "--seed", "7", *extra]


def test_ref_self_gives_zero_diff(tmp_path):
    qa, a, _ = _setup(tmp_path)
    out = _run(_args(a, qa, ["--ref-controlled", a[0], "--ref-query-matched", a[1], "--ref-normalized", a[2]]))
    rows = re.findall(r"^\s+Δ\w+\s+([+-]\d\.\d+)\s+\[([+-]\d\.\d+), ([+-]\d\.\d+)\]", out, re.M)
    assert len(rows) == 4
    assert all(float(x) == 0.0 for row in rows for x in row)


def test_ref_point_is_difference(tmp_path):
    qa, a, b = _setup(tmp_path)
    out_a, out_b = _run(_args(a, qa)), _run(_args(b, qa))
    out = _run(_args(a, qa, ["--ref-controlled", b[0], "--ref-query-matched", b[1], "--ref-normalized", b[2]]))
    for k in ("YL_ctrl", "stage1", "stage2", "residual"):
        assert abs(_point(out, f"Δ{k}") - (_point(out_a, k) - _point(out_b, k))) <= 0.0011


def test_no_ref_keeps_output(tmp_path):
    qa, a, _ = _setup(tmp_path)
    assert "paired 차이" not in _run(_args(a, qa))
