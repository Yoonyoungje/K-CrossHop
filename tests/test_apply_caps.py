"""dataset/apply_caps.py 가 PYTHONHASHSEED 와 무관하게 같은 결과를 내는지 (순서 독립 cap)."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FULL_IDS = ROOT / "data/construction/ids_410.json"


@pytest.mark.skipif(not FULL_IDS.exists(), reason="로컬 데이터 없음")
def test_cap_is_hash_seed_independent(tmp_path):
    kept = []
    for seed in ("0", "12345"):
        out = tmp_path / f"cap_{seed}.json"
        subprocess.run([sys.executable, str(ROOT / "dataset/apply_caps.py"), "--output", str(out)],
                       env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
                       capture_output=True, text=True)
        kept.append(json.loads(out.read_text(encoding="utf-8"))["kept_triple_ids"])
    assert kept[0] == kept[1]
    assert len(kept[0]) > 0
