#!/usr/bin/env python3
"""
analysis/self_translate_compare.py

self_translate 파일럿 모델 간 비교 표 (모델 호출 0회). 존재하는 궤적 파일만 집계한다.

  1) 모델마다 self_translate_check.py 를 돌려 재작성 CSV·점검 로그를 만든다 (이미 있으면 다시 만든다)
  2) 검색 쿼리(첫 줄) 형태를 질문 / 답 서술문 / 기타로 나누고, 인덱스 언어별로 표를 찍는다
     질문     : 물음표 포함
     답 서술문: 물음표 없음 ∧ (다·니다·요로 끝남 ∨ 한글 없이 마침표로 끝남)

  python3 analysis/self_translate_compare.py            # 기본 태그 _pilot50
  python3 analysis/self_translate_compare.py _pilot20    # 1차 파일럿
"""
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["llama", "aya", "exaone", "qwen", "sonnet", "gemini"]
TAG = sys.argv[1] if len(sys.argv) > 1 else "_pilot50"     # 예: _pilot50 / _pilot20
TRAJ = "data/self_translate/trajectories_{m}_self_tr" + TAG + ".jsonl"
CSV = "data/self_translate/rewrites_{m}" + TAG + ".csv"
LOG = "data/self_translate/logs/check_{m}" + TAG + ".txt"


def form(query: str) -> str:
    t = query.strip().strip('"“”\'').strip()
    if "?" in t or "？" in t:
        return "question"
    if re.search(r"(다|니다|요)[.!]?$", t) or (not re.search(r"[가-힣]", t) and t.endswith(".")):
        return "declarative"
    return "other"


def main():
    print(f"[태그 {TAG}] 재작성 홉 기준 (KO-index 생략 홉 제외)")
    print("| 모델 | 인덱스 | 목표 언어 | 질문 | 답 서술문 | 키워드·기타 | 여러 줄·머리말 | 빈 출력 | "
          "누출 V3 (V1) | finish=length |")
    print("|-|-|-|-|-|-|-|-|-|-|")
    for m in MODELS:
        traj = ROOT / TRAJ.format(m=m)
        if not traj.exists() or not traj.read_text(encoding="utf-8").strip():
            continue
        with open(ROOT / LOG.format(m=m), "w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(ROOT / "analysis/self_translate_check.py"),
                            "--trajectories", TRAJ.format(m=m), "--out-csv", CSV.format(m=m),
                            "--examples", "20"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = list(csv.DictReader(open(ROOT / CSV.format(m=m), encoding="utf-8-sig")))
        rows = [r for r in rows if r["status"] != "skipped_ko_index"]
        for lang in ("ko", "en"):
            s = [r for r in rows if r["index_lang"] == lang]
            n = len(s)
            if not n:
                continue
            pct = lambda k: f"{k}/{n} ({k / n:.0%})" if n else "0/0"  # noqa: E731
            cnt = lambda key: sum(r[key] == "1" for r in s)            # noqa: E731
            forms = [form(r["retrieval_query"]) for r in s]
            fin = sum(str(r.get("finish_reason", "")).lower() in ("length", "max_tokens") for r in s)
            print(f"| {m} | {lang} | {pct(cnt('script_ok'))} | {pct(forms.count('question'))} | "
                  f"{pct(forms.count('declarative'))} | {pct(forms.count('other'))} | "
                  f"{pct(cnt('explanation_mixed'))} | {pct(cnt('empty_or_error'))} | "
                  f"{cnt('leak_v3')} ({cnt('leak_v1')}) | {fin} |")
    print("\n모델별 예시 20개와 세부 점검: data/self_translate/logs/check_{model}" + TAG + ".txt")


if __name__ == "__main__":
    main()
