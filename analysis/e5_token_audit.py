#!/usr/bin/env python3
"""
analysis/e5_token_audit.py — E5 입력의 512 토큰 초과 비율 (E5 부록 truncation 문단). CPU, 토크나이저만 쓴다.
camera-ready 때 work/e5_gate/token_audit.py 에서 옮겼다 (경로 기본값만 바꿈).

e5-large의 max_seq_length는 512, BGE-M3는 8192다. 현재 chunk는 utils/wiki_api.py의
_chunk_words(350 words, overlap 50) — 즉 토큰이 아니라 **공백 분리 어절** 기준이다.
한국어는 XLM-R BPE에서 토큰 수가 크게 불어나므로 e5에서 잘려나갈 수 있고, 그러면 두
retriever가 서로 다른 "passage"를 보게 된다. 버그가 아니라 보고해야 할 설정 차이다.

실제 인코딩되는 문자열("passage: " + text) 기준을 정본으로 보고, 비교용으로 prefix 없는
raw 분포도 함께 낸다.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

E5_MODEL = "intfloat/multilingual-e5-large"
PASSAGE_PREFIX = "passage: "
LIMIT = 512


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def stats(lengths: np.ndarray, limit: int) -> dict:
    return {
        "n": int(lengths.size),
        "median": float(np.median(lengths)),
        "mean": round(float(lengths.mean()), 1),
        "p90": float(np.percentile(lengths, 90)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(lengths.max()),
        f"over_{limit}_count": int((lengths > limit).sum()),
        f"over_{limit}_ratio": round(float((lengths > limit).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages",   default="data/corpus/passages.jsonl")
    ap.add_argument("--model",      default=E5_MODEL)
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--out-json",   default="results/camera_ready/retriever_robustness/e5_token_audit.json")
    args = ap.parse_args()

    log(f"tokenizer 로딩 ({args.model}) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    log(f"  fast={tok.is_fast}  model_max_length={tok.model_max_length}")

    by_lang: dict[str, dict[str, list]] = {}
    buf_text: list[str] = []
    buf_lang: list[str] = []
    n_total = 0
    t0 = time.time()

    def flush():
        if not buf_text:
            return
        # add_special_tokens=True → <s> ... </s> 포함 (실제 인코딩 길이)
        raw = tok(buf_text, add_special_tokens=True, truncation=False)["input_ids"]
        pre = tok([PASSAGE_PREFIX + t for t in buf_text],
                  add_special_tokens=True, truncation=False)["input_ids"]
        for lang, r, p in zip(buf_lang, raw, pre):
            d = by_lang.setdefault(lang, {"raw": [], "prefixed": []})
            d["raw"].append(len(r))
            d["prefixed"].append(len(p))
        buf_text.clear()
        buf_lang.clear()

    with open(args.passages, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            buf_text.append(rec["text"])
            buf_lang.append(rec.get("lang", "ko"))
            n_total += 1
            if len(buf_text) >= args.batch_size:
                flush()
                if n_total % 50000 == 0:
                    log(f"  {n_total}건 ... ({time.time() - t0:.0f}s)")
    flush()
    elapsed = round(time.time() - t0, 1)
    log(f"완료 {n_total}건 / {elapsed}s")

    out = {
        "model": args.model,
        "passages_file": args.passages,
        "n_total": n_total,
        "limit": LIMIT,
        "passage_prefix": PASSAGE_PREFIX,
        "chunking": "utils/wiki_api.py _chunk_words(350 words, overlap 50) — 어절 기준",
        "per_lang": {},
    }
    for lang in sorted(by_lang):
        out["per_lang"][lang] = {
            "prefixed": stats(np.array(by_lang[lang]["prefixed"]), LIMIT),
            "raw":      stats(np.array(by_lang[lang]["raw"]), LIMIT),
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2, ensure_ascii=False), "utf-8")

    print()
    print(f'{"index":6} {"n passages":>11} {"median":>8} {"p90":>8} {"p99":>8} {"max":>8} {">512 ratio":>11}')
    print("-" * 66)
    for lang in sorted(out["per_lang"]):
        s = out["per_lang"][lang]["prefixed"]
        print(f'{lang.upper():6} {s["n"]:>11,} {s["median"]:>8.0f} {s["p90"]:>8.0f} '
              f'{s["p99"]:>8.0f} {s["max"]:>8,} {s["over_512_ratio"]:>10.2%}')
    print("\n(위 표는 실제 인코딩 문자열 'passage: ' + text 기준)")

    breached = [l for l in out["per_lang"]
                if out["per_lang"][l]["prefixed"]["over_512_ratio"] > 0.10]
    if breached:
        print(f"\nSTOP: >512 비율 10% 초과 — {breached}. re-chunking은 비교 대상 자체를 "
              f"바꾸는 결정이므로 지시 없이 진행하지 않는다.")
    else:
        print("\n>512 비율 전 언어 10% 이하 — Step 3 진행 가능")
    print(f"저장: {args.out_json}")

    sys.exit(3 if breached else 0)


if __name__ == "__main__":
    main()
