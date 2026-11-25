"""
utils/metrics.py
=================
파이프라인 공유 지표 계산 유틸리티.
pipeline/06_annotate.py 와 analysis/07_analyze.py 에서 동일 로직을 사용하도록 단일 소스화.
"""
import math
import statistics

FAITH_HALLU_THRESHOLD = 0.5

# 패턴별 transition direction 레이블
PATTERN_DIRECTION = {"a": "KO→KO", "b": "EN→KO", "c": "KO→EN", "d": "EN→EN"}


def valid_faith(x) -> bool:
    """hop_faithfulness가 실제 측정값인지 확인 (None·NaN → False)."""
    return x is not None and not (isinstance(x, float) and math.isnan(x))


# ─── 홉-레벨 지표 단일 소스 ───────────────────────────────────────────────────
# 아래 헬퍼들이 same/cross 분류·FaithRate·yield·PDR의 유일한 구현이다.
# 분석 스크립트(07~16)는 인라인으로 재구현하지 말고 반드시 이 함수들을 호출한다.

def index_class(h: dict) -> str:
    """gold_lang 기준 same(ko-index) / cross(en-index) 분류."""
    return "cross" if h.get("gold_lang") == "en" else "same"


def hop_faith(h: dict) -> float | None:
    """채점 가능한 홉의 FaithRate(=supported/total). 기권·애매·미측정 → None (분모 제외).

    FaithRate-LSHD 계열의 단일 값 소스. 06이 저장한 hop_faithfulness를 사용한다
    (claims 재계산 결과와 동일함이 검증됨)."""
    if h.get("abstained") or h.get("ambiguous_answer_marker"):
        return None
    f = h.get("hop_faithfulness")
    return f if valid_faith(f) else None


def hop_yield(h: dict) -> float | None:
    """Answer-yield 지표 y_h. 분모 규칙:
      - ambiguous 홉 → None (분모 제외, FaithRate·AFR과 동일 처리)
      - 기권 홉 → 0.0 (답을 내지 않음)
      - 비기권·미채점(claim 0 → faith None) → 0.0 (답은 냈으나 supported 0)
      - 그 외 → 1[faith ≥ THRESHOLD]
    canonical: y_h = 1[¬abstained ∧ ¬ambiguous ∧ faith ≥ 0.5], 분모 = non-ambiguous 전체."""
    if h.get("ambiguous_answer_marker"):
        return None
    if h.get("abstained"):
        return 0.0
    f = h.get("hop_faithfulness")
    if not valid_faith(f):
        return 0.0
    return 1.0 if f >= FAITH_HALLU_THRESHOLD else 0.0


def hop_claim_rates(h: dict) -> dict | None:
    """claims 기반 {faith, pdr, contra}. 기권·애매·claim없음 → None.
    패턴별 FaithRate/PDR/ContraRate 표용. faith 값은 hop_faith와 동일하다."""
    if h.get("abstained") or h.get("ambiguous_answer_marker") or not h.get("claims"):
        return None
    claims = h["claims"]
    total = len(claims)
    sup = sum(1 for c in claims if c["label"] == "supported")
    uns = sum(1 for c in claims if c["label"] == "unsupported")
    con = sum(1 for c in claims if c["label"] == "contradicted")
    return {"faith": sup / total, "pdr": uns / total, "contra": con / total}


def lshd(records: list[dict], value_fn, hop_filter=None
         ) -> tuple[float, float, float, int, int]:
    """일반 LSHD 집계: mean(same) − mean(cross).

    value_fn(h) → float|None (None이면 분모에서 제외).
    hop_filter(h) → bool 로 추가 조건 부여(예: 특정 hop_idx, v3 retrieval 성공).
    반환: (gap, same_mean, cross_mean, n_same, n_cross). 한쪽이라도 비면 gap=nan."""
    same, cross = [], []
    for r in records:
        for h in r["hops"]:
            if hop_filter is not None and not hop_filter(h):
                continue
            v = value_fn(h)
            if v is None:
                continue
            (cross if index_class(h) == "cross" else same).append(v)
    sm = statistics.mean(same) if same else float("nan")
    cr = statistics.mean(cross) if cross else float("nan")
    gap = sm - cr if not (math.isnan(sm) or math.isnan(cr)) else float("nan")
    return gap, sm, cr, len(same), len(cross)


def clepr_by_pattern(results: list[dict]) -> dict:
    """
    primary: 패턴별 direction CLEPR.

    CLEPR_KO→KO = P(Hallu_1 | Hallu_0, pattern=a)
    CLEPR_EN→KO = P(Hallu_1 | Hallu_0, pattern=b)
    CLEPR_KO→EN = P(Hallu_1 | Hallu_0, pattern=c)
    CLEPR_EN→EN = P(Hallu_1 | Hallu_0, pattern=d)

    aggregate:
      switch (b+c): EN→KO + KO→EN  (hop0↔hop1 언어 전환 있음)
      same   (a+d): KO→KO + EN→EN  (hop0↔hop1 언어 전환 없음)
    """
    by_pat: dict[str, list[int]] = {p: [0, 0] for p in "abcd"}  # [denom, numer]

    for r in results:
        hops = r["hops"]
        if len(hops) < 2:
            continue
        h0, h1 = hops[0], hops[1]
        if h0.get("abstained") or h1.get("abstained"):
            continue
        f0 = h0.get("hop_faithfulness")
        f1 = h1.get("hop_faithfulness")
        if not valid_faith(f0) or not valid_faith(f1):
            continue
        if f0 >= FAITH_HALLU_THRESHOLD:
            continue
        pat = r["pattern"]
        by_pat[pat][0] += 1
        if f1 < FAITH_HALLU_THRESHOLD:
            by_pat[pat][1] += 1

    out: dict = {}
    for pat in "abcd":
        d, n = by_pat[pat]
        out[pat] = {"rate": n / d if d else float("nan"), "n": d,
                    "label": PATTERN_DIRECTION[pat]}

    for key, pats in [("switch", "bc"), ("same", "ad")]:
        d = sum(by_pat[p][0] for p in pats)
        n = sum(by_pat[p][1] for p in pats)
        out[key] = {"rate": n / d if d else float("nan"), "n": d}

    return out


def clepr_rates(results: list[dict]) -> dict:
    """
    레거시: lang_switch 기반 aggregate CLEPR.
    clepr_by_pattern()이 primary.

    CLEPR_xling = P(k+1 unfaithful | k unfaithful, lang_switch@k)
    CLEPR_same  = P(k+1 unfaithful | k unfaithful, ~lang_switch@k)
    """
    xling_denom = xling_numer = 0
    same_denom  = same_numer  = 0

    for r in results:
        hops = r["hops"]
        for k in range(len(hops) - 1):
            h_k, h_k1 = hops[k], hops[k + 1]
            if h_k.get("abstained") or h_k1.get("abstained"):
                continue
            fk  = h_k.get("hop_faithfulness")
            fk1 = h_k1.get("hop_faithfulness")
            if not valid_faith(fk) or not valid_faith(fk1):
                continue
            if fk >= FAITH_HALLU_THRESHOLD:
                continue
            if h_k["lang_switch"]:
                xling_denom += 1
                if fk1 < FAITH_HALLU_THRESHOLD:
                    xling_numer += 1
            else:
                same_denom += 1
                if fk1 < FAITH_HALLU_THRESHOLD:
                    same_numer += 1

    return {
        "xling":   xling_numer / xling_denom if xling_denom else float("nan"),
        "same":    same_numer  / same_denom  if same_denom  else float("nan"),
        "xling_n": xling_denom,
        "same_n":  same_denom,
    }
