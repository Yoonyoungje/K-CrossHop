#!/usr/bin/env python3
"""
pipeline/06_annotate.py
===================
Agentic RAG 출력(궤적)에 홉별 충실성 라벨을 부여한다.
LLM-as-judge (Claude) 방식으로 판정.

입력:
  - data/processed/qa_pairs.jsonl     (QA 쌍 + gold evidence 정보)
  - data/trajectories/trajectories.jsonl  (에이전트 실행 결과, 별도 생성)

출력:
  - data/processed/annotated.jsonl    (claim별 LLM 판정 라벨 포함)

에이전트 출력 포맷 (trajectories.jsonl 각 줄):
  {
    "qid": "q00042",
    "model": "qwen3-8b",
    "pattern": "d",
    "trajectory": [
      {
        "hop": 0,
        "retrieved_passages": ["passage text 1", "passage text 2"],
        "retrieved_langs":    ["ko", "ko"],
        "generated_text":     "올드보이는 박찬욱 감독의 2003년 영화이다...",
        "gold_lang":          "ko",      # 필수 — 없으면 fail-fast
        "retrieval_success":  true,      # 필수 — 없으면 fail-fast
        "lang_switch":        false      # gold_lang 시퀀스로 재계산되나 05 출력에 포함됨
      },
      ...
    ],
    "final_answer": "칸영화제"
  }

실행:
  python pipeline/06_annotate.py --trajectories data/trajectories/trajectories_pilot.jsonl --api-key $KEY
"""

import asyncio
import json
import math
import random
import re
import time
import argparse
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field
import yaml
import anthropic
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.lang_switch import compute_lang_switch
from utils.metrics import valid_faith, clepr_rates as _clepr_rates
from utils.answer_parser import (
    ParseResult, _META_LINE, _ANSWER_MARKER, _CLEAR_ABSTAIN,
    _SOFT_ABSTAIN_NORMALIZED, ABSTAIN_KWS,
    _is_soft_abstain, _extract_sentences, _check_abstain, parse_generated_answer,
    is_judge_target, normalize_entity,
    PARSER_VERSION,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


LABEL_SUPPORTED    = "supported"
LABEL_NEUTRAL      = "unsupported"
LABEL_CONTRADICTED = "contradicted"

FAITH_HALLU_THRESHOLD  = 0.5  # utils/metrics.py와 동기화 유지
JUDGE_PROMPT_VERSION   = "v2"

# dynamic 파이프라인(05) trajectory가 홉에 기록하는 flag 필드.
# annotated 출력에 그대로 전파해 dynamic LSHD/CLEPR 세밀 분해에 사용 가능하게 한다.
# (계약: analysis/TRAJECTORY_FIELDS.md). 신/구 스키마 키를 모두 포함한다.
_DYNAMIC_FLAG_KEYS = (
    # 신 스키마 (_build_hop_state)
    "query_template_field", "query_template_lang", "design_query_index_match",
    "actual_query_lang_detected", "query_contains_gold_answer",
    "bridge_anchor_text", "bridge_anchor_lang", "bridge_anchor_source",
    "anchor_replacement_success", "bridge_insert_text", "bridge_insert_source",
    "normalization_applied", "normalization_status", "normalization_basis",
    "prev_answer_raw", "prev_answer_is_gold_bridge", "prev_answer_surface_lang",
    "prev_answer_abstained",
    # 구 스키마 (hop_state 도입 전 dynamic run)
    "bridge_entity", "bridge_correct", "bridge_lang",
)


def safe_faithfulness(x):
    """abstained/NaN/None faithfulness → None, else rounded float.
    비기권인데 claim이 0개라 faithfulness가 NaN인 홉이 집계를 오염시키지 않도록 한다."""
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else round(x, 4)


@dataclass
class ClaimLabel:
    claim_text: str
    label: str        # supported | unsupported | contradicted
    reasoning: str


@dataclass
class HopAnnotation:
    hop: int
    generated_text: str
    retrieved_langs: list[str]
    gold_lang: str
    lang_switch: bool
    abstained: bool
    claims: list[ClaimLabel]
    hop_faithfulness: float
    has_hallucination: bool
    claim_extraction_source: str = "none"
    answer_marker_count: int = 0
    ambiguous_answer_marker: bool = False
    parse_issue: str | None = None
    dynamic_flags: dict = field(default_factory=dict)


@dataclass
class TrajectoryAnnotation:
    qid: str
    model: str
    pattern: str
    hops: list[HopAnnotation]
    cross_lingual_error_propagation: bool
    abstained: bool
    retrieval_failed: bool
    final_answer_correct: bool


class LLMJudge:
    """LLM-as-judge 충실성 판정기 (OpenAI / Gemini / Claude 지원)."""

    # ── Static system prefix (≥1024 tokens → OpenAI prompt caching 적용) ────────
    # 모든 judge call에서 동일 → cached input ($0.075/1M) 적용 가능.
    # 동적 내용(question / passages / claims)은 user message 끝에만 위치.
    _SYSTEM = """\
You are a faithfulness judge for a multi-hop Korean–English cross-lingual QA system. \
Your task is to evaluate whether each model answer (claim) is supported by the retrieved \
passages as a correct and complete answer to the given question.

## Label Definitions

supported
  The retrieved passages explicitly confirm the claim as a correct answer to the question, \
and the answer satisfies the specific relation the question asks about. \
For example, if the question asks "Who directed X?", the answer must be confirmed as the \
director in the passages, not merely mentioned as a cast member or producer.

contradicted
  The retrieved passages explicitly state a different answer, making the claim factually \
incorrect according to the provided evidence. The contradiction must be direct, \
not merely the absence of confirmation.

unsupported
  The retrieved passages do not contain sufficient information to confirm or deny the claim. \
The answer may be plausible from world knowledge, but it is not verified by the passage text.

## Judging Rules

Rule 1 — Short answers are fully valid.
Entity names, dates, place names, organization names, person names, film and book titles, \
and other short noun phrases are valid complete answers in this task. \
Do NOT mark an answer unsupported or invalid merely because it is brief or not a full sentence.

Rule 2 — The answer must satisfy the asked relation.
An entity must fulfill the specific relation the question asks about. \
If a passage mentions a person, organization, or place without confirming they satisfy \
the questioned relation, that claim is unsupported. \
Example: if the question asks "Who founded company X?" and a passage says \
"Y is the CEO of X" without mentioning founding, Y is unsupported.

Rule 3 — Korean–English bilingual equivalence.
Korean and English surface forms of the same entity are interchangeable. \
"베이징" = "Beijing", "한국" = "South Korea", "서울" = "Seoul". \
Accept either language form as supported if the passages confirm the underlying entity \
in any language. Do not penalize an answer for using a different but equivalent surface form.

Rule 4 — Aliases and alternate names count.
Recognized alternate names, common abbreviations, and aliases of an entity are equivalent \
to the canonical name. If passages confirm the entity under any recognized alias, \
label the claim supported.

Rule 5 — Evidence must come from the passages only.
Do not use your own world knowledge to label a claim supported. \
Only the text in the provided passages constitutes valid evidence. \
If you believe the answer is correct from training knowledge but the passages do not \
confirm it, label it unsupported.

Rule 6 — Partial or one-of-many answers.
If a claim represents one valid answer among several possible answers, and the passages \
confirm it as correct, label it supported. \
Do not require the claim to be the only possible answer.

Rule 7 — Very short or generic claims.
Be cautious with very short (one or two characters) or highly generic terms \
such as pronouns, articles, or vague nouns. \
These should be unsupported unless the passages explicitly and unambiguously confirm them \
as the specific answer to the question.

## Output Format

Return ONLY a JSON array with one element per answer, in the same order as the input \
numbered list. Output nothing else — no markdown fences, no explanation text:
[{"label": "supported"|"contradicted"|"unsupported"}, ...]

## Examples

### Example 1 — English, basic relation check
Question: What is the capital city of South Korea?
Retrieved passages:
[1] Seoul is the capital and largest metropolis of South Korea. It has been the country's \
capital since the Joseon Dynasty and has a population of over 9 million in the city proper.
[2] Busan is the second-largest city in South Korea, located on the southeastern tip of the \
Korean Peninsula, and serves as the country's largest port.
Answers: 1. Seoul  2. Busan  3. Tokyo
Output: [{"label": "supported"}, {"label": "contradicted"}, {"label": "unsupported"}]

### Example 2 — Korean, director relation
Question: 올드보이를 연출한 감독은?
Retrieved passages:
[1] 올드보이(2003)는 박찬욱 감독이 연출한 한국 영화이다. 제57회 칸 영화제에서 심사위원 대상을 \
수상하며 국제적으로 큰 주목을 받았다.
[2] 봉준호 감독은 기생충(2019)으로 제92회 아카데미 시상식 작품상을 수상한 한국 감독이다.
Answers: 1. 박찬욱  2. 봉준호  3. 홍상수
Output: [{"label": "supported"}, {"label": "contradicted"}, {"label": "unsupported"}]

### Example 3 — Cross-lingual alias equivalence
Question: Where were the 2008 Summer Olympics held?
Retrieved passages:
[1] The 2008 Summer Olympics, officially the Games of the XXIX Olympiad, were held in \
Beijing, China from 8 to 24 August 2008. The games featured 10,942 athletes from 204 nations.
Answers: 1. 베이징  2. 상하이  3. Tokyo
Output: [{"label": "supported"}, {"label": "contradicted"}, {"label": "unsupported"}]

### Example 4 — Alias variants both accepted
Question: 기생충을 제작한 국가는?
Retrieved passages:
[1] 기생충(Parasite)은 2019년 봉준호 감독의 한국 영화로, 대한민국에서 제작되었으며 \
제92회 아카데미 시상식에서 작품상을 수상했다.
Answers: 1. 대한민국  2. 한국  3. 미국
Output: [{"label": "supported"}, {"label": "supported"}, {"label": "contradicted"}]\
"""

    # ── Dynamic user suffix: question이 있을 때 (v2 default) ─────────────────────
    _PROMPT = """\
Question: {question}

Retrieved passages:
{passages}

Answers to judge:
{claims}

Output ONLY a JSON array — no explanation, no text before or after:
[{{"label": "supported"|"contradicted"|"unsupported"}}, ...]"""

    # ── Dynamic user suffix: question 없을 때 fallback ────────────────────────────
    _PROMPT_NO_QUESTION = """\
Retrieved passages:
{passages}

Answers to judge:
{claims}

Output ONLY a JSON array — no explanation, no text before or after:
[{{"label": "supported"|"contradicted"|"unsupported"}}, ...]"""

    # OpenAI seed: 동일 seed + temperature=0 → 재현 가능한 출력
    _OPENAI_SEED = 42

    def __init__(self, model: str, api_key: str):
        self.model = model
        self.actual_model: str | None = None
        self._usage_stats: dict[str, int] = {
            "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0, "calls": 0,
        }
        if model.startswith("gemini"):
            from google import genai
            self._gemini = genai.Client(api_key=api_key)
            self._openai = None
            self._client = None
        elif model.startswith("gpt") or model.startswith("o"):
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(api_key=api_key, max_retries=0)
            self._gemini = None
            self._client = None
        else:
            self._gemini = None
            self._openai = None
            self._client = anthropic.AsyncAnthropic(api_key=api_key)

    def _gemini_call(self, prompt: str) -> str:
        """동기 Gemini 호출 (asyncio.to_thread로 실행)."""
        from google.genai import types as genai_types
        for attempt in range(4):
            try:
                response = self._gemini.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self._SYSTEM,
                        max_output_tokens=1024,
                        temperature=0,
                        thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                return response.text.strip()
            except Exception as e:
                err = str(e)
                if "429" in err and "PerDay" not in err and attempt < 3:
                    wait = 60 * (attempt + 1)
                    logger.warning(f"429 rate limit (분당), {wait}s 후 재시도 ({attempt+1}/3)")
                    time.sleep(wait)
                else:
                    raise

    async def judge_hop(self, claims: list[str], passages: list[str],
                        question: str | None = None) -> list[ClaimLabel]:
        """홉의 모든 claim을 한 번의 LLM 호출로 판정."""
        if not claims:
            return []
        if not passages:
            return [ClaimLabel(c, LABEL_NEUTRAL, "no passages") for c in claims]

        passages_text = "\n\n".join(f"[{i+1}] {p[:2500]}" for i, p in enumerate(passages))
        claims_text   = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        if question:
            prompt = self._PROMPT.format(question=question, passages=passages_text, claims=claims_text)
        else:
            prompt = self._PROMPT_NO_QUESTION.format(passages=passages_text, claims=claims_text)

        if self._gemini:
            text = await asyncio.to_thread(self._gemini_call, prompt)
        elif self._openai:
            from openai import RateLimitError
            for attempt in range(6):
                try:
                    response = await self._openai.chat.completions.create(
                        model=self.model,
                        temperature=0,
                        seed=self._OPENAI_SEED,
                        max_completion_tokens=4096,
                        messages=[
                            {"role": "system", "content": self._SYSTEM},
                            {"role": "user",   "content": prompt},
                        ],
                    )
                    if self.actual_model is None:
                        self.actual_model = response.model
                    text = response.choices[0].message.content.strip()
                    u = response.usage
                    self._usage_stats["prompt_tokens"]     += u.prompt_tokens
                    self._usage_stats["completion_tokens"] += u.completion_tokens
                    self._usage_stats["calls"]             += 1
                    details = getattr(u, "prompt_tokens_details", None)
                    if details:
                        self._usage_stats["cached_tokens"] += getattr(details, "cached_tokens", 0) or 0
                    break
                except RateLimitError:
                    if attempt >= 5:
                        raise
                    wait = 30 * (attempt + 1) + random.uniform(0, 10)
                    logger.warning(f"OpenAI 429 rate limit, {wait:.1f}s 후 재시도 ({attempt+1}/5)")
                    await asyncio.sleep(wait)
            else:
                raise RuntimeError("OpenAI rate limit 재시도 초과")
        else:
            response = await self._client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=0,
                system=self._SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )
            if self.actual_model is None:
                self.actual_model = response.model
            text = response.content[0].text.strip()
        text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
        # LLM이 생성한 유니코드 스마트 쿼트 → ASCII 정규화
        text = (text.replace('\u201c', '"').replace('\u201d', '"')
                    .replace('\u2018', "'").replace('\u2019', "'"))
        text = re.sub(r'(?<=")\)(?=\s*[,}\]])', '}', text)   # ") → "}
        text = re.sub(r'(?<=")\](?=\s*,\s*\{)', '}', text)   # "], { → "}, {

        verdicts = None
        try:
            verdicts = json.loads(text)
        except json.JSONDecodeError:
            # 산문 앞뒤에 JSON 배열이 묻혀 있을 경우 추출 시도
            m = re.search(r"\[.*\]", text, re.DOTALL)
            if m:
                try:
                    verdicts = json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        if not isinstance(verdicts, list):
            logger.warning(f"LLM 응답 JSON 파싱 실패 (claims={len(claims)})\n응답 원문:\n{text}")
            return [ClaimLabel(c, LABEL_NEUTRAL, "parse error") for c in claims]

        result = []
        for i, claim in enumerate(claims):
            if i < len(verdicts):
                v = verdicts[i]
                lbl = v.get("label", LABEL_NEUTRAL)
                if lbl not in (LABEL_SUPPORTED, LABEL_NEUTRAL, LABEL_CONTRADICTED):
                    lbl = LABEL_NEUTRAL
                reason = v.get("reason", "")
            else:
                lbl, reason = LABEL_NEUTRAL, "missing verdict"
            result.append(ClaimLabel(claim_text=claim, label=lbl, reasoning=reason))
        return result


def compute_clepr(annotations: list[HopAnnotation]) -> bool:
    """궤적 단위 bool: 언어전환 홉에서 환각 → 다음 홉도 환각인 쌍이 하나라도 존재하는지."""
    for i, hop in enumerate(annotations[:-1]):
        nxt = annotations[i + 1]
        # 측정 불가(기권/무claim → faithfulness None·NaN) 홉이 낀 쌍은 제외
        if not valid_faith(hop.hop_faithfulness) or not valid_faith(nxt.hop_faithfulness):
            continue
        if hop.lang_switch and hop.has_hallucination:
            if nxt.has_hallucination:
                return True
    return False



async def annotate_trajectory(traj: dict, judge: LLMJudge,
                             gold_answers: dict[str, str],
                             sub_q_map: dict[tuple, str] | None = None,
                             reuse: dict | None = None) -> TrajectoryAnnotation:
    """단일 궤적 전체에 라벨 부여. 홉별 LLM 판정은 병렬로 실행."""
    qid = traj.get("qid", "?")

    # 필수 필드 검증 (fail fast — 구 포맷 trajectory는 재생성 필요)
    for step in traj["trajectory"]:
        if "gold_lang" not in step:
            raise ValueError(
                f"trajectory({qid}) hop {step.get('hop','?')}에 gold_lang이 없습니다. "
                "pipeline/05_pilot_runner.py로 재생성된 trajectory가 필요합니다."
            )
        if "retrieval_success" not in step:
            raise ValueError(
                f"trajectory({qid}) hop {step.get('hop','?')}에 retrieval_success가 없습니다. "
                "pipeline/05_pilot_runner.py로 재생성된 trajectory가 필요합니다."
            )

    # lang_switch는 gold_lang 시퀀스에서 단일 소스로 재계산 (직전 홉 대비 전환)
    gold_langs = [step["gold_lang"] for step in traj["trajectory"]]
    switches   = compute_lang_switch(gold_langs)

    # dynamic flag 전파: trajectory 홉에 있는 flag만 추려 hop별로 보관
    hop_flags = {
        step["hop"]: {k: step[k] for k in _DYNAMIC_FLAG_KEYS if k in step}
        for step in traj["trajectory"]
    }

    hop_metadata = []
    for hop_data, gold_lang, switch in zip(traj["trajectory"], gold_langs, switches):
        hop_idx      = hop_data["hop"]
        gen_text     = hop_data.get("generated_text", "")
        passages     = hop_data.get("retrieved_passages", [])
        langs        = hop_data.get("retrieved_langs", [])
        sub_q        = (sub_q_map or {}).get((qid, hop_idx), "") or ""

        if hop_data.get("api_error"):
            hop_metadata.append((hop_idx, gen_text, passages, langs, gold_lang, switch,
                                 True, [], "api_error", False, "api_error", 0, sub_q))
            continue

        pr = parse_generated_answer(gen_text)
        parse = (pr.abstained, pr.ambiguous_answer_marker, pr.source,
                 pr.parse_issue, pr.answer_marker_count)
        src = _reused_hop(reuse, qid, hop_idx, hop_data)
        if src is not None:                   # seed 홉 — 파서 판정도 원본을 따른다
            parse = _reused_parse(src, *parse)
        hop_metadata.append((hop_idx, gen_text, passages, langs, gold_lang, switch,
                             parse[0], pr.claims, parse[2], parse[1], parse[3], parse[4],
                             sub_q))

    async def _judge_or_skip(hop_abstained, ambiguous, claims_text, passages, sub_q, hop_idx):
        reused = _reused_labels(reuse, qid, hop_idx,
                                next(s for s in traj["trajectory"] if s["hop"] == hop_idx))
        if reused is not None:
            return reused or None             # seed 홉 — judge 호출 없이 원본 라벨 재사용 (라벨 없음 → NaN)
        if hop_abstained or ambiguous or not claims_text:
            return None
        return await judge.judge_hop(claims_text, passages, question=sub_q or None)

    hop_results = []
    for (hop_i, _, psg, _, _, _, abs_, cls_, _, amb_, _, _, sq) in hop_metadata:
        hop_results.append(await _judge_or_skip(abs_, amb_, cls_, psg, sq, hop_i))

    hop_anns: list[HopAnnotation] = []
    for (hop_idx, gen_text, passages, langs, gold_lang, switch,
         hop_abstained, _, extract_src, ambiguous, parse_issue, marker_count, _sq), claim_labels in zip(hop_metadata, hop_results):
        if claim_labels is None:
            claim_labels = []
            faithfulness = float("nan")
            has_hallu    = False
        else:
            total        = len(claim_labels) or 1
            supported_n  = sum(1 for cl in claim_labels if cl.label == LABEL_SUPPORTED)
            faithfulness = supported_n / total
            has_hallu    = faithfulness < FAITH_HALLU_THRESHOLD

        hop_anns.append(HopAnnotation(
            hop=hop_idx,
            generated_text=gen_text,
            retrieved_langs=langs,
            gold_lang=gold_lang,
            lang_switch=switch,
            abstained=hop_abstained,
            claims=claim_labels,
            hop_faithfulness=faithfulness,
            has_hallucination=has_hallu,
            claim_extraction_source=extract_src,
            answer_marker_count=marker_count,
            ambiguous_answer_marker=ambiguous,
            parse_issue=parse_issue,
            dynamic_flags=hop_flags.get(hop_idx, {}),
        ))

    clepr      = compute_clepr(hop_anns)
    # top-level abstained: 마지막 홉 파서 판정으로 통일 (per-hop 로직과 일치)
    abstained  = (hop_anns[-1].abstained if hop_anns
                  else _check_abstain(traj.get("final_answer", "")))
    # 계약 필드 retrieval_success로 판정 (retrieved_langs 재계산 금지)
    ret_failed = any(not step["retrieval_success"] for step in traj["trajectory"])

    gold = gold_answers.get(traj["qid"], "")
    final_answer = traj.get("final_answer", "")
    norm_gold = normalize_entity(gold) if gold else ""
    answer_correct = bool(norm_gold and norm_gold in normalize_entity(final_answer))

    return TrajectoryAnnotation(
        qid=traj["qid"],
        model=traj.get("model", "unknown"),
        pattern=traj.get("pattern", "?"),
        hops=hop_anns,
        cross_lingual_error_propagation=clepr,
        abstained=abstained,
        retrieval_failed=ret_failed,
        final_answer_correct=answer_correct,
    )


def ann_to_dict(ann: TrajectoryAnnotation, save_rationale: bool = False) -> dict:
    """TrajectoryAnnotation → JSON 직렬화 가능한 dict."""
    d = {
        "qid": ann.qid,
        "model": ann.model,
        "pattern": ann.pattern,
        "cross_lingual_error_propagation": ann.cross_lingual_error_propagation,
        "abstained": ann.abstained,
        "retrieval_failed": ann.retrieval_failed,
        "final_answer_correct": ann.final_answer_correct,
        "hops": [],
    }
    for h in ann.hops:
        d["hops"].append({
            "hop": h.hop,
            "gold_lang": h.gold_lang,
            "lang_switch": h.lang_switch,
            "abstained": h.abstained,
            "retrieved_langs": h.retrieved_langs,
            "hop_faithfulness": safe_faithfulness(h.hop_faithfulness),
            "has_hallucination": h.has_hallucination,
            "claims": [
                {"text": c.claim_text, "label": c.label,
                 **( {"reason": c.reasoning} if save_rationale and c.reasoning else {} )}
                for c in h.claims
            ],
            "claim_extraction_source": h.claim_extraction_source,
            "answer_marker_count": h.answer_marker_count,
            "ambiguous_answer_marker": h.ambiguous_answer_marker,
            "parse_issue": h.parse_issue,
            **h.dynamic_flags,
        })
    return d


def _parse_verdicts(text: str | None, claims: list[str],
                    save_rationale: bool = False) -> list[ClaimLabel]:
    """Batch API 결과 텍스트 → ClaimLabel 목록."""
    if text is None:
        return [ClaimLabel(c, LABEL_NEUTRAL, "batch error") for c in claims]
    text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    text = (text.replace('“', '"').replace('”', '"')
                .replace('‘', "'").replace('’', "'"))
    verdicts = None
    try:
        verdicts = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if m:
            try:
                verdicts = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(verdicts, list):
        return [ClaimLabel(c, LABEL_NEUTRAL, "parse error") for c in claims]
    result = []
    for i, claim in enumerate(claims):
        if i < len(verdicts):
            v = verdicts[i]
            lbl = v.get("label", LABEL_NEUTRAL)
            if lbl not in (LABEL_SUPPORTED, LABEL_NEUTRAL, LABEL_CONTRADICTED):
                lbl = LABEL_NEUTRAL
            reason = v.get("reason", "") if save_rationale else ""
        else:
            lbl, reason = LABEL_NEUTRAL, "missing verdict"
        result.append(ClaimLabel(claim_text=claim, label=lbl, reasoning=reason))
    return result


def load_reused_hop_labels(ann_path: str, traj_path: str) -> dict:
    """seed 로 재사용한 hop0 의 judge 라벨을 그대로 가져온다 (재채점 없이).

    {(qid, 0): {"ann": 원본 annotated 홉, "source_hop": 원본 trajectory 홉}} 을 돌려주고,
    사용 시점에 현재 홉의 generated_text·retrieved_passages 가 원본과 같은지 확인한다.
    재사용 홉은 claim 라벨뿐 아니라 파서 판정(기권·모호·추출 출처)도 원본을 따른다 — 원본이 이전
    파서로 만들어졌으면 지금 파서와 기권 판정이 갈릴 수 있어서다(예: 'Unknown Worlds' 기권 오탐).
    hop0 만 싣는다 — seed 는 hop0 공유뿐이다(05 load_seed_hops). hop1 은 생성물이 우연히 같아도
    재채점한다(조건 간 대조축이라 judge 변동을 조건마다 따로 둔다)."""
    src_traj: dict[tuple, dict] = {}
    with open(traj_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                for h in t["trajectory"]:
                    src_traj[(t["qid"], int(h["hop"]))] = h
    out: dict[tuple, dict] = {}
    with open(ann_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                a = json.loads(line)
                for h in a["hops"]:
                    key = (a["qid"], int(h["hop"]))
                    if key[1] == 0 and key in src_traj:
                        out[key] = {"ann": h, "source_hop": src_traj[key]}
    return out


def _reused_hop(reuse: dict | None, qid: str, hop_idx: int, hop_data: dict) -> dict | None:
    """재사용 가능한 홉이면 원본 annotated 홉, 아니면 None. 생성물이 원본과 다르면 재사용하지 않는다."""
    ent = (reuse or {}).get((qid, hop_idx))
    if ent is None:
        return None
    src = ent["source_hop"]
    if (src.get("generated_text") != hop_data.get("generated_text")
            or src.get("retrieved_passages") != hop_data.get("retrieved_passages")):
        return None
    return ent["ann"]


def _reused_labels(reuse: dict | None, qid: str, hop_idx: int, hop_data: dict,
                   save_rationale: bool = False) -> list[ClaimLabel] | None:
    """재사용 가능한 홉이면 ClaimLabel 목록, 아니면 None."""
    ann = _reused_hop(reuse, qid, hop_idx, hop_data)
    if ann is None:
        return None
    return [ClaimLabel(c["text"], c["label"], c.get("reason") if save_rationale else None)
            for c in ann.get("claims", [])]


def _reused_parse(ann: dict, abstained: bool, ambiguous: bool, source: str,
                  parse_issue, marker_count: int) -> tuple:
    """재사용 홉의 파서 판정 (abstained, ambiguous, 추출 출처, parse_issue, marker 수) — 원본 값 우선."""
    return (ann.get("abstained", abstained), ann.get("ambiguous_answer_marker", ambiguous),
            ann.get("claim_extraction_source", source), ann.get("parse_issue", parse_issue),
            ann.get("answer_marker_count", marker_count))


def _collect_batch_requests(trajs: list[dict], judge: "LLMJudge",
                            sub_q_map: dict, reuse: dict | None = None) -> tuple[list[dict], list[tuple]]:
    """trajectory 목록 → (batch JSONL 행 목록, (traj, hop_parses) 목록).
    결정적·비-API. submit 시 pending을, collect 시 parsed를 재도출하는 단일 소스."""
    pending: list[dict] = []
    parsed: list[tuple] = []
    for traj in trajs:
        qid = traj.get("qid", "?")
        gold_langs = [s["gold_lang"] for s in traj["trajectory"]]
        switches   = compute_lang_switch(gold_langs)
        hop_parses = []
        for hop_data, gold_lang, switch in zip(traj["trajectory"], gold_langs, switches):
            hop_idx  = hop_data["hop"]
            passages = hop_data.get("retrieved_passages", [])
            langs    = hop_data.get("retrieved_langs", [])
            sub_q    = (sub_q_map or {}).get((qid, hop_idx), "") or ""

            if hop_data.get("api_error"):
                hop_parses.append((hop_idx, None, passages, langs, gold_lang, switch, sub_q))
                continue

            pr = parse_generated_answer(hop_data.get("generated_text", ""))
            hop_parses.append((hop_idx, pr, passages, langs, gold_lang, switch, sub_q))

            if _reused_labels(reuse, qid, hop_idx, hop_data) is not None:
                continue                      # seed 홉 — judge 호출 없이 원본 라벨을 그대로 쓴다
            if not pr.abstained and not pr.ambiguous_answer_marker and pr.claims and passages:
                passages_text = "\n\n".join(f"[{i+1}] {p[:2500]}" for i, p in enumerate(passages))
                claims_text   = "\n".join(f"{i+1}. {c}" for i, c in enumerate(pr.claims))
                user_content  = (
                    judge._PROMPT.format(question=sub_q, passages=passages_text, claims=claims_text)
                    if sub_q else
                    judge._PROMPT_NO_QUESTION.format(passages=passages_text, claims=claims_text)
                )
                pending.append({
                    "custom_id": f"{qid}__hop{hop_idx}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": judge.model,
                        "temperature": 0,
                        "seed": judge._OPENAI_SEED,
                        "max_completion_tokens": 1024,
                        "messages": [
                            {"role": "system", "content": judge._SYSTEM},
                            {"role": "user",   "content": user_content},
                        ],
                    },
                })
        parsed.append((traj, hop_parses))
    return pending, parsed


async def _submit_batch(judge: "LLMJudge", pending: list[dict]) -> str:
    """pending 요청을 업로드 + batch 생성 → batch_id. 업로드 임시파일은 생성 직후 삭제."""
    import os, tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for req in pending:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
        tmp_path = f.name
    try:
        with open(tmp_path, "rb") as fb:
            batch_file = await judge._openai.files.create(file=fb, purpose="batch")
        logger.info(f"파일 업로드 완료: {batch_file.id}")
        batch = await judge._openai.batches.create(
            input_file_id=batch_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
    finally:
        os.unlink(tmp_path)
    return batch.id


def _batch_ckpt_path(output_file: Path) -> Path:
    return output_file.with_name(output_file.stem + ".batch.json")


def _write_batch_ckpt(output_file: Path, batch_id: str, judge_model: str, n_pending: int) -> None:
    import datetime
    _batch_ckpt_path(output_file).write_text(json.dumps({
        "batch_id": batch_id,
        "judge_model": judge_model,
        "n_pending": n_pending,
        "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }, ensure_ascii=False, indent=2))


def _read_batch_ckpt(output_file: Path) -> dict:
    ckpt = _batch_ckpt_path(output_file)
    if not ckpt.exists():
        raise FileNotFoundError(f"batch 체크포인트 없음: {ckpt} — 먼저 --batch-submit 하세요")
    return json.loads(ckpt.read_text())


async def _download_batch_results(judge: "LLMJudge", batch) -> dict:
    """완료된 batch의 출력·에러 파일 → {custom_id: content|None}. 실패 항목은 None.
    실패 요청은 출력 파일이 아니라 에러 파일에만 실린다(예: 429 insufficient_quota)."""
    batch_results: dict[str, str | None] = {}
    if batch.output_file_id:
        result_content = await judge._openai.files.content(batch.output_file_id)
        for line in result_content.text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            cid  = item["custom_id"]
            if item["response"]["status_code"] == 200:
                batch_results[cid] = item["response"]["body"]["choices"][0]["message"]["content"].strip()
            else:
                logger.warning(f"Batch item 실패: {cid}")
                batch_results[cid] = None
    if getattr(batch, "error_file_id", None):
        err_content = await judge._openai.files.content(batch.error_file_id)
        for line in err_content.text.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            resp = item.get("response") or {}
            err  = item.get("error") or (resp.get("body") or {}).get("error") or {}
            logger.warning(f"Batch item 실패: {item['custom_id']} "
                           f"({resp.get('status_code')} {err.get('code') or err.get('type')})")
            batch_results[item["custom_id"]] = None
    logger.info(f"Batch 결과 수신: {sum(v is not None for v in batch_results.values())}건 "
                f"(실패 {sum(v is None for v in batch_results.values())}건)")
    return batch_results


async def _resolve_failed_requests(judge: "LLMJudge", pending: list[dict], batch_results: dict,
                                   retry: bool) -> None:
    """batch 결과가 없는 요청(실패·누락)을 처리한다. 결과 없이 조립하면 claim 이 전부 neutral
    ('batch error') → faith 0·환각으로 기록되므로 그대로 두지 않는다.
    retry 면 같은 요청 본문을 동기로 다시 보내 채우고(과금), 아니면 출력을 쓰기 전에 멈춘다."""
    failed = [r for r in pending if batch_results.get(r["custom_id"]) is None]
    if not failed:
        return
    ids = ", ".join(r["custom_id"] for r in failed[:10]) + (" …" if len(failed) > 10 else "")
    if not retry:
        raise RuntimeError(f"batch 결과가 없는 요청 {len(failed)}건 ({ids}) — 출력을 쓰지 않았다. "
                           "--batch-retry-failed 로 다시 수집하면 같은 요청을 동기로 재판정한다(과금).")
    for r in failed:
        resp = await judge._openai.chat.completions.create(**r["body"])
        batch_results[r["custom_id"]] = resp.choices[0].message.content.strip()
    logger.info(f"결과 없던 요청 {len(failed)}건을 같은 본문으로 동기 재판정: {ids}")


async def _poll_and_download(judge: "LLMJudge", batch_id: str, wait: bool) -> dict | None:
    """batch 상태 확인. wait=False면 미완료 시 None 반환(재실행 가능). 완료 시 결과 dict."""
    batch = await judge._openai.batches.retrieve(batch_id)
    while batch.status not in ("completed", "failed", "expired", "cancelled"):
        if not wait:
            rc = batch.request_counts
            logger.info(f"아직 처리 중: status={batch.status} 완료={rc.completed}/{rc.total} "
                        f"— 나중에 --batch-collect 재실행")
            return None
        await asyncio.sleep(30)
        batch = await judge._openai.batches.retrieve(batch_id)
        rc = batch.request_counts
        logger.info(f"  status={batch.status} 완료={rc.completed} 실패={rc.failed} 전체={rc.total}")
    if batch.status != "completed":
        raise RuntimeError(f"Batch 실패: status={batch.status}, id={batch_id}")
    return await _download_batch_results(judge, batch)


def _assemble_batch_output(parsed: list[tuple], batch_results: dict, gold_answers: dict,
                           output_file: Path, save_rationale: bool,
                           reuse: dict | None = None) -> None:
    """batch 결과 dict + parsed 홉 → annotated JSONL 출력 작성."""
    with open(output_file, "w", encoding="utf-8") as f:
        for traj, hop_parses in parsed:
            qid = traj.get("qid", "?")
            hop_anns: list[HopAnnotation] = []
            hop_flags = {
                step["hop"]: {k: step[k] for k in _DYNAMIC_FLAG_KEYS if k in step}
                for step in traj["trajectory"]
            }

            for hop_idx, pr, passages, langs, gold_lang, switch, sub_q in hop_parses:
                gen_text = traj["trajectory"][hop_idx].get("generated_text", "")

                if pr is None:  # api_error
                    claim_labels = []
                    faithfulness = float("nan")
                    has_hallu    = False
                    extract_src  = "api_error"
                    abstained    = True
                    ambiguous    = False
                    parse_issue  = "api_error"
                    marker_count = 0
                elif (src := _reused_hop(reuse, qid, hop_idx, traj["trajectory"][hop_idx])) is not None:
                    # seed 홉 — 원본 조건의 판정을 파서 판정까지 그대로 쓴다 (judge 요청 없음)
                    claim_labels = _reused_labels(reuse, qid, hop_idx, traj["trajectory"][hop_idx],
                                                  save_rationale)
                    faithfulness = (sum(1 for cl in claim_labels if cl.label == LABEL_SUPPORTED)
                                    / len(claim_labels)) if claim_labels else float("nan")
                    has_hallu    = bool(claim_labels) and faithfulness < FAITH_HALLU_THRESHOLD
                    abstained, ambiguous, extract_src, parse_issue, marker_count = _reused_parse(
                        src, pr.abstained, pr.ambiguous_answer_marker, pr.source,
                        pr.parse_issue, pr.answer_marker_count)
                elif pr.abstained or pr.ambiguous_answer_marker or not pr.claims:
                    claim_labels = []
                    faithfulness = float("nan")
                    has_hallu    = False
                    extract_src  = pr.source
                    abstained    = pr.abstained
                    ambiguous    = pr.ambiguous_answer_marker
                    parse_issue  = pr.parse_issue
                    marker_count = pr.answer_marker_count
                elif not passages:
                    claim_labels = [ClaimLabel(c, LABEL_NEUTRAL, "no passages") for c in pr.claims]
                    total        = len(claim_labels) or 1
                    faithfulness = sum(1 for cl in claim_labels if cl.label == LABEL_SUPPORTED) / total
                    has_hallu    = faithfulness < FAITH_HALLU_THRESHOLD
                    extract_src  = pr.source
                    abstained    = False
                    ambiguous    = pr.ambiguous_answer_marker
                    parse_issue  = pr.parse_issue
                    marker_count = pr.answer_marker_count
                else:
                    cid          = f"{qid}__hop{hop_idx}"
                    claim_labels = _parse_verdicts(batch_results.get(cid), pr.claims, save_rationale)
                    total        = len(claim_labels) or 1
                    faithfulness = sum(1 for cl in claim_labels if cl.label == LABEL_SUPPORTED) / total
                    has_hallu    = faithfulness < FAITH_HALLU_THRESHOLD
                    extract_src  = pr.source
                    abstained    = False
                    ambiguous    = pr.ambiguous_answer_marker
                    parse_issue  = pr.parse_issue
                    marker_count = pr.answer_marker_count

                hop_anns.append(HopAnnotation(
                    hop=hop_idx, generated_text=gen_text, retrieved_langs=langs,
                    gold_lang=gold_lang, lang_switch=switch, abstained=abstained,
                    claims=claim_labels, hop_faithfulness=faithfulness,
                    has_hallucination=has_hallu, claim_extraction_source=extract_src,
                    answer_marker_count=marker_count, ambiguous_answer_marker=ambiguous,
                    parse_issue=parse_issue, dynamic_flags=hop_flags.get(hop_idx, {}),
                ))

            clepr        = compute_clepr(hop_anns)
            final_ans    = traj.get("final_answer", "")
            gold         = gold_answers.get(qid, "")
            norm_gold    = normalize_entity(gold) if gold else ""
            answer_ok    = bool(norm_gold and norm_gold in normalize_entity(final_ans))
            ret_failed   = any(not s["retrieval_success"] for s in traj["trajectory"])

            ann = TrajectoryAnnotation(
                qid=qid, model=traj.get("model", "unknown"),
                pattern=traj.get("pattern", "?"), hops=hop_anns,
                cross_lingual_error_propagation=clepr,
                abstained=(hop_anns[-1].abstained if hop_anns else _check_abstain(final_ans)),
                retrieval_failed=ret_failed,
                final_answer_correct=answer_ok,
            )
            f.write(json.dumps(ann_to_dict(ann, save_rationale=save_rationale),
                               ensure_ascii=False, allow_nan=False) + "\n")
            f.flush()


async def submit_batch_annotation(trajs: list[dict], judge: "LLMJudge",
                                  sub_q_map: dict, output_file: Path,
                                  reuse: dict | None = None) -> str | None:
    """제출만 수행하고 batch_id를 체크포인트에 저장 후 반환 (24h 창 분리)."""
    pending, _ = _collect_batch_requests(trajs, judge, sub_q_map, reuse)
    logger.info(f"Batch 요청 {len(pending)}개 구성 / 전체 궤적 {len(trajs)}개")
    if not pending:
        logger.warning("판정 대상 홉이 없습니다 — 제출 생략")
        return None
    batch_id = await _submit_batch(judge, pending)
    _write_batch_ckpt(output_file, batch_id, judge.model, len(pending))
    logger.info(f"Batch 제출 완료: {batch_id}")
    logger.info("  → 동일 --output 으로 나중에 --batch-collect 실행해 결과를 수집하세요.")
    return batch_id


async def collect_batch_annotation(trajs: list[dict], judge: "LLMJudge",
                                   gold_answers: dict, sub_q_map: dict,
                                   output_file: Path, save_rationale: bool,
                                   wait: bool, reuse: dict | None = None,
                                   retry_failed: bool = False) -> bool:
    """제출된 batch 결과를 수집해 annotated 출력 작성. 미완료(wait=False)면 False 반환."""
    info = _read_batch_ckpt(output_file)
    batch_id = info["batch_id"]
    if info.get("judge_model") and info["judge_model"] != judge.model:
        logger.warning(f"judge_model 불일치: 제출={info['judge_model']} 현재={judge.model}")
    batch_results = await _poll_and_download(judge, batch_id, wait=wait)
    if batch_results is None:
        return False
    pending, parsed = _collect_batch_requests(trajs, judge, sub_q_map, reuse)
    await _resolve_failed_requests(judge, pending, batch_results, retry_failed)
    _assemble_batch_output(parsed, batch_results, gold_answers, output_file, save_rationale, reuse)
    logger.info(f"Batch annotation 완료: {output_file}")
    return True


async def run_batch_annotation(
    trajs: list[dict],
    judge: "LLMJudge",
    gold_answers: dict,
    sub_q_map: dict,
    output_file: Path,
    save_rationale: bool,
    reuse: dict | None = None,
    retry_failed: bool = False,
) -> None:
    """OpenAI Batch API blocking 모드 — 제출+대기+수집을 한 프로세스에서 수행."""
    pending, parsed = _collect_batch_requests(trajs, judge, sub_q_map, reuse)
    logger.info(f"Batch 요청 {len(pending)}개 구성 / 전체 궤적 {len(trajs)}개")
    if not pending:
        _assemble_batch_output(parsed, {}, gold_answers, output_file, save_rationale, reuse)
        logger.info(f"Batch annotation 완료: {output_file}")
        return

    batch_id = await _submit_batch(judge, pending)
    _write_batch_ckpt(output_file, batch_id, judge.model, len(pending))
    logger.info(f"Batch 생성: {batch_id} — 완료 시까지 30초 간격 polling")

    batch_results = await _poll_and_download(judge, batch_id, wait=True)
    await _resolve_failed_requests(judge, pending, batch_results, retry_failed)
    _assemble_batch_output(parsed, batch_results, gold_answers, output_file, save_rationale, reuse)
    logger.info(f"Batch annotation 완료: {output_file}")


def run_parser_diagnostic(trajs: list[dict], hop_output: Path | None,
                          qa_pairs_path: Path | None = None) -> None:
    """parse_generated_answer()만 실행하고 파서 통계를 출력. LLM judge 호출 없음."""
    from collections import Counter

    # gold_map: (qid, hop_idx) → gold_entity 문자열
    gold_map: dict[tuple, str] = {}
    if qa_pairs_path and qa_pairs_path.exists():
        for line in qa_pairs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            qa = json.loads(line)
            for h in qa.get("hops", []):
                gold_map[(qa["qid"], h["hop"])] = h.get("gold_entity", "")

    total_hops = 0
    source_counts: Counter = Counter()
    abstain_count = 0
    ambiguous_count = 0
    multi_answer_count = 0
    no_claim_non_abstain_count = 0
    execution_artifact_count = 0
    api_error_count = 0
    empty_content_count = 0
    empty_content_with_reasoning_count = 0
    thinking_tag_leak_count = 0
    parseable_answer_or_abstain_count = 0
    judge_target_count = 0
    gold_hit_count = 0
    gold_hit_eligible = 0  # judge_target이면서 gold_map에 키가 있는 홉
    abstain_gold_hit_count = 0  # 기권했는데 gold가 passage에 있던 홉
    parse_issue_counts: Counter = Counter()
    marker_dist: Counter = Counter()
    by_pattern: dict[str, dict] = defaultdict(lambda: {
        "total": 0, "abstain": 0, "ambiguous": 0,
        "judge_target": 0, "no_claim_non_abstain": 0, "execution_artifact": 0,
    })

    hop_rows: list[dict] = []

    for traj in trajs:
        qid     = traj.get("qid", "?")
        pattern = traj.get("pattern", "?")
        for step in traj.get("trajectory", []):
            gen_text = step.get("generated_text", "")
            hop_idx  = step.get("hop", total_hops)
            pr = parse_generated_answer(gen_text)
            jt = is_judge_target(pr, step)

            total_hops += 1
            source_counts[pr.source] += 1
            if pr.abstained:
                abstain_count += 1
            if pr.ambiguous_answer_marker:
                ambiguous_count += 1
            if pr.multi_answer:
                multi_answer_count += 1

            _ea = (
                step.get("api_error", False)
                or step.get("empty_content", False)
                or step.get("empty_content_with_reasoning", False)
                or step.get("thinking_tag_leak", False)
            )
            if _ea:
                execution_artifact_count += 1
            if step.get("api_error", False):
                api_error_count += 1
            if step.get("empty_content", False):
                empty_content_count += 1
            if step.get("empty_content_with_reasoning", False):
                empty_content_with_reasoning_count += 1
            if step.get("thinking_tag_leak", False):
                thinking_tag_leak_count += 1

            if not pr.abstained and not pr.ambiguous_answer_marker and not _ea and not pr.claims:
                no_claim_non_abstain_count += 1
            if pr.abstained or jt:
                parseable_answer_or_abstain_count += 1
            if jt:
                judge_target_count += 1

            # gold_hit: judge 대상 홉에서 파싱된 claim이 gold를 포함하는지
            gold = gold_map.get((qid, hop_idx), "")
            gold_hit: bool | None = None
            if jt and gold:
                norm_gold = normalize_entity(gold)
                gold_hit = any(
                    norm_gold in normalize_entity(c) or normalize_entity(c) in norm_gold
                    for c in pr.claims
                )
                gold_hit_eligible += 1
                if gold_hit:
                    gold_hit_count += 1

            # abstain_gold_alias_hit: 기권했는데 gold가 retrieved passage에 있었는지
            abstain_gold_alias_hit: bool | None = None
            if pr.abstained and gold:
                norm_gold = normalize_entity(gold)
                passages = step.get("retrieved_passages", [])
                abstain_gold_alias_hit = any(
                    norm_gold in normalize_entity(p) for p in passages if p
                )
                if abstain_gold_alias_hit:
                    abstain_gold_hit_count += 1

            if pr.parse_issue:
                parse_issue_counts[pr.parse_issue] += 1
            # 마커 수는 보조 진단 — V2.3 이후 마커 미사용 모델에서는 0이 정상
            mk = pr.answer_marker_count if pr.answer_marker_count <= 1 else "2+"
            marker_dist[mk] += 1

            bp = by_pattern[pattern]
            bp["total"] += 1
            if pr.abstained:
                bp["abstain"] += 1
            if pr.ambiguous_answer_marker:
                bp["ambiguous"] += 1
            if jt:
                bp["judge_target"] += 1
            if not pr.abstained and not pr.ambiguous_answer_marker and not _ea and not pr.claims:
                bp["no_claim_non_abstain"] += 1
            if _ea:
                bp["execution_artifact"] += 1

            if hop_output is not None:
                hop_rows.append({
                    "qid":                    qid,
                    "pattern":                pattern,
                    "hop":                    hop_idx,
                    "source":                 pr.source,
                    "abstained":              pr.abstained,
                    "ambiguous_answer_marker": pr.ambiguous_answer_marker,
                    "multi_answer":           pr.multi_answer,
                    "answer_marker_count":    pr.answer_marker_count,
                    "parse_issue":            pr.parse_issue,
                    "claim_count":            len(pr.claims),
                    "claims_preview":         [c[:100] for c in pr.claims],
                    "is_judge_target":        jt,
                    "gold_entity":            gold or None,
                    "gold_hit":               gold_hit,
                    "abstain_gold_alias_hit": abstain_gold_alias_hit,
                })

    n_trajs = len(trajs)
    pct = lambda n: f"{n/total_hops*100:5.1f}%" if total_hops else "  N/A "

    print("\n" + "=" * 52)
    print("  Parser Diagnostic")
    print("=" * 52)
    print(f"  궤적: {n_trajs}개  /  홉: {total_hops}개")

    print("\n[ source 분포 ]")
    source_order = [
        "answer_section", "answer_section_short", "answer_section_abstain",
        "multi_answer_ambiguous",
        "fallback_cleaned", "fallback_whole", "fallback_short", "fallback_abstain",
        "artifact", "none",
    ]
    for src in source_order + [s for s in source_counts if s not in source_order]:
        n = source_counts.get(src, 0)
        if n:
            print(f"  {src:<28}: {n:4}  ({pct(n)})")

    # primary: nonempty율, 기권율, 애매율 / secondary: 마커 수
    print("\n[ 마커 수 분포 ]")
    for key in [0, 1, "2+"]:
        n = marker_dist.get(key, 0)
        label = f"{key}개" if key != "2+" else "2개+"
        print(f"  {label:<6}: {n:4}  ({pct(n)})")

    print("\n[ 요약 ]")
    print(f"  기권 홉            : {abstain_count:4}  ({pct(abstain_count)})")
    print(f"  애매 홉            : {ambiguous_count:4}  ({pct(ambiguous_count)})")
    print(f"  복수 답변(multi)   : {multi_answer_count:4}  ({pct(multi_answer_count)})")
    print(f"  claim 없음(비기권) : {no_claim_non_abstain_count:4}  ({pct(no_claim_non_abstain_count)})")
    print(f"  실행 아티팩트      : {execution_artifact_count:4}  ({pct(execution_artifact_count)})")
    if execution_artifact_count:
        print(f"    api_error              : {api_error_count:4}")
        print(f"    empty_content          : {empty_content_count:4}")
        print(f"    empty_content_w_reason : {empty_content_with_reasoning_count:4}")
        print(f"    thinking_tag_leak      : {thinking_tag_leak_count:4}")
    print(f"  파서블(답변+기권)  : {parseable_answer_or_abstain_count:4}  ({pct(parseable_answer_or_abstain_count)})")
    print(f"  judge 대상 홉      : {judge_target_count:4}  ({pct(judge_target_count)})")
    if gold_hit_eligible:
        gold_hit_rate = gold_hit_count / gold_hit_eligible
        print(f"\n[ gold_hit (judge 대상 홉 중 gold 포함) ]")
        print(f"  gold_hit           : {gold_hit_count:4} / {gold_hit_eligible}  ({gold_hit_rate:.1%})")
    if abstain_count:
        pct_abs = lambda n: f"{n/abstain_count*100:5.1f}%" if abstain_count else "  N/A "
        print(f"\n[ 기권 과분석 — over-abstain 여부 ]")
        print(f"  기권 홉 총계                  : {abstain_count:4}")
        if gold_map:
            print(f"  기권 중 gold 있었던 홉 (over) : {abstain_gold_hit_count:4}  ({pct_abs(abstain_gold_hit_count)})")
            print(f"  기권 중 gold 없었던 홉 (적절) : {abstain_count - abstain_gold_hit_count:4}  ({pct_abs(abstain_count - abstain_gold_hit_count)})")
        else:
            print("  (--qa-pairs 없이 실행 — gold_hit 계산 불가)")
    if parse_issue_counts:
        print("\n[ parse_issue 분포 ]")
        for issue, n in parse_issue_counts.most_common():
            print(f"  {issue:<36}: {n:4}  ({pct(n)})")

    print("\n[ 패턴별 기권율 / 애매율 / judge대상율 ]")
    for pat in sorted(by_pattern):
        bp = by_pattern[pat]
        t  = bp["total"] or 1
        print(f"  {pat}: 기권 {bp['abstain']/t:.1%}  애매 {bp['ambiguous']/t:.1%}  judge대상 {bp['judge_target']/t:.1%}  (n={bp['total']})")

    # ── 파일 출력 ────────────────────────────────────────────────
    if hop_output is not None:
        hop_output.parent.mkdir(parents=True, exist_ok=True)
        with open(hop_output, "w", encoding="utf-8") as f:
            for row in hop_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\n  홉별 JSONL → {hop_output}")

        summary = {
            "total_trajs":        n_trajs,
            "total_hops":         total_hops,
            "abstain_count":      abstain_count,
            "ambiguous_count":    ambiguous_count,
            "multi_answer_count": multi_answer_count,
            "no_claim_non_abstain_count":          no_claim_non_abstain_count,
            "execution_artifact_count":            execution_artifact_count,
            "api_error_count":                     api_error_count,
            "empty_content_count":                 empty_content_count,
            "empty_content_with_reasoning_count":  empty_content_with_reasoning_count,
            "thinking_tag_leak_count":             thinking_tag_leak_count,
            "parseable_answer_or_abstain_count":   parseable_answer_or_abstain_count,
            "judge_target":                        judge_target_count,
            "gold_hit_count":                      gold_hit_count,
            "gold_hit_eligible":                   gold_hit_eligible,
            "gold_hit_rate":                       gold_hit_count / gold_hit_eligible if gold_hit_eligible else None,
            "abstain_gold_hit_count":              abstain_gold_hit_count,
            "source_counts":      dict(source_counts),
            "marker_count_dist":  {str(k): v for k, v in marker_dist.items()},
            "parse_issue_counts": dict(parse_issue_counts),
            "by_pattern":         {p: dict(v) for p, v in by_pattern.items()},
        }
        summary_path = hop_output.with_name(hop_output.stem + "_summary.json")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"  요약 JSON   → {summary_path}")

    print()


# judge 모델별 단가 ($/1M tok, input/output). 출처: frontier_test_plan §5.2 실측 확정 단가.
_JUDGE_PRICES = {
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4":      (2.50, 15.00),
    "claude":       (3.00, 15.00),
    "gemini":       (1.50, 9.00),
}


def preflight_summary(trajs: list[dict], output_file: Path, model: str) -> None:
    n_traj = len(trajs)
    n_hops = sum(len(t.get("trajectory", [])) for t in trajs)
    est_in = sum(
        sum(min(len(p), 2500) for p in h.get("retrieved_passages", [])) // 4
        + len(h.get("generated_text", "")) // 4
        for t in trajs for h in t.get("trajectory", [])
    )
    est_out  = n_hops * 60   # 홉당 평균 2 claims × 30 tokens
    # 단가는 judge 모델에 맞춘다. 이전에는 $3/$15(Sonnet)가 하드코딩돼 있어
    # 기본 judge(gpt-5.4-mini, $0.75/$4.50)의 비용을 4배 과대 추정했다.
    price_in, price_out = _JUDGE_PRICES.get(
        next((k for k in _JUDGE_PRICES if model.startswith(k)), ""), (3.0, 15.0))
    cost_usd = est_in * price_in * 1e-6 + est_out * price_out * 1e-6
    print("=" * 52)
    print(f"  모델       : {model}")
    print(f"  출력 파일  : {output_file}")
    print(f"  trajectory : {n_traj}개  /  API 호출 : {n_hops}회")
    print(f"  예상 토큰  : 입력 {est_in:,}  출력 {est_out:,}")
    print(f"  예상 비용  : ${cost_usd:.2f} USD  (±50%)")
    print("=" * 52)


async def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--trajectories", default=None, help="에이전트 실행 결과 JSONL (미지정 시 config.yaml run.trajectories 사용)")
    parser.add_argument("--output",       default=None, help="출력 파일 경로 (미지정 시 config.yaml output.annotated 사용)")
    parser.add_argument("--api-key",        default=os.environ.get("ANTHROPIC_API_KEY"), help="Anthropic API key")
    parser.add_argument("--gemini-api-key", default=os.environ.get("GEMINI_API_KEY"),   help="Google Gemini API key")
    parser.add_argument("--openai-api-key", default=os.environ.get("OPENAI_API_KEY"),   help="OpenAI API key")
    parser.add_argument("--max",          type=int, default=None, help="처리할 최대 궤적 수 (테스트용)")
    parser.add_argument("--dry-run",      action="store_true",   help="API 호출 없이 예상 비용 요약만 출력")
    parser.add_argument("-y", "--yes",    action="store_true",   help="실행 전 확인 프롬프트 스킵")
    parser.add_argument("--batch",        action="store_true",
                        help="OpenAI Batch API 사용 (비용 ~50%% 절감, 24h 완료 창, OpenAI 전용, blocking)")
    parser.add_argument("--batch-submit", action="store_true",
                        help="Batch 제출만 하고 batch_id를 {output}.batch.json에 저장 후 종료 (24h 창 분리)")
    parser.add_argument("--batch-collect", action="store_true",
                        help="제출된 batch 결과를 수집해 annotated 출력 작성 (미완료면 상태 보고 후 종료)")
    parser.add_argument("--batch-wait", action="store_true",
                        help="--batch-collect 시 완료까지 30초 간격으로 대기 (기본: 미완료면 즉시 종료)")
    parser.add_argument("--batch-retry-failed", action="store_true",
                        help="batch 에서 결과가 없는 요청(실패·누락)을 같은 본문으로 동기 재판정한다(과금). "
                             "없으면 그런 요청이 있을 때 출력을 쓰지 않고 멈춘다")
    parser.add_argument("--save-rationale", action="store_true",
                        help="judge 판정 이유(reason)를 annotated 파일에 저장 (기본: 생략하여 output token 절감)")
    parser.add_argument("--parser-diagnostic", action="store_true",
                        help="LLM judge 없이 parse_generated_answer()만 실행하고 파서 통계를 출력")
    parser.add_argument("--parser-diagnostic-output", default=None, metavar="PATH",
                        help="홉별 파서 결과를 저장할 JSONL 경로 (미지정 시 stdout만)")
    parser.add_argument("--reuse-hop-labels", nargs=2, default=None,
                        metavar=("ANNOTATED", "TRAJECTORIES"),
                        help="seed 로 재사용한 hop0 의 judge 라벨을 그대로 가져온다 (hop1 은 항상 재채점). "
                             "두 인자는 원본 조건의 annotated·trajectory 파일. 생성물이 다르면 그 홉은 정상 판정.")
    parser.add_argument("--qa-pairs", default=None, metavar="PATH",
                        help="gold_hit 계산용 QA pairs JSONL (미지정 시 config.yaml output.qa_pairs 사용)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    ann_cfg = cfg["annotation"]

    trajs_file = Path(args.trajectories or cfg["output"]["trajectories"])
    if not trajs_file.exists():
        logger.error(f"궤적 파일 없음: {trajs_file}")
        return

    output_file = Path(args.output or cfg["output"]["annotated"])
    output_file.parent.mkdir(parents=True, exist_ok=True)

    qa_file = Path(args.qa_pairs) if args.qa_pairs else Path(cfg["output"]["qa_pairs"])
    gold_answers: dict[str, str] = {}
    sub_q_map: dict[tuple, str] = {}   # (qid, hop_idx) → sub_q
    if qa_file.exists():
        for line in qa_file.read_text().splitlines():
            if line.strip():
                qa = json.loads(line)
                gold_answers[qa["qid"]] = qa["answer"].lower()
                for h in qa.get("hops", []):
                    sq = h.get("sub_q", "")
                    if sq:
                        sub_q_map[(qa["qid"], h["hop"])] = sq
        logger.info(f"gold answer {len(gold_answers)}개, sub_q {len(sub_q_map)}개 로드")
    else:
        logger.warning(f"QA 파일 없음: {qa_file} → final_answer_correct / sub_q 판정 불가")

    # ── parser-diagnostic: API key 불필요, 조기 종료 ────────────
    if args.parser_diagnostic:
        trajs = [json.loads(l) for l in trajs_file.read_text().splitlines() if l.strip()]
        if args.max:
            trajs = trajs[:args.max]
        hop_out = Path(args.parser_diagnostic_output) if args.parser_diagnostic_output else None
        qa_pairs_p = Path(args.qa_pairs) if args.qa_pairs else Path(cfg["output"]["qa_pairs"])
        run_parser_diagnostic(trajs, hop_out, qa_pairs_path=qa_pairs_p)
        return

    judge_model = ann_cfg["judge_model"]
    if judge_model.startswith("gemini"):
        if not args.gemini_api_key:
            parser.error("Gemini 모델에는 --gemini-api-key 또는 GEMINI_API_KEY 환경변수 필요")
        judge_key = args.gemini_api_key
    elif judge_model.startswith("gpt") or judge_model.startswith("o"):
        if not args.openai_api_key:
            parser.error("OpenAI 모델에는 --openai-api-key 또는 OPENAI_API_KEY 환경변수 필요")
        judge_key = args.openai_api_key
    else:
        if not args.api_key:
            parser.error("--api-key 또는 ANTHROPIC_API_KEY 환경변수 필요")
        judge_key = args.api_key

    judge = LLMJudge(model=judge_model, api_key=judge_key)

    reuse = None
    if args.reuse_hop_labels:
        ann_src, traj_src = args.reuse_hop_labels
        reuse = load_reused_hop_labels(ann_src, traj_src)
        logger.info(f"홉 라벨 재사용 후보 {len(reuse)}건 ← {ann_src} (+ {traj_src})")

    trajs = [json.loads(l) for l in trajs_file.read_text().splitlines() if l.strip()]
    if args.max:
        trajs = trajs[:args.max]

    # ── Annotation 메타 (재현성 보장) ────────────────────────────
    is_openai = judge_model.startswith("gpt") or judge_model.startswith("o")
    ann_meta = {
        "judge_model": judge_model,
        "judge_prompt_version": JUDGE_PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "temperature": 0,
        **({"seed": LLMJudge._OPENAI_SEED} if is_openai else {}),
    }
    meta_file = output_file.with_name(output_file.stem + ".meta.json")

    # ── 디커플링 Batch: submit / collect 분리 (24h 창 babysitting 제거) ──
    # full trajs를 사용하므로 done_qids resume 필터 이전에 처리한다.
    if args.batch_submit or args.batch_collect:
        if not is_openai:
            parser.error("--batch-submit/--batch-collect 는 OpenAI judge 모델 전용입니다.")
        if args.batch_submit and args.batch_collect:
            parser.error("--batch-submit 와 --batch-collect 는 동시에 사용할 수 없습니다.")
        if args.batch_submit:
            preflight_summary(trajs, output_file, judge_model)
            if args.dry_run:
                print("[dry-run] 제출 없이 종료합니다.")
                return
            if not args.yes:
                if input("batch 제출하시겠습니까? [y/N]: ").strip().lower() != "y":
                    print("취소됐습니다.")
                    return
            meta_file.write_text(json.dumps(ann_meta, ensure_ascii=False, indent=2))
            await submit_batch_annotation(trajs, judge, sub_q_map, output_file, reuse)
        else:
            ok = await collect_batch_annotation(
                trajs, judge, gold_answers, sub_q_map,
                output_file, args.save_rationale, wait=args.batch_wait, reuse=reuse,
                retry_failed=args.batch_retry_failed)
            if ok:
                print(f"\n라벨링 완료: {output_file}")
            else:
                print("아직 처리 중입니다 — 나중에 --batch-collect 를 다시 실행하세요.")
        return

    # ── Resume: 이미 완료된 QID 스킵 ─────────────────────────────
    done_qids: set[str] = set()
    if output_file.exists():
        for line in output_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_qids.add(json.loads(line)["qid"])
        if done_qids:
            logger.info(f"이미 완료: {len(done_qids)}개 스킵")
            # 재실행 시 설정 일치 여부 확인
            if meta_file.exists():
                existing = json.loads(meta_file.read_text())
                mismatches = {k: {"기존": existing.get(k), "현재": v}
                              for k, v in ann_meta.items() if existing.get(k) != v}
                if mismatches:
                    logger.error(f"재실행 설정 불일치: {mismatches}")
                    logger.error("다른 --output 경로를 지정하거나 기존 파일을 삭제 후 재실행하세요.")
                    return
    trajs = [t for t in trajs if t["qid"] not in done_qids]

    # 새 실행이면 메타 파일 기록
    if not done_qids:
        meta_file.write_text(json.dumps(ann_meta, ensure_ascii=False, indent=2))
        logger.info(f"Annotation 메타 기록: {meta_file}")

    # ── Pre-flight 요약 ───────────────────────────────────────────
    preflight_summary(trajs, output_file, judge_model)
    if args.dry_run:
        print("[dry-run] 실제 API 호출 없이 종료합니다.")
        return
    if not args.yes:
        ans = input("계속하시겠습니까? [y/N]: ").strip().lower()
        if ans != "y":
            print("취소됐습니다.")
            return

    # ── Batch API 모드 ────────────────────────────────────────────
    if args.batch:
        if not (judge_model.startswith("gpt") or judge_model.startswith("o")):
            parser.error("--batch는 OpenAI 모델 전용입니다.")
        if done_qids:
            logger.warning("--batch 모드는 resume을 지원하지 않습니다. 전체 재실행됩니다.")
        logger.info(f"Batch API 모드 시작: {len(trajs)}개 궤적")
        await run_batch_annotation(
            trajs, judge, gold_answers, sub_q_map, output_file, args.save_rationale, reuse,
            retry_failed=args.batch_retry_failed,
        )
    else:
        # ── Standard 비동기 모드 ──────────────────────────────────
        max_concurrent = ann_cfg.get("max_concurrent", 10)
        logger.info(f"궤적 {len(trajs)}개 처리 시작 (동시 처리: {max_concurrent})")
        sem        = asyncio.Semaphore(max_concurrent)
        write_lock = asyncio.Lock()

        write_mode = "a" if done_qids else "w"
        with open(output_file, write_mode, encoding="utf-8") as f:
            async def annotate_and_write(traj):
                await asyncio.sleep(random.uniform(0, 1.0))
                async with sem:
                    ann = await annotate_trajectory(traj, judge, gold_answers, sub_q_map, reuse)
                async with write_lock:
                    f.write(json.dumps(ann_to_dict(ann, args.save_rationale),
                                       ensure_ascii=False, allow_nan=False) + "\n")
                    f.flush()

            await atqdm.gather(*[annotate_and_write(t) for t in trajs], desc="궤적 라벨링")

    # ── 요약 통계 출력 ────────────────────────────────────────────
    results = [json.loads(l) for l in output_file.read_text().splitlines() if l.strip()]

    print("\n" + "="*60)
    print("패턴별 언어전환 홉 vs 비전환 홉 충실성 비교 (LSHD)")
    print("="*60)
    pat_faith: dict[str, dict] = defaultdict(lambda: {"switch": [], "no_switch": []})
    abstain_by_pat: dict[str, list] = defaultdict(list)

    for r in results:
        pat = r["pattern"]
        for h in r["hops"]:
            abstain_by_pat[pat].append(h.get("abstained", False))
            if h.get("abstained") or h["hop_faithfulness"] is None:
                continue
            if h["lang_switch"]:
                pat_faith[pat]["switch"].append(h["hop_faithfulness"])
            else:
                pat_faith[pat]["no_switch"].append(h["hop_faithfulness"])

    labels = {"a": "KO→KO→KO", "b": "KO→EN→KO", "c": "KO→KO→EN", "d": "KO→EN→EN"}
    for pat in sorted(pat_faith):
        sw   = pat_faith[pat]["switch"]
        nsw  = pat_faith[pat]["no_switch"]
        cr   = _clepr_rates([r for r in results if r["pattern"] == pat])
        abstain_rate = statistics.mean(abstain_by_pat[pat]) if abstain_by_pat[pat] else float("nan")

        print(f"  패턴 {pat} ({labels.get(pat,'?')}):")
        print(f"    비전환 홉 FaithRate: {statistics.mean(nsw):.3f}" if nsw else "    비전환 홉: 없음")
        print(f"    전환 홉   FaithRate: {statistics.mean(sw):.3f}" if sw else "    전환 홉: 없음")
        if sw and nsw:
            print(f"    LSHD: {statistics.mean(nsw) - statistics.mean(sw):.3f}")
        elif not sw:
            print(f"    LSHD: N/A (전환 홉 없음 — 베이스라인 패턴)")
        else:
            print(f"    LSHD: N/A (비전환 홉 없음)")
        xling_str = f"{cr['xling']:.3f} (n={cr['xling_n']})" if not math.isnan(cr["xling"]) else "N/A"
        same_str  = f"{cr['same']:.3f}  (n={cr['same_n']})"  if not math.isnan(cr["same"])  else "N/A"
        print(f"    CLEPR_xling: {xling_str}")
        print(f"    CLEPR_same:  {same_str}")
        print(f"    기권율: {abstain_rate:.1%}")

    # 전체 CLEPR_xling vs CLEPR_same (H2 검증)
    print("\n" + "="*60)
    print("전체 CLEPR_xling vs CLEPR_same (H2: 언어전환이 전파를 증폭하는가)")
    print("="*60)
    cr_all = _clepr_rates(results)
    if not math.isnan(cr_all["xling"]) and not math.isnan(cr_all["same"]):
        diff = cr_all["xling"] - cr_all["same"]
        print(f"  CLEPR_xling: {cr_all['xling']:.3f}  (n={cr_all['xling_n']})")
        print(f"  CLEPR_same:  {cr_all['same']:.3f}  (n={cr_all['same_n']})")
        print(f"  차이 (xling - same): {diff:+.3f}  {'← H2 지지' if diff > 0 else '← H2 불지지'}")
    else:
        print("  데이터 부족으로 계산 불가 (환각 홉 쌍이 없거나 기권 처리됨)")

    # actual_model 확정 후 메타 파일 업데이트
    if judge.actual_model and judge.actual_model != judge_model:
        ann_meta["actual_model"] = judge.actual_model
        meta_file.write_text(json.dumps(ann_meta, ensure_ascii=False, indent=2))
        logger.info(f"실제 모델 버전 기록: {judge.actual_model}")

    # Cache 통계 출력 (OpenAI 전용)
    if is_openai and judge._usage_stats["calls"] > 0:
        st = judge._usage_stats
        hit_rate = st["cached_tokens"] / st["prompt_tokens"] if st["prompt_tokens"] else 0
        print(f"\n[Cache 통계] calls={st['calls']}, "
              f"prompt={st['prompt_tokens']:,}, "
              f"cached={st['cached_tokens']:,} ({hit_rate:.1%}), "
              f"completion={st['completion_tokens']:,}")

    print(f"\n라벨링 완료: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
