#!/usr/bin/env python3
"""
pipeline/05_pilot_runner.py
===================
Agentic RAG 실행 스크립트. 파일럿(소규모)과 전체 실험 모두 지원.

1. qa_pairs.jsonl에서 패턴별 샘플링 (config.yaml run.per_pattern: null이면 전체)
2. 각 QA에 대해 agentic RAG (2홉 ReAct 루프) 실행
3. 궤적 저장 → pipeline/06_annotate.py로 LLM-as-judge 라벨링

사용 모델:
  - 리트리버: BGE-M3 + FAISS (pipeline/04_build_faiss.py로 빌드)
  - 에이전트: Anthropic API (config.yaml run.agent_model)

실행:
  python pipeline/05_pilot_runner.py
"""

import json
import re
import random
import argparse
import logging
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import yaml
import torch
import faiss
import requests
import anthropic as _anthropic_sdk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root (utils 패키지)

from utils.lang_switch import compute_lang_switch
from utils.answer_parser import _check_abstain, _ANSWER_MARKER

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BGE_M3_MODEL = "BAAI/bge-m3"


# ─── FAISS 시맨틱 검색 ────────────────────────────────────────────────────────

class FAISSRetriever:
    """임베딩 모델 + FAISS 언어별 패시지 검색기 (기본 BGE-M3).

    인덱스 폴더에 04 가 쓴 retriever.json(model / query_prefix / passage_prefix)이 있으면
    그 인코더를 쓴다 (예: multilingual-E5 는 "query: "/"passage: " 접두사 필수).
    없으면 BGE-M3·접두사 없음 — 정본 인덱스의 동작 그대로다.
    """

    def __init__(self, index_dir: Path):
        if not index_dir.exists():
            raise FileNotFoundError(
                f"FAISS 인덱스 없음: {index_dir}\n"
                "→ python pipeline/04_build_faiss.py 를 먼저 실행하세요."
            )
        manifest_path = index_dir / "retriever.json"
        manifest = (json.loads(manifest_path.read_text(encoding="utf-8"))
                    if manifest_path.exists() else {})
        model_name = manifest.get("model", BGE_M3_MODEL)
        self._query_prefix = manifest.get("query_prefix", "")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"임베딩 모델 로딩: {model_name} (device={device}, "
                    f"query_prefix={self._query_prefix!r})")
        self._model = SentenceTransformer(model_name, device=device)

        self._indexes: dict[str, faiss.Index] = {}
        self._metas:   dict[str, list[dict]]  = {}
        for lang in ["ko", "en"]:
            idx_path  = index_dir / f"{lang}_index.faiss"
            meta_path = index_dir / f"{lang}_meta.json"
            if idx_path.exists() and meta_path.exists():
                self._indexes[lang] = faiss.read_index(str(idx_path))
                self._metas[lang]   = json.loads(meta_path.read_text(encoding="utf-8"))
                logger.info(f"  [{lang}] {self._indexes[lang].ntotal}개 벡터 로드 ({idx_path})")
                if manifest:
                    self._check_encoder(lang, manifest.get("passage_prefix", ""))
            else:
                logger.warning(f"  [{lang}] 인덱스 파일 없음, 스킵")

    def _check_encoder(self, lang: str, passage_prefix: str) -> None:
        """인덱스와 인코더가 짝이 맞는지 확인한다. BGE-M3 와 E5-large 는 둘 다 1024차원이라
        엉뚱한 인코더를 물려도 에러 없이 검색 품질만 무너진다 — 첫 패시지가 자기 자신을 1위로
        찾는지로 잡는다."""
        index, meta = self._indexes[lang], self._metas[lang]
        dim = self._model.get_sentence_embedding_dimension()
        assert index.d == dim, f"[{lang}] 인덱스 차원 {index.d} ≠ 인코더 차원 {dim}"
        vec = self._model.encode([passage_prefix + meta[0]["text"]], normalize_embeddings=True,
                                 show_progress_bar=False).astype("float32")
        _, ids = index.search(vec, 1)
        top = meta[ids[0][0]]["text"] if 0 <= ids[0][0] < len(meta) else None
        assert top == meta[0]["text"], f"[{lang}] 인코더-인덱스 불일치: 자기검색 실패"

    def retrieve(self, query: str, lang: str,
                 top_k: int = 3) -> tuple[list[str], list[str]]:
        """query → (passage_texts, langs)."""
        index = self._indexes.get(lang)
        meta  = self._metas.get(lang)
        if index is None or not meta:
            return [], []

        # 접두사는 인코딩 입력에만 붙인다 (retrieval_query 등 기록되는 쿼리 문자열은 그대로).
        vec = self._model.encode(
            [self._query_prefix + query], normalize_embeddings=True,
            show_progress_bar=False).astype("float32")
        _, ids = index.search(vec, top_k)

        passages = [meta[i]["text"] for i in ids[0] if 0 <= i < len(meta)]
        return passages, [lang] * len(passages)


# ─── ReAct Agentic RAG ────────────────────────────────────────────────────────

# V1: 기존 생각:/검색어:/답변: 형식
SYSTEM_PROMPT_V1 = """당신은 멀티홉 질의응답 에이전트입니다.
제공된 문서를 참고하여 단계별로 추론하세요.
모르는 경우 "모르겠습니다"라고 답하세요.

각 단계에서 다음 형식으로 응답하세요:
생각: <추론 과정>
검색어: <다음 검색에 사용할 키워드>
답변: <현재 홉의 중간 답변>
"""

HOP_PROMPT_V1 = """질문: {question}

현재 단계: {hop_idx}/{total_hops}
검색된 문서:
{passages}

이전 단계 답변: {prev_answer}

위 문서를 바탕으로 현재 단계 답변을 작성하세요."""

# V2: 단순화된 답변: 형식 (생각:/검색어: 제거)
SYSTEM_PROMPT_V2 = """당신은 질문에 단계적으로 답하는 검색 에이전트입니다.
각 단계에서 제공된 문서를 바탕으로 답변하세요.
반드시 마지막 줄을 아래 형식으로 작성하세요:

답변: <이 단계에서 찾은 사실>

문서에서 답을 찾을 수 없으면:
답변: 모르겠습니다."""

HOP_PROMPT_V2 = """문서:
{passages}

질문: {question}"""

# V2.2: 형식 안정화 측정 프롬프트 (출력 노이즈 감소, 기권 강제 없음)
SYSTEM_PROMPT_V22 = """당신은 검색 문서 기반 멀티홉 질의응답 에이전트입니다.

목표:
- 현재 단계 질문에만 답하세요.
- 제공된 문서에 근거가 있는 내용만 사용하세요.
- 답이 확인되면 가장 짧은 이름, 장소, 기관, 작품명, 날짜, 또는 명사구로 답하세요.
- 문서에서 현재 단계의 답을 확인할 수 없을 때만 모르겠다고 답하세요.

출력 규칙:
- 출력은 반드시 한 줄이어야 합니다.
- 설명, 추론, 근거, 목록, 따옴표, 괄호 설명을 쓰지 마세요.
- 아래 형식에서 콜론 앞의 고정 단어를 정확히 한 번만 쓰세요.
- 답을 확인할 수 없을 때는 답 자리에는 모르겠습니다.만 쓰세요.

답변: <현재 단계의 짧은 답>
"""

HOP_PROMPT_V22 = """문서:
{passages}

현재 단계 질문:
{question}

이전 단계 답변:
{prev_answer}

위 문서만 사용해 현재 단계 질문의 답을 한 줄로 작성하세요."""

SYSTEM_PROMPT_V23 = """당신은 검색 문서 기반 멀티홉 질의응답 에이전트입니다.

목표:
- 현재 단계 질문에만 답하세요.
- 제공된 문서에 근거가 있는 내용만 사용하세요.
- 외부 지식이나 추측을 사용하지 마세요.
- 답이 확인되면 가장 짧은 이름, 장소, 기관, 작품명, 날짜, 또는 명사구로 답하세요.
- 문서에서 현재 단계의 답을 확인할 수 없을 때만 모르겠습니다라고 답하세요.

응답 방식:
- 짧은 답 또는 모르겠습니다만 출력하세요.
- 설명, 추론 과정, 근거 인용, 목록은 쓰지 마세요.
- 권장 형식:
답변: <짧은 답>

위 형식을 사용하지 않아도 되지만, 답변 내용은 반드시 짧게 유지하세요.
"""

HOP_PROMPT_V23 = """문서:
{passages}

현재 단계 질문:
{question}

이전 단계 답변:
{prev_answer}

위 문서만 사용해 현재 단계 질문에 답하세요."""

# 활성 프롬프트 (기본: V2)
SYSTEM_PROMPT = SYSTEM_PROMPT_V2
HOP_PROMPT    = HOP_PROMPT_V2


def _select_prompts(prompt_version: str) -> tuple[str, str, int, float | None, float | None]:
    """(sys_prompt, hop_template, max_tokens, temperature, top_p) 반환."""
    if prompt_version in ("v2.3", "v23"):
        return SYSTEM_PROMPT_V23, HOP_PROMPT_V23, 128, 0.0, 1.0
    elif prompt_version in ("v2.2", "v22"):
        return SYSTEM_PROMPT_V22, HOP_PROMPT_V22, 96, 0.0, 1.0
    elif prompt_version == "v1":
        return SYSTEM_PROMPT_V1, HOP_PROMPT_V1, 512, None, None
    else:  # v2 default
        return SYSTEM_PROMPT_V2, HOP_PROMPT_V2, 512, None, None


_MAX_API_RETRIES = 3


def build_prompt(hop_template: str, question: str, hop_idx: int,
                 total_hops: int, passages: list[str], prev_answer: str) -> str:
    """홉 프롬프트 문자열 구성 (동기/배치 공유)."""
    return hop_template.format(
        question=question,
        hop_idx=hop_idx + 1,
        total_hops=total_hops,
        passages="\n\n---\n\n".join(passages),
        prev_answer=prev_answer or "(없음)",
    )


def extract_prev_answer(gen_text: str) -> str:
    """생성 텍스트에서 prev_answer 추출 (06 파서와 동일 마커 규칙). 동기/배치 공유."""
    if _ANSWER_MARKER.search(gen_text):
        return _ANSWER_MARKER.split(gen_text)[-1].strip()
    return gen_text.strip()


def build_claude_kwargs(model: str, system_prompt: str, prompt: str, max_tokens: int,
                        temperature: float | None = None, top_p: float | None = None) -> dict:
    """Anthropic Messages 파라미터 구성 (동기 create·배치 params 공유).

    동기 경로와 byte-동일해야 8B↔frontier 비교 성립 → 단일 진실원.
    temperature/top_p 동시 지정 불가(temperature 우선)."""
    kwargs: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    elif top_p is not None:
        kwargs["top_p"] = top_p
    return kwargs


def build_gpt_body(model: str, system_prompt: str, prompt: str, max_tokens: int,
                   temperature: float | None = None, top_p: float | None = None,
                   gen_seed: int | None = None,
                   reasoning_effort: str | None = None) -> dict:
    """GPT /v1/chat/completions body 구성 (동기 POST·배치 request line 공유).

    gpt-5 계열은 max_completion_tokens + reasoning_effort, temperature/top_p 미지원.
    동기 경로와 byte-동일해야 8B↔frontier 비교가 성립 → 이 함수가 단일 진실원."""
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ],
    }
    if model.startswith("gpt-5"):
        body["max_completion_tokens"] = max_tokens
        if reasoning_effort is not None:
            body["reasoning_effort"] = reasoning_effort
        if gen_seed is not None:
            body["seed"] = gen_seed
    else:
        body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if gen_seed is not None:
            body["seed"] = gen_seed
    return body


def run_agent_hop(question: str, hop_idx: int, total_hops: int,
                  passages: list[str], prev_answer: str,
                  api_key: str, model: str,
                  local_endpoint: str | None = None,
                  system_prompt: str = SYSTEM_PROMPT,
                  hop_template: str = HOP_PROMPT,
                  max_tokens: int = 512,
                  temperature: float | None = None,
                  top_p: float | None = None,
                  gen_seed: int | None = None,
                  reasoning_effort: str | None = None) -> tuple[str, bool, dict | None]:
    """(generated_text, api_error, usage) 반환. api_error=True 시 trajectory에 기록 후 집계 제외.
    usage: 토큰 사용량 dict (gpt-5의 reasoning_tokens 포함) 또는 None (미지원 provider)."""
    prompt = build_prompt(hop_template, question, hop_idx, total_hops, passages, prev_answer)

    for attempt in range(_MAX_API_RETRIES):
        try:
            if model.startswith("claude"):
                kwargs = build_claude_kwargs(model, system_prompt, prompt, max_tokens,
                                             temperature, top_p)
                try:
                    client = _anthropic_sdk.Anthropic(api_key=api_key)
                    msg = client.messages.create(**kwargs)
                    usage = {"input_tokens": msg.usage.input_tokens,
                             "output_tokens": msg.usage.output_tokens,
                             "finish_reason": getattr(msg, "stop_reason", None)}
                    return msg.content[0].text, False, usage
                except _anthropic_sdk.RateLimitError:
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"Anthropic 429 — {wait}초 후 재시도 ({attempt+1}/{_MAX_API_RETRIES})")
                    time.sleep(wait)
                    continue
                except _anthropic_sdk.APIStatusError as e:
                    if e.status_code in (500, 502, 503, 504):
                        wait = 5 * (2 ** attempt)
                        logger.warning(f"Anthropic {e.status_code} — {wait}초 후 재시도 ({attempt+1}/{_MAX_API_RETRIES})")
                        time.sleep(wait)
                        continue
                    raise
            elif model.startswith("gpt-"):
                body = build_gpt_body(model, system_prompt, prompt, max_tokens,
                                      temperature, top_p, gen_seed, reasoning_effort)
                resp = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=60,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"API {resp.status_code} — {wait}초 후 재시도 ({attempt+1}/{_MAX_API_RETRIES})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                j = resp.json()
                usage = dict(j.get("usage") or {})
                usage["finish_reason"] = j["choices"][0].get("finish_reason")
                return j["choices"][0]["message"]["content"], False, usage
            elif model.startswith("gemini"):
                # pipeline/06_annotate.py와 동일 패턴. thinking_budget=0 (옵션 ①). tool 미사용이라
                # budget=0 정상 동작. 429 등은 아래 generic except에서 백오프 재시도.
                from google import genai
                from google.genai import types as gt
                config_kwargs: dict = {
                    "system_instruction": system_prompt,
                    "max_output_tokens": max_tokens,
                    "thinking_config": gt.ThinkingConfig(thinking_budget=0),
                }
                if temperature is not None:
                    config_kwargs["temperature"] = temperature
                if top_p is not None:
                    config_kwargs["top_p"] = top_p
                if gen_seed is not None:
                    config_kwargs["seed"] = gen_seed
                client = genai.Client(api_key=api_key)
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=gt.GenerateContentConfig(**config_kwargs),
                )
                um = getattr(resp, "usage_metadata", None)
                cands = getattr(resp, "candidates", None) or []
                usage = {"input_tokens": um.prompt_token_count,
                         "output_tokens": um.candidates_token_count,
                         "thoughts_tokens": um.thoughts_token_count} if um else {}
                usage["finish_reason"] = str(getattr(cands[0], "finish_reason", None)) if cands else None
                return (resp.text or ""), False, usage
            elif model.startswith("qwen3"):
                # Ollama native /api/chat: think:false 지원 (/v1/chat/completions는 미지원)
                base = (local_endpoint or "http://localhost:11434/v1/chat/completions")
                native_ep = base.replace("/v1/chat/completions", "/api/chat")
                opts: dict = {"num_predict": max_tokens}
                if temperature is not None:
                    opts["temperature"] = temperature
                if top_p is not None:
                    opts["top_p"] = top_p
                if gen_seed is not None:
                    opts["seed"] = gen_seed
                body: dict = {
                    "model": model,
                    "think": False,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "options": opts,
                    "stream": False,
                }
                resp = requests.post(
                    native_ep,
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=120,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"API {resp.status_code} — {wait}초 후 재시도 ({attempt+1}/{_MAX_API_RETRIES})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                j = resp.json()
                # Ollama native: done_reason("stop"/"length") + 토큰 수 → 상한 도달 판정용
                return j["message"]["content"], False, {
                    "input_tokens": j.get("prompt_eval_count"),
                    "output_tokens": j.get("eval_count"),
                    "finish_reason": j.get("done_reason")}
            else:
                endpoint = local_endpoint or "http://localhost:11434/v1/chat/completions"
                body: dict = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    "stream": False,
                    "max_tokens": max_tokens,
                }
                if temperature is not None:
                    body["temperature"] = temperature
                if top_p is not None:
                    body["top_p"] = top_p
                if gen_seed is not None:
                    body["seed"] = gen_seed
                resp = requests.post(
                    endpoint,
                    headers={"Content-Type": "application/json"},
                    json=body,
                    timeout=120,
                )
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = 5 * (2 ** attempt)
                    logger.warning(f"API {resp.status_code} — {wait}초 후 재시도 ({attempt+1}/{_MAX_API_RETRIES})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                j = resp.json()
                usage = dict(j.get("usage") or {})
                usage["finish_reason"] = j["choices"][0].get("finish_reason")
                return j["choices"][0]["message"]["content"], False, usage
        except Exception as e:
            logger.warning(f"에이전트 API 실패 (시도 {attempt+1}/{_MAX_API_RETRIES}): {e}")
            if attempt < _MAX_API_RETRIES - 1:
                time.sleep(5 * (2 ** attempt))

    logger.error("에이전트 API 최종 실패: api_error=True 기록")
    return "", True, None


def _detect_lang(text: str) -> str:
    """텍스트가 주로 KO인지 EN인지 판별."""
    if not text:
        return "unknown"
    ko_chars = sum(1 for c in text if '가' <= c <= '힣')
    return "ko" if ko_chars / max(len(text), 1) > 0.2 else "en"


def _is_bridge_correct(prev_answer: str, gold_entity: str) -> bool:
    """prev_answer가 gold_entity와 일치하는지 대소문자 무시 exact match."""
    if not prev_answer or not gold_entity:
        return False
    return prev_answer.strip().lower() == gold_entity.strip().lower()


# run-level policy metadata (retrieval_mode → trajectory flags)
_TRAJ_MODE_META: dict[str, dict] = {
    "dynamic_subq": {
        "trajectory_mode":      "dynamic_controlled",
        "query_policy":         "ko_dynamic",
        "template_policy":      "ko_template",
        "bridge_insert_policy": "prev_answer_raw",
        "normalization_policy": "none",
        "index_routing":        "oracle_gold_lang",
    },
    "dynamic_subq_qmatched": {
        "trajectory_mode":      "dynamic_ablation_raw",
        "query_policy":         "lang_matched_dynamic_raw",
        "template_policy":      "lang_matched_template",
        "bridge_insert_policy": "prev_answer_raw",
        "normalization_policy": "none",
        "index_routing":        "oracle_gold_lang",
    },
    "dynamic_subq_norm": {
        "trajectory_mode":      "dynamic_ablation_normalized",
        "query_policy":         "lang_matched_dynamic_normalized",
        "template_policy":      "lang_matched_template",
        "bridge_insert_policy": "prev_answer_normalized",
        "normalization_policy": "gold_bridge_alias_match",
        "index_routing":        "oracle_gold_lang",
    },
    # camera-ready: bridge-first 순서의 중간 칸 — KO 템플릿 + 정규화 bridge.
    # controlled → (이 모드) → dynamic_subq_norm 이 template-first 체인의 역순 분해가 된다.
    "dynamic_subq_bridgenorm": {
        "trajectory_mode":      "dynamic_bridge_only_normalized",
        "query_policy":         "ko_dynamic_normalized",
        "template_policy":      "ko_template",
        "bridge_insert_policy": "prev_answer_normalized",
        "normalization_policy": "gold_bridge_alias_match",
        "index_routing":        "oracle_gold_lang",
    },
}


# ─── 생성 질문 분리 조건 + self-translate ─────────────────────────
# 기존 dynamic 세 조건은 hop1 에서 생성 질문 = 검색 쿼리다(resolve_hop_query). 아래 모드들은
# 생성 질문을 controlled(dynamic_subq)와 같게 고정하고 **검색 쿼리만** 바꾼다.
#   dynamic_subq_qmatched_decoupled : 검색 = dynamic_subq_qmatched 쿼리
#   dynamic_subq_norm_decoupled     : 검색 = dynamic_subq_norm 쿼리
#   self_translate                  : 검색 = controlled 쿼리를 생성 모델이 인덱스 언어로 재작성한 문장
DECOUPLED_QUERY_BASE: dict[str, str] = {
    "dynamic_subq_qmatched_decoupled": "dynamic_subq_qmatched",
    "dynamic_subq_norm_decoupled":     "dynamic_subq_norm",
    "self_translate":                  "dynamic_subq",
}

# 2차 절차(2026-09-16 확정): EN-index 홉만 재작성한다. KO-index 홉은 원문 KO 쿼리를 그대로 검색에 쓴다.
SELF_TRANSLATE_SYSTEM_PROMPT = (
    "Rewrite the following question in {target_language} so that it can be used as a search "
    "query over a {target_language} Wikipedia passage index. Write every entity name in its "
    "standard {target_language} form; translating or standardizing the name of an entity already "
    "in the question does not count as adding information. Do not answer the question, and do not "
    "add any entity or fact that is not in the question. If the question is already a suitable "
    "{target_language} query, or cannot be rewritten, output it unchanged. Output only the "
    "rewritten question on a single line.")
SELF_TRANSLATE_TEMPLATE = "{question}"
SELF_TRANSLATE_TARGET = {"ko": "Korean", "en": "English"}

_GENERATION_QUESTION_POLICY = "controlled_ko_template_raw_bridge"
_TRAJ_MODE_META.update({
    "dynamic_subq_qmatched_decoupled": {
        **_TRAJ_MODE_META["dynamic_subq_qmatched"],
        "trajectory_mode":            "dynamic_ablation_raw_decoupled",
        "generation_question_policy": _GENERATION_QUESTION_POLICY,
    },
    "dynamic_subq_norm_decoupled": {
        **_TRAJ_MODE_META["dynamic_subq_norm"],
        "trajectory_mode":            "dynamic_ablation_normalized_decoupled",
        "generation_question_policy": _GENERATION_QUESTION_POLICY,
    },
    "self_translate": {
        **_TRAJ_MODE_META["dynamic_subq"],
        "trajectory_mode":            "dynamic_self_translate",
        "query_policy":               "model_rewrite_of_controlled_query",
        "generation_question_policy": _GENERATION_QUESTION_POLICY,
    },
})

_SELF_TRANSLATE_TRAJ_KEYS = (
    "generation_question", "self_translate_source_query", "self_translate_raw_output",
    "self_translate_status", "self_translate_target_lang", "self_translate_n_lines",
    "self_translate_usage",
)


def build_self_translate_prompt(gold_lang: str) -> str:
    """재작성 시스템 프롬프트 (동기/배치 공유). 목표 언어 = 그 홉의 인덱스 언어."""
    return SELF_TRANSLATE_SYSTEM_PROMPT.format(target_language=SELF_TRANSLATE_TARGET[gold_lang])


def apply_self_translate(hop_state: dict, hop_data: dict, raw_text: str,
                         api_error: bool, usage: dict | None) -> str:
    """재작성 결과를 hop_state 에 반영하고 검색 쿼리를 반환한다 (동기/배치 공유).

    검색 쿼리 = 비어 있지 않은 첫 줄. 호출 실패·빈 출력이면 원래 KO 쿼리로 검색하고 status 로 남긴다.
    쿼리 파생 flag 는 **실제로 던진 쿼리** 기준으로 다시 계산한다."""
    src = hop_state["query"]
    lines = [ln.strip() for ln in (raw_text or "").strip().splitlines() if ln.strip()]
    if api_error:
        status, query = "api_error_fallback", src
    elif not lines:
        status, query = "empty_fallback", src
    else:
        status, query = "ok", lines[0]
    gold_lang = hop_data["gold_lang"]
    if status == "ok":
        hop_state["query_template_field"] = "self_translate"
        hop_state["query_template_lang"] = gold_lang
    hop_state.update({
        "query":                       query,
        "design_query_index_match":    hop_state["query_template_lang"] == gold_lang,
        "actual_query_lang_detected":  _detect_lang(query),
        "query_contains_gold_answer":  hop_data["gold_entity"].lower() in query.lower(),
        "self_translate_source_query": src,
        "self_translate_raw_output":   raw_text or "",
        "self_translate_status":       status,
        "self_translate_target_lang":  gold_lang,
        "self_translate_n_lines":      len(lines),
        "self_translate_usage":        usage,
        "self_translate_pending":      False,
    })
    return query


def _normalize_bridge(prev_answer: str, hop_data: dict,
                      target_lang: str) -> tuple[str, str, str | None]:
    """Returns (normalized_text, normalization_status, normalization_basis).

    normalization_status:
      success_gold_bridge_alias — alias match 성공, target-lang canonical label로 치환
      failed_unlinked           — alias match 실패, raw prev_answer 반환
      skipped_abstain           — prev_answer 없음
    """
    if not prev_answer:
        return prev_answer, "skipped_abstain", None
    ko_label = hop_data.get("sub_q_bridge_label", "") or ""
    en_label = hop_data.get("sub_q_en_bridge_label", "") or ""
    pa = prev_answer.strip().lower()
    if pa and ((ko_label and pa == ko_label.lower()) or (en_label and pa == en_label.lower())):
        result = en_label if target_lang == "en" else ko_label
        return result, "success_gold_bridge_alias", "prev_answer_alias_match"
    return prev_answer, "failed_unlinked", None


def _build_hop_state(retrieval_mode: str, hop_idx: int, hop_data: dict,
                     prev_answer: str, prev_hop_data: dict | None) -> dict:
    """query + 쿼리 구성 flag를 담은 dict를 반환한다.

    반환 키: query, query_template_field, query_template_lang, query_template_text,
    bridge_anchor_text/lang/source, anchor_replacement_success,
    bridge_insert_text/source, normalization_applied/status/basis,
    design_query_index_match, actual_query_lang_detected, query_contains_gold_answer.
    """
    gold_lang = hop_data["gold_lang"]
    use_en    = retrieval_mode in ("dynamic_subq_qmatched", "dynamic_subq_norm") and gold_lang == "en"

    # template 선택
    if use_en:
        tmpl_field = "sub_q_en"
        tmpl_text  = hop_data.get("sub_q_en") or hop_data["gold_entity"]
        tmpl_lang  = "en"
    else:
        tmpl_field = "sub_q"
        tmpl_text  = hop_data["sub_q"]
        tmpl_lang  = "ko"

    design_match = (tmpl_lang == gold_lang)

    # hop0 또는 prev_answer 없음: template 그대로
    if hop_idx == 0 or not prev_answer:
        return {
            "query":                      tmpl_text,
            "query_template_field":       tmpl_field,
            "query_template_lang":        tmpl_lang,
            "query_template_text":        tmpl_text,
            "bridge_anchor_text":         None,
            "bridge_anchor_lang":         None,
            "bridge_anchor_source":       None,
            "anchor_replacement_success": None,
            "bridge_insert_text":         None,
            "bridge_insert_source":       "none",
            "normalization_applied":      False,
            "normalization_status":       "not_applicable",
            "normalization_basis":        None,
            "design_query_index_match":   design_match,
            "actual_query_lang_detected": tmpl_lang,
            "query_contains_gold_answer": hop_data["gold_entity"].lower() in tmpl_text.lower(),
        }

    # anchor 선택 (hop1+)
    prev_gold = (prev_hop_data or {}).get("gold_entity", "")
    if use_en:
        en_bl = hop_data.get("sub_q_en_bridge_label") or ""
        if en_bl and en_bl in tmpl_text:
            anchor, anchor_lang, anchor_src = en_bl, "en", "sub_q_en_bridge_label"
        elif prev_gold and prev_gold in tmpl_text:
            anchor, anchor_lang, anchor_src = prev_gold, _detect_lang(prev_gold), "prev_gold_entity"
        else:
            anchor, anchor_lang, anchor_src = None, None, None
    else:
        bl = hop_data.get("sub_q_bridge_label") or ""
        if bl and bl in tmpl_text:
            anchor, anchor_lang, anchor_src = bl, "ko", "sub_q_bridge_label"
        elif prev_gold and prev_gold in tmpl_text:
            anchor, anchor_lang, anchor_src = prev_gold, _detect_lang(prev_gold), "prev_gold_entity"
        else:
            anchor, anchor_lang, anchor_src = None, None, None

    # bridge insert (raw 또는 normalized)
    if retrieval_mode in ("dynamic_subq_norm", "dynamic_subq_bridgenorm"):
        bridge, norm_status, norm_basis = _normalize_bridge(prev_answer, hop_data, gold_lang)
        norm_applied = (norm_status == "success_gold_bridge_alias")
        insert_src   = "normalized_prev_answer" if norm_applied else "prev_answer_raw"
    else:
        bridge, norm_status, norm_basis = prev_answer, "not_applicable", None
        norm_applied, insert_src = False, "prev_answer_raw"

    # 치환
    if anchor and anchor in tmpl_text:
        query    = tmpl_text.replace(anchor, bridge, 1)
        replaced = True
    else:
        query    = tmpl_text
        replaced = False

    return {
        "query":                      query,
        "query_template_field":       tmpl_field,
        "query_template_lang":        tmpl_lang,
        "query_template_text":        tmpl_text,
        "bridge_anchor_text":         anchor,
        "bridge_anchor_lang":         anchor_lang,
        "bridge_anchor_source":       anchor_src,
        "anchor_replacement_success": replaced,
        "bridge_insert_text":         bridge,
        "bridge_insert_source":       insert_src,
        "normalization_applied":      norm_applied,
        "normalization_status":       norm_status,
        "normalization_basis":        norm_basis,
        "design_query_index_match":   design_match,
        "actual_query_lang_detected": _detect_lang(query),
        "query_contains_gold_answer": hop_data["gold_entity"].lower() in query.lower(),
    }


def _build_dynamic_query(retrieval_mode: str, hop_idx: int,
                         hop_data: dict, prev_answer: str,
                         prev_hop_data: dict | None) -> str:
    """_build_hop_state 래퍼 — query 문자열만 반환 (테스트 호환용)."""
    return _build_hop_state(retrieval_mode, hop_idx, hop_data, prev_answer, prev_hop_data)["query"]


def resolve_hop_query(retrieval_mode: str, qa: dict, hop_idx: int,
                      hop_data: dict, prev_answer: str) -> dict:
    """홉 쿼리/언어/bridge flags를 retrieval_mode별로 구성 (동기/배치 공유).

    반환: query, lang, agent_question, hop_state, bridge_entity, bridge_correct, bridge_lang.
    """
    gold_lang = hop_data["gold_lang"]
    agent_question = hop_data["sub_q"]  # default; dynamic_subq hop1+에서 override됨
    bridge_entity  = None
    bridge_correct = None
    bridge_lang    = None
    hop_state      = None   # _build_hop_state 결과 (dynamic 모드 전용)

    if retrieval_mode == "gold_entity":
        query = hop_data["gold_entity"]
        lang  = gold_lang
    elif retrieval_mode == "ko_subq":
        query = hop_data["sub_q"]
        lang  = gold_lang
    elif retrieval_mode in ("dynamic_subq", "dynamic_subq_qmatched", "dynamic_subq_norm",
                            "dynamic_subq_bridgenorm"):
        prev_hop  = qa["hops"][hop_idx - 1] if hop_idx > 0 else None
        hop_state = _build_hop_state(retrieval_mode, hop_idx, hop_data, prev_answer, prev_hop)
        query     = hop_state["query"]
        if hop_idx > 0 and prev_answer:
            agent_question = query
            bridge_entity  = prev_answer
            bridge_correct = _is_bridge_correct(
                prev_answer, qa["hops"][hop_idx - 1]["gold_entity"])
            bridge_lang    = _detect_lang(prev_answer)
        lang = gold_lang
    elif retrieval_mode in DECOUPLED_QUERY_BASE:
        prev_hop  = qa["hops"][hop_idx - 1] if hop_idx > 0 else None
        hop_state = _build_hop_state(DECOUPLED_QUERY_BASE[retrieval_mode], hop_idx, hop_data,
                                     prev_answer, prev_hop)
        query     = hop_state["query"]
        if hop_idx > 0 and prev_answer:
            # 생성 질문은 controlled 와 같은 KO 템플릿 + raw bridge — 검색 쿼리와 분리
            agent_question = _build_hop_state("dynamic_subq", hop_idx, hop_data,
                                              prev_answer, prev_hop)["query"]
            bridge_entity  = prev_answer
            bridge_correct = _is_bridge_correct(
                prev_answer, qa["hops"][hop_idx - 1]["gold_entity"])
            bridge_lang    = _detect_lang(prev_answer)
        hop_state["generation_question"] = agent_question
        # KO-index 홉은 재작성하지 않는다 (2차 절차). skip 사유를 status 로 남긴다.
        if retrieval_mode == "self_translate":
            if gold_lang == "en":
                hop_state["self_translate_pending"] = True
            else:
                hop_state["self_translate_pending"] = False
                hop_state["self_translate_source_query"] = query
                hop_state["self_translate_raw_output"]   = ""
                hop_state["self_translate_status"]       = "skipped_ko_index"
                hop_state["self_translate_target_lang"]  = gold_lang
                hop_state["self_translate_n_lines"]      = 0
                hop_state["self_translate_usage"]        = None
        lang = gold_lang
    else:
        # sub_q (Ablation): EN-index 홉은 sub_q_en 사용
        if gold_lang == "en":
            query = hop_data.get("sub_q_en") or hop_data["gold_entity"]
        else:
            query = hop_data["sub_q"]
        lang = gold_lang

    return {"query": query, "lang": lang, "agent_question": agent_question,
            "hop_state": hop_state, "bridge_entity": bridge_entity,
            "bridge_correct": bridge_correct, "bridge_lang": bridge_lang}


def build_traj_entry(hop_idx: int, passages: list[str], retrieved_langs: list[str],
                     gen_text: str, lang_switch: bool, hop_data: dict,
                     query: str, lang: str, retrieval_mode: str, api_error: bool,
                     hop_state: dict | None, bridge_entity, bridge_correct,
                     bridge_lang, gen_usage) -> dict:
    """trajectory 홉 엔트리 조립 (출력 스키마). 동기/배치 공유."""
    traj_entry: dict = {
        "hop":                  hop_idx,
        "retrieved_passages":   passages,
        "retrieved_langs":      retrieved_langs,
        "generated_text":       gen_text,
        "retrieval_success":    len(passages) >= 1,
        "lang_switch":          lang_switch,
        "gold_entity":          hop_data["gold_entity"],
        "gold_lang":            hop_data["gold_lang"],
        "retrieval_query":      query,
        "retrieval_query_lang": lang,
        "retrieval_index_used": lang,
        "retrieval_mode":       retrieval_mode,
        "api_error":            api_error,
    }
    # dynamic 모드: 쿼리 구성 flags + bridge flags
    if hop_state is not None:
        for key in ("query_template_field", "query_template_lang",
                    "design_query_index_match", "actual_query_lang_detected",
                    "query_contains_gold_answer",
                    "bridge_anchor_text", "bridge_anchor_lang", "bridge_anchor_source",
                    "anchor_replacement_success",
                    "bridge_insert_text", "bridge_insert_source",
                    "normalization_applied", "normalization_status", "normalization_basis"):
            traj_entry[key] = hop_state[key]
        for key in _SELF_TRANSLATE_TRAJ_KEYS:      # 분리 조건 전용 — 기존 모드엔 키가 없다
            if key in hop_state:
                traj_entry[key] = hop_state[key]
    if bridge_entity is not None:
        traj_entry["prev_answer_raw"]          = bridge_entity
        traj_entry["prev_answer_is_gold_bridge"] = bridge_correct
        traj_entry["prev_answer_surface_lang"] = bridge_lang
        traj_entry["prev_answer_abstained"]    = _check_abstain(bridge_entity)
    if gen_usage is not None:
        traj_entry["gen_usage"] = gen_usage   # 토큰 사용량(gpt-5 reasoning_tokens 포함), 비용 교정용
    return traj_entry


def load_seed_hops(path: str | Path) -> dict[str, dict[int, dict]]:
    """다른 실행의 trajectory 파일 → {qid: {0: hop0 엔트리}}. hop0 공유 전용 (재실행 없이 재사용).

    hop1 이후는 조건마다 쿼리가 달라지므로 싣지 않는다. 실으면 배치 경로는 실행할 라운드가
    사라지고, 동기 경로는 쿼리가 우연히 같은 홉만 재사용해 한 파일 안에서 출처가 섞인다."""
    seed: dict[str, dict[int, dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                t = json.loads(line)
                seed[t["qid"]] = {0: h for h in t["trajectory"] if int(h["hop"]) == 0}
    return seed


def assemble_trajectory(qa: dict, trajectory: list[dict], prev_answer: str,
                        model: str, retrieval_mode: str,
                        seed_source: str | None = None) -> dict:
    """홉 엔트리 리스트 → 최종 trajectory dict (abstain_reason + item/run 메타). 동기/배치 공유."""
    _abstained = _check_abstain(prev_answer)
    abstain_reason = {
        "no_passage": _abstained and any(
            not h["retrieval_success"] for h in trajectory
        ),
        "passage_mismatch": _abstained and any(
            h["retrieval_success"] and
            not any(h["gold_entity"].lower() in p.lower()
                    for p in h["retrieved_passages"])
            for h in trajectory
        ),
        "lang_switch_fail": _abstained and any(
            h["lang_switch"] and not h["retrieval_success"]
            for h in trajectory
        ),
    }

    gold_langs_list = [h["gold_lang"] for h in qa["hops"]]
    run_meta = _TRAJ_MODE_META.get(retrieval_mode, {})
    if seed_source is not None:      # 재사용한 홉의 출처 (재현용). 미사용 시 키 자체가 없다.
        run_meta = {**run_meta, "seeded_hops_from": seed_source}

    return {
        # ── item-level identifiers ────────────────────────────────────────────
        "qid":                   qa["qid"],
        "origin_qid":            qa.get("origin_qid"),
        "path_cluster_id":       qa.get("path_cluster_id"),
        "bridge_cluster_id":     qa.get("bridge_cluster_id"),
        # ── pattern metadata ──────────────────────────────────────────────────
        "model":                 model,
        "pattern":               qa["pattern"],
        "lang_pattern":          qa["pattern"],
        "lang_sequence":         qa.get("lang_sequence", []),
        "hop0_gold_lang":        gold_langs_list[0] if gold_langs_list else None,
        "hop1_gold_lang":        gold_langs_list[1] if len(gold_langs_list) > 1 else None,
        "transition_direction":  "→".join(gl.upper() for gl in gold_langs_list),
        # ── run-level metadata ────────────────────────────────────────────────
        "run_metadata":          run_meta,
        # ── trajectory ───────────────────────────────────────────────────────
        "trajectory":            trajectory,
        "final_answer":          prev_answer,
        "abstain_reason":        abstain_reason,
    }


def run_pilot_trajectory(qa: dict, retriever: FAISSRetriever,
                         api_key: str, model: str,
                         local_endpoint: str | None = None,
                         top_k: int = 5,
                         retrieval_mode: str = "sub_q",
                         prompt_version: str = "v2",
                         gen_seed: int | None = None,
                         max_output_tokens: int | None = None,
                         reasoning_effort: str | None = None,
                         seed_hops: dict[str, dict[int, dict]] | None = None,
                         seed_source: str | None = None) -> dict:
    trajectory = []
    prev_answer = ""

    sys_prompt, hop_tmpl, max_tokens, temperature, top_p = _select_prompts(prompt_version)
    if max_output_tokens is not None:
        max_tokens = max_output_tokens

    gold_langs = [h["gold_lang"] for h in qa["hops"]]
    switches   = compute_lang_switch(gold_langs)

    for hop_data, lang_switch in zip(qa["hops"], switches):
        hop_idx = hop_data["hop"]

        seeded = (seed_hops or {}).get(qa["qid"], {}).get(hop_idx)
        if seeded is not None:
            # 다른 실행의 홉을 그대로 재사용한다 (요청이 문자열까지 같아야 한다 — fail fast).
            expect = resolve_hop_query(retrieval_mode, qa, hop_idx, hop_data, prev_answer)
            want = (expect["hop_state"]["query"]
                    if (expect["hop_state"] or {}).get("self_translate_pending")
                    else expect["query"])
            got = (seeded.get("self_translate_source_query")
                   if (expect["hop_state"] or {}).get("self_translate_pending")
                   else seeded.get("retrieval_query"))
            if got != want:
                raise ValueError(
                    f"seed 홉 불일치 ({qa['qid']} hop{hop_idx}): 재사용 {got!r} vs 이 모드 {want!r}")
            # 조건 라벨만 현재 모드로 맞춘다 (요청·출력은 그대로). 출처는 run_metadata 에 남는다.
            trajectory.append({**seeded, "retrieval_mode": retrieval_mode})
            if not seeded.get("api_error"):
                prev_answer = extract_prev_answer(seeded.get("generated_text", ""))
            continue

        q = resolve_hop_query(retrieval_mode, qa, hop_idx, hop_data, prev_answer)
        if q["hop_state"] is not None and q["hop_state"].get("self_translate_pending"):
            # 재작성도 생성과 같은 모델·같은 디코딩 설정(max_tokens 포함)
            rw_text, rw_err, rw_usage = run_agent_hop(
                q["query"], hop_idx, len(qa["hops"]), [], "", api_key, model, local_endpoint,
                system_prompt=build_self_translate_prompt(hop_data["gold_lang"]),
                hop_template=SELF_TRANSLATE_TEMPLATE,
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                gen_seed=gen_seed, reasoning_effort=reasoning_effort)
            q["query"] = apply_self_translate(q["hop_state"], hop_data, rw_text, rw_err, rw_usage)
        query, lang, agent_question = q["query"], q["lang"], q["agent_question"]
        hop_state = q["hop_state"]
        bridge_entity, bridge_correct, bridge_lang = (
            q["bridge_entity"], q["bridge_correct"], q["bridge_lang"])

        passages, retrieved_langs = retriever.retrieve(query, lang, top_k=top_k)

        if prompt_version == "v1":
            agent_question = qa["question"]

        gen_text, api_error, gen_usage = run_agent_hop(
            agent_question, hop_idx, len(qa["hops"]),
            passages, prev_answer, api_key, model, local_endpoint,
            system_prompt=sys_prompt, hop_template=hop_tmpl,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            gen_seed=gen_seed, reasoning_effort=reasoning_effort)

        if not api_error:
            prev_answer = extract_prev_answer(gen_text)

        traj_entry = build_traj_entry(
            hop_idx, passages, retrieved_langs, gen_text, lang_switch, hop_data,
            query, lang, retrieval_mode, api_error, hop_state,
            bridge_entity, bridge_correct, bridge_lang, gen_usage)
        trajectory.append(traj_entry)

    return assemble_trajectory(qa, trajectory, prev_answer, model, retrieval_mode, seed_source)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    import os
    from dotenv import load_dotenv
    load_dotenv()  # 프로젝트 루트 .env 파일 자동 로드

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--api-key",     default=None,
                        help="미지정 시 모델에 따라 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY 자동 사용")
    parser.add_argument("--agent-model", default=None, help="미지정 시 config.yaml run.agent_model 사용")
    parser.add_argument("--seed",        type=int, default=None, help="미지정 시 config.yaml pilot.seed 사용")
    parser.add_argument("--output",      default=None, help="궤적 출력 파일 경로 (미지정 시 config.yaml output.trajectories 사용)")
    parser.add_argument("--index-dir",   default=None, help="FAISS 인덱스 디렉토리 (미지정 시 config.yaml 사용)")
    parser.add_argument("--qa-pairs",    default=None, help="QA 파일 경로 (미지정 시 config.yaml output.qa_pairs 사용)")
    parser.add_argument("--top-k",       type=int, default=None, help="패시지 검색 수 (미지정 시 config.yaml run.top_k 사용)")
    parser.add_argument("--retrieval-mode", default=None,
                        choices=["gold_entity", "ko_subq", "sub_q",
                                 "dynamic_subq", "dynamic_subq_qmatched", "dynamic_subq_norm",
                                 "dynamic_subq_bridgenorm",
                                 "dynamic_subq_qmatched_decoupled", "dynamic_subq_norm_decoupled",
                                 "self_translate"],
                        help="*_decoupled / self_translate: 생성 질문은 dynamic_subq 와 같고 검색 쿼리만 다름 "
                             "(self_translate = 생성 모델이 controlled 쿼리를 인덱스 언어로 재작성) / "
                             "ko_subq: Main protocol — 항상 KO sub_q+gold_lang index / "
                             "sub_q: Ablation 1 — EN hop은 sub_q_en 사용 / "
                             "gold_entity: diagnostic upper bound / "
                             "dynamic_subq: dyn_ctrl — hop0=KO sub_q, hop1+=KO template+prev_answer / "
                             "dynamic_subq_qmatched: dyn_abl_raw — hop0=lang-matched, hop1+=lang-matched template+raw bridge / "
                             "dynamic_subq_norm: dyn_abl_norm — hop0=lang-matched, hop1+=lang-matched template+normalized bridge / "
                             "dynamic_subq_bridgenorm: bridge-only — hop0=KO sub_q, hop1+=KO template+normalized bridge "
                             "(미지정 시 config.yaml run.retrieval_mode 사용)")
    parser.add_argument("--seed-trajectories", default=None,
                        help="다른 실행의 trajectory JSONL. 거기 있는 홉(보통 hop0)은 재생성하지 않고 그대로 쓰고 "
                             "이후 홉만 실행한다. 요청 문자열이 다르면 즉시 중단.")
    parser.add_argument("--prompt-version", default=None, choices=["v1", "v2", "v2.2", "v22", "v2.3", "v23"],
                        help="v1: 생각:/검색어:/답변: 형식 / v2: 단순화된 답변: 형식 / v2.2(=v22): 형식 안정화 측정 프롬프트 / v2.3(=v23): 자연 측정 프롬프트, 마커 권장만 (미지정 시 config.yaml run.prompt_version 사용)")
    parser.add_argument("--per-pattern",  type=int, default=None,
                        help="패턴당 샘플 수 (미지정 시 config.yaml run.per_pattern 사용)")
    parser.add_argument("--per-qt", type=int, default=None,
                        help="query_type당 샘플 수. 지정 시 --per-pattern 무시하고 qt 균등 샘플링.")
    parser.add_argument("--max-output-tokens", type=int, default=None,
                        help="생성 max token 오버라이드 (미지정 시 prompt_version 기본값). frontier는 256 권장.")
    parser.add_argument("--reasoning-effort", default=None,
                        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="GPT-5 계열 reasoning_effort. 모델별 최소값 상이: gpt-5.4는 'none'이 floor"
                             "(minimal 미지원), 일부 gpt-5는 'minimal'. frontier 통제는 최소값 사용. 비-GPT-5엔 무시.")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    pilot_cfg = cfg["pilot"]

    agent_model    = args.agent_model or pilot_cfg["agent_model"]
    local_endpoint = pilot_cfg.get("local_endpoint")
    seed           = args.seed if args.seed is not None else pilot_cfg["seed"]
    traj_file      = Path(args.output or cfg["output"]["trajectories"])
    top_k          = args.top_k          if args.top_k          is not None else pilot_cfg.get("top_k", 5)
    retrieval_mode = args.retrieval_mode if args.retrieval_mode is not None else pilot_cfg.get("retrieval_mode", "sub_q")
    prompt_version = args.prompt_version if args.prompt_version is not None else pilot_cfg.get("prompt_version", "v2.3")

    # api_key: CLI 미지정 시 모델에 따라 환경변수 자동 선택
    api_key = args.api_key
    if not api_key:
        if agent_model.startswith("gpt-"):
            api_key = os.environ.get("OPENAI_API_KEY")
        elif agent_model.startswith("gemini"):
            api_key = os.environ.get("GEMINI_API_KEY")
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

    if agent_model.startswith("claude") and not api_key:
        raise ValueError("Claude 모델에는 ANTHROPIC_API_KEY 필요: .env에 추가하세요.")
    if agent_model.startswith("gpt-") and not api_key:
        raise ValueError("GPT 모델에는 OPENAI_API_KEY 필요: .env에 추가하세요.")
    if agent_model.startswith("gemini") and not api_key:
        raise ValueError("Gemini 모델에는 GEMINI_API_KEY 필요: .env에 추가하세요.")

    # GPT-5 계열인데 reasoning_effort 미지정 → API 기본값(medium)이 적용되어
    # frontier 통제 조건(minimal)과 어긋날 수 있다. 잊은 플래그로 인한 조건 오염 방지.
    if agent_model.startswith("gpt-5") and args.reasoning_effort is None:
        logger.warning("gpt-5 계열인데 --reasoning-effort 미지정 → API 기본값 적용됨. "
                       "frontier 통제 시 --reasoning-effort minimal 권장.")

    random.seed(seed)

    qa_file   = Path(args.qa_pairs or cfg["output"]["qa_pairs"])
    traj_file.parent.mkdir(parents=True, exist_ok=True)

    # Resume: 이미 완료된 QID 스킵
    done_qids: set[str] = set()
    if traj_file.exists():
        for line in traj_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_qids.add(json.loads(line)["qid"])
        if done_qids:
            logger.info(f"Resume: 이미 완료된 {len(done_qids)}개 스킵")

    if not qa_file.exists():
        logger.error("QA 파일 없음 → pipeline/03_generate_qa.py 먼저 실행하세요.")
        return

    retriever = FAISSRetriever(Path(args.index_dir or cfg["output"]["faiss_index"]))

    # 샘플링
    all_qa = [json.loads(l) for l in qa_file.read_text().splitlines() if l.strip()]

    pilot_qas = []

    if args.per_qt is not None:
        by_qt = defaultdict(list)
        for qa in all_qa:
            by_qt[qa["query_type"]].append(qa)
        for qt in sorted(by_qt):
            pool    = by_qt[qt]
            sampled = random.sample(pool, min(args.per_qt, len(pool)))
            pilot_qas.extend(sampled)
            logger.info(f"query_type {qt}: {len(sampled)}개 (목표 {args.per_qt}, 풀 {len(pool)})")
    else:
        by_pattern = defaultdict(list)
        for qa in all_qa:
            by_pattern[qa["pattern"]].append(qa)
        per_pattern = args.per_pattern if args.per_pattern is not None else pilot_cfg.get("per_pattern")
        for pat in ["a", "b", "c", "d"]:
            pool    = by_pattern.get(pat, [])
            sampled = random.sample(pool, min(per_pattern, len(pool))) if per_pattern else pool
            pilot_qas.extend(sampled)
            logger.info(f"패턴 {pat}: {len(sampled)}개" + (f" (샘플링, 목표 {per_pattern})" if per_pattern else " (전체)"))

    # done_qids 제외
    pilot_qas = [qa for qa in pilot_qas if qa["qid"] not in done_qids]
    logger.info(f"총 {len(pilot_qas)}개 실행 시작")

    logger.info(f"프롬프트 버전: {prompt_version} / retrieval_mode: {retrieval_mode} / top_k: {top_k}")

    seed_hops = None
    if args.seed_trajectories:
        seed_hops = load_seed_hops(args.seed_trajectories)
        n_h = sum(len(v) for v in seed_hops.values())
        logger.info(f"홉 재사용: {args.seed_trajectories} (qid {len(seed_hops)}, 홉 {n_h})")
    with open(traj_file, "a", encoding="utf-8") as f:
        for qa in tqdm(pilot_qas, desc="파일럿 실행"):
            traj = run_pilot_trajectory(qa, retriever, api_key, agent_model, local_endpoint,
                                        top_k=top_k, retrieval_mode=retrieval_mode,
                                        prompt_version=prompt_version, gen_seed=seed,
                                        max_output_tokens=args.max_output_tokens,
                                        reasoning_effort=args.reasoning_effort,
                                        seed_hops=seed_hops,
                                        seed_source=args.seed_trajectories)
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")

    logger.info(f"궤적 저장: {traj_file}")
    print(f"\n파일럿 완료. 다음 단계:")
    print(f"   python pipeline/06_annotate.py --trajectories {traj_file}")


if __name__ == "__main__":
    main()
