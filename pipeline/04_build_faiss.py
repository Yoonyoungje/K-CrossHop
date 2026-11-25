#!/usr/bin/env python3
"""
pipeline/04_build_faiss.py
==================
passages.jsonl → BGE-M3 임베딩 → ko/en 분리 FAISS IndexFlatIP

출력:
  data/processed/faiss_index/ko_index.faiss
  data/processed/faiss_index/en_index.faiss
  data/processed/faiss_index/ko_meta.json   # passage_id / text / entity_qid / entity_label
  data/processed/faiss_index/en_meta.json

실행:
  python pipeline/04_build_faiss.py [--config config.yaml] [--batch-size 64]
"""

import json
import argparse
import logging
import numpy as np
from pathlib import Path
import yaml
import faiss
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BGE_M3_MODEL = "BAAI/bge-m3"


def load_passages_by_lang(passages_file: Path) -> dict[str, list[dict]]:
    by_lang: dict[str, list[dict]] = {"ko": [], "en": []}
    with open(passages_file, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            lang = p.get("lang", "ko")
            if lang in by_lang:
                by_lang[lang].append(p)
    logger.info(f"로드: KO {len(by_lang['ko'])}개, EN {len(by_lang['en'])}개 패시지")
    return by_lang


def encode_passages(model: SentenceTransformer, passages: list[dict],
                    batch_size: int, prefix: str = "") -> np.ndarray:
    texts = [prefix + p["text"] for p in passages]
    vecs = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        normalize_embeddings=True)
    return vecs.astype("float32")


def build_and_save(vecs: np.ndarray, passages: list[dict],
                   index_path: Path, meta_path: Path) -> None:
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    faiss.write_index(index, str(index_path))
    logger.info(f"  인덱스 저장: {index_path} ({index.ntotal}개)")

    meta = [
        {
            "passage_id":   p["passage_id"],
            "entity_qid":   p["entity_qid"],
            "entity_label": p["entity_label"],
            "lang":         p["lang"],
            "text":         p["text"],
        }
        for p in passages
    ]
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    logger.info(f"  메타데이터 저장: {meta_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--passages",   default=None, help="패시지 JSONL (미지정 시 config 사용)")
    parser.add_argument("--index-dir",  default=None, help="FAISS 인덱스 출력 디렉토리 (미지정 시 config 사용)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device",     default=None, help="cpu | cuda:0 | ... (기본: GPU 자동 감지)")
    parser.add_argument("--embed-model", default=BGE_M3_MODEL,
                        help="임베딩 모델 (예: intfloat/multilingual-e5-large)")
    parser.add_argument("--passage-prefix", default="", help='패시지 접두사 (E5: "passage: ")')
    parser.add_argument("--query-prefix",   default="", help='쿼리 접두사 (E5: "query: "). '
                        "05 가 retriever.json 에서 읽는다")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    passages_file = Path(args.passages  or cfg["output"]["passages"])
    index_dir     = Path(args.index_dir or cfg["output"]["faiss_index"])
    index_dir.mkdir(parents=True, exist_ok=True)

    if not passages_file.exists():
        logger.error(f"패시지 파일 없음: {passages_file} → pipeline/02_fetch_passages.py 먼저 실행하세요.")
        return

    import torch
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"임베딩 모델 로딩: {args.embed_model} (device={device}, "
                f"passage_prefix={args.passage_prefix!r})")
    model = SentenceTransformer(args.embed_model, device=device)

    by_lang = load_passages_by_lang(passages_file)
    for lang in ["ko", "en"]:
        passages = by_lang[lang]
        if not passages:
            logger.warning(f"{lang} 패시지 없음, 스킵")
            continue
        logger.info(f"\n[{lang.upper()}] {len(passages)}개 임베딩 중...")
        vecs = encode_passages(model, passages, args.batch_size, args.passage_prefix)
        build_and_save(
            vecs, passages,
            index_dir / f"{lang}_index.faiss",
            index_dir / f"{lang}_meta.json",
        )

    # 05 의 FAISSRetriever 가 이 파일로 인코더·접두사를 맞춘다 (없으면 BGE-M3 로 간주).
    (index_dir / "retriever.json").write_text(json.dumps({
        "model": args.embed_model,
        "query_prefix": args.query_prefix,
        "passage_prefix": args.passage_prefix,
        "dim": int(model.get_sentence_embedding_dimension()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print(f"완료: {index_dir}")
    print(f"  ko_index.faiss: {len(by_lang['ko'])}개")
    print(f"  en_index.faiss: {len(by_lang['en'])}개")


if __name__ == "__main__":
    main()
