# This file builds embedding and BM25 indexes for atomic knowledge cards.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
# -*- coding: utf-8 -*-
"""Build dense and sparse retrieval indexes for TiMedLM atomic cards.

The script reads atomic-card JSONL files, builds a BM25 index over the same
retrieval text used by dense retrieval, and stores L2-normalized BGE-M3 dense
embeddings aligned with the card order.
"""

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import List

import jieba
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi
from tqdm import tqdm


DEFAULT_MODEL_PATH = "BAAI/bge-m3"
DEFAULT_KB_DIR = "data/atomic_cards"
DEFAULT_EMBED_OUT_PATH = "cache/atoms_all_bge.pkl"
DEFAULT_BM25_OUT_PATH = "cache/atoms_bm25.pkl"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_LENGTH = 512

CARD_FILES_IN_ORDER = [
    "atoms_lll.jsonl",
    "atoms_sbyd.jsonl",
    "atoms_ywyz.jsonl",
    "atoms_jzbc.jsonl",
    "atoms_zyyx.jsonl",
    "diag_manual.jsonl",
]

# Keep this aligned with retrieval.py. Add domain stopwords if needed.
STOPWORDS = set()


def read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_all_cards(kb_dir: str, card_files: List[str]) -> List[dict]:
    cards = []
    for fname in card_files:
        fpath = os.path.join(kb_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing card file: {fpath}")
        batch = read_jsonl(fpath)
        cards.extend(batch)
        print(f"  {fname}: {len(batch)} cards")
    return cards


def build_text(card: dict) -> str:
    """Build the retrieval text shared by BM25 and dense retrieval."""
    retrieval_info = card.get("retrieval_enhancement", {}) or {}
    structured_fields = card.get("structured_fields", {}) or {}

    attribute_value = structured_fields.get("attribute_value", [])
    if isinstance(attribute_value, str):
        attribute_value = [attribute_value]

    keywords = retrieval_info.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]

    parts = [
        card.get("title", ""),
        card.get("content", ""),
        " ".join(str(x) for x in keywords),
        " ".join(str(x) for x in attribute_value),
    ]
    return " ".join(str(x) for x in parts if x).strip()


def tokenize(text: str) -> List[str]:
    return [
        tok for tok in jieba.cut(text)
        if tok.strip() and tok not in STOPWORDS and len(tok) > 1
    ]


def build_bm25(texts: List[str]) -> BM25Okapi:
    tokenized_corpus = [tokenize(t) for t in tqdm(texts, desc="Tokenizing")]
    return BM25Okapi(tokenized_corpus), tokenized_corpus


def build_dense_embeddings(
    texts: List[str],
    model_path: str,
    output_path: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model = BGEM3FlagModel(model_path, use_fp16=True)
    ckpt_path = output_path + ".ckpt"

    if os.path.exists(ckpt_path):
        with open(ckpt_path, "rb") as f:
            all_embeddings = pickle.load(f)
        start_idx = len(all_embeddings)
        print(f"Resume embeddings from checkpoint: {start_idx}/{len(texts)}")
    else:
        all_embeddings = []
        start_idx = 0

    for i in tqdm(range(start_idx, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[i:i + batch_size]
        result = model.encode(batch_texts, batch_size=batch_size, max_length=max_length)
        all_embeddings.extend(result["dense_vecs"])

        if (i // batch_size) % 10 == 0:
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            with open(ckpt_path, "wb") as f:
                pickle.dump(all_embeddings, f)

    embeddings = np.asarray(all_embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    embeddings = embeddings / norms

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    return embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Build TiMedLM retrieval indexes.")
    parser.add_argument("--kb_dir", default=DEFAULT_KB_DIR)
    parser.add_argument("--embedding_model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--embed_out", default=DEFAULT_EMBED_OUT_PATH)
    parser.add_argument("--bm25_out", default=DEFAULT_BM25_OUT_PATH)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_bm25", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cards = load_all_cards(args.kb_dir, CARD_FILES_IN_ORDER)
    print(f"Total cards: {len(cards)}")

    texts = [build_text(card) for card in cards]

    if not args.skip_bm25:
        print("Building BM25 index...")
        bm25, tokenized_corpus = build_bm25(texts)
        Path(args.bm25_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.bm25_out, "wb") as f:
            pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)
        print(f"BM25 index saved to: {args.bm25_out}")

    if not args.skip_dense:
        print("Building dense embeddings...")
        embeddings = build_dense_embeddings(
            texts=texts,
            model_path=args.embedding_model,
            output_path=args.embed_out,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        Path(args.embed_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.embed_out, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"Embedding shape: {embeddings.shape}")
        print(f"Dense embeddings saved to: {args.embed_out}")


if __name__ == "__main__":
    main()
