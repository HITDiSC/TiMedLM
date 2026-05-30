# This file builds a chunk-level knowledge base from source books.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
# -*- coding: utf-8 -*-
"""
Build a chunk-based knowledge base from raw OCR book txt files.

Outputs:
1. chunks jsonl: each row is a chunk dict with card_id/title/content/source fields.
2. dense embedding pkl: numpy array aligned with the jsonl row order.

Example:
CUDA_VISIBLE_DEVICES=3 python build_chunk_kb.py \
  --books_dir data/books \
  --output_jsonl data/chunk_knowledgebase/book_chunks_500_100.jsonl \
  --output_emb cache/book_chunks_500_100_bge.pkl
"""

import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from tqdm import tqdm


DEFAULT_BOOKS_DIR = "data/books"
DEFAULT_OUTPUT_JSONL = "data/chunk_knowledgebase/book_chunks_500_100.jsonl"
DEFAULT_OUTPUT_EMB = "cache/book_chunks_500_100_bge.pkl"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"


BOOK_NAME_MAP = {
    "Blue Beryl": "lll",
    "Four Medical Tantras": "sbyd",
    "Somaratsa": "ywyz",
    "Jing Zhu Ben Cao": "jzbc",
    "Introduction to Tibetan Medicine": "zyyx",
}

def read_text_auto(path: str) -> str:
    for enc in ["utf-8", "utf-8-sig", "gb18030", "gbk"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_book_name(filename: str) -> str:
    name = Path(filename).stem
    name = re.sub(r"^(?:\[?OCR\]?|OCR_?)", "", name)
    name = name.strip("_-[]() ")
    return name or Path(filename).stem


def book_code(book_name: str) -> str:
    for key, val in BOOK_NAME_MAP.items():
        if key in book_name:
            return val
    cleaned = re.sub(r"\W+", "_", book_name, flags=re.UNICODE).strip("_")
    return cleaned[:24] or "book"


def split_paragraphs(text: str) -> List[str]:
    paras = []
    for block in re.split(r"\n\s*\n", text):
        block = re.sub(r"\s+", " ", block).strip()
        if block:
            paras.append(block)
    return paras


def char_chunks(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    if len(text) <= chunk_size:
        yield text
        return
    step = max(1, chunk_size - overlap)
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start += step


def build_chunks_for_book(path: str, chunk_size: int, overlap: int, min_chars: int) -> List[Dict]:
    raw = normalize_text(read_text_auto(path))
    book = clean_book_name(os.path.basename(path))
    code = book_code(book)
    paras = split_paragraphs(raw)

    chunks = []
    buf = ""
    part_id = 0

    def flush_buffer():
        nonlocal buf, part_id
        text = buf.strip()
        buf = ""
        if not text:
            return
        for sub in char_chunks(text, chunk_size, overlap):
            if len(sub) < min_chars:
                continue
            part_id += 1
            chunks.append({
                "card_id": f"chunk_{code}_{part_id:06d}",
                "title": f"{book} chunk {part_id}",
                "card_type": "chunk",
                "source_book": book,
                "source_file": os.path.basename(path),
                "chunk_index": part_id,
                "content": sub,
                "text": sub,
            })

    for para in paras:
        if not buf:
            buf = para
        elif len(buf) + 1 + len(para) <= chunk_size:
            buf = buf + "\n" + para
        else:
            flush_buffer()
            buf = para

    flush_buffer()
    return chunks


def build_embedding_text(chunk: Dict) -> str:
    return " ".join([
        str(chunk.get("source_book", "")),
        str(chunk.get("title", "")),
        str(chunk.get("content", "")),
    ]).strip()


def write_jsonl(rows: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_embeddings(chunks: List[Dict], model_path: str, batch_size: int, max_length: int) -> np.ndarray:
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_path, use_fp16=True)
    texts = [build_embedding_text(c) for c in chunks]
    all_vecs = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[start:start + batch_size]
        out = model.encode(batch, batch_size=batch_size, max_length=max_length)
        vecs = np.asarray(out["dense_vecs"], dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
        vecs = vecs / norms
        all_vecs.append(vecs)

    if not all_vecs:
        return np.zeros((0, 1024), dtype=np.float32)
    return np.vstack(all_vecs).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--books_dir", default=DEFAULT_BOOKS_DIR)
    parser.add_argument("--output_jsonl", default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--output_emb", default=DEFAULT_OUTPUT_EMB)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--min_chars", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--skip_embedding", action="store_true")
    args = parser.parse_args()

    book_paths = sorted(
        str(p) for p in Path(args.books_dir).glob("*.txt")
        if p.is_file()
    )
    if not book_paths:
        raise FileNotFoundError(f"No .txt files found in {args.books_dir}")

    all_chunks = []
    book_stats = {}
    for path in book_paths:
        chunks = build_chunks_for_book(path, args.chunk_size, args.overlap, args.min_chars)
        all_chunks.extend(chunks)
        book_stats[clean_book_name(os.path.basename(path))] = len(chunks)

    write_jsonl(all_chunks, args.output_jsonl)
    print(f"Books: {len(book_paths)}")
    print(f"Chunks: {len(all_chunks)}")
    print(json.dumps(book_stats, ensure_ascii=False, indent=2))
    print(f"Chunk jsonl saved to: {args.output_jsonl}")

    if args.skip_embedding:
        print("Skip embedding.")
        return

    emb = build_embeddings(all_chunks, args.embedding_model, args.batch_size, args.max_length)
    Path(args.output_emb).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_emb, "wb") as f:
        pickle.dump(emb, f)
    print(f"Embedding shape: {emb.shape}")
    print(f"Embedding pkl saved to: {args.output_emb}")


if __name__ == "__main__":
    main()


