# This file extracts atomic knowledge cards from source-book text.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
# -*- coding: utf-8 -*-
"""Extract atomic knowledge cards from OCR text files with an OpenAI-compatible LLM.

This script implements the atomic-card extraction step described in the paper.
It reads page-marked OCR text files, sends each page to a prompted LLM, parses
JSON/JSONL card outputs, and writes one JSONL file per source book.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List

from openai import OpenAI


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-max-latest"
DEFAULT_MAX_CONTEXT_LEN = 2000
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SLEEP_SECONDS = 0.5
DEFAULT_REQUEST_TIMEOUT = 120.0

DEFAULT_BOOKS = [
    {"name": "Blue Beryl", "abbr": "lll", "filename": "[OCR]_蓝琉璃.txt"},
    {"name": "The Four Medical Tantras", "abbr": "sbyd", "filename": "[OCR]_四部医典.txt"},
    {"name": "Somaratsa", "abbr": "ywyz", "filename": "[OCR]_月王药诊.txt"},
    {"name": "Jing Zhu Ben Cao", "abbr": "jzbc", "filename": "[OCR]_晶珠本草.txt"},
    {"name": "Introduction to Tibetan Medicine", "abbr": "zyyx", "filename": "[OCR]_藏医药学.txt"},
]


def read_txt_with_pages(path: Path) -> List[Dict]:
    raw = path.read_text(encoding="utf-8")

    # Supports page markers like "=== 12 ===". If no marker is found, treat the
    # whole file as one page.
    pattern = r"={2,}\s*(\d+)\s*={2,}"
    page_marks = list(re.finditer(pattern, raw))
    pages = []

    if not page_marks:
        text = clean_page_text(raw)
        return [{"page": 1, "text": text}] if text else []

    for i, mark in enumerate(page_marks):
        page_num = int(mark.group(1))
        start = mark.end()
        end = page_marks[i + 1].start() if i + 1 < len(page_marks) else len(raw)
        content = clean_page_text(raw[start:end])
        if content:
            pages.append({"page": page_num, "text": content})
    return pages


def clean_page_text(text: str) -> str:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not re.match(r"^\d+$", line.strip())
    ]
    content = "\n".join(lines).strip()
    return content if len(content) >= 50 else ""


def make_client(api_key: str, base_url: str, timeout: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)


def query_llm(
    client: OpenAI,
    model: str,
    system_prompt: str,
    content: str,
    temperature: float,
    max_tokens: int,
) -> str:
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [retry {attempt}] LLM call failed: {exc}")
            if attempt < 3:
                time.sleep(2 ** attempt)
    return ""


def parse_cards(text: str) -> List[Dict]:
    if not text:
        return []

    cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [x for x in parsed if isinstance(x, dict) and x]
        if isinstance(parsed, dict):
            return [parsed]
    except Exception:
        pass

    cards = []
    for line in text.splitlines():
        line = line.strip().strip("`").rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj:
                cards.append(obj)
        except Exception:
            continue
    return cards


def load_prompt(prompt_path: Path) -> Dict:
    with prompt_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_prompts(prompt_json: Dict, book_name: str, page_no: int, text: str, max_context_len: int):
    system_prompt = "\n".join(prompt_json["task"]) + "\n" + "\n".join(prompt_json["requirements"])
    context_template = "\n".join(prompt_json["context_text"])
    user_prompt = context_template.format(
        book_name=book_name,
        page_no=page_no,
        context=text[:max_context_len],
    )
    return system_prompt, user_prompt


def extract_page_cards(client: OpenAI, args, prompt_json: Dict, page: Dict, book_name: str) -> List[Dict]:
    system_prompt, user_prompt = build_prompts(
        prompt_json=prompt_json,
        book_name=book_name,
        page_no=page["page"],
        text=page["text"],
        max_context_len=args.max_context_len,
    )

    for attempt in range(1, 4):
        response = query_llm(
            client=client,
            model=args.model,
            system_prompt=system_prompt,
            content=user_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        cards = parse_cards(response)
        if cards:
            return cards
        if attempt < 3:
            print(f"    Parse failed; retrying page {page['page']} ({attempt}/3)")
            time.sleep(1)
    return []


def processed_pages(output_jsonl: Path) -> set:
    done = set()
    if not output_jsonl.exists():
        return done
    with output_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                parts = str(obj.get("card_id", "")).split("_")
                if len(parts) >= 3:
                    done.add(int(parts[2]))
            except Exception:
                continue
    return done


def extract_from_book(client: OpenAI, args, prompt_json: Dict, book: Dict) -> int:
    book_path = Path(args.books_dir) / book["filename"]
    output_jsonl = Path(args.output_dir) / f"atoms_{book['abbr']}.jsonl"
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    pages = read_txt_with_pages(book_path)
    done = processed_pages(output_jsonl)
    print(f"\nStart {book['name']}: {len(pages)} pages, {len(done)} already processed")

    count = 0
    with output_jsonl.open("a", encoding="utf-8") as fw:
        for page in pages:
            page_no = page["page"]
            if page_no in done:
                continue
            print(f"  Extracting page {page_no}...")
            cards = extract_page_cards(client, args, prompt_json, page, book["name"])
            if cards:
                for card in cards:
                    fw.write(json.dumps(card, ensure_ascii=False) + "\n")
                fw.flush()
                count += len(cards)
                print(f"    extracted {len(cards)} cards")
            else:
                print("    no valid cards")
            time.sleep(args.sleep_seconds)
    return count


def parse_args():
    parser = argparse.ArgumentParser(description="Extract atomic knowledge cards from OCR books.")
    parser.add_argument("--books_dir", default="data/books")
    parser.add_argument("--output_dir", default="data/atomic_cards")
    parser.add_argument("--prompt_path", default="prompts/atomic_card_extraction_prompt.json")
    parser.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--base_url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--max_context_len", type=int, default=DEFAULT_MAX_CONTEXT_LEN)
    parser.add_argument("--sleep_seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--request_timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--max_workers", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Please set {args.api_key_env} before running.")

    prompt_json = load_prompt(Path(args.prompt_path))
    client = make_client(api_key=api_key, base_url=args.base_url, timeout=args.request_timeout)

    total = 0
    if args.max_workers <= 1:
        for book in DEFAULT_BOOKS:
            total += extract_from_book(client, args, prompt_json, book)
    else:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(extract_from_book, client, args, prompt_json, book) for book in DEFAULT_BOOKS]
            for future in as_completed(futures):
                total += future.result()

    print(f"\nDone. Extracted {total} cards in this run.")


if __name__ == "__main__":
    main()
