# -*- coding: utf-8 -*-
"""
Fair strong-model Single-RAG MCQ evaluation, answer-only version.

Conservative strong-model single-RAG baseline:
- same dataset and single-choice filter
- single retrieval round
- retrieval query: question + all option texts
- top-6 evidence
- plain evidence format
- final output must be exactly one option letter
"""

import os
import re
import sys
import json
import time
from datetime import datetime
from collections import defaultdict, Counter

from tqdm import tqdm
from openai import OpenAI


GR_API_KEY = os.environ.get("GR_API_KEY", "")
BASE_URL = os.environ.get("GR_BASE_URL", "https://endpoint.wendalog.com")
GPT_MODEL = os.environ.get("GR_MODEL", "claude-sonnet-4-0")

DATA_PATH = os.environ.get("TIMEDLM_MCQ_TEST_PATH", "data/samples/mcq_eval_sample.json")
RESULT_DIR = os.environ.get("TIMEDLM_MCQ_RESULT_DIR", "results/mcq/closed_model_single_rag")

MODE_NAME = "strong_model_single_rag_answer_only_top6"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"{RESULT_DIR}/{MODE_NAME}_{timestamp}.json"
CKPT_PATH = f"{RESULT_DIR}/{MODE_NAME}_ckpt.json"

RETRIEVE_TOP_K = 6
SAVE_EVERY = 20
DEBUG_SAMPLES = 3
MAX_RETRY = 3
RETRY_SLEEP = 3

os.makedirs(RESULT_DIR, exist_ok=True)

RETRIEVAL_ROOT = os.environ.get(
    "RETRIEVAL_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "timedlm", "retrieval")),
)
sys.path.append(RETRIEVAL_ROOT)
from retrieval import retrieve_with_scores


SYSTEM_PROMPT = (
    "You are a Tibetan medicine exam assistant. "
    "Use the provided evidence and follow the user's output format exactly."
)


def query_gr(prompt: str) -> str:
    if not GR_API_KEY:
        raise RuntimeError("GR_API_KEY is empty. Please set it before running.")

    client = OpenAI(api_key=GR_API_KEY, base_url=BASE_URL)
    last_err = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(f"Strong-model query failed after {MAX_RETRY} retries: {last_err}")


def get_card_content(card):
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


def score_from_meta(meta):
    if isinstance(meta, dict):
        return float(meta.get("score", meta.get("dense_raw", 0.0)))
    return float(meta)


def build_retrieval_query(question: str, options: dict) -> str:
    option_text = "；".join(
        f"{k}.{v}" for k, v in (options or {}).items() if str(v).strip()
    )
    return f"{question}；{option_text}".strip("；")


def retrieve_evidence(question: str, options: dict):
    query = build_retrieval_query(question, options)
    raw = retrieve_with_scores(query, top_k=RETRIEVE_TOP_K)

    cards = []
    for item in raw:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        card, meta = item
        if not isinstance(card, dict):
            continue
        cards.append({
            "card_id": card.get("card_id", ""),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", ""),
            "content": get_card_content(card),
            "score": score_from_meta(meta),
        })

    cards.sort(key=lambda x: x["score"], reverse=True)
    return query, cards[:RETRIEVE_TOP_K]


def format_evidence(cards):
    if not cards:
        return "未检索到相关证据。"

    lines = []
    for i, c in enumerate(cards, start=1):
        lines.append(
            f"[{i}] card_id: {c['card_id']}\n"
            f"title: {c['title']}\n"
            f"type: {c['card_type']}\n"
            f"score: {c['score']:.4f}\n"
            f"content: {c['content']}"
        )

    return "\n\n".join(lines)


def build_prompt(question: str, options: dict, evidence_text: str, valid_options) -> str:
    option_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    valid_str = "/".join(sorted(valid_options))

    return f"""请根据题目、选项和检索证据选择最可能正确的答案。

题目：
{question}

选项：
{option_str}

检索证据：
{evidence_text}

输出要求：
- 只输出一个选项字母。
- 必须是 {valid_str} 中的一个。
- 不要分析，不要解释，不要输出 JSON，不要输出标点。
"""


def extract_pred(content: str, valid_options=None):
    if valid_options is None:
        valid_options = {"A", "B", "C", "D"}

    valid_options = {str(x).strip().upper() for x in valid_options}
    valid_pat = "".join(sorted(valid_options))

    if not content:
        return None

    content = re.sub(r"```json|```", "", str(content)).strip()
    content_upper = content.upper()

    if content_upper in valid_options:
        return content_upper

    m = re.search(rf'"answer"\s*:\s*"([{valid_pat}])"', content_upper)
    if m:
        return m.group(1)

    patterns = [
        rf"答案[是为：:]\s*([{valid_pat}])",
        rf"正确答案[是为：:]\s*([{valid_pat}])",
        rf"最终答案[是为：:]\s*([{valid_pat}])",
        rf"answer\s*[:：]\s*([{valid_pat}])",
    ]
    for pat in patterns:
        m = re.search(pat, content_upper, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    m = re.search(rf"\b([{valid_pat}])\b", content_upper[-100:])
    if m:
        return m.group(1)

    matches = re.findall(rf"[{valid_pat}]", content_upper[-50:])
    return matches[-1] if matches else None


def load_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return [], set()

    try:
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        rows = ckpt.get("results", [])
        done_ids = {str(r["question_num"]) for r in rows}
        print(f"[checkpoint] Resume from {len(rows)} completed samples.\n")
        return rows, done_ids
    except Exception as e:
        print(f"[checkpoint] Failed to read checkpoint: {e}. Restarting.\n")
        return [], set()


def save_checkpoint(results, dataset_total):
    ckpt = {
        "timestamp": timestamp,
        "mode": MODE_NAME,
        "model": GPT_MODEL,
        "data_path": DATA_PATH,
        "progress": f"{len(results)}/{dataset_total}",
        "results": results,
    }
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = [d for d in dataset if d.get("type") == "单选"]

    print(f"单选题共 {len(dataset)} 道")
    print(f"Mode: {MODE_NAME}")
    print(f"Model: {GPT_MODEL}")
    print(f"Retrieve top-k: {RETRIEVE_TOP_K}\n")

    all_results, done_ids = load_checkpoint()

    correct = sum(1 for r in all_results if r.get("is_correct"))
    total = len(all_results)
    ch_total = defaultdict(int)
    ch_correct = defaultdict(int)
    answer_counter = Counter()
    error_counts = Counter()

    for r in all_results:
        ch = r.get("chapter", "未知")
        ch_total[ch] += 1
        answer_counter[r.get("pred")] += 1
        error_counts[r.get("error_type", "unknown")] += 1
        if r.get("is_correct"):
            ch_correct[ch] += 1

    for i, sample in enumerate(tqdm(dataset, desc="Evaluating")):
        qnum = str(sample.get("question_num", i + 1))
        if qnum in done_ids:
            continue

        question = sample.get("query", sample.get("question", ""))
        options = sample.get("options", {}) or {}
        gt = str(sample["answer"]).strip().upper()
        ch = str(sample.get("chapter", "未知"))
        valid_opts = {str(k).strip().upper() for k in options.keys()} or {"A", "B", "C", "D"}

        try:
            retrieval_query, cards = retrieve_evidence(question, options)
            evidence_text = format_evidence(cards)
            raw_output = query_gr(build_prompt(question, options, evidence_text, valid_opts))
            pred = extract_pred(raw_output, valid_opts)
        except Exception as e:
            tqdm.write(f"\n[ERROR] Question {qnum} failed: {repr(e)}")
            retrieval_query = None
            cards = []
            raw_output = ""
            pred = None

        if pred:
            pred = pred.strip().upper()

        is_correct = pred == gt
        valid_answer = pred in valid_opts if pred else False

        if is_correct:
            error_type = "correct"
        elif pred is None:
            error_type = "null_output"
        elif not valid_answer:
            error_type = "invalid_answer"
        else:
            error_type = "wrong"

        total += 1
        ch_total[ch] += 1
        answer_counter[pred] += 1
        error_counts[error_type] += 1

        if is_correct:
            correct += 1
            ch_correct[ch] += 1
        else:
            tqdm.write(f"\nQuestion: {qnum} GT: {gt} Pred: {pred} Error: {error_type}")

        if i < DEBUG_SAMPLES:
            tqdm.write("\n" + "=" * 60)
            tqdm.write(f"Question {qnum}: {question}")
            tqdm.write(f"Retrieval query: {retrieval_query}")
            for c in cards:
                tqdm.write(f"  {c['card_id']} | {c['title']} | score={c['score']:.4f}")
            tqdm.write(f"Output: {raw_output[:300]}")
            tqdm.write(f"GT: {gt} Pred: {pred}")
            tqdm.write("=" * 60)

        all_results.append({
            "question_num": qnum,
            "question": question,
            "options": options,
            "gt": gt,
            "pred": pred,
            "raw_output": raw_output[:1000],
            "is_correct": is_correct,
            "valid_answer": valid_answer,
            "error_type": error_type,
            "chapter": ch,
            "retrieval_query": retrieval_query,
            "retrieved_cards": cards,
            "best_score": max((c["score"] for c in cards), default=0.0),
            "mode": MODE_NAME,
        })

        if total % SAVE_EVERY == 0:
            save_checkpoint(all_results, len(dataset))
            tqdm.write(f"[checkpoint] Saved {total}/{len(dataset)}")

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    valid_answer_count = sum(1 for r in all_results if r.get("valid_answer"))
    valid_answer_rate = valid_answer_count / total if total else 0.0

    print("\n" + "=" * 60)
    print(f"Mode = {MODE_NAME}")
    print(f"Accuracy = {correct / total:.4f} ({correct}/{total})")
    print(f"Valid Answer Rate = {valid_answer_rate:.4f}")
    print(f"Answer Distribution = {dict(answer_counter)}")
    print(f"Error Distribution = {dict(error_counts)}")

    output = {
        "mode": MODE_NAME,
        "model": GPT_MODEL,
        "base_url": BASE_URL,
        "data_path": DATA_PATH,
        "timestamp": timestamp,
        "rag_type": "single_round_rag",
        "config": {
            "retrieve_top_k": RETRIEVE_TOP_K,
            "retrieval_query": "question + all option texts",
            "evidence_format": "plain_text_cards",
            "answer_only_prompting": True,
            "temperature": 0,
        },
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "valid_answer_rate": round(valid_answer_rate, 4),
        "answer_distribution": dict(answer_counter),
        "error_distribution": dict(error_counts),
        "chapter_stats": {
            ch: {
                "correct": ch_correct[ch],
                "total": ch_total[ch],
                "accuracy": round(ch_correct[ch] / ch_total[ch], 4),
            }
            for ch in ch_total
        },
        "results": all_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nResult saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()
