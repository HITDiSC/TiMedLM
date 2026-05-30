# -*- coding: utf-8 -*-
# This file evaluates a closed model on MCQ with multi-round RAG.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
Fair strong-model Multi-RAG MCQ evaluation.

Designed for the comparison:
    local model multi-round RAG (final_fix/eval/new_eval.py)
    vs.
    strong model multi-round RAG

The retrieval budget is intentionally aligned with new_eval.py:
- max rounds: 3
- retrieve top-k per generated query: 4
- max evidence cards shown per round: 6
- final answer stage: answer-only A/B/C/D

The strong model is used as the retrieval controller and final answerer. It does
not receive the gold answer.

Environment:
    export GR_API_KEY="..."
    export GR_MODEL="gpt-4o"  # optional
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


# ============================================================
# Config
# ============================================================

GR_API_KEY = os.environ.get("GR_API_KEY", "")
BASE_URL = os.environ.get("GR_BASE_URL", "https://endpoint.wendalog.com")
GPT_MODEL = os.environ.get("GR_MODEL", "gpt-4o")

DATA_PATH = os.environ.get("TIMEDLM_MCQ_TEST_PATH", "data/samples/mcq_eval_sample.json")
RESULT_DIR = os.environ.get("TIMEDLM_MCQ_RESULT_DIR", "results/mcq/closed_model_multi_rag")

MODE_NAME = "strong_model_fair_multirag_answer_only"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"{RESULT_DIR}/{MODE_NAME}_{timestamp}.json"
CKPT_PATH = f"{RESULT_DIR}/{MODE_NAME}_ckpt.json"

MAX_ROUNDS = 3
RETRIEVE_TOP_K_PER_QUERY = 4
MAX_QUERIES_PER_ROUND = 2
MAX_CARDS_PER_ROUND = 6

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


# ============================================================
# API
# ============================================================

SYSTEM_PROMPT = (
    "You are a Tibetan medicine exam assistant. "
    "Use retrieval evidence carefully and follow the requested output format exactly."
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


# ============================================================
# Retrieval helpers
# ============================================================

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


def build_initial_query(question: str, options: dict) -> str:
    option_text = "；".join(str(v).strip() for v in (options or {}).values() if str(v).strip())
    return f"{question}；{option_text}".strip("；")


def retrieve_for_queries(queries):
    all_scored = []
    seen = set()

    for q in queries[:MAX_QUERIES_PER_ROUND]:
        try:
            raw = retrieve_with_scores(q, top_k=RETRIEVE_TOP_K_PER_QUERY)
        except Exception:
            raw = []

        for item in raw:
            if not isinstance(item, tuple) or len(item) != 2:
                continue
            card, meta = item
            if not isinstance(card, dict):
                continue
            cid = card.get("card_id") or get_card_content(card)[:120]
            if cid in seen:
                continue
            seen.add(cid)
            all_scored.append((card, score_from_meta(meta), q))

    all_scored.sort(key=lambda x: x[1], reverse=True)
    return all_scored[:MAX_CARDS_PER_ROUND]


def format_cards(scored_cards, include_query=False):
    rows = []
    for card, score, query in scored_cards:
        row = {
            "card_id": card.get("card_id", ""),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", ""),
            "score": round(float(score), 4),
            "content": get_card_content(card),
        }
        if include_query:
            row["retrieval_query"] = query
        rows.append(row)
    return json.dumps(rows, ensure_ascii=False)


def option_lines(options: dict) -> str:
    return "\n".join(f"{k}. {v}" for k, v in (options or {}).items())


# ============================================================
# Multi-round controller
# ============================================================

def parse_query_control(content: str):
    text = re.sub(r"```json|```", "", content or "").strip()

    try:
        obj = json.loads(text)
        stop = bool(obj.get("stop", False))
        queries = obj.get("queries", obj.get("query", []))
        if isinstance(queries, str):
            queries = [queries]
        queries = [str(q).strip() for q in queries if str(q).strip()]
        return stop, queries[:MAX_QUERIES_PER_ROUND], obj
    except Exception:
        pass

    if re.search(r"\b(STOP|FINAL|ENOUGH)\b", text, re.IGNORECASE):
        return True, [], {"raw": content}

    lines = [x.strip() for x in text.splitlines() if x.strip()]
    return False, lines[:MAX_QUERIES_PER_ROUND], {"raw": content}


def build_query_prompt(question, options, rounds_log, round_num):
    evidence_blocks = []
    for rd in rounds_log:
        evidence_blocks.append(
            f"Round {rd['round']} query: {' | '.join(rd['queries'])}\n"
            f"Evidence:\n{format_cards(rd['cards'])}"
        )
    evidence_text = "\n\n".join(evidence_blocks) if evidence_blocks else "No evidence retrieved yet."

    return f"""You are controlling retrieval for a Tibetan medicine multiple-choice question.

Question:
{question}

Options:
{option_lines(options)}

Current retrieved evidence:
{evidence_text}

Round: {round_num}/{MAX_ROUNDS}

Generate retrieval queries that cover the question core concept and option keywords.
If the evidence is already enough to answer, set stop=true.

Output only JSON:
{{"stop": false, "queries": ["query 1", "query 2"], "note": "short reason"}}

Rules:
- Produce at most {MAX_QUERIES_PER_ROUND} queries.
- Do not answer the question here.
- Do not include the gold answer or any option letter as a conclusion.
"""


def run_multirag(question, options):
    rounds_log = []

    first_query = build_initial_query(question, options)
    next_queries = [first_query]

    for round_num in range(1, MAX_ROUNDS + 1):
        cards = retrieve_for_queries(next_queries)
        round_info = {
            "round": round_num,
            "queries": next_queries,
            "cards": cards,
            "control_raw": None,
            "stop": False,
        }
        rounds_log.append(round_info)

        if round_num >= MAX_ROUNDS:
            break

        control_prompt = build_query_prompt(question, options, rounds_log, round_num)
        control_raw = query_gr(control_prompt)
        stop, queries, control_obj = parse_query_control(control_raw)

        round_info["control_raw"] = control_raw
        round_info["control"] = control_obj
        round_info["stop"] = stop

        if stop or not queries:
            break

        next_queries = queries

    return rounds_log


# ============================================================
# Final answer
# ============================================================

def build_final_prompt(question, options, rounds_log, valid_options):
    evidence_blocks = []
    for rd in rounds_log:
        evidence_blocks.append(
            f"Round {rd['round']} query: {' | '.join(rd['queries'])}\n"
            f"Evidence:\n{format_cards(rd['cards'], include_query=True)}"
        )
    evidence_text = "\n\n".join(evidence_blocks)
    valid_str = "/".join(sorted(valid_options))

    return f"""Answer the Tibetan medicine multiple-choice question using the retrieved evidence.

Question:
{question}

Options:
{option_lines(options)}

Retrieved evidence:
{evidence_text}

Final answer rules:
- Output exactly one option letter.
- The option letter must be one of: {valid_str}
- Do not explain.
- Do not output JSON.
- Do not output punctuation.
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

    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and "answer" in obj:
            raw = str(obj["answer"]).upper()
            matches = re.findall(rf"[{valid_pat}]", raw)
            if matches:
                return matches[0]
    except Exception:
        pass

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


def answer_question(question, options, rounds_log, valid_options):
    prompt = build_final_prompt(question, options, rounds_log, valid_options)
    raw = query_gr(prompt)
    pred = extract_pred(raw, valid_options)

    if pred:
        return pred, raw

    valid_str = "/".join(sorted(valid_options))
    retry_prompt = prompt + f"\n\nYour previous output was invalid. Output only one letter from {valid_str}:"
    raw2 = query_gr(retry_prompt)
    pred2 = extract_pred(raw2, valid_options)
    return pred2, raw2


# ============================================================
# Checkpoint
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = [d for d in dataset if d.get("type") == "单选"]

    print(f"单选题共 {len(dataset)} 道")
    print(f"Mode: {MODE_NAME}")
    print(f"Model: {GPT_MODEL}")
    print(f"Max rounds: {MAX_ROUNDS}")
    print(f"Top-k per query: {RETRIEVE_TOP_K_PER_QUERY}")
    print(f"Max cards per round: {MAX_CARDS_PER_ROUND}\n")

    all_results, done_ids = load_checkpoint()

    correct = sum(1 for r in all_results if r.get("is_correct"))
    total = len(all_results)
    total_rounds = sum(r.get("retrieval_rounds", 0) for r in all_results)

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
            rounds_log = run_multirag(question, options)
            pred, raw_output = answer_question(question, options, rounds_log, valid_opts)
        except Exception as e:
            tqdm.write(f"\n[ERROR] Question {qnum} failed: {repr(e)}")
            rounds_log = []
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

        n_rounds = len(rounds_log)
        total += 1
        total_rounds += n_rounds
        ch_total[ch] += 1
        answer_counter[pred] += 1
        error_counts[error_type] += 1

        if is_correct:
            correct += 1
            ch_correct[ch] += 1
        else:
            tqdm.write(
                f"\nQuestion: {qnum} GT: {gt} Pred: {pred} "
                f"Rounds: {n_rounds} Error: {error_type} Text: {question[:80]}"
            )

        if i < DEBUG_SAMPLES:
            tqdm.write("\n" + "=" * 60)
            tqdm.write(f"Question {qnum}: {question}")
            for rd in rounds_log:
                tqdm.write(f"Round {rd['round']} queries: {' | '.join(rd['queries'])}")
                for c, score, query in rd["cards"]:
                    tqdm.write(f"  {c.get('card_id', '')} | {c.get('title', '')} | score={score:.4f}")
            tqdm.write(f"Output: {raw_output[:500]}")
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
            "retrieval_rounds": n_rounds,
            "total_retrieved": sum(len(rd["cards"]) for rd in rounds_log),
            "rounds_log": [
                {
                    "round": rd["round"],
                    "queries": rd["queries"],
                    "control_raw": rd.get("control_raw"),
                    "stop": rd.get("stop", False),
                    "retrieved_cards": [
                        {
                            "card_id": c.get("card_id", ""),
                            "title": c.get("title", ""),
                            "card_type": c.get("card_type", ""),
                            "score": round(float(score), 4),
                            "query": query,
                            "content": get_card_content(c)[:300],
                        }
                        for c, score, query in rd["cards"]
                    ],
                }
                for rd in rounds_log
            ],
            "mode": MODE_NAME,
        })

        if total % SAVE_EVERY == 0:
            save_checkpoint(all_results, len(dataset))
            tqdm.write(f"[checkpoint] Saved {total}/{len(dataset)}")

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    valid_answer_count = sum(1 for r in all_results if r.get("valid_answer"))
    valid_answer_rate = valid_answer_count / total if total else 0.0
    avg_rounds = total_rounds / total if total else 0.0

    print("\n" + "=" * 60)
    print(f"Mode = {MODE_NAME}")
    print(f"Accuracy = {correct / total:.4f} ({correct}/{total})")
    print(f"Avg Rounds = {avg_rounds:.2f}")
    print(f"Valid Answer Rate = {valid_answer_rate:.4f}")
    print(f"Answer Distribution = {dict(answer_counter)}")
    print(f"Error Distribution = {dict(error_counts)}")

    print("\nChapter stats:")
    for ch in sorted(ch_total.keys()):
        t = ch_total[ch]
        c = ch_correct[ch]
        print(f"  Chapter {ch}: {c / t:.4f} ({c}/{t})")

    output = {
        "mode": MODE_NAME,
        "model": GPT_MODEL,
        "base_url": BASE_URL,
        "data_path": DATA_PATH,
        "timestamp": timestamp,
        "rag_type": "fair_multi_round_rag",
        "config": {
            "max_rounds": MAX_ROUNDS,
            "retrieve_top_k_per_query": RETRIEVE_TOP_K_PER_QUERY,
            "max_queries_per_round": MAX_QUERIES_PER_ROUND,
            "max_cards_per_round": MAX_CARDS_PER_ROUND,
            "final_answer_only": True,
            "temperature": 0,
        },
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_rounds": round(avg_rounds, 2),
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
