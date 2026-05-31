# -*- coding: utf-8 -*-
"""
原始 Qwen3-8B + Native Single-RAG 评估脚本（nothink）

设置：
- 原始 Qwen3-8B
- 无 LoRA
- 单轮 RAG
- top-6 evidence
- enable_thinking=False
- 不使用多轮 plan-query-judge
- 不使用 forced 候选打分

运行：
    CUDA_VISIBLE_DEVICES=0 python eval_qwen_native_rag_nothink.py
"""

import os
import re
import sys
import json
import torch
from datetime import datetime
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ─────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get("TIMEDLM_BASE_MODEL_PATH", "Qwen/Qwen3-8B")
DATA_PATH = os.environ.get("TIMEDLM_MCQ_TEST_PATH", "data/samples/mcq_eval_sample.json")

RESULT_DIR = os.environ.get("TIMEDLM_MCQ_RESULT_DIR", "results/mcq/qwen8b_single_rag")
os.makedirs(RESULT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"{RESULT_DIR}/eval_qwen_native_rag_nothink_{timestamp}.json"

# 检索配置
RETRIEVE_TOP_K = 6

# 生成配置
MAX_NEW_TOKENS = 512
DEBUG_SAMPLES = 3


# ─────────────────────────────────────────────
# 检索模块
# ─────────────────────────────────────────────

RETRIEVAL_ROOT = os.environ.get(
    "RETRIEVAL_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "timedlm", "retrieval")),
)
sys.path.append(RETRIEVAL_ROOT)
from retrieval import retrieve_with_scores


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一位精通藏医学的专家，拥有深厚的藏医理论和临床知识。
你需要根据给定的藏医知识库证据回答单选题。
最终答案必须是 A、B、C、D 中的一个。
"""

USER_PROMPT_TEMPLATE = """\
以下是一道藏医学单选题。请根据题目、选项和检索到的知识库证据作答。

题目：
{question}

选项：
{option_str}

知识库证据：
{evidence_text}

要求：
1. 请先简要分析每个选项。
2. 最终必须以 JSON 格式输出：
{{"analysis": "你的简要分析", "answer": "正确答案字母"}}
3. answer 字段只能是 A、B、C、D 中的一个。
"""


# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────

print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载原始 Qwen3-8B，无 LoRA...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print("模型加载完成\n")


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def build_retrieval_query(question, options):
    """
    Native RAG 单轮检索 query。
    尽量覆盖题干和选项，但不做多轮规划。
    """
    option_text = "；".join([f"{k}.{v}" for k, v in options.items()])
    return f"{question}；{option_text}"


def get_card_content(card):
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


def retrieve_evidence(question, options, top_k=6):
    query = build_retrieval_query(question, options)

    try:
        scored_cards = retrieve_with_scores(query, top_k=top_k)
    except Exception as e:
        print(f"[检索失败] query={query} error={e}")
        return query, [], []

    cards = []
    for card, score in scored_cards:
        cards.append({
            "card_id": card.get("card_id", ""),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", ""),
            "content": get_card_content(card),
            "score": float(score),
        })

    return query, scored_cards, cards


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


def generate(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def extract_pred(content):
    """
    只允许 A/B/C/D。
    如果模型输出 E 或其他内容，不当作合法答案。
    """
    if not content:
        return None

    content = re.sub(r"```json|```", "", content).strip()

    # 1. 完整 JSON
    try:
        obj = json.loads(content)
        raw = str(obj.get("answer", "")).upper()
        matches = re.findall(r"[ABCD]", raw)
        if matches:
            return matches[0]
    except Exception:
        pass

    # 2. answer 字段正则
    m = re.search(r'"answer"\s*:\s*"([ABCD])"', content.upper())
    if m:
        return m.group(1)

    # 3. 中文格式
    patterns = [
        r"答案[是为：:]\s*([ABCD])",
        r"正确答案[是为：:]\s*([ABCD])",
        r"最终答案[是为：:]\s*([ABCD])",
        r"应?选\s*([ABCD])",
        r"故选\s*([ABCD])",
    ]

    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 4. 末尾 50 字符兜底
    matches = re.findall(r"[ABCD]", content[-50:].upper())
    return matches[-1] if matches else None


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = [d for d in dataset if d.get("type") == "单选"]

    print(f"单选题共 {len(dataset)} 道\n")
    print("===== 原始 Qwen3-8B + Native RAG no-think 评估 =====\n")

    correct = 0
    total = 0

    ch_total = defaultdict(int)
    ch_correct = defaultdict(int)

    all_results = []

    for i, sample in enumerate(tqdm(dataset, desc="评估中")):
        question = sample.get("query", sample.get("question", ""))
        options = sample.get("options", {})
        option_str = "\n".join([f"{k}. {v}" for k, v in options.items()])
        gt = sample["answer"].strip().upper()
        ch = str(sample.get("chapter", "未知"))
        qnum = sample.get("question_num", str(i + 1))

        retrieval_query, scored_cards, cards = retrieve_evidence(
            question,
            options,
            top_k=RETRIEVE_TOP_K,
        )

        evidence_text = format_evidence(cards)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    question=question,
                    option_str=option_str,
                    evidence_text=evidence_text,
                )
            },
        ]

        content = generate(messages)
        pred = extract_pred(content)

        if pred:
            pred = pred.strip().upper()

        is_correct = pred == gt

        total += 1
        ch_total[ch] += 1

        if is_correct:
            correct += 1
            ch_correct[ch] += 1
        else:
            tqdm.write("\n----------------------")
            tqdm.write(f"题号: {qnum}  GT: {gt}  Pred: {pred}")
            tqdm.write(f"题目: {question[:80]}")

        if i < DEBUG_SAMPLES:
            print(f"\n{'=' * 60}")
            print(f"【题目 {qnum}】{question}")
            print(f"【检索 query】{retrieval_query}")
            print(f"【top evidence】")
            for c in cards:
                print(f"  {c['card_id']} | {c['title']} | score={c['score']:.4f}")
            print(f"【输出】{content[:500]}")
            print(f"【GT】{gt}  【Pred】{pred}")
            print(f"{'=' * 60}")

        all_results.append({
            "question_num": qnum,
            "question": question,
            "options": options,
            "gt": gt,
            "pred": pred,
            "is_correct": is_correct,
            "raw_output": content[:1000],
            "retrieval_query": retrieval_query,
            "retrieved_cards": cards,
            "best_score": max([c["score"] for c in cards], default=0.0),
            "chapter": ch,
            "mode": "qwen_native_rag_nothink",
        })

    valid_answer_count = sum(
        1 for r in all_results
        if r.get("pred") in {"A", "B", "C", "D"}
    )
    e_output_count = sum(
        1 for r in all_results
        if r.get("pred") == "E"
    )
    invalid_count = total - valid_answer_count

    valid_answer_rate = valid_answer_count / total if total else 0.0
    e_error_rate = e_output_count / total if total else 0.0

    print(f"\n{'=' * 50}")
    print(f"Accuracy = {correct / total:.4f} ({correct}/{total})")
    print(f"Valid Answer Rate = {valid_answer_rate:.4f}")
    print(f"E-error Rate = {e_error_rate:.4f}")
    print(f"Invalid Count = {invalid_count}")

    print("\n按章节统计：")
    for ch in sorted(ch_total.keys()):
        t = ch_total[ch]
        c = ch_correct[ch]
        print(f"  第{ch}章: {c / t:.4f} ({c}/{t})")

    output = {
        "mode": "qwen_native_rag_nothink",
        "timestamp": timestamp,
        "data_path": DATA_PATH,
        "model_path": MODEL_PATH,
        "rag_type": "native_single_rag",
        "retrieve_top_k": RETRIEVE_TOP_K,
        "thinking": False,
        "lora": False,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "valid_answer_rate": round(valid_answer_rate, 4),
        "e_error_rate": round(e_error_rate, 4),
        "invalid_count": invalid_count,
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

    print(f"\n结果已保存到：{RESULT_PATH}")


if __name__ == "__main__":
    main()