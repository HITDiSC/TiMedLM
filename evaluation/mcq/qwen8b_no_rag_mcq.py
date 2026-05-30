# -*- coding: utf-8 -*-
# This file evaluates the base Qwen model on MCQ without RAG.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
基线模型评估脚本（nothink模式）
Qwen3-8B原始模型，无RAG，无LoRA
与 eval_baseline_think.py 的唯一区别：enable_thinking=False，MAX_NEW_TOKENS 缩短

运行：
    CUDA_VISIBLE_DEVICES=0 python eval_baseline_nothink.py
"""

import os
import re
import json
import torch
from datetime import datetime
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ─────────────────────────────────────────────
# 配置（路径与think版保持完全一致）
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get("TIMEDLM_BASE_MODEL_PATH", "Qwen/Qwen3-8B")
DATA_PATH = os.environ.get("TIMEDLM_MCQ_TEST_PATH", "data/samples/mcq_eval_sample.json")       # ⚠️ 三个脚本必须用同一份数据

MAX_NEW_TOKENS = 512    # nothink无思考链，256~512已足够
DEBUG_SAMPLES  = 3

timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.environ.get("TIMEDLM_MCQ_RESULT_DIR", "results/mcq/qwen8b_no_rag")
RESULT_PATH = f"{RESULT_DIR}/eval_nothink_baseline_{timestamp}.json"
os.makedirs(RESULT_DIR, exist_ok=True)

# system prompt 与 think 版完全一致，控制变量
SYSTEM_PROMPT = """\
你是一位精通藏医学的专家，拥有深厚的藏医理论和临床知识。
回答问题时，请仔细分析每个选项，结合藏医理论进行推理，最终给出最准确的答案。"""

# nothink 模式下模型不会自动展开思维链，
# 保留 analysis 字段让模型仍输出推理过程，行为与 think 版对齐
USER_PROMPT_TEMPLATE = """\
以下是一道藏医学单选题，请根据你的藏医学知识认真分析每个选项后作答。

题目：{question}

选项：
{option_str}

请先对每个选项进行分析，然后以JSON格式输出最终答案：
{{"analysis": "你的分析过程", "answer": "正确答案字母"}}"""

# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────

print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("加载模型（原始Qwen3-8B，无LoRA）...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("模型加载完成\n")

# ─────────────────────────────────────────────
# 推理与答案提取
# ─────────────────────────────────────────────

def generate(messages):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,   # ← 唯一差异
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None, top_p=None, top_k=None,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # nothink 模式无 thinking token，直接解码全部新 token
    output_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def extract_pred(content: str):
    """
    与三个 nothink 脚本完全一致的提取逻辑：
    1. 完整 JSON → answer 字段
    2. answer 字段正则
    3. 末尾 50 字符找字母
    """
    content = re.sub(r"```json|```", "", content).strip()
    try:
        raw = json.loads(content)["answer"]
        matches = re.findall(r"[ABCDE]", str(raw).upper())
        return matches[0] if matches else None
    except Exception:
        pass
    m = re.search(r'"answer"\s*:\s*"([ABCDE])', content.upper())
    if m:
        return m.group(1)
    matches = re.findall(r"[ABCDE]", content[-50:].upper())
    return matches[0] if matches else None

# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    dataset = [d for d in dataset if d.get("type") == "单选"]
    print(f"单选题共 {len(dataset)} 道\n")
    print("===== 基线模型评估（Qwen3-8B，nothink，无RAG）=====\n")

    correct    = 0
    total      = 0
    ch_total   = defaultdict(int)
    ch_correct = defaultdict(int)
    all_results = []

    for i, sample in enumerate(tqdm(dataset, desc="评估中")):
        question   = sample.get("query", sample.get("question", ""))
        option_str = "\n".join([f"{k}. {v}" for k, v in sample["options"].items()])
        gt         = sample["answer"].strip().upper()
        ch         = str(sample.get("chapter", "未知"))

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT_TEMPLATE.format(
                question=question, option_str=option_str
            )},
        ]

        content = generate(messages)
        pred    = extract_pred(content)
        if pred:
            pred = pred.strip().upper()

        if i < DEBUG_SAMPLES:
            print(f"\n{'='*60}\n【题目{i+1}】{question}")
            print(f"【输出】{content[:200]}")
            print(f"【GT】{gt}  【Pred】{pred}")

        is_correct    = (pred == gt)
        total        += 1
        ch_total[ch] += 1
        if is_correct:
            correct        += 1
            ch_correct[ch] += 1
        else:
            tqdm.write(f"\n----------------------")
            tqdm.write(f"题号: {total}  GT: {gt}  Pred: {pred}")
            tqdm.write(f"题目: {question[:50]}")

        all_results.append({
            "question_num": sample.get("question_num", str(total)),
            "question":     question,
            "options":      sample.get("options", {}),
            "gt":           gt,
            "pred":         pred,
            "is_correct":   is_correct,
            "raw_output":   content[:500],
            "chapter":      ch,
            "mode":         "baseline_nothink",
        })

    print(f"\n{'='*50}")
    print(f"Accuracy = {correct/total:.4f} ({correct}/{total})")

    print("\n按章节统计：")
    for ch in sorted(ch_total.keys()):
        t = ch_total[ch]
        c = ch_correct[ch]
        print(f"  第{ch}章: {c/t:.4f} ({c}/{t})")

    output = {
        "mode":          "baseline_nothink",
        "timestamp":     timestamp,
        "data_path":     DATA_PATH,
        "model_path":    MODEL_PATH,
        "total":         total,
        "correct":       correct,
        "accuracy":      round(correct / total, 4),
        "chapter_stats": {ch: {"correct": ch_correct[ch], "total": ch_total[ch]}
                          for ch in ch_total},
        "results":       all_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到：{RESULT_PATH}")


if __name__ == "__main__":
    main()
