# This file builds open-ended QA SFT examples.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
问答对轨迹处理脚本（6000条加权修复版）
- 原始问答约1000条时，默认随机抽取200条作为测试集
- 测试集输出纯 Q&A 格式（input / reference）
- 剩余约800条作为训练候选池
- 训练集按不同问答格式加权采样，默认共1500条：
    qa_plain    400
    qa_brief    400
    qa_points   300
    qa_expert   200
    qa_detailed 200
- 同一条原始问答可以出现在不同提示词格式中；同一提示词内默认不重复采样
"""

import json
import random
from collections import defaultdict
from typing import Dict, List


QA_PROMPTS = {
    "qa_plain": "",
    "qa_detailed": "请结合藏医典籍详细回答，并注明引用来源。",
    "qa_points": "请分点说明，条理清晰地回答以下问题。",
    "qa_brief": "请简要回答以下问题，100字以内。",
    "qa_expert": "请作为藏医专家，用专业术语回答以下问题。",
}

DEFAULT_PROMPT_COUNTS = {
    "qa_plain": 400,
    "qa_brief": 400,
    "qa_points": 300,
    "qa_expert": 200,
    "qa_detailed": 200,
}

CONTROL_TAGS = ("<plan>", "<query>", "<judge>")


def extract_qa(item: dict) -> dict:
    """从轨迹中提取纯Q&A对，用于测试集。"""
    messages = item.get("messages", [])

    user_content = ""
    for msg in messages:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break

    reference = ""
    for msg in reversed(messages):
        content = msg.get("content", "").strip()
        if msg.get("role") == "assistant" and not content.startswith(CONTROL_TAGS):
            reference = content
            break

    return {
        "input": user_content,
        "reference": reference,
        "question_type": item.get("question_type", ""),
        "seed_card_ids": item.get("seed_card_ids", []),
    }


def insert_prompt(item: dict, prompt_id: str, prompt_text: str) -> dict:
    """在训练集轨迹的user消息前插入提示词，返回新item。"""
    new_messages = []
    inserted = False

    for msg in item.get("messages", []):
        if msg.get("role") == "user" and prompt_text and not inserted:
            new_content = prompt_text + "\n" + msg.get("content", "")
            new_messages.append({**msg, "content": new_content})
            inserted = True
        else:
            new_messages.append(msg)

    return {
        **item,
        "messages": new_messages,
        "format_id": prompt_id,
        "source": "qa_original",
    }


def validate_test_set(test_set: List[dict]) -> None:
    empty_input = sum(1 for x in test_set if not x["input"])
    empty_ref = sum(1 for x in test_set if not x["reference"])

    if empty_input or empty_ref:
        print(f"  警告：测试集 input为空 {empty_input} 条，reference为空 {empty_ref} 条")
    else:
        print("  测试集校验通过，无空字段")


def main(
    input_path: str,
    out_train: str,
    out_test: str,
    seed: int,
    n_test: int,
    prompt_counts: Dict[str, int]
) -> None:
    rng = random.Random(seed)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"问答对总数: {len(data)}")

    if n_test >= len(data):
        raise ValueError(f"n_test={n_test} 必须小于问答总数 {len(data)}")

    indices = list(range(len(data)))
    rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_pool_idx = indices[n_test:]

    print(f"测试集: {len(test_idx)} 条")
    print(f"训练候选池: {len(train_pool_idx)} 条")

    test_set = [extract_qa(data[i]) for i in test_idx]

    train_set = []
    actual_counts = defaultdict(int)

    for prompt_id, target in prompt_counts.items():
        prompt_text = QA_PROMPTS[prompt_id]
        pool = train_pool_idx.copy()
        rng.shuffle(pool)

        if target > len(pool):
            print(
                f"  警告: {prompt_id} 目标 {target} 条超过训练候选池 {len(pool)} 条，"
                "同一提示词内将允许有放回采样。"
            )
            selected = [rng.choice(pool) for _ in range(target)]
        else:
            selected = pool[:target]

        for idx in selected:
            train_set.append(insert_prompt(data[idx], prompt_id, prompt_text))
            actual_counts[prompt_id] += 1

        print(f"  {prompt_id}: {actual_counts[prompt_id]} 条")

    rng.shuffle(train_set)

    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)

    with open(out_test, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"\n训练集: {len(train_set)} 条（完整轨迹+加权提示词） → {out_train}")
    print(f"测试集: {len(test_set)} 条（纯Q&A） → {out_test}")

    validate_test_set(test_set)


# ─────────────────────────────────────────────
# 路径与参数配置区：只需要改这里，然后直接运行
# ─────────────────────────────────────────────

INPUT_PATH = r"qa_trajectory.json"          # 原始问答轨迹文件，约1000条
OUT_TRAIN_PATH = r"qa_train_weighted.json"  # 输出问答训练集
OUT_TEST_PATH = r"qa_test.json"             # 输出问答测试集

SEED = 42
N_TEST = 200

PROMPT_COUNTS = {
    "qa_plain": 400,
    "qa_brief": 400,
    "qa_points": 300,
    "qa_expert": 200,
    "qa_detailed": 200,
}


if __name__ == "__main__":
    main(
        input_path=INPUT_PATH,
        out_train=OUT_TRAIN_PATH,
        out_test=OUT_TEST_PATH,
        seed=SEED,
        n_test=N_TEST,
        prompt_counts=PROMPT_COUNTS,
    )