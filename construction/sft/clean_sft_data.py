# -*- coding: utf-8 -*-
# This file cleans SFT trajectory data before training.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
清洗merged_train.json：
1. 只保留messages字段
2. tool消息里只保留query和refined_result，删掉card_id和citation_text
3. 删掉seed_card_ids、rounds、citations、book、type、gold、format_id等元数据

运行：
    python clean_merged.py
"""

import json
import re

INPUT_PATH  = r"final_train.json"
OUTPUT_PATH = r"final_train_v5.json"


def clean_tool_content(content_str):
    """清洗tool消息的content，只保留query和refined_result"""
    try:
        items = json.loads(content_str)
        cleaned = []
        for item in items:
            cleaned.append({
                "query":          item.get("query", ""),
                "refined_result": item.get("refined_result", ""),
            })
        return json.dumps(cleaned, ensure_ascii=False)
    except Exception:
        return content_str


def clean_messages(messages):
    """清洗messages列表"""
    cleaned = []
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "")

        if role == "tool":
            content = clean_tool_content(content)

        cleaned.append({"role": role, "content": content})
    return cleaned


def main():
    print(f"加载：{INPUT_PATH}")
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"原始数据：{len(data)} 条")

    cleaned_data = []
    for item in data:
        cleaned_data.append({
            "messages": clean_messages(item["messages"]),
            "format_id": item.get("format_id", ""),
        })

    print(f"清洗后：{len(cleaned_data)} 条")

    # 统计token节省情况（粗略估计）
    original_len = sum(len(json.dumps(item, ensure_ascii=False)) for item in data)
    cleaned_len  = sum(len(json.dumps(item, ensure_ascii=False)) for item in cleaned_data)
    print(f"原始大小：{original_len/1024/1024:.1f} MB")
    print(f"清洗后大小：{cleaned_len/1024/1024:.1f} MB")
    print(f"压缩比：{cleaned_len/original_len:.2%}")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"\n已保存到：{OUTPUT_PATH}")

    # 打印一条样例确认格式
    print("\n样例（第1条）：")
    sample = cleaned_data[0]
    for msg in sample["messages"]:
        role    = msg["role"]
        content = msg["content"][:100].replace("\n", " ")
        print(f"  [{role}] {content}...")


if __name__ == "__main__":
    main()