# This file merges SFT data sources into a single training file.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
数据合并脚本
将 mcq_train.json / qa_train.json / belle_train.json 合并为最终训练集

用法：
    python merge_train_data.py  --mcq    mcq_train.json   --qa     qa_train.json --belle  belle_train.json --output final_train.json --seed   42

预期输出：
    选择题  9种 × 400条 = 3600条
    问答对  5种 × 400条 = 2000条
    BELLE   1种 × 400条 =  400条
    合计                  6000条
"""

import json
import random
import argparse
from collections import defaultdict


def load(path, label):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  {label}: {len(data)} 条")
    return data


def main(mcq_path, qa_path, belle_path, output_path, seed):
    rng = random.Random(seed)

    print("读取各数据源：")
    mcq   = load(mcq_path,   "选择题轨迹")
    qa    = load(qa_path,    "问答对轨迹")
    belle = load(belle_path, "BELLE通用题")

    merged = mcq + qa + belle
    rng.shuffle(merged)

    # 统计各格式分布
    dist = defaultdict(int)
    for item in merged:
        dist[item.get("format_id", "unknown")] += 1

    print(f"\n合并后总条数: {len(merged)}")
    print("各格式分布：")
    for fid, cnt in sorted(dist.items()):
        print(f"  {fid:30s}: {cnt}")

    # 检查是否有format_id缺失
    missing = sum(1 for x in merged if not x.get("format_id"))
    if missing:
        print(f"\n警告：{missing} 条缺少 format_id 字段")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"\n已写出: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq",    default="mcq_new_train.json")
    parser.add_argument("--qa",     default="qa_train.json")
    parser.add_argument("--belle",  default="belle_train.json")
    parser.add_argument("--output", default="final_train.json")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    main(args.mcq, args.qa, args.belle, args.output, args.seed)
