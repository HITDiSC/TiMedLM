"""
BELLE通用数据处理脚本（6000条加权修复版）
- 默认从BELLE数据集中筛选500条通用题
- 排除藏医/中医/民族医学相关内容，避免与专业数据混淆
- 改写为轨迹格式，plan/judge固定为“无需检索，直接回答”
- 不含query和tool消息
"""

import json
import random
import re
from typing import List, Optional, Tuple


EXCLUDE_KEYWORDS = [
    "藏医", "藏药", "藏族", "中医", "中药", "针灸", "经络", "穴位",
    "民族医", "蒙医", "苗医", "维医", "傣医",
    "隆病", "赤巴", "培根", "三因", "尿诊",
    "今天", "今日", "最新", "最近", "现在几点", "天气",
    "股票", "基金", "汇率", "比特币",
]

SYSTEM_PROMPT = (
    "你是一个知识丰富的AI助手，能够回答各类常识、科学、文化等方面的问题。"
    "\n\n【回答规则】\n"
    "对于通用知识问题，可以直接基于已有知识回答；不要编造实时信息。"
)


def is_valid(instruction: str, output: str) -> bool:
    """过滤不适合作为通用指令遵循数据的条目。"""
    instruction = (instruction or "").strip()
    output = (output or "").strip()
    text = instruction + output

    if any(kw in text for kw in EXCLUDE_KEYWORDS):
        return False

    if len(instruction) < 5 or len(instruction) > 300:
        return False

    if len(output) < 10 or len(output) > 1000:
        return False

    # 过滤明显依赖实时日期的信息
    if re.search(r"\d{4}年|\d+月\d+日", instruction):
        return False

    return True


def to_trajectory(instruction: str, output: str) -> dict:
    """将instruction/output对转换为轨迹格式。"""
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": instruction.strip()
            },
            {
                "role": "assistant",
                "content": "<plan>该问题属于通用知识，不需要检索藏医典籍，可以直接回答。</plan>",
            },
            {
                "role": "assistant",
                "content": "<judge>无需检索，直接回答。</judge>"
            },
            {
                "role": "assistant",
                "content": output.strip()
            },
        ],
        "format_id": "general_direct",
        "source": "belle",
    }


def load_from_hf(n_target: int) -> List[Tuple[str, str]]:
    """从HuggingFace下载BELLE数据集并筛选。"""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("请先安装datasets：pip install datasets --break-system-packages") from exc

    print("从HuggingFace加载BELLE数据集...")
    ds = load_dataset("BelleGroup/train_0.5M_CN", split="train", streaming=True)

    candidates: List[Tuple[str, str]] = []
    checked = 0

    for item in ds:
        checked += 1

        inst = item.get("instruction", "").strip()
        output = item.get("output", "").strip()

        if is_valid(inst, output):
            candidates.append((inst, output))

        if len(candidates) >= n_target * 5:
            break

        if checked % 10000 == 0:
            print(f"  已检查 {checked} 条，有效候选 {len(candidates)} 条")

    print(f"  共检查 {checked} 条，有效候选 {len(candidates)} 条")
    return candidates


def load_from_local(path: str) -> List[Tuple[str, str]]:
    """从本地jsonl文件加载。"""
    candidates: List[Tuple[str, str]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"  跳过JSON解析失败行: {line_no}")
                continue

            inst = item.get("instruction", "").strip()
            output = item.get("output", "").strip()

            if is_valid(inst, output):
                candidates.append((inst, output))

    print(f"  本地文件有效候选: {len(candidates)} 条")
    return candidates


def main(
    input_path: Optional[str],
    output_path: str,
    seed: int,
    n_target: int
) -> None:
    rng = random.Random(seed)

    if input_path:
        candidates = load_from_local(input_path)
    else:
        candidates = load_from_hf(n_target)

    if len(candidates) < n_target:
        print(f"警告：有效候选只有 {len(candidates)} 条，不足 {n_target}，全部保留")
        selected = candidates
    else:
        selected = rng.sample(candidates, n_target)

    result = [to_trajectory(inst, out) for inst, out in selected]
    rng.shuffle(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n已写出: {len(result)} 条 → {output_path}")


# ─────────────────────────────────────────────
# 路径与参数配置区：只需要改这里，然后直接运行
# ─────────────────────────────────────────────

# 本地BELLE jsonl路径。若想自动从HuggingFace下载，设为 None。
INPUT_PATH = None

# 示例：
# INPUT_PATH = r"belle_raw.jsonl"

OUTPUT_PATH = r"belle_train_weighted.json"

SEED = 42
N_TARGET = 500


if __name__ == "__main__":
    main(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        seed=SEED,
        n_target=N_TARGET,
    )