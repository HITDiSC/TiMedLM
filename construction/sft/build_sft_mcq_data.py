"""
错题/选择题轨迹处理脚本（6000条加权修复版）

目标：修复“15种格式平均采样”导致的选择题能力负迁移。
- 不再使用“每种格式固定400条”
- 按format_id设置不同目标条数
- 默认选择题模块生成4000条：
    plain_letter       1831  # 1831道题全量覆盖强约束格式
    option_content      550
    one_sentence        550
    natural_sentence    350
    json                150
    with_retrieval      150
    json_with_source    100
    evidence_citation   170
    book_citation       149
- 默认开启 --force_four_options：如果题面含E且gold在A-D，则删除E选项；gold为E的题跳过
- 所有选择题user指令都会替换为带“答案空间约束”的新指令
"""

import copy
import json
import random
import re
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple


EXISTING_FORMAT_IDS = [
    "plain_letter",
    "option_content",
    "json",
    "one_sentence",
    "with_retrieval",
    "json_with_source",
]

NEW_FORMAT_IDS = [
    "natural_sentence",
    "evidence_citation",
    "book_citation",
]

ALL_FORMAT_IDS = EXISTING_FORMAT_IDS + NEW_FORMAT_IDS

# 6000条修复版：选择题模块默认4000条
DEFAULT_FORMAT_COUNTS = {
    "plain_letter": 1831,
    "option_content": 550,
    "one_sentence": 550,
    "natural_sentence": 350,
    "json": 150,
    "with_retrieval": 150,
    "json_with_source": 100,
    "evidence_citation": 170,
    "book_citation": 149,
}

CARD_ID_TO_BOOK = {
    "lll": "《蓝琉璃》",
    "sbyd": "《四部医典》",
    "ywyz": "《月王药诊》",
    "jzbc": "《晶珠本草》",
    "zyyx": "《藏医药学概论》",
}

CONTROL_TAGS = ("<plan>", "<query>", "<judge>")
OPTION_RE = re.compile(r"^([A-E])\.\s*(.+)")


def get_book_from_card_id(card_id: str) -> str:
    prefix = card_id.split("_")[1] if "_" in card_id else ""
    return CARD_ID_TO_BOOK.get(prefix, "藏医典籍")


def get_user_content(messages: Sequence[dict]) -> str:
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""


def get_option_letters(messages: Sequence[dict]) -> List[str]:
    letters = []
    for line in get_user_content(messages).split("\n"):
        m = OPTION_RE.match(line.strip())
        if m:
            letters.append(m.group(1))
    return letters


def option_space_text(messages: Sequence[dict]) -> str:
    letters = get_option_letters(messages)
    if not letters:
        letters = ["A", "B", "C", "D"]
    return "、".join(letters)


def get_instruction(format_id: str, messages: Sequence[dict]) -> str:
    options = option_space_text(messages)
    base = (
        f"请从题目给出的选项中选择唯一正确答案。本题只有 {options} 这些选项，"
        f"必须从 {options} 中选择，不得输出未给出的选项。"
    )
    if "E" not in options.split("、"):
        base += "不得输出 E。"

    instructions = {
        "plain_letter": base + f"只能输出 {options} 中的一个大写字母，不要解释，不要输出标点或其他内容。",
        "option_content": base + "请按“选项.内容”的格式作答，例如“B. 面部油腻”。",
        "json": base + "请输出JSON格式答案，如 {\"B\": \"面部油腻\"}。JSON的键必须是题目给出的选项之一。",
        "one_sentence": base + "请输出正确答案并用一句话说明理由，理由不超过30字。格式：答案：X。理由：……",
        "with_retrieval": base + "请输出答案并说明检索依据，依据必须与题目选项匹配。",
        "json_with_source": base + "请输出带来源的JSON格式答案，字段包含answer、content和source。",
        "natural_sentence": base + "请用完整句子回答，格式如“正确答案是X，因为……”，理由不超过40字。",
        "evidence_citation": base + "请根据检索到的典籍内容回答，并注明支持该答案的关键证据。",
        "book_citation": base + "请给出答案，并注明该知识点出自哪部藏医典籍。",
    }
    return instructions[format_id]


def get_correct_option_text(messages: Sequence[dict], gold: str) -> str:
    for line in get_user_content(messages).split("\n"):
        if line.strip().startswith(f"{gold}."):
            return line.strip()[len(gold) + 1:].strip()
    return ""


def get_all_options(messages: Sequence[dict]) -> List[Tuple[str, str]]:
    options = []
    for line in get_user_content(messages).split("\n"):
        m = OPTION_RE.match(line.strip())
        if m:
            options.append((m.group(1), m.group(2).strip()))
    return options


def replace_user_instruction(messages: Sequence[dict], new_instruction: str) -> List[dict]:
    """替换user消息里的第一行指令。"""
    new_messages = []
    replaced = False
    for msg in messages:
        if msg.get("role") == "user" and not replaced:
            content = msg.get("content", "")
            lines = content.strip().split("\n", 1)
            if len(lines) > 1:
                new_content = new_instruction + "\n" + lines[1]
            else:
                new_content = new_instruction + "\n" + content
            new_messages.append({**msg, "content": new_content})
            replaced = True
        else:
            new_messages.append(copy.deepcopy(msg))
    return new_messages


def replace_last_assistant(messages: Sequence[dict], new_content: str) -> List[dict]:
    new_messages = list(copy.deepcopy(messages))
    for i in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[i]
        content = msg.get("content", "")
        if msg.get("role") == "assistant" and not content.startswith(CONTROL_TAGS):
            new_messages[i] = {**msg, "content": new_content}
            break
    return new_messages


def remove_e_option_from_user(content: str) -> str:
    """删除题面中的E选项行，保留其他内容。"""
    kept_lines = []
    for line in content.split("\n"):
        if re.match(r"^\s*E\.\s*", line):
            continue
        kept_lines.append(line)
    return "\n".join(kept_lines)


def force_four_option_item(item: dict) -> Optional[dict]:
    """
    将五选项题转换成四选项题。
    - gold为A-D：删除E选项行
    - gold为E：返回None，避免训练四选项模型时继续学习E
    """
    gold = item.get("gold", "")
    if gold == "E":
        return None

    new_item = copy.deepcopy(item)
    new_messages = []
    for msg in new_item.get("messages", []):
        if msg.get("role") == "user":
            new_messages.append({
                **msg,
                "content": remove_e_option_from_user(msg.get("content", ""))
            })
        else:
            new_messages.append(msg)
    new_item["messages"] = new_messages
    return new_item


def normalize_item_for_format(
    item: dict,
    format_id: str,
    force_four_options: bool
) -> Optional[dict]:
    """清理选项空间，并统一替换为强约束user指令。"""
    work = force_four_option_item(item) if force_four_options else copy.deepcopy(item)
    if work is None:
        return None

    instruction = get_instruction(format_id, work.get("messages", []))
    work["messages"] = replace_user_instruction(work.get("messages", []), instruction)
    return work


def extract_tool_items(messages: Sequence[dict]) -> List[dict]:
    all_items = []
    for msg in messages:
        if msg.get("role") == "tool":
            try:
                parsed = json.loads(msg.get("content", ""))
                if isinstance(parsed, list):
                    all_items.extend(parsed)
            except Exception:
                pass
    return all_items


def pick_evidence(messages: Sequence[dict], gold_text: str) -> str:
    all_items = extract_tool_items(messages)
    evidence = ""
    keywords = list(gold_text[:8])

    for item in all_items:
        result = item.get("refined_result", "")
        if any(kw in result for kw in keywords):
            evidence = result.rstrip("。，、")
            break

    if not evidence and all_items:
        evidence = all_items[0].get("refined_result", "").rstrip("。，、")

    return evidence


def build_natural_sentence_answer(messages: Sequence[dict], gold: str, gold_text: str) -> str:
    evidence = pick_evidence(messages, gold_text)
    reason = evidence if evidence else gold_text
    if len(reason) > 35:
        reason = reason[:35]
    return f"正确答案是{gold}，因为{reason}。"


def build_evidence_citation_answer(messages: Sequence[dict], gold: str, gold_text: str) -> str:
    evidence = pick_evidence(messages, gold_text)
    if evidence:
        return f"答案是{gold}。根据检索结果，{evidence}，直接支持{gold}项的描述，故选{gold}。"
    return f"答案是{gold}。根据藏医典籍记载，{gold_text}，故选{gold}。"


def build_book_citation_answer(messages: Sequence[dict], gold: str, gold_text: str) -> str:
    book = "藏医典籍"
    all_items = extract_tool_items(messages)
    keywords = list(gold_text[:8])

    for item in all_items:
        result = item.get("refined_result", "")
        if any(kw in result for kw in keywords):
            book = get_book_from_card_id(item.get("card_id", ""))
            break

    if book == "藏医典籍" and all_items:
        book = get_book_from_card_id(all_items[0].get("card_id", ""))

    return f"答案是{gold}。该知识点出自{book}，记载了{gold_text}。"


def build_existing_format_answer(
    messages: Sequence[dict],
    gold: str,
    gold_text: str,
    format_id: str
) -> Optional[str]:
    """
    对原始1~4类格式重写最后答案，避免旧轨迹中的冗余解释继续污染强约束训练。
    with_retrieval/json_with_source 保留原始来源型答案，因为它们本来承担检索/来源学习。
    """
    if not gold or not gold_text:
        return None

    if format_id == "plain_letter":
        return gold

    if format_id == "option_content":
        return f"{gold}.{gold_text}"

    if format_id == "json":
        return json.dumps({gold: gold_text}, ensure_ascii=False)

    if format_id == "one_sentence":
        evidence = pick_evidence(messages, gold_text)
        reason = evidence if evidence else gold_text
        reason = re.sub(r"[。；;，,]*$", "", reason)
        if len(reason) > 26:
            reason = reason[:26]
        return f"答案：{gold}。理由：{reason}。"

    return None


def maybe_rewrite_existing_answer(item: dict, format_id: str, rewrite: bool) -> dict:
    if not rewrite:
        return item

    if format_id not in {"plain_letter", "option_content", "json", "one_sentence"}:
        return item

    messages = item.get("messages", [])
    gold = item.get("gold", "")
    gold_text = get_correct_option_text(messages, gold)
    new_answer = build_existing_format_answer(messages, gold, gold_text, format_id)

    if not new_answer:
        return item

    new_item = copy.deepcopy(item)
    new_item["messages"] = replace_last_assistant(messages, new_answer)
    return new_item


def build_new_format_item(
    base_item: dict,
    format_id: str,
    force_four_options: bool
) -> Optional[dict]:
    work = normalize_item_for_format(base_item, format_id, force_four_options)
    if work is None:
        return None

    messages = work.get("messages", [])
    gold = work.get("gold", "")
    gold_text = get_correct_option_text(messages, gold)

    if not gold or not gold_text:
        return None

    if format_id == "natural_sentence":
        new_answer = build_natural_sentence_answer(messages, gold, gold_text)
    elif format_id == "evidence_citation":
        if not get_all_options(messages):
            return None
        new_answer = build_evidence_citation_answer(messages, gold, gold_text)
    elif format_id == "book_citation":
        new_answer = build_book_citation_answer(messages, gold, gold_text)
    else:
        raise ValueError(f"不支持的新格式: {format_id}")

    work["messages"] = replace_last_assistant(messages, new_answer)
    work["format_id"] = format_id
    work["source"] = "mcq_rewritten"
    return work


def get_group_gold(group: Sequence[dict]) -> str:
    for item in group:
        gold = item.get("gold", "")
        if gold:
            return gold
    return ""


def group_raw_items(raw: Sequence[dict], group_size: int) -> List[List[dict]]:
    groups = []
    for i in range(0, len(raw), group_size):
        group = list(raw[i: i + group_size])
        if len(group) == group_size:
            groups.append(group)
        else:
            print(f"  警告：末尾不完整分组 {len(group)} 条，跳过")
    return groups


def sample_without_replacement(
    pool: List[dict],
    target: int,
    rng: random.Random,
    label: str
) -> List[dict]:
    if target <= 0:
        return []

    if len(pool) < target:
        print(f"  警告: {label} 只有 {len(pool)} 条，不足 {target}，将全部保留")
        return pool.copy()

    return rng.sample(pool, target)


def main(
    input_path: str,
    output_path: str,
    seed: int,
    group_size: int,
    format_counts: Dict[str, int],
    force_four_options: bool,
    rewrite_existing_answers: bool
) -> None:
    rng = random.Random(seed)

    print(f"读取数据: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"  原始轨迹总条数: {len(raw)}")

    groups = group_raw_items(raw, group_size)
    print(f"  题目总数: {len(groups)}")

    if force_four_options:
        e_gold_groups = sum(1 for g in groups if get_group_gold(g) == "E")
        print(f"  force_four_options=True：gold为E的题将跳过，数量: {e_gold_groups}")

    # 原始格式1~6索引
    format_pool = defaultdict(list)
    for group in groups:
        for item in group:
            fid = item.get("format_id", "unknown")
            if fid in EXISTING_FORMAT_IDS:
                normalized = normalize_item_for_format(item, fid, force_four_options)
                if normalized is not None:
                    format_pool[fid].append(normalized)

    print("  各原始格式可用数量:")
    for fid in EXISTING_FORMAT_IDS:
        print(f"    {fid}: {len(format_pool[fid])}")

    result = []

    # 采样原始格式1~6
    for fid in EXISTING_FORMAT_IDS:
        target = format_counts.get(fid, 0)
        sampled = sample_without_replacement(format_pool[fid], target, rng, fid)
        for item in sampled:
            item = maybe_rewrite_existing_answer(item, fid, rewrite_existing_answers)
            result.append({**item, "format_id": fid, "source": "mcq_original_weighted"})
        print(f"  {fid}: 目标 {target}，实际 {len(sampled)}")

    # 新格式8/9/10：按目标数量分别抽题。默认不同新格式之间不重复用同一题。
    available_groups = groups.copy()
    rng.shuffle(available_groups)
    used_group_ids = set()
    skipped_new = defaultdict(int)

    for fmt_id in NEW_FORMAT_IDS:
        target = format_counts.get(fmt_id, 0)
        made = 0

        for group in available_groups:
            if made >= target:
                break

            gid = id(group)
            if gid in used_group_ids:
                continue

            base_item = next(
                (x for x in group if x.get("format_id") == "plain_letter"),
                group[0]
            )

            new_item = build_new_format_item(base_item, fmt_id, force_four_options)
            if new_item is None:
                skipped_new[fmt_id] += 1
                continue

            result.append(new_item)
            used_group_ids.add(gid)
            made += 1

        if made < target:
            print(f"  警告: {fmt_id} 目标 {target}，实际只生成 {made}")
        else:
            print(f"  {fmt_id}: 目标 {target}，实际 {made}")

    if skipped_new:
        print(f"  新格式跳过数量: {dict(skipped_new)}")

    rng.shuffle(result)

    print(f"\n最终训练集总条数: {len(result)}")
    dist = defaultdict(int)
    for item in result:
        dist[item.get("format_id", "unknown")] += 1

    print("各格式分布:")
    for fid in ALL_FORMAT_IDS:
        print(f"  {fid}: {dist.get(fid, 0)}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n已写出: {output_path}")


# ─────────────────────────────────────────────
# 路径与参数配置区：只需要改这里，然后直接运行
# ─────────────────────────────────────────────

INPUT_PATH = r"trajectory.json"           # 原始选择题轨迹文件
OUTPUT_PATH = r"mcq_train_weighted.json"  # 输出的加权选择题训练集

SEED = 42
GROUP_SIZE = 6

# True：删除E选项，gold=E的题跳过；四选项训练推荐 True
FORCE_FOUR_OPTIONS = True

# True：重写 plain_letter / option_content / json / one_sentence 的最后答案，减少冗余解释污染
REWRITE_EXISTING_ANSWERS = True

FORMAT_COUNTS = {
    "plain_letter": 1831,
    "option_content": 550,
    "one_sentence": 550,
    "natural_sentence": 350,
    "json": 150,
    "with_retrieval": 150,
    "json_with_source": 100,
    "evidence_citation": 170,
    "book_citation": 149,
}


if __name__ == "__main__":
    main(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        seed=SEED,
        group_size=GROUP_SIZE,
        format_counts=FORMAT_COUNTS,
        force_four_options=FORCE_FOUR_OPTIONS,
        rewrite_existing_answers=REWRITE_EXISTING_ANSWERS,
    )