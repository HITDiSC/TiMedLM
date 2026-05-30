# -*- coding: utf-8 -*-
# This file evaluates TiMedLM on MCQ with multi-round RAG.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
微调后模型 + 多轮 RAG 评估脚本（think模式优化版 v3）

相对上一版主要优化：
1. plan / query / judge 阶段仍使用 think 模式
2. 最终答案阶段关闭 think 模式，强制只输出 A/B/C/D
3. 初始检索提示加入“必须覆盖题干核心概念 + 每个选项关键词”
4. 检索结果增加选项感知 rerank，降低泛化卡片和无关高分卡片干扰
5. judge_sufficient 增强：第二轮以后，若高分证据覆盖题干/选项关键词，可避免过多 forced
6. forced 阶段改为候选打分：分别计算 A/B/C/D 的条件概率，取最高者
7. 保留断点保存 / 断点恢复机制
"""

import os
import re
import sys
import json
import math
import torch
from datetime import datetime
from collections import defaultdict

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

RETRIEVAL_ROOT = os.environ.get(
    "RETRIEVAL_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "timedlm", "retrieval")),
)
sys.path.append(RETRIEVAL_ROOT)
from retrieval import retrieve_with_scores


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

MODEL_PATH = os.environ.get("TIMEDLM_MODEL_PATH", "models/timedlm-sft-v5")
# 当前 LoRA 路径

LORA_PATH = os.environ.get("TIMEDLM_LORA_PATH", "models/timedlm-lora")


# 测试集路径
DATA_PATH = os.environ.get("TIMEDLM_MCQ_TEST_PATH", "data/samples/mcq_eval_sample.json")

MAX_ROUNDS = 3

# plan/query/judge 阶段允许较长输出
MAX_NEW_TOKENS = 2048

# 最终答案阶段必须非常短
FINAL_MAX_NEW_TOKENS = 4

DEBUG_SAMPLES = 3

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_DIR = os.environ.get("TIMEDLM_MCQ_RESULT_DIR", "results/mcq/timedlm_multi_rag")
# sft
# RESULT_PATH = f"{RESULT_DIR}/eval_finetuned_rag_think_final_strict_v3_{timestamp}.json"
RESULT_PATH = f"{RESULT_DIR}/eval_final280_rag_think_final_strict_{timestamp}.json"


# 新断点文件，避免加载旧版本结果
# CKPT_PATH = f"{RESULT_DIR}/eval_finetuned_rag_think_final_strict_v5_ckpt.json"

CKPT_PATH = f"{RESULT_DIR}/eval_final280_rag_think_final_strict_ckpt.json"

SAVE_EVERY = 20
os.makedirs(RESULT_DIR, exist_ok=True)

# 检索参数
RETRIEVE_TOP_K_PER_QUERY = 4
MAX_CARDS_PER_ROUND = 6

# forced 阶段是否使用 A/B/C/D 候选 logprob 打分
USE_LOGPROB_FOR_FORCED = True

# 如果严格最终输出失败，是否也用 logprob 兜底
USE_LOGPROB_FOR_FINAL_FALLBACK = True


SYSTEM_PROMPT = """\
你是一个面向藏医知识问答与考试解题的AI助手。你必须严格遵循以下规则。

【强制检索规则（最高优先级）】
无论题目涉及任何内容，你都必须先通过检索获取藏医知识，再作答。
绝对禁止在未检索的情况下直接输出答案。

【题型覆盖范围】
本考试涵盖以下所有题型，每类都需要检索：
- 疾病症状与诊断（隆病、赤巴病、培根病等三因疾病）
- 药物功效与性味（清热解毒、温中散寒、药物分类等）
- 治疗用药原则（剂型选择、性味配伍、禁忌）
- 外治法（艾灸、拔罐、放血等主要作用）
- 人体生理（脉诊部位、命脉定义、不同年龄生理特点）
- 时间医学与医学理论

【检索流程】
第一步：生成检索计划 <plan>...</plan>，说明需要检索什么知识。
第二步：生成检索词 <query>多个检索词用；分隔</query>，每轮必须生成query。
第三步：收到检索结果后，判断证据是否充分 <judge>...</judge>。
第四步：若充分则给出答案；否则调整检索词进行下一轮。

【query生成要求】
生成 query 时必须同时覆盖：
1. 题干中的核心概念；
2. 每个选项中的关键术语；
3. 如果是药物、方剂、性味、剂型、适应症题，必须分别检索候选药物/方剂/性味/剂型与病症之间的关系；
4. 如果是“不包括”“不属于”“错误的是”等否定题，必须检索各选项是否属于正确范围。

【最终答案规则】
最终答案只能是题目给出的选项字母之一，例如 A、B、C 或 D。
最终答案阶段不得输出 <plan>、<query>、<judge>。
最终答案阶段不得继续检索。
"""


# ─────────────────────────────────────────────
# 加载模型
# ─────────────────────────────────────────────

print("加载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载基座模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("加载LoRA权重...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()
print("模型加载完成\n")


# ─────────────────────────────────────────────
# 生成函数
# ─────────────────────────────────────────────

def generate(
    messages,
    stop_at=None,
    max_new_tokens=None,
    enable_thinking=True,
    repetition_penalty=1.1,
):
    """
    通用生成函数。
    - plan/query/judge 阶段：enable_thinking=True
    - final answer 阶段：enable_thinking=False
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

    # 兼容 Qwen3 think 模式：若存在 </think> token，则只取其后的内容
    # 151668 沿用你原脚本中的 token id
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    response = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True
    ).strip()

    if stop_at:
        for tag in stop_at:
            if tag in response:
                response = response[:response.index(tag) + len(tag)]
                break

    return response


# ─────────────────────────────────────────────
# 答案提取与标签解析
# ─────────────────────────────────────────────

def extract_pred(content: str, valid_options=None):
    """
    提取预测答案，支持：
    - A
    - 答案是A
    - 正确答案：A
    - 选A
    - {"answer": "A"}
    - {"A": "..."}
    """
    if valid_options is None:
        valid_options = {"A", "B", "C", "D"}

    valid_options = {x.upper() for x in valid_options}
    valid_pat = "".join(sorted(valid_options))

    if not content:
        return None

    content = re.sub(r"```json|```", "", content).strip()
    content_upper = content.upper()

    # 1. 直接单字母
    if content_upper in valid_options:
        return content_upper

    # 2. JSON格式
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            if "answer" in obj:
                raw = str(obj["answer"]).upper()
                matches = re.findall(rf"[{valid_pat}]", raw)
                if matches:
                    return matches[0]

            # 支持 {"A": "..."} 格式
            for k in obj.keys():
                k_upper = str(k).upper()
                if k_upper in valid_options:
                    return k_upper
    except Exception:
        pass

    # 3. "answer": "X"
    m = re.search(rf'"answer"\s*:\s*"([{valid_pat}])"', content_upper)
    if m:
        return m.group(1)

    # 4. 中文自然语言格式
    patterns = [
        rf"答案[是为：:]\s*([{valid_pat}])",
        rf"正确答案[是为：:]\s*([{valid_pat}])",
        rf"最终答案[是为：:]\s*([{valid_pat}])",
        rf"应?选\s*([{valid_pat}])",
        rf"故选\s*([{valid_pat}])",
        rf"选项\s*([{valid_pat}])\s*[是为正确]",
    ]

    for pat in patterns:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_options:
                return letter

    # 5. 末尾100字符中找孤立合法字母
    tail = content_upper[-100:]
    m = re.search(rf"\b([{valid_pat}])\b", tail)
    if m:
        return m.group(1)

    # 6. 兜底：末尾50字符最后出现的合法字母
    matches = re.findall(rf"[{valid_pat}]", content_upper[-50:])
    return matches[-1] if matches else None


def extract_tag(text, tag):
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def has_control_tag(text: str) -> bool:
    if not text:
        return False
    control_tags = [
        "<plan>", "</plan>",
        "<query>", "</query>",
        "<judge>", "</judge>",
    ]
    return any(tag in text for tag in control_tags)


def get_card_content(card):
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


def format_cards(cards):
    results = []
    for card in cards:
        evidence = card.get("evidence", {}) or {}
        results.append({
            "card_id": card.get("card_id", ""),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", ""),
            "content": get_card_content(card),
            "citation_text": evidence.get("citation_text", ""),
        })
    return json.dumps(results, ensure_ascii=False)


# ─────────────────────────────────────────────
# 文本关键词辅助
# ─────────────────────────────────────────────

STOP_WORDS = {
    "以下", "哪种", "哪个", "哪些", "主要", "属于", "不属于", "包括", "不包括",
    "藏医", "认为", "治疗", "常用", "药物", "方剂", "疾病", "患者",
    "中的", "是", "为", "与", "有关", "进行", "选择", "正确", "错误",
    "____", "一种", "适用于", "主要功能"
}


def clean_question_text(text: str) -> str:
    text = text or ""
    text = text.replace("请只输出正确答案的选项字母。", "")
    text = text.replace("### 考试题目", "")
    text = text.replace("____", "")
    text = re.sub(r"^\s*[A-Z]\.\s*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def simple_terms_from_text(text: str):
    """
    轻量关键词抽取，避免额外依赖。
    """
    text = clean_question_text(text)
    text = re.sub(r"[，。；、：:？！\?\s]+", "|", text)
    raw_terms = [x.strip() for x in text.split("|") if x.strip()]

    terms = []
    for t in raw_terms:
        if len(t) < 2:
            continue
        if t in STOP_WORDS:
            continue
        # 去掉明显模板词
        bad = False
        for sw in STOP_WORDS:
            if t == sw:
                bad = True
                break
        if not bad:
            terms.append(t)

    return list(dict.fromkeys(terms))


def option_terms(options: dict):
    terms = []
    for k, v in (options or {}).items():
        value = str(v).strip()
        if value:
            terms.append(value)

            # 对 “甘、酸、咸” 这类选项做拆分
            for part in re.split(r"[、，,/\s]+", value):
                part = part.strip()
                if part and len(part) >= 1:
                    terms.append(part)

    return list(dict.fromkeys(terms))


def keyword_hit_in_cards(raw_question: str, options: dict, cards: list) -> bool:
    """
    判断检索卡片是否覆盖题干或选项关键词。
    """
    terms = option_terms(options)
    terms.extend(simple_terms_from_text(raw_question))

    # 只保留非空
    terms = [t for t in terms if t]

    if not terms:
        return False

    card_text = "\n".join(
        (c.get("title", "") + " " + get_card_content(c))
        for c in cards
    )

    hit_count = 0
    for t in terms:
        if t and t in card_text:
            hit_count += 1

    return hit_count >= 1


def option_coverage_count(options: dict, cards: list) -> int:
    """
    统计 top cards 覆盖了多少个选项文本。
    """
    card_text = "\n".join(
        (c.get("title", "") + " " + get_card_content(c))
        for c in cards
    )

    count = 0
    for v in (options or {}).values():
        v = str(v).strip()
        if v and v in card_text:
            count += 1

    return count


# ─────────────────────────────────────────────
# 检索重排
# ─────────────────────────────────────────────

def rerank_cards_by_options(raw_question: str, options: dict, scored_cards: list):
    """
    在 embedding 分数基础上，加入轻量规则：
    - 命中选项内容：加分
    - 命中题干关键词：加分
    - 非诊断题中泛化 diag_manual 卡片：轻微降权
    - 卡片标题直接命中选项/题干：额外加分
    """
    q_terms = simple_terms_from_text(raw_question)
    o_terms = option_terms(options)

    reranked = []

    for item in scored_cards:
        # item 可能是 (card, score)，也可能已经带扩展字段
        if len(item) == 2:
            card, raw_score = item
        else:
            card, raw_score = item[0], item[1]

        title = card.get("title", "") or ""
        content = get_card_content(card)
        text = title + " " + content
        cid = card.get("card_id", "") or ""

        bonus = 0.0

        # 命中完整选项内容，加分较高
        for term in o_terms:
            if not term:
                continue
            if term in title:
                bonus += 0.06
            elif term in text:
                bonus += 0.04

        # 命中题干关键词
        for term in q_terms:
            if not term:
                continue
            if term in title:
                bonus += 0.03
            elif term in text:
                bonus += 0.01

        # 泛化诊断卡片降权：非诊断/寒热题时，diag_manual 容易占位
        q = raw_question or ""
        if cid.startswith("diag_manual"):
            if not any(x in q for x in ["诊断", "寒热", "尿诊", "脉诊", "病性", "症状"]):
                bonus -= 0.04

        # case 卡片在通用概念题中轻微降权
        if card.get("card_type") == "case":
            if not any(x in q for x in ["患者", "症状", "表现", "诊断", "病例"]):
                bonus -= 0.02

        adjusted = raw_score + bonus
        adjusted = max(0.0, min(1.0, float(adjusted)))

        reranked.append((card, adjusted, float(raw_score), float(bonus)))

    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked


def retrieve_cards_by_query(query_content, raw_question=None, options=None, verbose=False):
    queries = [
        q.strip()
        for q in re.split(r"[；;]", query_content)
        if q.strip()
    ]

    all_cards_scored = []

    for q in queries:
        try:
            all_cards_scored.extend(
                retrieve_with_scores(q, top_k=RETRIEVE_TOP_K_PER_QUERY)
            )
        except Exception as e:
            if verbose:
                print(f"【检索失败】query={q} error={e}")

    seen = set()
    uniq_scored = []

    for card, score in all_cards_scored:
        cid = card.get("card_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            uniq_scored.append((card, float(score)))

    if raw_question is not None and options is not None:
        reranked = rerank_cards_by_options(raw_question, options, uniq_scored)
    else:
        reranked = [(c, float(s), float(s), 0.0) for c, s in uniq_scored]

    top_scored = reranked[:MAX_CARDS_PER_ROUND]
    cards_this_round = [c for c, _, _, _ in top_scored]
    best_score = max((s for _, s, _, _ in top_scored), default=0.0)

    return reranked, top_scored, cards_this_round, best_score


# ─────────────────────────────────────────────
# 最终答案候选打分
# ─────────────────────────────────────────────

def score_candidate_choice(messages, choice: str) -> float:
    """
    计算候选字母 choice 作为 assistant 输出的平均 logprob。
    仅用于 forced 或最终兜底，不用于 plan/query/judge。
    """
    prefix_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    full_text = prefix_text + choice

    prefix_ids = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(model.device)

    full_ids = tokenizer(
        full_text,
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"].to(model.device)

    prefix_len = prefix_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prefix_len:
        return -1e9

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)

    token_logps = []
    for pos in range(prefix_len, full_len):
        target_id = full_ids[0, pos]
        prev_pos = pos - 1
        if prev_pos < 0:
            continue
        token_logps.append(log_probs[0, prev_pos, target_id].item())

    if not token_logps:
        return -1e9

    return sum(token_logps) / len(token_logps)


def choose_by_logprob(messages, valid_options, verbose=False):
    """
    对 A/B/C/D 分别打分，选择条件概率最高者。
    """
    valid_options = sorted({x.upper() for x in valid_options})
    scores = {}

    for opt in valid_options:
        try:
            scores[opt] = score_candidate_choice(messages, opt)
        except Exception as e:
            scores[opt] = -1e9
            if verbose:
                print(f"【候选打分失败】{opt}: {e}")

    best = max(scores.items(), key=lambda x: x[1])[0]

    if verbose:
        print(f"【候选打分】{scores} → {best}")

    return best, scores


# ─────────────────────────────────────────────
# 最终答案严格生成
# ─────────────────────────────────────────────

def generate_final_choice(messages, valid_options, verbose=False):
    """
    证据充分后使用。
    关键点：
    - enable_thinking=False
    - max_new_tokens 很小
    - 不让模型继续 plan/query/judge
    """
    valid_options = {x.upper() for x in valid_options}
    valid_str = "/".join(sorted(valid_options))

    final_prompt = (
        f"现在给出最终答案。\n"
        f"请在内部比较各选项与检索证据的一致性，但不要输出分析过程。\n"
        f"规则：\n"
        f"1. 不要输出 <plan>、<query>、<judge>。\n"
        f"2. 不要继续检索。\n"
        f"3. 不要解释，不要展示推理。\n"
        f"4. 只能输出一个选项字母，必须是 {valid_str} 中的一个。\n"
        f"5. 不要输出标点或其他任何内容。"
    )

    final_messages = messages + [
        {"role": "user", "content": final_prompt}
    ]

    final = generate(
        final_messages,
        max_new_tokens=FINAL_MAX_NEW_TOKENS,
        enable_thinking=False,
        repetition_penalty=1.0,
    )

    if verbose:
        print(f"【严格最终输出1】{final}")

    pred = extract_pred(final, valid_options)

    if pred and not has_control_tag(final):
        return pred, final

    # 如果仍然输出控制标签或无法提取，追加一次更直接的纠错请求
    final_messages.append({"role": "assistant", "content": final})
    final_messages.append({
        "role": "user",
        "content": f"你的输出不合法。只输出一个字母，必须是 {valid_str} 中的一个："
    })

    final2 = generate(
        final_messages,
        max_new_tokens=2,
        enable_thinking=False,
        repetition_penalty=1.0,
    )

    if verbose:
        print(f"【严格最终输出2】{final2}")

    pred2 = extract_pred(final2, valid_options)

    if pred2 and not has_control_tag(final2):
        return pred2, final2

    # 最后兜底：候选打分
    if USE_LOGPROB_FOR_FINAL_FALLBACK:
        score_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"现在只输出最终答案。只能输出 {valid_str} 中的一个字母，"
                    f"不要解释，不要标点："
                )
            }
        ]
        pred3, scores = choose_by_logprob(score_messages, valid_options, verbose=verbose)
        return pred3, f"[logprob_final_fallback] {pred3} scores={scores}"

    return pred2, final2


def generate_forced_choice(messages, valid_options, evidence_quality_label, verbose=False):
    """
    MAX_ROUNDS结束后仍未得到答案时使用。

    v3 优化：
    - 默认使用 A/B/C/D 候选 logprob 打分，而不是自由生成
    - 避免 forced 阶段随机被证据噪声带偏后输出长文本
    """
    valid_options = {x.upper() for x in valid_options}
    valid_str = "/".join(sorted(valid_options))

    if evidence_quality_label == "high":
        prefix = "已检索到较高相关度的藏医知识证据。"
    elif evidence_quality_label == "medium":
        prefix = "已检索到部分相关藏医知识证据。"
    else:
        prefix = "检索证据不足，但仍需从题目给出的选项中选择最合理答案。"

    forced_prompt = (
        f"{prefix}\n"
        f"现在请根据上方题目、选项和检索证据，选择最可能正确的答案。\n"
        f"只输出一个字母，必须是 {valid_str} 中的一个。\n"
        f"不要解释，不要分析，不要输出 <plan>、<query>、<judge>。"
    )

    forced_messages = messages + [
        {"role": "user", "content": forced_prompt}
    ]

    if USE_LOGPROB_FOR_FORCED:
        pred, scores = choose_by_logprob(
            forced_messages,
            valid_options,
            verbose=verbose
        )
        final = f"[logprob_forced] {pred} scores={scores}"
        if verbose:
            print(f"【强制候选打分输出】{final}")
        return pred, final

    # 如果关闭 logprob，则退回短生成
    final = generate(
        forced_messages,
        max_new_tokens=FINAL_MAX_NEW_TOKENS,
        enable_thinking=False,
        repetition_penalty=1.0,
    )

    if verbose:
        print(f"【强制输出1】{final}")

    pred = extract_pred(final, valid_options)

    if pred and not has_control_tag(final):
        return pred, final

    forced_messages.append({"role": "assistant", "content": final})
    forced_messages.append({
        "role": "user",
        "content": f"输出不合法。只输出一个字母，必须是 {valid_str} 中的一个："
    })

    final2 = generate(
        forced_messages,
        max_new_tokens=2,
        enable_thinking=False,
        repetition_penalty=1.0,
    )

    if verbose:
        print(f"【强制输出2】{final2}")

    pred2 = extract_pred(final2, valid_options)
    return pred2, final2


# ─────────────────────────────────────────────
# 错误分类与证据质量
# ─────────────────────────────────────────────

def classify_error(pred, gt, retrieval_rounds, rounds_log, answer_source):
    if pred == gt:
        return "correct"

    if pred is None:
        return "null_output"

    if retrieval_rounds == 0:
        return "no_retrieval"

    total_cards = sum(len(r.get("retrieved_cards", [])) for r in rounds_log)
    if total_cards == 0:
        return "empty_retrieval"

    if answer_source == "forced":
        return "forced_output_wrong"

    if str(answer_source).startswith("final_invalid"):
        return "final_invalid_wrong"

    return "retrieved_wrong"


SCORE_HIGH = 0.75
SCORE_MEDIUM = 0.50


def evidence_quality(rounds_log_with_scores):
    best = max(
        (r.get("best_score", 0.0) for r in rounds_log_with_scores),
        default=0.0
    )

    if best >= SCORE_HIGH:
        return "high"
    if best >= SCORE_MEDIUM:
        return "medium"
    return "low"


def is_judge_sufficient(
    judge_content: str,
    best_score: float,
    round_num: int,
    raw_question: str = "",
    options: dict = None,
    cards: list = None,
) -> bool:
    """
    v3 规则：
    1. judge 明确充分且没有否定词 → True
    2. 第一轮如果 judge 否定，仍保守，不直接充分
    3. 第二轮以后，如果 best_score 高且检索卡片命中题干/选项关键词 → True
    4. 第二轮以后，如果 best_score 极高且覆盖至少一个选项 → True
    """
    judge_content = judge_content or ""
    options = options or {}
    cards = cards or []

    negative_words = {
        "不充分", "无法", "缺乏", "没有找到", "未找到", "不足",
        "不能回答", "无法回答", "仍需", "继续检索", "尚未", "不明确"
    }

    positive_words = {
        "证据充分", "可以回答", "能够回答", "足以回答", "充分"
    }

    has_negative = any(w in judge_content for w in negative_words)
    has_positive = any(w in judge_content for w in positive_words)

    if has_positive and not has_negative:
        return True

    # 第一轮保持谨慎，避免过早停止
    if round_num == 1:
        return False

    # 第二轮以后，若高分证据覆盖题干或选项关键词，可以停止，减少 forced
    if best_score >= 0.90:
        if keyword_hit_in_cards(raw_question, options, cards):
            return True

    # 极高分并且覆盖至少一个选项，也可以停止
    if best_score >= 0.96 and option_coverage_count(options, cards) >= 1:
        return True

    return False


# ─────────────────────────────────────────────
# 断点保存 / 加载
# ─────────────────────────────────────────────

def load_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return [], set()

    try:
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        done_results = ckpt.get("results", [])
        done_ids = {str(r["question_num"]) for r in done_results}

        print(f"[断点恢复] 发现断点文件，已完成 {len(done_results)} 道题，继续评估...\n")
        return done_results, done_ids

    except Exception as e:
        print(f"[断点恢复] 读取断点失败（{e}），从头开始\n")
        return [], set()


def save_checkpoint(all_results, dataset_total):
    ckpt = {
        "timestamp": timestamp,
        "data_path": DATA_PATH,
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH,
        "progress": f"{len(all_results)}/{dataset_total}",
        "results": all_results,
    }

    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────
# 检索辅助
# ─────────────────────────────────────────────

def fallback_query_from_question(question: str) -> str:
    """
    第一轮模型没有生成query时的兜底检索词。
    """
    text = question

    # 去掉固定指令
    text = text.replace("请只输出正确答案的选项字母。", "")
    text = text.replace("### 考试题目", "")

    # 去掉选项行
    lines = []
    for line in text.splitlines():
        if re.match(r"^\s*[A-Z]\.\s*", line):
            continue
        lines.append(line)

    text = "\n".join(lines)

    # 去掉常见填空符
    text = text.replace("____", "")
    text = text.replace("（ ）", "")
    text = text.replace("()", "")

    text = text.strip()

    if not text:
        return "藏医基础理论"

    return text[:100]


def build_initial_instruction(full_question: str, raw_question: str, options: dict) -> str:
    option_lines = []
    for k, v in (options or {}).items():
        option_lines.append(f"{k}. {v}")

    option_text = "\n".join(option_lines)

    return (
        f"{full_question}\n\n"
        f"【重要】你必须先检索藏医知识库，不可直接作答。\n"
        f"生成 query 时必须满足：\n"
        f"1. 覆盖题干核心概念：{raw_question}\n"
        f"2. 覆盖每个选项关键词：\n{option_text}\n"
        f"3. 如果是药物、方剂、性味、剂型、适应症题，必须分别检索每个候选项与题干概念的关系。\n"
        f"4. 如果题目含“不包括”“不属于”“错误的是”，必须检索各选项是否属于正确范围。\n"
        f"请立即开始：先输出 <plan>检索计划</plan>，再输出 <query>检索词</query>。"
    )


# ─────────────────────────────────────────────
# RAG 推理
# ─────────────────────────────────────────────

def rag_answer(full_question, raw_question="", options=None, verbose=False, valid_options=None):
    if valid_options is None:
        valid_options = {"A", "B", "C", "D"}

    options = options or {}
    valid_options = {x.upper() for x in valid_options}

    initial_instruction = build_initial_instruction(
        full_question=full_question,
        raw_question=raw_question,
        options=options,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_instruction},
    ]

    rounds_log = []
    total_rounds = 0
    answer_source = "forced"

    for round_num in range(1, MAX_ROUNDS + 1):
        round_info = {
            "round": round_num,
            "query": "",
            "retrieved_cards": [],
            "retrieved_count": 0,
            "judge": "",
            "judge_sufficient": False,
            "best_score": 0.0,
            "option_coverage": 0,
        }

        # 1. 生成 plan
        plan_resp = generate(
            messages,
            stop_at=["</plan>"],
            enable_thinking=True,
            max_new_tokens=512,
        )

        if verbose:
            print(f"\n--- 第{round_num}轮 ---")
            print(f"【plan】{extract_tag(plan_resp, 'plan')}")

        messages.append({"role": "assistant", "content": plan_resp})

        # 2. 生成 query
        query_resp = generate(
            messages,
            stop_at=["</query>"],
            enable_thinking=True,
            max_new_tokens=512,
        )

        if "<query>" not in query_resp:
            if verbose:
                print(f"【无query响应】{query_resp[:200]}")

            if round_num == 1:
                fallback_query = fallback_query_from_question(full_question)
                query_content = fallback_query
                round_info["query"] = fallback_query
                round_info["forced_query"] = True

                if verbose:
                    print(f"【兜底检索词】{fallback_query}")

                messages.append({"role": "assistant", "content": query_resp})
                messages.append({
                    "role": "user",
                    "content": f"请先检索以下关键词，再作答：<query>{fallback_query}</query>"
                })
            else:
                # 非第一轮没有query，尝试直接提取答案
                pred = extract_pred(query_resp, valid_options)
                if pred:
                    answer_source = f"no_query_round{round_num}"
                    return pred, rounds_log, total_rounds, answer_source

                messages.append({"role": "assistant", "content": query_resp})
                continue
        else:
            query_content = extract_tag(query_resp, "query")
            round_info["query"] = query_content

            # 合并 plan + query 到同一条 assistant，保持对话轨迹自然
            messages[-1]["content"] = plan_resp + "\n" + query_resp[query_resp.index("<query>"):]

        if verbose:
            print(f"【query】{query_content}")

        # 3. 检索 + 重排
        reranked, top_scored, cards_this_round, best_score = retrieve_cards_by_query(
            query_content,
            raw_question=raw_question,
            options=options,
            verbose=verbose
        )

        round_info["retrieved_cards"] = [
            {
                "card_id": c.get("card_id"),
                "title": c.get("title"),
                "card_type": c.get("card_type"),
                "content": get_card_content(c)[:300],
                "score": round(float(score), 4),
                "raw_score": round(float(raw_score), 4),
                "rerank_bonus": round(float(bonus), 4),
            }
            for c, score, raw_score, bonus in top_scored
        ]

        round_info["retrieved_count"] = len(cards_this_round)
        round_info["best_score"] = round(float(best_score), 4)
        round_info["option_coverage"] = option_coverage_count(options, cards_this_round)

        if verbose:
            print(f"【检索到{len(reranked)}张卡片，最高分={best_score:.3f}】")
            for c, s, raw_s, bonus in top_scored:
                print(
                    f"  [{c.get('card_type')}] "
                    f"{c.get('title')} "
                    f"({c.get('card_id')}) "
                    f"score={s:.3f} raw={raw_s:.3f} bonus={bonus:.3f}"
                )

        messages.append({
            "role": "tool",
            "content": format_cards(cards_this_round)
        })

        # 4. 生成 judge
        judge_resp = generate(
            messages,
            stop_at=["</judge>"],
            enable_thinking=True,
            max_new_tokens=512,
        )

        judge_content = extract_tag(judge_resp, "judge")
        sufficient = is_judge_sufficient(
            judge_content=judge_content,
            best_score=best_score,
            round_num=round_num,
            raw_question=raw_question,
            options=options,
            cards=cards_this_round,
        )

        round_info["judge"] = judge_content
        round_info["judge_sufficient"] = sufficient

        if verbose:
            print(f"【judge】{judge_content}  →  sufficient={sufficient}")

        messages.append({"role": "assistant", "content": judge_resp})
        rounds_log.append(round_info)
        total_rounds = round_num

        # 5. 如果证据充分，严格最终作答
        if sufficient:
            answer_source = f"sufficient_round{round_num}"

            pred, final = generate_final_choice(
                messages,
                valid_options,
                verbose=verbose
            )

            if verbose:
                print(f"【最终答案】{final}")

            if pred:
                return pred, rounds_log, total_rounds, answer_source

            # 证据已充分但最终输出不合法，不再进入下一轮
            answer_source = f"final_invalid_round{round_num}"
            return pred, rounds_log, total_rounds, answer_source

    # 6. MAX_ROUNDS 后 forced
    answer_source = "forced"
    eq = evidence_quality(rounds_log)

    if verbose:
        print(f"【强制输出模式】证据质量={eq}")

    pred, final = generate_forced_choice(
        messages,
        valid_options,
        eq,
        verbose=verbose
    )

    if verbose:
        print(f"【强制最终答案】{final}")

    return pred, rounds_log, total_rounds, answer_source


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    dataset = [d for d in dataset if d.get("type") == "单选"]

    print(f"单选题共 {len(dataset)} 道\n")

    all_results, done_ids = load_checkpoint()

    correct = sum(1 for r in all_results if r["is_correct"])
    total = len(all_results)
    total_rounds = sum(r.get("retrieval_rounds", 0) for r in all_results)

    ch_total = defaultdict(int)
    ch_correct = defaultdict(int)
    error_counts = defaultdict(int)

    for r in all_results:
        ch = r.get("chapter", "未知")
        ch_total[ch] += 1
        error_counts[r.get("error_type", "unknown")] += 1
        if r["is_correct"]:
            ch_correct[ch] += 1

    print("===== 微调后模型 + 多轮RAG think + 严格最终答案评估 v3 =====\n")

    for i, sample in enumerate(tqdm(dataset, desc="评估中")):
        qnum = str(sample.get("question_num", i + 1))

        if qnum in done_ids:
            continue

        question = sample.get("query", sample.get("question", ""))

        options = sample.get("options", {}) or {}

        option_str = "\n".join([
            f"{k}. {v}"
            for k, v in options.items()
        ])

        full_q = (
            f"请只输出正确答案的选项字母。\n"
            f"### 考试题目\n"
            f"{question}\n"
            f"{option_str}"
        )

        gt = sample["answer"].strip().upper()
        ch = str(sample.get("chapter", "未知"))

        valid_opts = {
            k.strip().upper()
            for k in options.keys()
        } or {"A", "B", "C", "D"}

        verbose = total < DEBUG_SAMPLES

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"【题目 {qnum}】{question}")
            print(f"{'=' * 60}")

        pred, rounds_log, n_rounds, answer_source = rag_answer(
            full_question=full_q,
            raw_question=question,
            options=options,
            verbose=verbose,
            valid_options=valid_opts
        )

        if pred:
            pred = pred.strip().upper()

        is_correct = pred == gt
        error_type = classify_error(
            pred,
            gt,
            n_rounds,
            rounds_log,
            answer_source
        )

        total += 1
        total_rounds += n_rounds
        ch_total[ch] += 1
        error_counts[error_type] += 1

        if is_correct:
            correct += 1
            ch_correct[ch] += 1
        else:
            tqdm.write("\n----------------------")
            tqdm.write(
                f"题号: {qnum}  "
                f"GT: {gt}  "
                f"Pred: {pred}  "
                f"轮数: {n_rounds}  "
                f"来源: {answer_source}  "
                f"错误类型: {error_type}"
            )
            tqdm.write(f"题目: {question[:80]}")

        all_results.append({
            "question_num": qnum,
            "question": question,
            "options": options,
            "gt": gt,
            "pred": pred,
            "is_correct": is_correct,
            "error_type": error_type,
            "answer_source": answer_source,
            "retrieval_rounds": n_rounds,
            "total_retrieved": sum(
                r.get("retrieved_count", 0)
                for r in rounds_log
            ),
            "best_score": max(
                (r.get("best_score", 0.0) for r in rounds_log),
                default=0.0
            ),
            "max_option_coverage": max(
                (r.get("option_coverage", 0) for r in rounds_log),
                default=0
            ),
            "evidence_quality": evidence_quality(rounds_log) if rounds_log else "none",
            "rounds_log": rounds_log,
            "chapter": ch,
        })

        if total % SAVE_EVERY == 0:
            save_checkpoint(all_results, len(dataset))
            tqdm.write(f"[断点] 已保存 {total}/{len(dataset)} 道题")

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
        print("[断点] 评估完成，断点文件已清除")

    avg_rounds = total_rounds / total if total > 0 else 0

    error_analysis = {
        "correct": error_counts["correct"],
        "no_retrieval": error_counts["no_retrieval"],
        "empty_retrieval": error_counts["empty_retrieval"],
        "retrieved_wrong": error_counts["retrieved_wrong"],
        "forced_output_wrong": error_counts["forced_output_wrong"],
        "final_invalid_wrong": error_counts["final_invalid_wrong"],
        "null_output": error_counts["null_output"],
        "no_query_wrong": sum(
            error_counts[f"no_query_round{r}"]
            for r in range(1, 4)
        ),
    }

    source_dist = defaultdict(int)
    for r in all_results:
        source_dist[r["answer_source"]] += 1

    valid_answer_count = sum(
        1
        for r in all_results
        if r.get("pred") in set(map(str.upper, r.get("options", {}).keys()))
    )

    e_output_count = sum(
        1
        for r in all_results
        if r.get("pred") == "E"
    )

    valid_answer_rate = valid_answer_count / total if total > 0 else 0
    e_error_rate = e_output_count / total if total > 0 else 0

    forced_total = source_dist.get("forced", 0)
    forced_wrong = error_counts.get("forced_output_wrong", 0)
    forced_acc = None
    if forced_total > 0:
        forced_acc = round((forced_total - forced_wrong) / forced_total, 4)

    print(f"\n{'=' * 50}")
    print(f"Accuracy = {correct / total:.4f} ({correct}/{total})")
    print(f"平均检索轮数 = {avg_rounds:.2f}")
    print(f"Valid Answer Rate = {valid_answer_rate:.4f}")
    print(f"E-error Rate = {e_error_rate:.4f}")
    print(f"Forced Total = {forced_total}")
    print(f"Forced Accuracy = {forced_acc}")

    print("\n【错误类型分布】")
    for k, v in error_analysis.items():
        print(f"  {k}: {v}")

    print("\n【答案来源分布】")
    for k, v in sorted(source_dist.items()):
        print(f"  {k}: {v}")

    print("\n【按章节统计】")
    for ch in sorted(ch_total.keys()):
        t = ch_total[ch]
        c = ch_correct[ch]
        print(f"  第{ch}章: {c / t:.4f} ({c}/{t})")

    output = {
        # "mode": "finetuned_rag_think_final_strict_v3",
        "mode": "dpo_v1_rag_think_final_strict",
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH,
        "data_path": DATA_PATH,
        "timestamp": timestamp,
        "config": {
            "max_rounds": MAX_ROUNDS,
            "retrieve_top_k_per_query": RETRIEVE_TOP_K_PER_QUERY,
            "max_cards_per_round": MAX_CARDS_PER_ROUND,
            "use_logprob_for_forced": USE_LOGPROB_FOR_FORCED,
            "use_logprob_for_final_fallback": USE_LOGPROB_FOR_FINAL_FALLBACK,
        },
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4),
        "avg_rounds": round(avg_rounds, 2),
        "valid_answer_rate": round(valid_answer_rate, 4),
        "e_error_rate": round(e_error_rate, 4),
        "forced_total": forced_total,
        "forced_accuracy": forced_acc,
        "error_analysis": error_analysis,
        "answer_source_dist": dict(source_dist),
        "chapter_stats": {
            ch: {
                "correct": ch_correct[ch],
                "total": ch_total[ch],
                "accuracy": round(ch_correct[ch] / ch_total[ch], 4)
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