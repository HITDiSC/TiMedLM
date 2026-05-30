# -*- coding: utf-8 -*-
# This file runs five-step MCQ retrieval rollouts for GRPO.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
Old-eval-compatible MCQ rollout sanity for GRPO.

Protocol aligned to the legacy final eval:
- plan/query are generated in one assistant turn.
- retrieval evidence is injected as a user message, not tool role.
- judge uses enable_thinking=False.
- missing query goes to forced answer, matching old eval behavior.
- forced/logprob answers are used for reward only, not as train traces.
"""

import json
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from retrieval import retrieve_with_scores


CONFIG = {
    "model_path": os.environ.get("TIMEDLM_BASE_MODEL", "Qwen/Qwen3-8B"),
    "lora_path": None,
    "data_file": "data/grpo/mcq/mcq_grpo_train_prompts_targeted50.jsonl",
    "output_file": "outputs/grpo/logs/mcq_rollout_sanity_preds.jsonl",
    "report_file": "outputs/grpo/logs/mcq_rollout_sanity_report.json",
    "max_items": 50,
    "num_generations": 4,
    "max_rounds": 3,
    "retrieve_top_k_per_query": 6,
    "max_cards_per_round": 8,
    "plan_query_max_new_tokens": 512,
    "judge_max_new_tokens": 64,
    "final_max_new_tokens": 4,
    "temperature": 0.7,
    "top_p": 0.9,
    "use_logprob_for_forced": True,
    "use_logprob_for_final_fallback": True,
    "suff_threshold": 0.5,
    "suff_need_max": 3,
    "missing_query_penalty": -1.0,
    "plan_as_query_penalty": -0.15,
    "query_retry_penalty": -0.3,
    "seed": 42,
}


SYSTEM_PROMPT = """\
你是一个面向藏医知识问答与考试解题的AI助手。你必须严格遵循以下规则。

【强制检索规则（最高优先级）】
无论题目涉及任何内容，你都必须先通过检索获取藏医知识，再作答。
绝对禁止在未检索的情况下直接输出答案。

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


STOP_WORDS = {
    "以下", "哪种", "哪个", "哪些", "主要", "属于", "不属于", "包括", "不包括",
    "藏医", "认为", "治疗", "常用", "药物", "方剂", "疾病", "患者",
    "中的", "是", "为", "与", "有关", "进行", "选择", "正确", "错误",
    "____", "一种", "适用于", "主要功能",
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSON parse error at line {line_no}: {e}") from e
    return rows


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: List[Dict[str, Any]], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_example(x: Dict[str, Any]) -> Dict[str, Any]:
    if "question" not in x:
        x["question"] = x.get("input") or x.get("query") or ""
    if "gold_answer" not in x:
        x["gold_answer"] = x.get("answer") or x.get("gt") or ""
    if "options" not in x:
        x["options"] = {}
    if "gold_fact_ids" not in x:
        x["gold_fact_ids"] = x.get("gold_facts") or x.get("citations") or []
    return x


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if CONFIG.get("lora_path"):
        model = PeftModel.from_pretrained(model, CONFIG["lora_path"])
    model.eval()
    return model, tokenizer


def apply_chat_template(tokenizer, messages, enable_thinking=True):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@torch.no_grad()
def generate(
    model,
    tokenizer,
    messages,
    stop_at=None,
    max_new_tokens=None,
    enable_thinking=True,
    repetition_penalty=1.1,
    do_sample=True,
):
    model.eval()
    text = apply_chat_template(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": CONFIG["temperature"],
            "top_p": CONFIG["top_p"],
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        })

    outputs = model.generate(**inputs, **gen_kwargs)
    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    response = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()

    if stop_at:
        for tag in stop_at:
            if tag in response:
                response = response[:response.index(tag) + len(tag)]
                break

    return response


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def extract_tag_loose(text: str, tag: str) -> str:
    if not text:
        return ""
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(rf"<{tag}>(.*)", text, re.DOTALL | re.IGNORECASE)
    if m:
        content = m.group(1).strip()
        return re.split(r"</?(plan|query|judge)>", content, flags=re.IGNORECASE)[0].strip()
    m = re.search(rf"{tag}\s*[:：]\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def has_control_tag(text: str) -> bool:
    return any(t in (text or "") for t in ["<plan>", "</plan>", "<query>", "</query>", "<judge>", "</judge>"])


def extract_pred(content: str, valid_options=None):
    valid_options = {x.upper() for x in (valid_options or {"A", "B", "C", "D"})}
    valid_pat = "".join(sorted(valid_options))
    if not content:
        return None
    content = re.sub(r"```json|```", "", str(content)).strip()
    content_upper = content.upper()
    if content_upper in valid_options:
        return content_upper
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            if "answer" in obj:
                matches = re.findall(rf"[{valid_pat}]", str(obj["answer"]).upper())
                if matches:
                    return matches[0]
            for k in obj:
                if str(k).upper() in valid_options:
                    return str(k).upper()
    except Exception:
        pass
    for pat in [
        rf"答案[是为：:]\s*([{valid_pat}])",
        rf"正确答案[是为：:]\s*([{valid_pat}])",
        rf"最终答案[是为：:]\s*([{valid_pat}])",
        rf"应?选\s*([{valid_pat}])",
        rf"故选\s*([{valid_pat}])",
    ]:
        m = re.search(pat, content, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    m = re.search(rf"\b([{valid_pat}])\b", content_upper[-100:])
    if m:
        return m.group(1)
    matches = re.findall(rf"[{valid_pat}]", content_upper[-50:])
    return matches[-1] if matches else None


def get_card_content(card: Dict[str, Any]) -> str:
    evidence = card.get("evidence", {}) or {}
    return card.get("content") or card.get("refined_result") or evidence.get("citation_text") or ""


def format_cards(cards: List[Dict[str, Any]]) -> str:
    rows = []
    for c in cards:
        evidence = c.get("evidence", {}) or {}
        rows.append({
            "card_id": c.get("card_id", ""),
            "title": c.get("title", ""),
            "card_type": c.get("card_type", ""),
            "content": get_card_content(c),
            "citation_text": evidence.get("citation_text", ""),
        })
    return json.dumps(rows, ensure_ascii=False)


def simple_terms_from_text(text: str):
    text = re.sub(r"^\s*[A-Z]\.\s*.*$", "", text or "", flags=re.MULTILINE)
    text = re.sub(r"[，。；、：:？！\?\s]+", "|", text)
    terms = []
    for t in text.split("|"):
        t = t.strip()
        if len(t) >= 2 and t not in STOP_WORDS:
            terms.append(t)
    return list(dict.fromkeys(terms))


def option_terms(options: Dict[str, Any]):
    terms = []
    for v in (options or {}).values():
        value = str(v).strip()
        if value:
            terms.append(value)
            terms.extend([p.strip() for p in re.split(r"[、，,/\s]+", value) if p.strip()])
    return list(dict.fromkeys(terms))


def option_coverage_count(options: Dict[str, Any], cards: List[Dict[str, Any]]) -> int:
    card_text = "\n".join((c.get("title", "") + " " + get_card_content(c)) for c in cards)
    return sum(1 for v in (options or {}).values() if str(v).strip() and str(v).strip() in card_text)


def keyword_hit_in_cards(question: str, options: Dict[str, Any], cards: List[Dict[str, Any]]) -> bool:
    terms = option_terms(options) + simple_terms_from_text(question)
    card_text = "\n".join((c.get("title", "") + " " + get_card_content(c)) for c in cards)
    return any(t and t in card_text for t in terms)


def _meta_score(meta):
    if isinstance(meta, dict):
        return float(meta.get("score", meta.get("dense_raw", 0.0)))
    return float(meta)


def _meta_dense(meta):
    if isinstance(meta, dict):
        return float(meta.get("dense_raw", meta.get("score", 0.0)))
    return float(meta)


def rerank_cards_by_options(question: str, options: Dict[str, Any], scored_cards: List[Tuple[dict, Any]]):
    q_terms = simple_terms_from_text(question)
    o_terms = option_terms(options)
    out = []
    for card, meta in scored_cards:
        fusion = _meta_score(meta)
        dense_raw = _meta_dense(meta)
        bm25_raw = float(meta.get("bm25_raw", 0.0)) if isinstance(meta, dict) else 0.0
        title = card.get("title", "") or ""
        text = title + " " + get_card_content(card)
        bonus = 0.0
        for t in o_terms:
            if t in title:
                bonus += 0.06
            elif t in text:
                bonus += 0.04
        for t in q_terms:
            if t in title:
                bonus += 0.03
            elif t in text:
                bonus += 0.01
        cid = card.get("card_id", "") or ""
        if cid.startswith("diag_manual") and not any(x in question for x in ["诊断", "寒热", "尿诊", "脉诊", "病性", "症状"]):
            bonus -= 0.04
        if card.get("card_type") == "case" and not any(x in question for x in ["患者", "症状", "表现", "诊断", "病例"]):
            bonus -= 0.02
        adjusted = max(0.0, min(1.0, fusion + bonus))
        meta_dict = meta if isinstance(meta, dict) else {"score": fusion, "dense_raw": dense_raw, "bm25_raw": bm25_raw}
        out.append((card, adjusted, fusion, dense_raw, bm25_raw, bonus, meta_dict))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def retrieve_cards_by_query(query_content: str, question: str, options: Dict[str, Any], verbose=False):
    queries = [q.strip() for q in re.split(r"[；;]", str(query_content)) if q.strip()]
    all_scored = []
    for q in queries:
        try:
            all_scored.extend(retrieve_with_scores(q, top_k=CONFIG["retrieve_top_k_per_query"]))
        except Exception as e:
            if verbose:
                print(f"[retrieve failed] query={q} error={e}")

    seen = set()
    uniq = []
    for item in all_scored:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        card, meta = item
        cid = card.get("card_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            uniq.append((card, meta))

    reranked = rerank_cards_by_options(question, options, uniq)
    top_scored = reranked[:CONFIG["max_cards_per_round"]]
    cards = [x[0] for x in top_scored]
    best_score = max((x[1] for x in top_scored), default=0.0)
    best_dense_raw = max((x[3] for x in top_scored), default=0.0)
    best_fusion_score = max((x[2] for x in top_scored), default=0.0)
    cards_with_scores = [(x[0], x[6]) for x in top_scored]
    rq = retrieval_quality(cards_with_scores)
    return reranked, top_scored, cards, best_score, best_dense_raw, best_fusion_score, rq, cards_with_scores


def retrieval_quality(cards_with_scores):
    """
    Local quality helper compatible with both old and new retrieval.py.

    Old retrieval.py returns [(card, fusion_score), ...].
    New retrieval.py returns [(card, meta_dict), ...].
    """
    if not cards_with_scores:
        return "empty"

    scores = []
    top_fact_cnt = 0

    for i, item in enumerate(cards_with_scores[:8]):
        if not isinstance(item, tuple) or len(item) != 2:
            continue

        card, meta = item

        try:
            if isinstance(meta, dict):
                score = float(meta.get("dense_raw", meta.get("score", 0.0)))
            else:
                score = float(meta)
        except Exception:
            continue

        scores.append(score)

        cid = str(card.get("card_id", ""))
        if card.get("card_type") == "fact" or cid.startswith("fact"):
            if i < 5:
                top_fact_cnt += 1

    if not scores:
        return "low"

    best = max(scores)
    avg_top3 = sum(scores[:3]) / max(1, len(scores[:3]))

    # Old fusion scores can be around 1.0+, while new dense_raw is usually lower.
    if best >= 0.75 and top_fact_cnt >= 2:
        return "high"
    if best >= 0.60 and top_fact_cnt >= 2:
        return "mid"
    if avg_top3 >= 0.60 and top_fact_cnt >= 2:
        return "mid"
    return "low"


def build_initial_instruction(full_question: str, question: str, options: Dict[str, Any]) -> str:
    option_text = "\n".join(f"{k}. {v}" for k, v in (options or {}).items())
    return (
        f"{full_question}\n\n"
        "【重要】你必须先检索藏医知识库，不可直接作答。\n"
        "生成 query 时必须满足：\n"
        f"1. 覆盖题干核心概念：{question}\n"
        f"2. 覆盖每个选项关键词：\n{option_text}\n"
        "3. 如果是药物、方剂、性味、剂型、适应症题，必须分别检索每个候选项与题干概念的关系。\n"
        "4. 如果题目含“不包括”“不属于”“错误的是”，必须检索各选项是否属于正确范围。\n"
        "请立即开始：先输出 <plan>检索计划</plan>，再输出 <query>检索词</query>。"
    )


def is_judge_sufficient(judge_content: str, best_score: float, best_dense_raw: float, rq: str, round_num: int, question: str, options: Dict[str, Any], cards: List[Dict[str, Any]]) -> bool:
    text = judge_content or ""
    neg = any(w in text for w in ["不充分", "不足", "无法", "缺乏", "没有找到", "未找到", "继续检索", "尚未"])
    pos = any(w in text for w in ["证据充分", "可以回答", "能够回答", "足以回答", "充分"])
    if pos and not neg:
        return True
    if round_num == 1:
        return False
    if rq in {"high", "mid"} and best_dense_raw >= 0.62 and keyword_hit_in_cards(question, options, cards):
        return True
    if best_score >= 0.96 and option_coverage_count(options, cards) >= 1:
        return True
    return False


def evidence_quality(rounds_log: List[Dict[str, Any]]) -> str:
    best_dense = max((r.get("best_dense_raw", 0.0) for r in rounds_log), default=0.0)
    if best_dense >= 0.70:
        return "high"
    if best_dense >= 0.62:
        return "medium"
    return "low"


def score_candidate_choice(model, tokenizer, messages, choice: str) -> float:
    prefix_text = apply_chat_template(tokenizer, messages, enable_thinking=False)
    full_text = prefix_text + choice
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    prefix_len = prefix_ids.shape[1]
    if full_ids.shape[1] <= prefix_len:
        return -1e9
    with torch.no_grad():
        logits = model(full_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    vals = [log_probs[0, pos - 1, full_ids[0, pos]].item() for pos in range(prefix_len, full_ids.shape[1])]
    return sum(vals) / len(vals) if vals else -1e9


def choose_by_logprob(model, tokenizer, messages, valid_options):
    scores = {}
    for opt in sorted({x.upper() for x in valid_options}):
        try:
            scores[opt] = score_candidate_choice(model, tokenizer, messages, opt)
        except Exception:
            scores[opt] = -1e9
    best = max(scores.items(), key=lambda x: x[1])[0]
    return best, scores


def generate_final_choice(model, tokenizer, messages, valid_options, do_sample=False):
    valid_options = {x.upper() for x in valid_options}
    valid_str = "/".join(sorted(valid_options))
    final_prompt = (
        "现在给出最终答案。\n"
        "请在内部比较各选项与检索证据的一致性，但不要输出分析过程。\n"
        "规则：\n"
        "1. 不要输出 <plan>、<query>、<judge>。\n"
        "2. 不要继续检索。\n"
        "3. 不要解释，不要展示推理。\n"
        f"4. 只能输出一个选项字母，必须是 {valid_str} 中的一个。\n"
        "5. 不要输出标点或其他任何内容。"
    )
    final_messages = messages + [{"role": "user", "content": final_prompt}]
    final = generate(model, tokenizer, final_messages, max_new_tokens=CONFIG["final_max_new_tokens"], enable_thinking=False, repetition_penalty=1.0, do_sample=do_sample)
    pred = extract_pred(final, valid_options)
    if pred and not has_control_tag(final):
        trace = {"kind": "final", "messages": final_messages, "completion": final, "enable_thinking": False}
        return pred, final, "final_generate", trace

    final_messages_retry = final_messages + [
        {"role": "assistant", "content": final},
        {"role": "user", "content": f"你的输出不合法。只输出一个字母，必须是 {valid_str} 中的一个："},
    ]
    final2 = generate(model, tokenizer, final_messages_retry, max_new_tokens=2, enable_thinking=False, repetition_penalty=1.0, do_sample=False)
    pred2 = extract_pred(final2, valid_options)
    if pred2 and not has_control_tag(final2):
        trace = {"kind": "final_retry", "messages": final_messages_retry, "completion": final2, "enable_thinking": False}
        return pred2, final2, "final_retry", trace

    if CONFIG["use_logprob_for_final_fallback"]:
        score_messages = messages + [{"role": "user", "content": f"现在只输出最终答案。只能输出 {valid_str} 中的一个字母，不要解释，不要标点："}]
        pred3, scores = choose_by_logprob(model, tokenizer, score_messages, valid_options)
        return pred3, f"[logprob_final_fallback] {pred3} scores={scores}", "logprob_final_fallback", None
    return pred2, final2, "final_invalid", None


def generate_forced_choice(model, tokenizer, messages, valid_options, quality: str):
    valid_options = {x.upper() for x in valid_options}
    valid_str = "/".join(sorted(valid_options))
    prefix = {
        "high": "已检索到较高相关度的藏医知识证据。",
        "medium": "已检索到部分相关藏医知识证据。",
    }.get(quality, "检索证据不足，但仍需从题目给出的选项中选择最合理答案。")
    forced_messages = messages + [{
        "role": "user",
        "content": (
            f"{prefix}\n现在请根据上方题目、选项和检索证据，选择最可能正确的答案。\n"
            f"只输出一个字母，必须是 {valid_str} 中的一个。\n"
            "不要解释，不要分析，不要输出 <plan>、<query>、<judge>。"
        ),
    }]
    if CONFIG["use_logprob_for_forced"]:
        pred, scores = choose_by_logprob(model, tokenizer, forced_messages, valid_options)
        return pred, f"[logprob_forced] {pred} scores={scores}", "logprob_forced"
    final = generate(model, tokenizer, forced_messages, max_new_tokens=CONFIG["final_max_new_tokens"], enable_thinking=False, repetition_penalty=1.0, do_sample=False)
    return extract_pred(final, valid_options), final, "forced_generate"


def sufficiency_from_ids(retrieved_ids, gold_ids, suff_need_max=3):
    gold = set(gold_ids or [])
    if not gold:
        return 0.0
    need = min(suff_need_max, len(gold))
    return min(1.0, len(set(retrieved_ids) & gold) / need) if need > 0 else 0.0


def compute_step_reward(prev_suff, new_suff, prev_hit, new_hit):
    gain = max(0.0, new_suff - prev_suff)
    new_hits = max(0, new_hit - prev_hit)
    reward = 0.5 * gain + 0.15 * new_hits - 0.03
    if gain <= 0 and new_hits == 0:
        reward -= 0.05
    return reward


def compute_judge_reward(judge_sufficient, suff):
    actual = suff >= CONFIG["suff_threshold"]
    if judge_sufficient and actual:
        return 0.15, "correct_sufficient"
    if (not judge_sufficient) and (not actual):
        return 0.10, "correct_insufficient"
    if judge_sufficient and suff < 0.3:
        return -0.20, "early_stop_severe"
    return -0.05, "wrong"


def compute_answer_reward(pred, gold, suff, num_rounds):
    correct = pred == gold
    if correct:
        reward = 0.4 + 0.8 * suff
        if suff >= CONFIG["suff_threshold"]:
            reward += 0.4
        if 1 <= num_rounds <= 2:
            reward += 0.05
        return reward, "correct"
    reward = -1.2 + (0.1 if suff >= CONFIG["suff_threshold"] else 0.0)
    return reward, "wrong"


def run_one_rollout(model, tokenizer, example: Dict[str, Any], gen_id: int, verbose=False):
    question = example["question"]
    options = example.get("options", {}) or {}
    gold_answer = str(example["gold_answer"]).strip().upper()
    gold_fact_ids = set(example.get("gold_fact_ids", []))
    valid_options = {str(k).strip().upper() for k in options} or {"A", "B", "C", "D"}
    option_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    full_q = f"请只输出正确答案的选项字母。\n### 考试题目\n{question}\n{option_str}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_instruction(full_q, question, options)},
    ]
    rounds_log, steps, train_traces = [], [], []
    retrieved_ids = set()
    total_reward = 0.0
    pred = None
    final_text = ""
    answer_source = "forced"
    stop_reason = "unknown"

    for round_num in range(1, CONFIG["max_rounds"] + 1):
        pq_messages = messages if round_num == 1 else messages + [{
            "role": "user",
            "content": "证据仍不足。请继续生成新的 <plan> 和 <query>，不要直接输出最终答案。",
        }]
        pq_messages_for_loss = [dict(m) for m in pq_messages]
        pq_resp = generate(model, tokenizer, pq_messages, stop_at=["</query>"], max_new_tokens=CONFIG["plan_query_max_new_tokens"], enable_thinking=True, repetition_penalty=1.05, do_sample=True)
        plan = extract_tag_loose(pq_resp, "plan")
        query = extract_tag_loose(pq_resp, "query")
        used_plan_as_query = False

        # Legacy final eval used the plan as a conservative query fallback when
        # the model produced <plan> but omitted <query>. Keep that behavior here
        # so rollout/train match the old evaluator more closely.
        if not query and plan:
            query = plan
            used_plan_as_query = True

        if round_num > 1:
            messages.append(pq_messages[-1])
        messages.append({"role": "assistant", "content": pq_resp})

        plan_reward = 0.0
        if not plan:
            plan_reward -= 0.2
        if used_plan_as_query:
            plan_reward += CONFIG["plan_as_query_penalty"]
        elif not query:
            plan_reward += CONFIG["missing_query_penalty"]
        total_reward += plan_reward
        steps.append({
            "type": "plan_query",
            "round": round_num,
            "completion": pq_resp,
            "plan_valid": bool(plan),
            "query_valid": bool(query),
            "used_plan_as_query": used_plan_as_query,
            "reward": plan_reward,
        })

        # Only train successful plan/query generations. If a trajectory later gets
        # a positive reward because forced guessing is correct, this prevents
        # reinforcing malformed outputs that omitted <query>.
        if pq_resp and query:
            train_traces.append({"kind": "plan_query", "messages": pq_messages_for_loss, "completion": pq_resp, "enable_thinking": True})

        if not query:
            retry_instruction = (
                "你的上一条回复缺少 <query>。请只补充输出 "
                "<query>检索词</query>，不要输出答案，不要解释。"
            )
            retry_messages = messages + [{"role": "user", "content": retry_instruction}]
            retry_resp = generate(
                model,
                tokenizer,
                retry_messages,
                stop_at=["</query>"],
                max_new_tokens=128,
                enable_thinking=False,
                repetition_penalty=1.0,
                do_sample=True,
            )
            retry_query = extract_tag_loose(retry_resp, "query")

            steps.append({
                "type": "query_retry",
                "round": round_num,
                "completion": retry_resp,
                "query_valid": bool(retry_query),
                "reward": CONFIG["query_retry_penalty"] if retry_query else 0.0,
            })

            if retry_query:
                query = retry_query.strip()
                total_reward += CONFIG["query_retry_penalty"]
                messages.append({"role": "user", "content": retry_instruction})
                messages.append({"role": "assistant", "content": retry_resp})
            else:
                answer_source = f"missing_query_round{round_num}_forced"
                stop_reason = "missing_query"
                break

        prev_suff = sufficiency_from_ids(retrieved_ids, gold_fact_ids, CONFIG["suff_need_max"])
        prev_hit = len(retrieved_ids & gold_fact_ids)
        reranked, top_scored, cards, best_score, best_dense, best_fusion, rq, cards_with_scores = retrieve_cards_by_query(query, question, options, verbose=verbose)
        for c in cards:
            cid = c.get("card_id")
            if cid:
                retrieved_ids.add(cid)
        new_suff = sufficiency_from_ids(retrieved_ids, gold_fact_ids, CONFIG["suff_need_max"])
        new_hit = len(retrieved_ids & gold_fact_ids)
        query_reward = compute_step_reward(prev_suff, new_suff, prev_hit, new_hit)
        total_reward += query_reward

        evidence_json = format_cards(cards)
        retrieval_message = (
            f"【检索结果】\n{evidence_json}\n\n"
            "请只判断这些证据是否足以回答题目。\n"
            "只能输出以下两种格式之一：\n"
            "<judge>证据充分</judge>\n"
            "<judge>证据不足</judge>\n"
            "不要解释，不要分析，不要输出 <think>。"
        )
        judge_messages = messages + [{"role": "user", "content": retrieval_message}]
        judge_messages_for_loss = [dict(m) for m in judge_messages]
        judge_resp = generate(model, tokenizer, judge_messages, stop_at=["</judge>"], max_new_tokens=CONFIG["judge_max_new_tokens"], enable_thinking=False, repetition_penalty=1.0, do_sample=True)
        judge_content = extract_tag_loose(judge_resp, "judge") or judge_resp
        sufficient = is_judge_sufficient(judge_content, best_score, best_dense, rq, round_num, question, options, cards)
        judge_reward, judge_quality = compute_judge_reward(sufficient, new_suff)
        total_reward += judge_reward

        round_info = {
            "round": round_num,
            "query": query,
            "retrieved_count": len(cards),
            "retrieved_cards": [{"card_id": c.get("card_id"), "title": c.get("title"), "card_type": c.get("card_type"), "content": get_card_content(c)[:300]} for c in cards],
            "best_score": best_score,
            "best_dense_raw": best_dense,
            "best_fusion_score": best_fusion,
            "retrieval_quality": rq,
            "option_coverage": option_coverage_count(options, cards),
            "judge": judge_content,
            "judge_sufficient": sufficient,
            "sufficiency_after": new_suff,
            "query_reward": query_reward,
            "judge_reward": judge_reward,
        }
        rounds_log.append(round_info)
        steps.append({"type": "query_retrieve", "round": round_num, "query": query, "reward": query_reward, **round_info})
        steps.append({"type": "judge", "round": round_num, "completion": judge_resp, "judge_quality": judge_quality, "reward": judge_reward})
        if judge_resp:
            train_traces.append({"kind": "judge", "messages": judge_messages_for_loss, "completion": judge_resp, "enable_thinking": False})

        messages = judge_messages + [{"role": "assistant", "content": judge_resp}]

        if sufficient:
            pred, final_text, answer_source, final_trace = generate_final_choice(model, tokenizer, messages, valid_options, do_sample=False)
            if final_trace is not None:
                train_traces.append(final_trace)
            stop_reason = "answered"
            break

    if pred is None:
        eq = evidence_quality(rounds_log)
        pred, final_text, forced_source = generate_forced_choice(model, tokenizer, messages, valid_options, eq)
        if not str(answer_source).startswith("missing_query"):
            answer_source = forced_source
        stop_reason = stop_reason if stop_reason != "unknown" else "forced"

    final_suff = sufficiency_from_ids(retrieved_ids, gold_fact_ids, CONFIG["suff_need_max"])
    answer_reward, answer_quality = compute_answer_reward(pred, gold_answer, final_suff, len(rounds_log))
    total_reward += answer_reward
    steps.append({"type": "answer", "completion": final_text, "pred_answer": pred, "gold_answer": gold_answer, "answer_quality": answer_quality, "reward": answer_reward})

    return {
        "question_id": example.get("question_id") or example.get("id") or example.get("qid"),
        "generation_id": gen_id,
        "question": question,
        "options": options,
        "gold_answer": gold_answer,
        "gold_fact_ids": list(gold_fact_ids),
        "steps": steps,
        "rounds_log": rounds_log,
        "final": {
            "pred_answer": pred,
            "correct": pred == gold_answer,
            "answer_source": answer_source,
            "num_rounds": len(rounds_log),
            "retrieved_fact_ids": list(retrieved_ids),
            "gold_hit_count": len(retrieved_ids & gold_fact_ids),
            "final_sufficiency": final_suff,
            "stop_reason": stop_reason,
            "total_reward": total_reward,
        },
        "train_traces": train_traces,
    }


def main():
    random.seed(CONFIG["seed"])
    data = [normalize_example(x) for x in load_jsonl(CONFIG["data_file"])]
    random.shuffle(data)
    if CONFIG["max_items"] and CONFIG["max_items"] > 0:
        data = data[:CONFIG["max_items"]]

    model, tokenizer = load_model_and_tokenizer()
    all_trajs = []
    stop_counter, source_counter, answer_counter = Counter(), Counter(), Counter()
    correct = 0
    group_ranges = []

    for ex in tqdm(data, desc="old-eval rollout sanity"):
        rewards = []
        for g in range(CONFIG["num_generations"]):
            traj = run_one_rollout(model, tokenizer, ex, g, verbose=False)
            all_trajs.append(traj)
            rewards.append(traj["final"]["total_reward"])
            correct += int(bool(traj["final"].get("correct")))
            stop_counter[traj["final"].get("stop_reason")] += 1
            source_counter[traj["final"].get("answer_source")] += 1
            answer_counter[traj["final"].get("pred_answer")] += 1
        group_ranges.append(max(rewards) - min(rewards))

    total = len(all_trajs)
    report = {
        "config": CONFIG,
        "question_count": len(data),
        "trajectory_count": total,
        "accuracy_on_trajectories": correct / total if total else 0.0,
        "avg_total_reward": sum(t["final"]["total_reward"] for t in all_trajs) / total if total else 0.0,
        "avg_group_reward_range": sum(group_ranges) / len(group_ranges) if group_ranges else 0.0,
        "distinguishable_group_rate": sum(1 for x in group_ranges if x > 1e-6) / len(group_ranges) if group_ranges else 0.0,
        "stop_reason_distribution": dict(stop_counter),
        "answer_source_distribution": dict(source_counter),
        "answer_distribution": dict(answer_counter),
    }
    save_jsonl(all_trajs, CONFIG["output_file"])
    save_json(report, CONFIG["report_file"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
