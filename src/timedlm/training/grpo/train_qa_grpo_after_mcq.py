# -*- coding: utf-8 -*-
"""
QA-GRPO training after MCQ-GRPO.

Goal:
- Keep the better retrieval coverage learned by MCQ-GRPO.
- Recover/improve QA answer quality, reference coverage, and concise evidence use.

Default start point:
SFT-merged base + MCQ-GRPO targeted50 LoRA.
"""

import json
import math
import os
import random
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from retrieval import retrieve_with_scores


CONFIG = {
    "model_path": os.environ.get("TIMEDLM_BASE_MODEL", "Qwen/Qwen3-8B"),
    "init_lora_path": os.environ.get("TIMEDLM_MCQ_GRPO_LORA", "outputs/grpo/qwen3-8b-mcq-grpo-lora"),
    "train_file": "data/grpo/qa/qa_grpo_train_targeted50.jsonl",
    "output_dir": "outputs/grpo/qwen3-8b-mcq-qa-grpo-lora",
    "rollout_log_file": "outputs/grpo/logs/qa_grpo_rollout_log.jsonl",
    "train_report_file": "outputs/grpo/logs/qa_grpo_train_report.json",
    "max_train_items": 50,
    "num_generations": 4,
    "num_epochs": 1,
    "learning_rate": 1e-7,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 1,
    "max_seq_length": 3072,
    "normalize_logprob_by_len": True,
    "advantage_eps": 1e-6,
    "clip_advantage": 5.0,
    "max_rounds": 3,
    "retrieve_top_k_per_query": 4,
    "max_cards_per_round": 10,
    "plan_max_new_tokens": 512,
    "query_max_new_tokens": 768,
    "judge_max_new_tokens": 256,
    "answer_max_new_tokens": 900,
    "temperature": 0.7,
    "top_p": 0.9,
    "missing_query_penalty": -0.8,
    "forced_penalty": -0.25,
    "premature_stop_penalty": -0.4,
    "too_long_penalty": -0.4,
    "too_short_penalty": -0.5,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "seed": 42,
    "log_every_groups": 5,
    "save_every_groups": 50,
    "verbose_rollout": False,
}


SYSTEM_PROMPT = """\
你是一个面向藏医知识问答的检索增强助手。你必须先检索本地藏医知识库，再回答问题。

流程：
1. 输出 <plan>...</plan>，说明需要检索哪些知识。
2. 输出 <query>...</query>，多个检索词用分号分隔。
3. 收到检索结果后，输出 <judge>...</judge> 判断证据是否足够。
4. 如果证据不足，继续下一轮检索；如果证据足够，给出最终回答。

要求：
- query 必须覆盖问题中的核心疾病、症状、药物、方剂、诊断要点和治疗要点。
- 如果问题有多个子问题，必须分别检索，不要只检索一个笼统问题。
- 最终回答必须完整覆盖用户问题的每个方面。
- 最终回答需要引用检索到的真实 card_id，不得编造来源。
"""


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_dtype():
    if CONFIG["bf16"]:
        return torch.bfloat16
    if CONFIG["fp16"]:
        return torch.float16
    return torch.float32


def normalize_example(x: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(x)
    out["question"] = x.get("question") or x.get("input") or x.get("query") or ""
    out["reference_answer"] = x.get("reference_answer") or x.get("reference") or x.get("answer") or ""
    out["question_type"] = x.get("question_type") or x.get("type") or "unknown"
    out["gold_fact_ids"] = list(dict.fromkeys(x.get("gold_fact_ids") or x.get("gold_evidence_ids") or []))
    out["citation_fact_ids"] = list(dict.fromkeys(x.get("citation_fact_ids") or out["gold_fact_ids"]))
    out["silver_fact_ids"] = list(dict.fromkeys(x.get("silver_fact_ids") or []))
    out["source_queries"] = x.get("source_queries") or []
    out["source_tool_cards"] = x.get("source_tool_cards") or []
    out["question_id"] = x.get("question_id") or x.get("id") or str(abs(hash(out["question"])))
    return out


def print_trainable_parameters(model):
    trainable, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"trainable params: {trainable:,} / {total:,} = {100 * trainable / total:.4f}%")


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"],
        trust_remote_code=True,
        torch_dtype=get_dtype(),
        device_map="auto",
    )
    if CONFIG["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    init_lora_path = CONFIG.get("init_lora_path")
    if init_lora_path:
        print("Loading init LoRA:", init_lora_path)
        model = PeftModel.from_pretrained(model, init_lora_path, is_trainable=True)
    else:
        print("Creating new LoRA on SFT merged base.")
        lora_config = LoraConfig(
            r=CONFIG["lora_r"],
            lora_alpha=CONFIG["lora_alpha"],
            target_modules=CONFIG["target_modules"],
            lora_dropout=CONFIG["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    print_trainable_parameters(model)
    return model, tokenizer


def build_chat_text(tokenizer, messages, enable_thinking=True):
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


@torch.no_grad()
def generate(model, tokenizer, messages, stop_at=None, max_new_tokens=512, enable_thinking=True, repetition_penalty=1.05):
    model.eval()
    text = build_chat_text(tokenizer, messages, enable_thinking=enable_thinking)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=CONFIG["temperature"],
        top_p=CONFIG["top_p"],
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
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
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def has_control_tag(text: str) -> bool:
    return any(t in (text or "") for t in ["<plan>", "</plan>", "<query>", "</query>", "<judge>", "</judge>"])


def get_card_content(card: dict) -> str:
    evidence = card.get("evidence", {}) or {}
    return card.get("content") or card.get("refined_result") or evidence.get("citation_text") or ""


def get_card_id(card: dict) -> str:
    return str(card.get("card_id") or card.get("id") or "")


def score_from_meta(meta: Any) -> float:
    if isinstance(meta, dict):
        for k in ["score", "fusion_score", "dense_raw", "bm25_raw"]:
            if k in meta:
                return float(meta[k])
        return 0.0
    return float(meta)


def format_cards(cards: List[dict]) -> str:
    rows = []
    for c in cards:
        evidence = c.get("evidence", {}) or {}
        rows.append({
            "card_id": get_card_id(c),
            "title": c.get("title", ""),
            "card_type": c.get("card_type", ""),
            "content": get_card_content(c),
            "citation_text": evidence.get("citation_text", ""),
        })
    return json.dumps(rows, ensure_ascii=False)


def fallback_query(example: Dict[str, Any]) -> str:
    source_queries = example.get("source_queries") or []
    if source_queries:
        return "；".join(str(q).strip() for q in source_queries[:2] if str(q).strip())
    return str(example.get("question", ""))[:120]


def split_queries(query_content: str) -> List[str]:
    qs = [q.strip() for q in re.split(r"[；;]", query_content or "") if q.strip()]
    return qs[:8]


def retrieve_cards(query_content: str) -> Tuple[List[dict], List[Dict[str, Any]], float]:
    all_rows = []
    for q in split_queries(query_content):
        try:
            all_rows.extend(retrieve_with_scores(q, top_k=CONFIG["retrieve_top_k_per_query"]))
        except Exception as e:
            print(f"[retrieve failed] {q}: {e}")

    seen = set()
    uniq = []
    for item in all_rows:
        if not isinstance(item, tuple) or len(item) != 2:
            continue
        card, meta = item
        cid = get_card_id(card)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        score = score_from_meta(meta)
        uniq.append((card, score))

    uniq.sort(key=lambda x: x[1], reverse=True)
    top = uniq[:CONFIG["max_cards_per_round"]]
    cards = [c for c, _ in top]
    log_rows = [
        {
            "card_id": get_card_id(c),
            "title": c.get("title", ""),
            "card_type": c.get("card_type", ""),
            "score": round(float(s), 4),
            "content": get_card_content(c)[:300],
        }
        for c, s in top
    ]
    best_score = max((s for _, s in top), default=0.0)
    return cards, log_rows, float(best_score)


def citation_ids(text: str) -> List[str]:
    if not text:
        return []
    pat = r"(?:fact|case)_[A-Za-z0-9]+(?:_[0-9]+)*|diag_manual_[0-9]+"
    return list(dict.fromkeys(re.findall(pat, text)))


def clean_metric_text(text: str) -> str:
    text = re.sub(r"(?:引用来源|来源|References?)[:：].*$", "", text or "", flags=re.S | re.I)
    text = re.sub(r"(?:fact|case)_[A-Za-z0-9]+(?:_[0-9]+)*|diag_manual_[0-9]+", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", "", text)


def char_tokens(text: str) -> List[str]:
    return [c for c in clean_metric_text(text) if not c.isspace()]


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = char_tokens(pred), char_tokens(ref)
    if not p or not r:
        return 0.0
    lcs = lcs_len(p, r)
    prec = lcs / len(p)
    rec = lcs / len(r)
    return 2 * prec * rec / (prec + rec + 1e-9)


def keyword_recall(pred: str, ref: str) -> float:
    ref_clean = clean_metric_text(ref)
    pred_clean = clean_metric_text(pred)
    terms = set()
    for n in [2, 3, 4]:
        for i in range(max(0, len(ref_clean) - n + 1)):
            g = ref_clean[i:i + n]
            if re.search(r"[\u4e00-\u9fff]", g):
                terms.add(g)
    if not terms:
        return 0.0
    # sample a stable subset to keep this cheap
    terms = sorted(terms)[:300]
    return sum(1 for t in terms if t in pred_clean) / len(terms)


def answer_length_reward(pred: str, ref: str, qtype: str) -> Tuple[float, str]:
    p_len = len(clean_metric_text(pred))
    r_len = max(1, len(clean_metric_text(ref)))
    if p_len < 60:
        return CONFIG["too_short_penalty"], "too_short"
    ratio = p_len / r_len
    upper = 4.0 if qtype in {"diagnostic", "case"} else 3.0
    if ratio > upper:
        return CONFIG["too_long_penalty"], "too_long"
    if 0.45 <= ratio <= upper:
        return 0.25, "ok"
    return 0.0, "neutral"


def build_initial_instruction(example: Dict[str, Any]) -> str:
    return (
        f"问题：\n{example['question']}\n\n"
        f"问题类型：{example.get('question_type', 'unknown')}\n\n"
        "请先检索本地藏医知识库，再回答。生成 query 时必须覆盖每个子问题，"
        "尤其是症状诊断、药物功效、治疗方法、禁忌和理论依据。"
        "请立即开始：先输出 <plan>检索计划</plan>，再输出 <query>检索词</query>。"
    )


def is_judge_sufficient(judge: str, best_score: float, round_num: int, retrieved_recall: float) -> bool:
    positive = any(w in (judge or "") for w in ["充分", "足够", "可以回答", "能够回答"])
    negative = any(w in (judge or "") for w in ["不充分", "不足", "继续检索", "缺乏", "无法"])
    if positive and not negative:
        return True
    # During training, allow stopping when retrieval has already covered some gold evidence.
    if round_num >= 2 and retrieved_recall >= 0.35:
        return True
    if round_num >= 2 and best_score >= 0.95:
        return True
    return False


def final_prompt(allowed_ids: Sequence[str], forced: bool = False) -> str:
    allowed = "；".join(allowed_ids)
    prefix = "已经完成多轮检索。" if forced else "证据已经较充分。"
    return (
        f"{prefix}现在请基于检索证据回答问题。\n"
        "要求：\n"
        "1. 回答要准确、完整，但不要无关扩写。\n"
        "2. 必须覆盖问题中的每个子问题。\n"
        "3. 优先引用与答案直接相关的 fact/case 证据。\n"
        "4. 不要输出 <plan>、<query>、<judge>。\n"
        "5. 不要编造 card_id，只能引用下面列表中出现过的 card_id：\n"
        f"{allowed}\n"
        "6. 末尾写：引用来源：[card_id1, card_id2]\n"
    )


def compute_reward(example: Dict[str, Any], answer: str, rounds_log: List[Dict[str, Any]], answer_source: str, missing_query_count: int) -> Tuple[float, Dict[str, Any]]:
    ref = example.get("reference_answer", "")
    gold_ids = set(example.get("gold_fact_ids") or [])
    citation_gold_ids = set(example.get("citation_fact_ids") or []) or gold_ids

    retrieved_ids = []
    for r in rounds_log:
        for c in r.get("retrieved_cards", []):
            cid = c.get("card_id")
            if cid:
                retrieved_ids.append(cid)
    retrieved_set = set(retrieved_ids)

    cited = citation_ids(answer)
    cited_set = set(cited)

    rouge = rouge_l(answer, ref)
    key_rec = keyword_recall(answer, ref)
    retrieved_recall = len(retrieved_set & gold_ids) / len(gold_ids) if gold_ids else 0.0
    gold_citation_recall = len(cited_set & citation_gold_ids) / len(citation_gold_ids) if citation_gold_ids else 0.0
    citation_coverage = 1.0 if cited else 0.0
    hallucinated = [cid for cid in cited if cid not in retrieved_set]
    citation_validity = 1.0 if cited and not hallucinated else 0.0

    len_reward, len_label = answer_length_reward(answer, ref, example.get("question_type", ""))

    reward = 0.0
    reward += 1.8 * rouge
    reward += 1.0 * key_rec
    reward += 2.3 * gold_citation_recall
    reward += 1.1 * retrieved_recall
    reward += 0.4 * citation_validity
    reward += 0.2 * citation_coverage
    reward += len_reward

    if has_control_tag(answer):
        reward -= 1.0
    if clean_metric_text(answer).upper() in {"A", "B", "C", "D"}:
        reward -= 2.0
    if answer_source == "forced":
        reward += CONFIG["forced_penalty"]
    if missing_query_count:
        reward += CONFIG["missing_query_penalty"] * missing_query_count
    if len(rounds_log) == 1 and retrieved_recall < 0.2:
        reward += CONFIG["premature_stop_penalty"]

    metrics = {
        "rouge_l": round(rouge, 4),
        "keyword_recall": round(key_rec, 4),
        "retrieved_recall": round(retrieved_recall, 4),
        "gold_citation_recall": round(gold_citation_recall, 4),
        "citation_validity": citation_validity,
        "citation_coverage": citation_coverage,
        "hallucinated_citations": hallucinated,
        "length_label": len_label,
        "answer_len": len(clean_metric_text(answer)),
        "ref_len": len(clean_metric_text(ref)),
        "total_reward": round(float(reward), 4),
    }
    return float(reward), metrics


def run_one_rollout(model, tokenizer, example: Dict[str, Any], gen_id: int, verbose=False) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_initial_instruction(example)},
    ]
    train_traces = []
    rounds_log = []
    missing_query_count = 0
    answer_source = "forced"

    for round_num in range(1, CONFIG["max_rounds"] + 1):
        round_info = {
            "round": round_num,
            "query": "",
            "retrieved_cards": [],
            "retrieved_count": 0,
            "best_score": 0.0,
            "judge": "",
            "judge_sufficient": False,
        }

        plan_messages = list(messages)
        plan_resp = generate(model, tokenizer, plan_messages, stop_at=["</plan>"], max_new_tokens=CONFIG["plan_max_new_tokens"], enable_thinking=True)
        messages.append({"role": "assistant", "content": plan_resp})

        query_messages = list(messages)
        query_resp = generate(model, tokenizer, query_messages, stop_at=["</query>"], max_new_tokens=CONFIG["query_max_new_tokens"], enable_thinking=True)

        query = extract_tag(query_resp, "query")
        if query:
            messages[-1]["content"] = plan_resp + "\n" + query_resp[query_resp.index("<query>"):]
            train_traces.append({
                "kind": "plan",
                "messages": plan_messages,
                "completion": plan_resp,
                "enable_thinking": True,
            })
            train_traces.append({
                "kind": "query",
                "messages": query_messages,
                "completion": query_resp,
                "enable_thinking": True,
            })
        else:
            missing_query_count += 1
            query = fallback_query(example)
            messages.append({"role": "assistant", "content": query_resp})
            messages.append({"role": "user", "content": f"请先检索以下关键词：<query>{query}</query>"})

        round_info["query"] = query

        cards, card_log, best_score = retrieve_cards(query)
        round_info["retrieved_cards"] = card_log
        round_info["retrieved_count"] = len(cards)
        round_info["best_score"] = round(best_score, 4)

        messages.append({"role": "tool", "content": format_cards(cards)})

        judge_messages = list(messages)
        judge_resp = generate(model, tokenizer, judge_messages, stop_at=["</judge>"], max_new_tokens=CONFIG["judge_max_new_tokens"], enable_thinking=True)
        judge = extract_tag(judge_resp, "judge")
        current_retrieved = {c["card_id"] for r in rounds_log + [round_info] for c in r.get("retrieved_cards", [])}
        gold_ids = set(example.get("gold_fact_ids") or [])
        retrieved_recall = len(current_retrieved & gold_ids) / len(gold_ids) if gold_ids else 0.0
        sufficient = is_judge_sufficient(judge, best_score, round_num, retrieved_recall)

        round_info["judge"] = judge
        round_info["judge_sufficient"] = sufficient
        messages.append({"role": "assistant", "content": judge_resp})
        rounds_log.append(round_info)

        if judge_resp:
            train_traces.append({
                "kind": "judge",
                "messages": judge_messages,
                "completion": judge_resp,
                "enable_thinking": True,
            })

        if verbose:
            print(f"[round {round_num}] query={query[:120]} best={best_score:.3f} recall={retrieved_recall:.3f} suff={sufficient}")

        if sufficient:
            answer_source = f"sufficient_round{round_num}"
            f_messages = messages + [{"role": "user", "content": final_prompt(list(current_retrieved), forced=False)}]
            answer = generate(model, tokenizer, f_messages, max_new_tokens=CONFIG["answer_max_new_tokens"], enable_thinking=False, repetition_penalty=1.05)
            train_traces.append({
                "kind": "final",
                "messages": f_messages,
                "completion": answer,
                "enable_thinking": False,
            })
            reward, metrics = compute_reward(example, answer, rounds_log, answer_source, missing_query_count)
            return {
                "question_id": example.get("question_id"),
                "gen_id": gen_id,
                "question": example.get("question"),
                "answer": answer,
                "rounds_log": rounds_log,
                "train_traces": train_traces,
                "final": {
                    **metrics,
                    "answer_source": answer_source,
                    "stop_reason": "answered",
                    "total_reward": reward,
                    "retrieval_rounds": len(rounds_log),
                    "missing_query_count": missing_query_count,
                },
            }

        if round_num < CONFIG["max_rounds"]:
            messages.append({
                "role": "user",
                "content": "上一轮证据仍不完整。下一轮 query 必须补充尚未覆盖的子问题，不要重复已经检索到的内容。",
            })

    all_retrieved = {c["card_id"] for r in rounds_log for c in r.get("retrieved_cards", [])}
    f_messages = messages + [{"role": "user", "content": final_prompt(list(all_retrieved), forced=True)}]
    answer = generate(model, tokenizer, f_messages, max_new_tokens=CONFIG["answer_max_new_tokens"], enable_thinking=False, repetition_penalty=1.05)
    train_traces.append({
        "kind": "final",
        "messages": f_messages,
        "completion": answer,
        "enable_thinking": False,
    })
    reward, metrics = compute_reward(example, answer, rounds_log, answer_source, missing_query_count)
    return {
        "question_id": example.get("question_id"),
        "gen_id": gen_id,
        "question": example.get("question"),
        "answer": answer,
        "rounds_log": rounds_log,
        "train_traces": train_traces,
        "final": {
            **metrics,
            "answer_source": answer_source,
            "stop_reason": "forced",
            "total_reward": reward,
            "retrieval_rounds": len(rounds_log),
            "missing_query_count": missing_query_count,
        },
    }


def sequence_logprob(model, tokenizer, messages, completion: str, enable_thinking: bool):
    if not completion:
        return None
    prefix_text = build_chat_text(tokenizer, messages, enable_thinking=enable_thinking)
    prefix_ids = tokenizer(prefix_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    completion_ids = tokenizer(completion, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if completion_ids.numel() == 0:
        return None

    max_len = CONFIG["max_seq_length"]
    comp_len = completion_ids.numel()
    if comp_len >= max_len:
        completion_ids = completion_ids[-max_len + 1:]
        comp_len = completion_ids.numel()
        prefix_ids = prefix_ids[-1:]
    else:
        prefix_ids = prefix_ids[-(max_len - comp_len):]

    input_ids = torch.cat([prefix_ids, completion_ids], dim=0).unsqueeze(0).to(model.device)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    prefix_len = prefix_ids.numel()

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probs = torch.log_softmax(outputs.logits, dim=-1)

    token_logps = []
    for pos in range(prefix_len, input_ids.shape[1]):
        token_logps.append(log_probs[0, pos - 1, input_ids[0, pos]])
    if not token_logps:
        return None
    token_logps = torch.stack(token_logps)
    return token_logps.mean() if CONFIG["normalize_logprob_by_len"] else token_logps.sum()


def compute_group_loss(model, tokenizer, trajectories: List[Dict[str, Any]]):
    rewards = torch.tensor([t["final"]["total_reward"] for t in trajectories], dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std(unbiased=False)
    advantages = torch.clamp((rewards - mean) / (std + CONFIG["advantage_eps"]), -CONFIG["clip_advantage"], CONFIG["clip_advantage"])

    losses = []
    train_trace_count = 0
    allowed = {"query", "judge", "final"}
    model.train()

    for traj, adv in zip(trajectories, advantages):
        if abs(float(adv.item())) < 1e-8:
            continue
        traces = [t for t in traj.get("train_traces", []) if t.get("kind") in allowed]
        # Keep training cheap and avoid overfitting one long answer trace.
        traces = traces[:2] + traces[-1:]
        for trace in traces:
            logp = sequence_logprob(
                model=model,
                tokenizer=tokenizer,
                messages=trace["messages"],
                completion=trace["completion"],
                enable_thinking=trace["enable_thinking"],
            )
            if logp is None:
                continue
            losses.append(-adv.to(logp.device) * logp)
            train_trace_count += 1

    info = {
        "reward_mean": float(mean.item()),
        "reward_std": float(std.item()),
        "train_trace_count": train_trace_count,
        "advantages": [float(x) for x in advantages.tolist()],
    }
    if not losses:
        return None, info
    return torch.stack(losses).mean(), info


def build_optimizer_and_scheduler(model, total_steps: int):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])
    warmup_steps = max(1, int(total_steps * CONFIG["warmup_ratio"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )
    return optimizer, scheduler


def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    if os.path.exists(CONFIG["rollout_log_file"]):
        os.remove(CONFIG["rollout_log_file"])

    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    data = [normalize_example(x) for x in load_jsonl(CONFIG["train_file"])]
    random.shuffle(data)
    if CONFIG["max_train_items"]:
        data = data[:CONFIG["max_train_items"]]
    print(f"Loaded QA train prompts: {len(data)}")

    model, tokenizer = load_model_and_tokenizer()
    total_groups = len(data) * CONFIG["num_epochs"]
    optim_steps = math.ceil(total_groups / CONFIG["gradient_accumulation_steps"])
    optimizer, scheduler = build_optimizer_and_scheduler(model, optim_steps)

    global_group_step = 0
    global_optim_step = 0
    metric_counter = Counter()
    reward_means, reward_stds, group_ranges, trace_counts = [], [], [], []
    avg_rouge, avg_gcr, avg_ret_recall = [], [], []

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(CONFIG["num_epochs"]):
        random.shuffle(data)
        for ex in tqdm(data, desc=f"epoch {epoch + 1}"):
            global_group_step += 1
            trajectories = []
            group_rewards = []

            for g in range(CONFIG["num_generations"]):
                traj = run_one_rollout(model, tokenizer, ex, gen_id=g, verbose=CONFIG["verbose_rollout"])
                trajectories.append(traj)
                append_jsonl(traj, CONFIG["rollout_log_file"])
                group_rewards.append(traj["final"]["total_reward"])
                metric_counter[f"source/{traj['final'].get('answer_source')}"] += 1
                metric_counter[f"stop/{traj['final'].get('stop_reason')}"] += 1
                metric_counter[f"length/{traj['final'].get('length_label')}"] += 1

            loss, loss_info = compute_group_loss(model, tokenizer, trajectories)
            if loss is not None:
                (loss / CONFIG["gradient_accumulation_steps"]).backward()
                trace_counts.append(loss_info["train_trace_count"])

            reward_means.append(loss_info["reward_mean"])
            reward_stds.append(loss_info["reward_std"])
            group_ranges.append(max(group_rewards) - min(group_rewards))
            avg_rouge.append(sum(t["final"]["rouge_l"] for t in trajectories) / len(trajectories))
            avg_gcr.append(sum(t["final"]["gold_citation_recall"] for t in trajectories) / len(trajectories))
            avg_ret_recall.append(sum(t["final"]["retrieved_recall"] for t in trajectories) / len(trajectories))

            if global_group_step % CONFIG["gradient_accumulation_steps"] == 0:
                if loss is not None:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CONFIG["max_grad_norm"])
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_optim_step += 1
                torch.cuda.empty_cache()

            if global_group_step % CONFIG["log_every_groups"] == 0:
                n = CONFIG["log_every_groups"]
                msg = {
                    "group_step": global_group_step,
                    "optim_step": global_optim_step,
                    "recent_reward_mean": sum(reward_means[-n:]) / max(1, len(reward_means[-n:])),
                    "recent_reward_std": sum(reward_stds[-n:]) / max(1, len(reward_stds[-n:])),
                    "recent_group_range": sum(group_ranges[-n:]) / max(1, len(group_ranges[-n:])),
                    "recent_rouge_l": sum(avg_rouge[-n:]) / max(1, len(avg_rouge[-n:])),
                    "recent_gold_citation_recall": sum(avg_gcr[-n:]) / max(1, len(avg_gcr[-n:])),
                    "recent_retrieved_recall": sum(avg_ret_recall[-n:]) / max(1, len(avg_ret_recall[-n:])),
                    "recent_train_trace_count": sum(trace_counts[-n:]) / max(1, len(trace_counts[-n:])) if trace_counts else 0,
                    "lr": scheduler.get_last_lr()[0],
                }
                print(json.dumps(msg, ensure_ascii=False, indent=2))

            if global_group_step % CONFIG["save_every_groups"] == 0:
                ckpt_dir = os.path.join(CONFIG["output_dir"], f"checkpoint-group-{global_group_step}")
                model.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print("Saved checkpoint to:", ckpt_dir)

    if global_group_step % CONFIG["gradient_accumulation_steps"] != 0:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CONFIG["max_grad_norm"])
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_optim_step += 1

    model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])

    report = {
        "config": CONFIG,
        "global_group_step": global_group_step,
        "global_optim_step": global_optim_step,
        "output_dir": CONFIG["output_dir"],
        "rollout_log_file": CONFIG["rollout_log_file"],
        "summary": {
            "avg_reward_mean": sum(reward_means) / len(reward_means) if reward_means else 0.0,
            "avg_reward_std": sum(reward_stds) / len(reward_stds) if reward_stds else 0.0,
            "avg_group_range": sum(group_ranges) / len(group_ranges) if group_ranges else 0.0,
            "avg_train_trace_count": sum(trace_counts) / len(trace_counts) if trace_counts else 0.0,
            "avg_rouge_l": sum(avg_rouge) / len(avg_rouge) if avg_rouge else 0.0,
            "avg_gold_citation_recall": sum(avg_gcr) / len(avg_gcr) if avg_gcr else 0.0,
            "avg_retrieved_recall": sum(avg_ret_recall) / len(avg_ret_recall) if avg_ret_recall else 0.0,
            "metric_counter": dict(metric_counter),
        },
    }
    save_json(report, CONFIG["train_report_file"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
