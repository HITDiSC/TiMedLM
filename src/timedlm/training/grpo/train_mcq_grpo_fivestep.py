# -*- coding: utf-8 -*-
"""
Old-eval-compatible MCQ GRPO training.

This trains a LoRA adapter on top of the SFT merged model. The rollout protocol is
imported from debug_old_eval_rollout_sanity.py so training and sanity/eval stay aligned.
"""

import json
import math
import os
import random
from collections import Counter
from typing import Any, Dict, List

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

import mcq_fivestep_rollout as roll


CONFIG = {
    "model_path": os.environ.get("TIMEDLM_BASE_MODEL", "Qwen/Qwen3-8B"),
    "init_lora_path": None,
    "train_file": "data/grpo/mcq/mcq_grpo_train_prompts_targeted280.jsonl",
    "output_dir": "outputs/grpo/qwen3-8b-mcq-grpo-lora",
    "rollout_log_file": "outputs/grpo/logs/mcq_grpo_rollout_log.jsonl",
    "train_report_file": "outputs/grpo/logs/mcq_grpo_train_report.json",
    "max_train_items": 280,
    "num_generations": 4,
    "num_epochs": 1,
    "learning_rate": 1e-7,
    "weight_decay": 0.0,
    "warmup_ratio": 0.03,
    "max_grad_norm": 1.0,
    "gradient_accumulation_steps": 1,
    "max_seq_length": 2048,
    "normalize_logprob_by_len": True,
    "advantage_eps": 1e-6,
    "clip_advantage": 5.0,
    "missing_query_penalty": -1.0,
    "plan_as_query_penalty": -0.15,
    "query_retry_penalty": -0.3,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "bf16": True,
    "fp16": False,
    "gradient_checkpointing": True,
    "seed": 42,
    "log_every_groups": 10,
    "save_every_groups": 70,
    "verbose_rollout": False,
}


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def sync_rollout_config():
    for k in [
        "model_path",
        "max_rounds",
        "retrieve_top_k_per_query",
        "max_cards_per_round",
        "plan_query_max_new_tokens",
        "judge_max_new_tokens",
        "final_max_new_tokens",
        "temperature",
        "top_p",
        "use_logprob_for_forced",
        "use_logprob_for_final_fallback",
        "suff_threshold",
        "suff_need_max",
        "missing_query_penalty",
        "plan_as_query_penalty",
        "query_retry_penalty",
    ]:
        if k in CONFIG and k in roll.CONFIG:
            roll.CONFIG[k] = CONFIG[k]


def print_trainable_parameters(model):
    trainable = 0
    total = 0
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
    allowed_kinds = {"plan_query", "judge", "final", "final_retry"}
    model.train()

    for traj, adv in zip(trajectories, advantages):
        adv_value = float(adv.item())
        if abs(adv_value) < 1e-8:
            continue
        traces = [t for t in traj.get("train_traces", []) if t.get("kind") in allowed_kinds]
        traces = traces[:3]
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
    sync_rollout_config()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    if os.path.exists(CONFIG["rollout_log_file"]):
        os.remove(CONFIG["rollout_log_file"])

    print(json.dumps(CONFIG, ensure_ascii=False, indent=2))
    data = [roll.normalize_example(x) for x in roll.load_jsonl(CONFIG["train_file"])]
    random.shuffle(data)
    if CONFIG["max_train_items"] and CONFIG["max_train_items"] > 0:
        data = data[:CONFIG["max_train_items"]]
    print(f"Loaded train prompts: {len(data)}")

    model, tokenizer = load_model_and_tokenizer()
    total_groups = len(data) * CONFIG["num_epochs"]
    optim_steps = math.ceil(total_groups / CONFIG["gradient_accumulation_steps"])
    optimizer, scheduler = build_optimizer_and_scheduler(model, optim_steps)

    global_group_step = 0
    global_optim_step = 0
    metric_counter = Counter()
    reward_means, reward_stds, group_ranges, group_correct_rates, trace_counts = [], [], [], [], []

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(CONFIG["num_epochs"]):
        random.shuffle(data)
        for ex in tqdm(data, desc=f"epoch {epoch + 1}"):
            global_group_step += 1
            trajectories = []
            group_rewards = []
            group_correct = 0

            for g in range(CONFIG["num_generations"]):
                traj = roll.run_one_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    example=ex,
                    gen_id=g,
                    verbose=CONFIG["verbose_rollout"],
                )
                trajectories.append(traj)
                append_jsonl(traj, CONFIG["rollout_log_file"])
                group_rewards.append(traj["final"]["total_reward"])
                group_correct += int(bool(traj["final"].get("correct")))
                metric_counter[f"stop/{traj['final'].get('stop_reason')}"] += 1
                metric_counter[f"answer_source/{traj['final'].get('answer_source')}"] += 1
                metric_counter[f"answer/{traj['final'].get('pred_answer')}"] += 1

            loss, loss_info = compute_group_loss(model, tokenizer, trajectories)
            if loss is not None:
                (loss / CONFIG["gradient_accumulation_steps"]).backward()
                trace_counts.append(loss_info["train_trace_count"])

            reward_means.append(loss_info["reward_mean"])
            reward_stds.append(loss_info["reward_std"])
            group_ranges.append(max(group_rewards) - min(group_rewards))
            group_correct_rates.append(group_correct / len(trajectories))

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
                    "recent_group_correct_rate": sum(group_correct_rates[-n:]) / max(1, len(group_correct_rates[-n:])),
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
            "avg_group_correct_rate": sum(group_correct_rates) / len(group_correct_rates) if group_correct_rates else 0.0,
            "avg_train_trace_count": sum(trace_counts) / len(trace_counts) if trace_counts else 0.0,
            "metric_counter": dict(metric_counter),
        },
    }
    save_json(report, CONFIG["train_report_file"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
