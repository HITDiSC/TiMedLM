# -*- coding: utf-8 -*-
# This file trains TiMedLM with DPO preference optimization.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
import argparse
import json
import os
from collections import Counter
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


DEFAULT_DPO_DATA = Path("data/dpo/dpo_rag_decision.jsonl")
DEFAULT_OUTPUT_DIR = Path("outputs/qwen3-8b-tibetan-dpo")
DEFAULT_LOGGING_DIR = Path("outputs/logs-dpo")


def parse_args():
    parser = argparse.ArgumentParser(description="Train TiMedLM DPO LoRA.")
    parser.add_argument("--base_model", required=True, help="Base model path or Hugging Face model id.")
    parser.add_argument("--sft_lora", required=True, help="SFT LoRA checkpoint used as the DPO policy initialization.")
    parser.add_argument("--dpo_data", type=Path, default=DEFAULT_DPO_DATA)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging_dir", type=Path, default=DEFAULT_LOGGING_DIR)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.03)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-6)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--per_device_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=25)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                print(f"[WARN] JSON parse failed at line {line_no}: {exc}")
    return rows


def prepare_dataset(path: Path):
    rows = load_jsonl(path)
    data = []
    type_counter = Counter()
    error_counter = Counter()
    bad_empty = 0
    bad_same = 0

    for row in rows:
        prompt = str(row.get("prompt", "")).strip()
        chosen = str(row.get("chosen", "")).strip()
        rejected = str(row.get("rejected", "")).strip()

        if not prompt or not chosen or not rejected:
            bad_empty += 1
            continue
        if chosen == rejected:
            bad_same += 1
            continue

        type_counter[row.get("type", "unknown")] += 1
        error_counter[row.get("meta", {}).get("error_type", "unknown")] += 1
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print("\n[data]")
    print(f"raw rows: {len(rows)}")
    print(f"usable rows: {len(data)}")
    print(f"bad_empty: {bad_empty}")
    print(f"bad_same: {bad_same}")

    print("\n[type counts]")
    for key, value in type_counter.most_common():
        print(f"{key}: {value}")

    print("\n[error_type counts]")
    for key, value in error_counter.most_common():
        print(f"{key}: {value}")

    return Dataset.from_list(data)


def check_token_lengths(dataset, tokenizer, max_length: int, max_prompt_length: int):
    prompt_over = 0
    chosen_total_over = 0
    rejected_total_over = 0
    prompt_lens = []
    chosen_total_lens = []
    rejected_total_lens = []

    for item in dataset:
        prompt_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(item["chosen"], add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(item["rejected"], add_special_tokens=False)["input_ids"]

        prompt_lens.append(len(prompt_ids))
        chosen_total_lens.append(len(prompt_ids) + len(chosen_ids))
        rejected_total_lens.append(len(prompt_ids) + len(rejected_ids))

        prompt_over += len(prompt_ids) > max_prompt_length
        chosen_total_over += len(prompt_ids) + len(chosen_ids) > max_length
        rejected_total_over += len(prompt_ids) + len(rejected_ids) > max_length

    total = len(dataset)

    def pct(value):
        return round(value / total * 100, 2) if total else 0

    print("\n[token length check]")
    print(f"total: {total}")
    print(f"prompt > {max_prompt_length}: {prompt_over} ({pct(prompt_over)}%)")
    print(f"prompt+chosen > {max_length}: {chosen_total_over} ({pct(chosen_total_over)}%)")
    print(f"prompt+rejected > {max_length}: {rejected_total_over} ({pct(rejected_total_over)}%)")

    if prompt_lens:
        print(f"prompt max: {max(prompt_lens)}")
        print(f"chosen total max: {max(chosen_total_lens)}")
        print(f"rejected total max: {max(rejected_total_lens)}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    if not args.dpo_data.exists():
        raise FileNotFoundError(f"DPO data not found: {args.dpo_data}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)

    dataset = prepare_dataset(args.dpo_data)

    print("\n[load tokenizer]")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    check_token_lengths(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    print("\n[load policy base model]")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("\n[load SFT LoRA as trainable policy]")
    model = PeftModel.from_pretrained(
        base_model,
        args.sft_lora,
        is_trainable=True,
    )
    model.config.use_cache = False

    print("\n[load reference base model]")
    ref_base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print("\n[load SFT LoRA as frozen reference]")
    ref_model = PeftModel.from_pretrained(
        ref_base_model,
        args.sft_lora,
        is_trainable=False,
    )
    ref_model.config.use_cache = False
    ref_model.eval()

    print("\n[training args]")
    print(f"beta: {args.beta}")
    print(f"lr: {args.learning_rate}")
    print(f"epochs: {args.epochs}")
    print(f"warmup_ratio: {args.warmup_ratio}")
    print(f"max_length: {args.max_length}")
    print(f"max_prompt_length: {args.max_prompt_length}")
    print(f"per_device_batch_size: {args.per_device_batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")

    training_args = DPOConfig(
        output_dir=str(args.output_dir),
        logging_dir=str(args.logging_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        seed=args.seed,
        ddp_find_unused_parameters=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\n[start DPO training]")
    trainer.train()

    print("\n[save final LoRA]")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n[DONE] DPO LoRA saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
