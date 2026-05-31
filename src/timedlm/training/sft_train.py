# -*- coding: utf-8 -*-
"""
LoRA supervised fine-tuning script for TiMedLM.
"""

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_MODEL_PATH = "Qwen/Qwen3-8B"
DEFAULT_DATA_PATH = "data/final_train_v5.json"
DEFAULT_OUTPUT_DIR = "outputs/qwen3-8b-timedlm-sft-v5"
DEFAULT_LOGGING_DIR = "outputs/logs-sft-v5"
MAX_LENGTH = 4096

LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    bias="none",
)


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA SFT for TiMedLM-8B.")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--logging_dir", default=DEFAULT_LOGGING_DIR)
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_training_args(args):
    return TrainingArguments(
        output_dir=args.output_dir,
        logging_dir=args.logging_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_accumulation_steps=8,
        per_device_eval_batch_size=1,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        report_to="none",
        ddp_find_unused_parameters=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
    )


def load_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def messages_to_text(messages: list, tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def preprocess(examples, tokenizer):
    input_ids_list = []
    labels_list = []
    attention_masks = []

    for messages in examples["messages"]:
        full_text = messages_to_text(messages, tokenizer)
        tokenized = tokenizer(
            full_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]
        labels = [-100] * len(input_ids)
        current_len = 0

        for i, msg in enumerate(messages):
            partial_messages = messages[: i + 1]
            partial_text = messages_to_text(partial_messages, tokenizer)
            partial_ids = tokenizer(
                partial_text,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
                return_tensors=None,
            )["input_ids"]
            new_len = len(partial_ids)

            if msg["role"] == "assistant":
                for j in range(current_len, min(new_len, len(labels))):
                    labels[j] = input_ids[j]

            current_len = new_len

        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attention_masks.append([1] * len(input_ids))

    return {
        "input_ids": input_ids_list,
        "labels": labels_list,
        "attention_mask": attention_masks,
    }


def verify_mask(tokenized_dataset, num_samples=5):
    """Print the proportion of tokens that contribute to the loss."""
    print("\nVerifying loss mask; effective token ratio should usually be moderate.")
    ratios = []
    for i in range(min(num_samples, len(tokenized_dataset))):
        sample = tokenized_dataset[i]
        labels = sample["labels"]
        total = len(labels)
        non_mask = sum(1 for label in labels if label != -100)
        ratio = non_mask / total if total > 0 else 0
        ratios.append(ratio)
        print(f"  sample {i + 1}: {non_mask}/{total} = {ratio:.2%}")

    avg = sum(ratios) / len(ratios)
    print(f"  average effective token ratio: {avg:.2%}")
    if avg < 0.05:
        print("  Warning: effective token ratio is low; check assistant message masking.")
    elif avg > 0.6:
        print("  Warning: effective token ratio is high; tool messages may not be masked.")
    else:
        print("  Loss mask looks normal.")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    training_args = build_training_args(args)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print("Applying LoRA...")
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    print("Loading data...")
    raw_data = load_data(args.data_path)
    print(f"  Total samples: {len(raw_data)}")

    dataset = Dataset.from_list(raw_data)

    print("Preprocessing data...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
    )

    tokenized_dataset = tokenized_dataset.filter(
        lambda x: len(x["input_ids"]) > 10
    )
    print(f"  Valid samples: {len(tokenized_dataset)}")

    verify_mask(tokenized_dataset)

    split = tokenized_dataset.train_test_split(test_size=0.05, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Eval samples: {len(eval_dataset)}")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
    )

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    trainer.train()

    print("Saving model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
