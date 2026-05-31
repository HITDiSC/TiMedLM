# -*- coding: utf-8 -*-
"""
IFEval 测评脚本 3：
Ours = Qwen3-8B + LoRA，默认 no-think，无 RAG

说明：
IFEval 是通用指令遵循测试，不需要 RAG。
这里测的是领域 SFT 后，模型是否保持通用指令遵循能力。

运行：
CUDA_VISIBLE_DEVICES=0 python eval_ifeval_ours_lora.py
"""

import os
import sys
import json
import glob
import subprocess
from datetime import datetime
from typing import List, Dict, Any

import torch
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.environ.get("TIMEDLM_MODEL_PATH", "models/timedlm-sft-v5")
LORA_PATH = os.environ.get("TIMEDLM_LORA_PATH", "models/timedlm-lora")


IFEVAL_INPUT_PATH = os.environ.get("IFEVAL_INPUT_PATH", "data/ifeval/input_data.jsonl")
OFFICIAL_IFEVAL_DIR = os.environ.get("OFFICIAL_IFEVAL_DIR", "external/instruction_following_eval")

RESULT_DIR = os.environ.get("IFEVAL_RESULT_DIR", "results/ifeval/timedlm")

os.makedirs(RESULT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# RESPONSE_PATH = f"{RESULT_DIR}/ours_lora_ifeval_responses_{timestamp}.jsonl"
# CKPT_RESPONSE_PATH = f"{RESULT_DIR}/ours_lora_ifeval_responses_ckpt.jsonl"

RESPONSE_PATH = f"{RESULT_DIR}/qa_grpo_ifeval_responses_{timestamp}.jsonl"
CKPT_RESPONSE_PATH = f"{RESULT_DIR}/qa_grpo_ifeval_responses_ckpt.jsonl"

EVAL_OUTPUT_DIR = f"{RESULT_DIR}/official_eval_{timestamp}"
os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

# SUMMARY_PATH = f"{RESULT_DIR}/ours_lora_ifeval_summary_{timestamp}.json"
SUMMARY_PATH = f"{RESULT_DIR}/qa_grpo_ifeval_summary_{timestamp}.json"

MAX_NEW_TOKENS = 1024
SAVE_EVERY = 20
DEBUG_SAMPLES = 3

# 建议 Ours 用 no-think，保持和你的任务最终回答阶段一致
ENABLE_THINKING = False


print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载基座模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("加载 LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

print("模型加载完成\n")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_done_responses(path: str):
    if not os.path.exists(path):
        return [], set()

    results = []
    done_prompts = set()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results.append(obj)
            done_prompts.add(obj.get("prompt", ""))

    print(f"[断点恢复] 已生成 {len(results)} 条")
    return results, done_prompts


def append_jsonl(path: str, obj: Dict[str, Any]):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def generate_response(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=ENABLE_THINKING,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(
        output_ids,
        skip_special_tokens=True
    ).strip()

    return response


def run_official_ifeval():
    eval_main = os.path.join(OFFICIAL_IFEVAL_DIR, "evaluation_main.py")
    ifeval_parent_dir = os.path.dirname(OFFICIAL_IFEVAL_DIR.rstrip(os.sep))

    if not os.path.exists(eval_main):
        raise FileNotFoundError(
            f"找不到 evaluation_main.py，请检查 OFFICIAL_IFEVAL_DIR: {OFFICIAL_IFEVAL_DIR}"
        )

    cmd = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={IFEVAL_INPUT_PATH}",
        f"--input_response_data={RESPONSE_PATH}",
        f"--output_dir={EVAL_OUTPUT_DIR}",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = ifeval_parent_dir + ":" + env.get("PYTHONPATH", "")

    print("\n开始调用官方 IFEval evaluator...")
    print(" ".join(cmd))

    subprocess.run(
        cmd,
        cwd=ifeval_parent_dir,
        env=env,
        check=True,
    )


def parse_eval_file(path: str):
    prompt_total = 0
    prompt_correct = 0
    inst_total = 0
    inst_correct = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)

            follow_all = obj.get("follow_all_instructions", None)
            follow_list = obj.get("follow_instruction_list", [])

            if follow_all is not None:
                prompt_total += 1
                if follow_all:
                    prompt_correct += 1

            if isinstance(follow_list, list):
                inst_total += len(follow_list)
                inst_correct += sum(1 for x in follow_list if x)

    return {
        "prompt_total": prompt_total,
        "prompt_correct": prompt_correct,
        "prompt_accuracy": prompt_correct / prompt_total if prompt_total else None,
        "instruction_total": inst_total,
        "instruction_correct": inst_correct,
        "instruction_accuracy": inst_correct / inst_total if inst_total else None,
    }


def parse_official_results():
    strict_files = glob.glob(os.path.join(EVAL_OUTPUT_DIR, "*strict*.jsonl"))
    loose_files = glob.glob(os.path.join(EVAL_OUTPUT_DIR, "*loose*.jsonl"))

    summary = {
        # "mode": "ours_lora",
        "mode": "dpo_v1",
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH,
        "input_path": IFEVAL_INPUT_PATH,
        "response_path": RESPONSE_PATH,
        "official_eval_dir": EVAL_OUTPUT_DIR,
        "thinking": ENABLE_THINKING,
        "lora": True,
        "rag": False,
    }

    if strict_files:
        strict_path = strict_files[0]
        strict_result = parse_eval_file(strict_path)
        summary["strict_result_file"] = strict_path
        summary["prompt_level_strict_acc"] = strict_result["prompt_accuracy"]
        summary["instruction_level_strict_acc"] = strict_result["instruction_accuracy"]
        summary["strict_detail"] = strict_result
    else:
        summary["prompt_level_strict_acc"] = None
        summary["instruction_level_strict_acc"] = None

    if loose_files:
        loose_path = loose_files[0]
        loose_result = parse_eval_file(loose_path)
        summary["loose_result_file"] = loose_path
        summary["prompt_level_loose_acc"] = loose_result["prompt_accuracy"]
        summary["instruction_level_loose_acc"] = loose_result["instruction_accuracy"]
        summary["loose_detail"] = loose_result
    else:
        summary["prompt_level_loose_acc"] = None
        summary["instruction_level_loose_acc"] = None

    return summary


def main():
    data = load_jsonl(IFEVAL_INPUT_PATH)
    print(f"IFEval 样本数: {len(data)}")

    results, done_prompts = load_done_responses(CKPT_RESPONSE_PATH)

    for idx, item in enumerate(tqdm(data, desc="生成 IFEval 回答")):
        prompt = item.get("prompt", "")

        if not prompt:
            continue

        if prompt in done_prompts:
            continue

        response = generate_response(prompt)

        obj = {
            "prompt": prompt,
            "response": response,
        }

        append_jsonl(CKPT_RESPONSE_PATH, obj)
        results.append(obj)
        done_prompts.add(prompt)

        if len(results) <= DEBUG_SAMPLES:
            print("\n" + "=" * 80)
            print(f"【样本 {len(results)}】")
            print("Prompt:")
            print(prompt[:1000])
            print("\nResponse:")
            print(response[:1000])

        if len(results) % SAVE_EVERY == 0:
            print(f"[断点] 已保存 {len(results)}/{len(data)}")

    os.rename(CKPT_RESPONSE_PATH, RESPONSE_PATH)
    print(f"\n回答已保存到：{RESPONSE_PATH}")

    run_official_ifeval()

    summary = parse_official_results()

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nIFEval 测评完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsummary 保存到：{SUMMARY_PATH}")


if __name__ == "__main__":
    main()
