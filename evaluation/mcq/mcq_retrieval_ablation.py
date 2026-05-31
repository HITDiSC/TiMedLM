# -*- coding: utf-8 -*-
"""
Table 6: Ablation study on retrieval design for MCQ.

Variants:
1. Chunk + Embedding + single RAG
2. Card + Embedding + single RAG
3. Card + BM25 + single RAG
4. Card + Hybrid + single RAG

This script keeps the answer model/prompt fixed and only changes retrieval.
It reports Hit@6, Recall@6, and MCQ accuracy.
"""

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = os.environ.get("TIMEDLM_BASE_MODEL_PATH", "Qwen/Qwen3-8B")
DEFAULT_LORA_PATH = None
DEFAULT_DATA_PATH = os.environ.get("TIMEDLM_MCQ_TEST_PATH", "data/samples/mcq_eval_sample.json")
DEFAULT_RETRIEVAL_DIR = os.environ.get("RETRIEVAL_ROOT", str(Path(__file__).resolve().parents[2] / "src" / "timedlm" / "retrieval"))
DEFAULT_OUT = "evaluation/table6_mcq_retrieval_ablation.json"


def load_json_or_jsonl(path: str) -> List[dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ["data", "results", "items"]:
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError(f"Unsupported data format: {path}")


def get_gold_evidence_ids(item: dict) -> Set[str]:
    ids = set()
    for key in [
        "gold_fact_ids",
        "silver_fact_ids",
        "gold_evidence_ids",
        "seed_card_ids",
        "card_ids",
        "citations",
    ]:
        val = item.get(key)
        if not val:
            continue
        if isinstance(val, str):
            ids.add(val)
        elif isinstance(val, list):
            for x in val:
                if isinstance(x, str):
                    ids.add(x)
                elif isinstance(x, dict):
                    cid = x.get("card_id") or x.get("id")
                    if cid:
                        ids.add(str(cid))
    return ids


def normalize_score(meta: Any) -> float:
    if isinstance(meta, dict):
        for key in ["score", "fusion_score", "dense_raw", "bm25_raw"]:
            if key in meta:
                return float(meta[key])
        return 0.0
    return float(meta)


def get_card_id(card: dict) -> str:
    return str(card.get("card_id") or card.get("id") or "")


def get_card_content(card: dict) -> str:
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("text")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


def format_evidence(cards: Sequence[Tuple[dict, float]]) -> str:
    rows = []
    for card, score in cards:
        rows.append({
            "card_id": get_card_id(card),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", card.get("type", "")),
            "score": round(float(score), 4),
            "content": get_card_content(card),
        })
    return json.dumps(rows, ensure_ascii=False)


def build_query(question: str, options: Dict[str, Any]) -> str:
    option_terms = "；".join(str(v).strip() for v in (options or {}).values() if str(v).strip())
    return f"{question}；{option_terms}".strip("；")


def extract_pred(content: str, valid_options=None) -> Optional[str]:
    valid_options = {str(x).strip().upper() for x in (valid_options or {"A", "B", "C", "D"})}
    valid_pat = "".join(sorted(valid_options))
    if not content:
        return None
    content = re.sub(r"```json|```", "", str(content)).strip()
    up = content.upper()
    if up in valid_options:
        return up
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            if "answer" in obj:
                m = re.findall(rf"[{valid_pat}]", str(obj["answer"]).upper())
                if m:
                    return m[0]
            for k in obj:
                if str(k).upper() in valid_options:
                    return str(k).upper()
    except Exception:
        pass
    patterns = [
        rf"答案[是为：:]\s*([{valid_pat}])",
        rf"正确答案[是为：:]\s*([{valid_pat}])",
        rf"最终答案[是为：:]\s*([{valid_pat}])",
        rf"应?选\s*([{valid_pat}])",
        rf"故选\s*([{valid_pat}])",
    ]
    for pat in patterns:
        m = re.search(pat, content, re.I)
        if m:
            return m.group(1).upper()
    m = re.search(rf"\b([{valid_pat}])\b", up[-100:])
    if m:
        return m.group(1)
    m = re.findall(rf"[{valid_pat}]", up[-50:])
    return m[-1] if m else None


def compute_hit_recall(retrieved_ids: List[str], gold_ids: Set[str], k: int = 6) -> Tuple[Optional[float], Optional[float]]:
    if not gold_ids:
        return None, None
    top = set(x for x in retrieved_ids[:k] if x)
    hit = 1.0 if top & gold_ids else 0.0
    recall = len(top & gold_ids) / len(gold_ids)
    return hit, recall


class CardRetrievers:
    def __init__(self, retrieval_dir: str):
        sys.path.insert(0, retrieval_dir)
        import retrieval as ret
        self.ret = ret

    def dense(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        rows = self.ret.dense_search(query, top_k=top_k)
        return [(c, float(s)) for c, s in rows]

    def bm25(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        rows = self.ret.bm25_search(query, top_k=top_k)
        return [(c, float(s)) for c, s in rows]

    def hybrid(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        rows = self.ret.retrieve_with_scores(query, top_k=top_k)
        return [(c, normalize_score(s)) for c, s in rows]


class ChunkDenseRetriever:
    def __init__(self, chunk_jsonl: str, chunk_emb_cache: str, model_path: str):
        from FlagEmbedding import BGEM3FlagModel

        self.chunks = load_json_or_jsonl(chunk_jsonl)
        with open(chunk_emb_cache, "rb") as f:
            arr = pickle.load(f)
        self.emb = np.asarray(arr, dtype=np.float32)
        if self.emb.shape[0] != len(self.chunks):
            raise ValueError(f"chunk embeddings {self.emb.shape[0]} != chunks {len(self.chunks)}")
        self.model = BGEM3FlagModel(model_path, use_fp16=True)

    def _query_emb(self, query: str) -> np.ndarray:
        out = self.model.encode([query], batch_size=1, max_length=512)
        vec = np.asarray(out["dense_vecs"][0], dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def dense(self, query: str, top_k: int) -> List[Tuple[dict, float]]:
        q = self._query_emb(query)
        scores = self.emb @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in idx]


def load_model(model_path: str, lora_path: Optional[str]):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def generate_answer(model, tokenizer, question: str, options: dict, evidence_json: str, valid_options: Set[str]) -> Tuple[Optional[str], str]:
    valid_str = "/".join(sorted(valid_options))
    option_str = "\n".join(f"{k}. {v}" for k, v in options.items())
    prompt = (
        "你是一名藏医考试答题助手。\n"
        "请根据题目、选项和检索证据选择最可能正确的答案。\n"
        "不要输出分析过程，不要解释，不要输出推理步骤。\n"
        f"最终只输出一个选项字母，必须是 {valid_str} 中的一个。\n\n"
        f"题目：\n{question}\n\n"
        f"选项：\n{option_str}\n\n"
        f"检索证据：\n{evidence_json}\n"
    )
    messages = [
        {"role": "system", "content": "You are a Tibetan medicine exam assistant. Only output one option letter."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return extract_pred(raw, valid_options), raw


def safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def evaluate_variant(name: str, retriever_fn, dataset: List[dict], model, tokenizer, top_k: int, max_items: Optional[int]) -> Tuple[dict, List[dict]]:
    results = []
    correct = 0
    total = 0
    for i, item in enumerate(tqdm(dataset[:max_items] if max_items else dataset, desc=name)):
        if item.get("type") and item.get("type") != "单选":
            continue
        qnum = str(item.get("question_num", item.get("id", item.get("question_id", i + 1))))
        question = item.get("query") or item.get("question") or item.get("input") or ""
        options = item.get("options", {}) or {}
        gt = str(item.get("answer") or item.get("gold_answer") or item.get("gt") or "").strip().upper()
        valid_options = {str(k).strip().upper() for k in options.keys()} or {"A", "B", "C", "D"}

        query = build_query(question, options)
        cards = retriever_fn(query, top_k)
        retrieved_ids = [get_card_id(c) for c, _ in cards]
        gold_ids = get_gold_evidence_ids(item)
        hit6, recall6 = compute_hit_recall(retrieved_ids, gold_ids, k=6)

        pred, raw = generate_answer(model, tokenizer, question, options, format_evidence(cards), valid_options)
        if pred:
            pred = pred.strip().upper()
        is_correct = pred == gt
        total += 1
        correct += int(is_correct)
        results.append({
            "question_num": qnum,
            "question": question,
            "gt": gt,
            "pred": pred,
            "raw_output": raw,
            "is_correct": is_correct,
            "retrieval_query": query,
            "retrieved_ids": retrieved_ids,
            "gold_ids": list(gold_ids),
            "hit6": hit6,
            "recall6": recall6,
        })

    summary = {
        "variant": name,
        "total": total,
        "correct": correct,
        "mcq_acc": round(correct / total, 4) if total else None,
        "hit6": round(safe_mean(r["hit6"] for r in results), 4) if safe_mean(r["hit6"] for r in results) is not None else None,
        "recall6": round(safe_mean(r["recall6"] for r in results), 4) if safe_mean(r["recall6"] for r in results) is not None else None,
    }
    return summary, results


def print_table(summaries: List[dict]):
    print("\nTable 6: Ablation study on retrieval design for MCQ.")
    print("-" * 72)
    print(f"{'Variant':42s} {'Hit@6':>8s} {'Recall@6':>10s} {'MCQ Acc.':>10s}")
    print("-" * 72)
    for s in summaries:
        hit = "N/A" if s["hit6"] is None else f"{s['hit6']:.4f}"
        rec = "N/A" if s["recall6"] is None else f"{s['recall6']:.4f}"
        acc = "N/A" if s["mcq_acc"] is None else f"{s['mcq_acc']:.4f}"
        print(f"{s['variant']:42s} {hit:>8s} {rec:>10s} {acc:>10s}")
    print("-" * 72)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--lora_path", default=DEFAULT_LORA_PATH)
    parser.add_argument("--retrieval_dir", default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--top_k", type=int, default=6)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--chunk_jsonl", default=None)
    parser.add_argument("--chunk_emb_cache", default=None)
    parser.add_argument("--embedding_model_path", default=os.environ.get("TIMEDLM_EMBEDDING_MODEL", "BAAI/bge-m3"))
    args = parser.parse_args()

    dataset = load_json_or_jsonl(args.data_path)
    dataset = [x for x in dataset if not x.get("type") or x.get("type") == "单选"]

    model, tokenizer = load_model(args.model_path, args.lora_path)
    card_ret = CardRetrievers(args.retrieval_dir)

    variants = []
    if args.chunk_jsonl and args.chunk_emb_cache and os.path.exists(args.chunk_jsonl) and os.path.exists(args.chunk_emb_cache):
        chunk_ret = ChunkDenseRetriever(args.chunk_jsonl, args.chunk_emb_cache, args.embedding_model_path)
        variants.append(("Chunk + Embedding + single RAG", chunk_ret.dense))
    else:
        print("[WARN] chunk_jsonl/chunk_emb_cache not provided or missing; skip Chunk + Embedding + single RAG")

    variants.extend([
        ("Card + Embedding + single RAG", card_ret.dense),
        ("Card + BM25 + single RAG", card_ret.bm25),
        ("Card + Hybrid + single RAG", card_ret.hybrid),
    ])

    all_summaries = []
    all_results = {}
    for name, fn in variants:
        summary, rows = evaluate_variant(name, fn, dataset, model, tokenizer, args.top_k, args.max_items)
        all_summaries.append(summary)
        all_results[name] = rows
        print_table(all_summaries)

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": vars(args),
        "summaries": all_summaries,
        "results": all_results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print_table(all_summaries)
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
