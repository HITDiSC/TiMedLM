# -*- coding: utf-8 -*-
"""
问答 baseline 3：
Qwen3-8B native single-RAG no-think，无 LoRA

指标：
- ROUGE-L
- BLEU-4
- BERTScore-F1
- Hit@10
- Recall@10
- Citation Coverage
- Citation Validity
- Gold Citation Recall

运行：
CUDA_VISIBLE_DEVICES=0 python eval_qa_qwen_native_rag_nothink.py
"""

import os
import re
import sys
import json
import math
import string
import torch
from datetime import datetime
from collections import Counter
from typing import List, Optional, Tuple, Set, Dict

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = os.environ.get("TIMEDLM_BASE_MODEL_PATH", "Qwen/Qwen3-8B")
QA_TEST_PATH = os.environ.get("TIMEDLM_QA_TEST_PATH", "data/samples/oqa_eval_sample.json")

RESULT_DIR = os.environ.get("TIMEDLM_QA_RESULT_DIR", "results/qa/qwen8b")
os.makedirs(RESULT_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PATH = f"{RESULT_DIR}/qa_eval_qwen_native_rag_nothink_{timestamp}.json"
CKPT_PATH = f"{RESULT_DIR}/qa_eval_qwen_native_rag_nothink_ckpt.json"

RETRIEVE_TOP_K = 10

MAX_NEW_TOKENS = 1400
SAVE_EVERY = 10
DEBUG_SAMPLES = 3

USE_BERTSCORE = True
BERTSCORE_MODEL_TYPE = os.environ.get("TIMEDLM_BERTSCORE_MODEL_TYPE", "hfl/chinese-roberta-wwm-ext")
BERTSCORE_NUM_LAYERS = 12
MAX_BERT_TOKENS = 510

RETRIEVAL_ROOT = os.environ.get(
    "RETRIEVAL_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "timedlm", "retrieval")),
)
sys.path.append(RETRIEVAL_ROOT)
from retrieval import retrieve_with_scores


CARD_ID_RE = re.compile(
    r"\b(?:fact_[a-zA-Z0-9]+_\d{3}_\d{3}|case_[a-zA-Z0-9]+_\d{3}_\d{3}|diag_manual_\d{3})\b"
)


SYSTEM_PROMPT = """\
你是一位精通藏医学的专家，拥有深厚的藏医理论和临床知识。
你需要根据给定的藏医知识库证据回答问题。
请优先依据知识库证据。
不能编造不存在的 card_id。
"""

USER_PROMPT_TEMPLATE = """\
请根据问题和知识库证据回答。

问题：
{question}

知识库证据：
{evidence_text}

要求：
1. 回答应准确、完整、条理清晰。
2. 必须优先依据知识库证据。
3. 如果使用了证据，请在末尾写“引用来源：card_id1, card_id2”。
4. 引用来源只能使用上方知识库证据中出现的 card_id，不能编造。
"""


print("加载 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("加载原始 Qwen3-8B，无 LoRA...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

print("模型加载完成\n")


def generate(messages):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
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
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def extract_card_ids(text: str) -> List[str]:
    if not text:
        return []
    return list(dict.fromkeys(CARD_ID_RE.findall(text)))


def get_gold_evidence_ids(item: Dict) -> Set[str]:
    ids = set()

    ref = item.get("reference", "") or item.get("answer", "")
    ids.update(extract_card_ids(ref))

    for k in ["citations", "gold_card_ids"]:
        vals = item.get(k, [])
        if isinstance(vals, list):
            ids.update(str(x) for x in vals if x)
        elif isinstance(vals, str) and vals:
            ids.update(extract_card_ids(vals))
            if CARD_ID_RE.fullmatch(vals):
                ids.add(vals)

    if not ids:
        vals = item.get("seed_card_ids", [])
        if isinstance(vals, list):
            ids.update(str(x) for x in vals if x)
        elif isinstance(vals, str) and vals:
            ids.add(vals)

    return ids


def get_card_content(card: Dict) -> str:
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


def retrieve_cards(question: str, top_k: int = 10):
    query = question.strip()

    try:
        scored_cards = retrieve_with_scores(query, top_k=top_k)
    except Exception as e:
        print(f"[检索失败] query={query} error={e}")
        return query, []

    cards = []
    for card, score in scored_cards:
        cards.append({
            "card_id": card.get("card_id", ""),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", ""),
            "content": get_card_content(card),
            "score": float(score),
        })

    return query, cards


def format_evidence(cards):
    if not cards:
        return "未检索到相关证据。"

    lines = []
    for i, c in enumerate(cards, start=1):
        lines.append(
            f"[{i}] card_id: {c['card_id']}\n"
            f"title: {c['title']}\n"
            f"type: {c['card_type']}\n"
            f"score: {c['score']:.4f}\n"
            f"content: {c['content']}"
        )

    return "\n\n".join(lines)


def remove_card_ids_for_metric(text: str) -> str:
    if not text:
        return ""

    text = CARD_ID_RE.sub("[CARD_ID]", text)
    text = re.sub(r"引用来源[:：]\s*\[?.*?\]?\s*$", "", text, flags=re.S)
    text = re.sub(r"citations?[:：]\s*\[?.*?\]?\s*$", "", text, flags=re.I | re.S)

    return text.strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[，。！？；：“”‘’、（）《》【】\[\]]", "", text)
    return text


def tokenize_zh(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        return list(normalize_text(text))


def lcs_length(x: List[str], y: List[str]) -> int:
    if not x or not y:
        return 0

    m, n = len(x), len(y)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def rouge_l_score(pred: str, ref: str) -> float:
    pred = remove_card_ids_for_metric(pred)
    ref = remove_card_ids_for_metric(ref)

    pred_tokens = tokenize_zh(pred)
    ref_tokens = tokenize_zh(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)
    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)

    if recall + precision == 0:
        return 0.0

    beta = precision / (recall + 1e-12)

    return float(((1 + beta ** 2) * precision * recall) / (
        recall + beta ** 2 * precision + 1e-12
    ))


def ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4_score(pred: str, ref: str) -> float:
    pred = remove_card_ids_for_metric(pred)
    ref = remove_card_ids_for_metric(ref)

    pred_tokens = tokenize_zh(pred)
    ref_tokens = tokenize_zh(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []

    for n in range(1, 5):
        pred_ngrams = ngram_counts(pred_tokens, n)
        ref_ngrams = ngram_counts(ref_tokens, n)

        if not pred_ngrams:
            precisions.append(1e-9)
            continue

        overlap = 0
        total = sum(pred_ngrams.values())

        for ng, count in pred_ngrams.items():
            overlap += min(count, ref_ngrams.get(ng, 0))

        precisions.append((overlap + 1) / (total + 1))

    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))
    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)
    bp = 1.0 if pred_len > ref_len else math.exp(1 - ref_len / max(pred_len, 1))

    return float(bp * math.exp(log_precision))


def compute_hit_recall_at_k(retrieved_ids: List[str], gold_ids: Set[str], k: int = 10):
    if not gold_ids:
        return None, None

    top_ids = retrieved_ids[:k]
    retrieved_set = set(top_ids)

    hit = 1.0 if retrieved_set & gold_ids else 0.0
    recall = len(retrieved_set & gold_ids) / len(gold_ids)

    return hit, recall


def compute_citation_metrics(pred: str, retrieved_ids: List[str], gold_ids: Set[str]):
    cited_ids = extract_card_ids(pred)
    retrieved_set = set(retrieved_ids)

    if not cited_ids:
        return {
            "citation_coverage": 0.0,
            "citation_existence_rate": None,
            "citation_from_retrieval_rate": None,
            "citation_validity": None,
            "gold_citation_recall": 0.0 if gold_ids else None,
            "cited_ids": [],
            "non_retrieved_cited_ids": [],
        }

    from_retrieval_ids = [cid for cid in cited_ids if cid in retrieved_set]
    non_retrieved_cited_ids = [cid for cid in cited_ids if cid not in retrieved_set]

    citation_from_retrieval_rate = len(from_retrieval_ids) / len(cited_ids)

    # native RAG baseline 严格定义：所有引用必须来自本次检索
    citation_validity = 1.0 if len(from_retrieval_ids) == len(cited_ids) else 0.0

    gold_citation_recall = None
    if gold_ids:
        gold_citation_recall = len(set(cited_ids) & gold_ids) / len(gold_ids)

    return {
        "citation_coverage": 1.0,
        "citation_existence_rate": citation_from_retrieval_rate,
        "citation_from_retrieval_rate": citation_from_retrieval_rate,
        "citation_validity": citation_validity,
        "gold_citation_recall": gold_citation_recall,
        "cited_ids": cited_ids,
        "non_retrieved_cited_ids": non_retrieved_cited_ids,
    }


def token_truncate(text: str, bert_tokenizer, max_tokens: int = 510) -> str:
    if not text:
        return ""

    ids = bert_tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )

    return bert_tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def compute_bertscore_batch(preds: List[str], refs: List[str]) -> Tuple[List[Optional[float]], int, int]:
    if not USE_BERTSCORE:
        return [None] * len(preds), 0, 0

    try:
        from bert_score import score
        from transformers import AutoTokenizer as BertAutoTokenizer
    except Exception as e:
        print(f"未安装 bert_score 或 transformers，跳过 BERTScore。error={e}")
        return [None] * len(preds), 0, 0

    try:
        print("加载 BERTScore tokenizer...")
        bert_tokenizer = BertAutoTokenizer.from_pretrained(
            BERTSCORE_MODEL_TYPE,
            use_fast=True,
        )

        clean_preds = []
        clean_refs = []
        pred_trunc_count = 0
        ref_trunc_count = 0

        for p, r in zip(preds, refs):
            p = remove_card_ids_for_metric(p)
            r = remove_card_ids_for_metric(r)

            p_ids = bert_tokenizer.encode(p, add_special_tokens=False)
            r_ids = bert_tokenizer.encode(r, add_special_tokens=False)

            if len(p_ids) > MAX_BERT_TOKENS:
                pred_trunc_count += 1
            if len(r_ids) > MAX_BERT_TOKENS:
                ref_trunc_count += 1

            clean_preds.append(token_truncate(p, bert_tokenizer, MAX_BERT_TOKENS))
            clean_refs.append(token_truncate(r, bert_tokenizer, MAX_BERT_TOKENS))

        print(f"BERTScore prediction 截断数量: {pred_trunc_count}")
        print(f"BERTScore reference 截断数量: {ref_trunc_count}")

        _, _, f1 = score(
            clean_preds,
            clean_refs,
            model_type=BERTSCORE_MODEL_TYPE,
            num_layers=BERTSCORE_NUM_LAYERS,
            verbose=True,
            rescale_with_baseline=False,
            batch_size=8,
        )

        return [float(x) for x in f1.cpu().tolist()], pred_trunc_count, ref_trunc_count

    except Exception as e:
        print(f"BERTScore 计算失败，跳过。error={e}")
        return [None] * len(preds), 0, 0


def safe_mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def load_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return [], set()

    try:
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        results = ckpt.get("results", [])
        done_ids = {str(x["sample_id"]) for x in results}

        print(f"[断点恢复] 已完成 {len(results)} 条，继续评估...\n")
        return results, done_ids
    except Exception as e:
        print(f"[断点恢复失败] {e}，从头开始\n")
        return [], set()


def save_checkpoint(results, total):
    ckpt = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "qa_test_path": QA_TEST_PATH,
        "progress": f"{len(results)}/{total}",
        "results": results,
    }

    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


def main():
    with open(QA_TEST_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"问答测试集数量: {len(dataset)}")
    print("===== Qwen3-8B native RAG no-think 问答评估 =====\n")

    results, done_ids = load_checkpoint()

    for idx, item in enumerate(tqdm(dataset, desc="问答评估中")):
        sample_id = str(item.get("id", item.get("question_id", idx + 1)))

        if sample_id in done_ids:
            continue

        question = item.get("input") or item.get("question") or item.get("query") or ""
        reference = item.get("reference") or item.get("answer") or ""
        gold_ids = get_gold_evidence_ids(item)

        retrieval_query, cards = retrieve_cards(question, RETRIEVE_TOP_K)
        evidence_text = format_evidence(cards)
        retrieved_ids = [c["card_id"] for c in cards if c.get("card_id")]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    question=question,
                    evidence_text=evidence_text,
                ),
            },
        ]

        pred = generate(messages)

        rouge_l = rouge_l_score(pred, reference) if reference else None
        bleu4 = bleu4_score(pred, reference) if reference else None
        hit10, recall10 = compute_hit_recall_at_k(retrieved_ids, gold_ids, k=10)
        citation_info = compute_citation_metrics(pred, retrieved_ids, gold_ids)

        result = {
            "sample_id": sample_id,
            "question": question,
            "reference": reference,
            "prediction": pred,
            "question_type": item.get("question_type", ""),
            "retrieval_query": retrieval_query,
            "retrieved_card_ids": retrieved_ids,
            "retrieved_cards": cards,
            "gold_evidence_ids": list(gold_ids),
            "rouge_l": rouge_l,
            "bleu4": bleu4,
            "bertscore_f1": None,
            "hit10": hit10,
            "recall10": recall10,
            **citation_info,
            "mode": "qwen_native_rag_nothink",
        }

        results.append(result)

        if len(results) <= DEBUG_SAMPLES:
            print("\n" + "=" * 80)
            print(f"【样本 {sample_id}】")
            print("问题：", question)
            print("检索：", retrieved_ids)
            print("模型回答：", pred[:800])
            print("参考答案：", reference[:800])
            print(f"ROUGE-L={rouge_l}, BLEU-4={bleu4}, Hit@10={hit10}, Recall@10={recall10}")

        if len(results) % SAVE_EVERY == 0:
            save_checkpoint(results, len(dataset))

    print("\n开始计算 BERTScore...")
    preds = [r["prediction"] for r in results]
    refs = [r["reference"] for r in results]
    bert_scores, pred_trunc_count, ref_trunc_count = compute_bertscore_batch(preds, refs)

    for r, bs in zip(results, bert_scores):
        r["bertscore_f1"] = bs

    summary = {
        "total": len(results),
        "rouge_l": round(safe_mean([r["rouge_l"] for r in results]), 4),
        "bleu4": round(safe_mean([r["bleu4"] for r in results]), 4),
        "bertscore_f1": round(safe_mean([r["bertscore_f1"] for r in results]), 4)
        if safe_mean([r["bertscore_f1"] for r in results]) is not None else None,
        "hit10": round(safe_mean([r["hit10"] for r in results]), 4),
        "recall10": round(safe_mean([r["recall10"] for r in results]), 4),
        "citation_coverage": round(safe_mean([r["citation_coverage"] for r in results]), 4),
        "citation_existence_rate": round(safe_mean([r["citation_existence_rate"] for r in results]), 4),
        "citation_from_retrieval_rate": round(safe_mean([r["citation_from_retrieval_rate"] for r in results]), 4),
        "citation_validity": round(safe_mean([r["citation_validity"] for r in results]), 4),
        "gold_citation_recall": round(safe_mean([r["gold_citation_recall"] for r in results]), 4),
        "bertscore_model_type": BERTSCORE_MODEL_TYPE,
        "bertscore_num_layers": BERTSCORE_NUM_LAYERS,
        "bertscore_max_tokens": MAX_BERT_TOKENS,
        "bertscore_truncated_pred_count": pred_trunc_count,
        "bertscore_truncated_ref_count": ref_trunc_count,
        "bertscore_added_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    output = {
        "mode": "qwen_native_rag_nothink",
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "qa_test_path": QA_TEST_PATH,
        "retrieve_top_k": RETRIEVE_TOP_K,
        "config": {
            "use_bertscore": USE_BERTSCORE,
            "bertscore_model_type": BERTSCORE_MODEL_TYPE,
            "bertscore_num_layers": BERTSCORE_NUM_LAYERS,
            "bertscore_max_tokens": MAX_BERT_TOKENS,
            "rag": True,
            "rag_type": "native_single_rag",
            "thinking": False,
            "lora": False,
            "citation_validity_definition": "all_cited_ids_must_come_from_retrieval",
        },
        "summary": summary,
        "results": results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    print("\n评估完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"结果保存到：{RESULT_PATH}")


if __name__ == "__main__":
    main()