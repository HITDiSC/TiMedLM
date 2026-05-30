# -*- coding: utf-8 -*-
# This file evaluates a closed model on open-ended QA without RAG.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""
Closed/open-source model QA evaluation without RAG.

This script mirrors the broad style of the single-RAG QA evaluation script,
but evaluates the model under a no-retrieval setting:
- calls an OpenAI-compatible endpoint for a chat model
- does not call retrieval.retrieve_with_scores
- asks the model to answer only from its own parametric knowledge
- saves checkpoint and final JSON
- reports ROUGE-L, BLEU-4, and BERTScore; retrieval/citation fields are kept as None
"""

import os
import re
import json
import math
import time
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm
from openai import OpenAI


# ============================================================
# Config
# ============================================================

# OpenAI-compatible API for the open-source model.
# Examples:
#   export GR_API_KEY=xxx
#   export QA_MODEL_NAME=Qwen3-8B
#   export QA_BASE_URL=http://127.0.0.1:8000/v1
API_KEY = os.environ.get("GR_API_KEY", "")
BASE_URL = os.environ.get("QA_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
MODEL_NAME = os.environ.get("QA_MODEL_NAME", "glm-4.5-air")

DATA_PATH = os.environ.get("QA_TEST_PATH", "data/samples/oqa_eval_sample.json")
RESULT_DIR = os.environ.get(
    "QA_RESULT_DIR",
    f"results/qa/{MODEL_NAME.replace('/', '_')}_no_rag",
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
MODE_NAME = f"{MODEL_NAME.replace('/', '_')}_qa_no_rag"
RESULT_PATH = f"{RESULT_DIR}/{MODE_NAME}_{timestamp}.json"
CKPT_PATH = f"{RESULT_DIR}/{MODE_NAME}_ckpt1.json"

MAX_RETRY = int(os.environ.get("QA_MAX_RETRY", "3"))
RETRY_SLEEP = int(os.environ.get("QA_RETRY_SLEEP", "3"))
SAVE_EVERY = int(os.environ.get("QA_SAVE_EVERY", "10"))
DEBUG_SAMPLES = int(os.environ.get("QA_DEBUG_SAMPLES", "3"))

TEMPERATURE = float(os.environ.get("QA_TEMPERATURE", "0"))
MAX_TOKENS = int(os.environ.get("QA_MAX_TOKENS", "1400"))

# Optional full card file, only used to check whether cited card_ids exist.
KNOWLEDGE_CARDS_PATH = os.environ.get("QA_KNOWLEDGE_CARDS_PATH") or None

USE_BERTSCORE = os.environ.get("QA_USE_BERTSCORE", "1") != "0"
BERTSCORE_MODEL_TYPE = os.environ.get(
    "QA_BERTSCORE_MODEL_TYPE",
    "hfl/chinese-roberta-wwm-ext",
)
BERTSCORE_NUM_LAYERS = int(os.environ.get("QA_BERTSCORE_NUM_LAYERS", "12"))
BERTSCORE_BATCH_SIZE = int(os.environ.get("QA_BERTSCORE_BATCH_SIZE", "8"))
BERTSCORE_MAX_TOKENS = int(os.environ.get("QA_BERTSCORE_MAX_TOKENS", "510"))

os.makedirs(RESULT_DIR, exist_ok=True)


# ============================================================
# Card id utilities
# ============================================================

# Keep the same card_id pattern as the RAG script, so references that contain
# card ids can still be cleaned before text-similarity metrics are computed.
CARD_ID_RE = re.compile(
    r"\b(?:fact_[a-zA-Z0-9]+_\d{3}_\d{3}|case_[a-zA-Z0-9]+_\d{3}_\d{3}|diag_manual_\d{3})\b"
)


def get_card_content(card: Dict) -> str:
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


# ============================================================
# Model API
# ============================================================

def query_model(prompt: str) -> str:
    if not API_KEY:
        raise RuntimeError("API key is empty. Please set GR_API_KEY or change API_KEY.")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    last_err = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一个严谨的藏医知识问答助手。"
                            "请直接回答问题，不要编造引用来源或 card_id。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            last_err = e
            if attempt < MAX_RETRY:
                time.sleep(RETRY_SLEEP * attempt)

    raise RuntimeError(f"model query failed after {MAX_RETRY} retries: {last_err}")


# ============================================================
# Prompt
# ============================================================

def build_no_rag_prompt(
    question: str,
    question_type: str,
) -> str:
    qtype = question_type or "未标注"

    return f"""你正在回答一个开放式藏医问答题。

要求：
1. 只能进行一次回答，不要输出检索计划、query、judge 或多轮控制标签。
2. 不提供检索证据，请直接根据模型自身知识作答。
3. 回答要完整、直接、可用于评分，不要只写一句泛泛结论。
4. 不要编造文献来源、检索证据或 card_id。
5. 如果确实不确定，请给出最稳妥的回答，并说明不确定之处。

问题类型：{qtype}

问题：
{question}
"""


def build_no_rag_citation_info(answer: str) -> Dict:
    """Keep citation-related keys for JSON compatibility, but do not score them.

    In the no-RAG setting, the model is not given allowed card_ids, so citation
    validity and retrieval-based citation metrics are not meaningful.
    """
    return {
        "cited_ids": extract_card_ids(answer),
        "citation_coverage": None,
        "citation_existence_rate": None,
        "citation_from_retrieval_rate": None,
        "citation_validity": None,
        "non_existing_cited_ids": [],
        "retrieved_cited_ids": [],
    }


# ============================================================
# Data and checkpoint
# ============================================================

def load_dataset(path: str) -> List[Dict]:
    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for key in ("data", "items", "samples"):
            if isinstance(data.get(key), list):
                return data[key]
        return list(data.values())

    return data


def get_sample_id(item: Dict, idx: int) -> str:
    for key in ("sample_id", "id", "question_num", "qid"):
        if item.get(key) is not None:
            return str(item[key])
    return str(idx + 1)


def get_question(item: Dict) -> str:
    return (
        item.get("question")
        or item.get("query")
        or item.get("prompt")
        or item.get("input")
        or ""
    )


def get_reference(item: Dict) -> str:
    return item.get("reference") or item.get("answer") or item.get("gold") or ""


def get_question_type(item: Dict) -> str:
    return item.get("type") or item.get("question_type") or item.get("category") or ""


def load_checkpoint() -> Tuple[List[Dict], Set[str]]:
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


def save_checkpoint(results: List[Dict], total: int):
    ckpt = {
        "timestamp": timestamp,
        "mode": MODE_NAME,
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "data_path": DATA_PATH,
        "progress": f"{len(results)}/{total}",
        "results": results,
    }
    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


# ============================================================
# Metrics
# ============================================================

def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：、“”‘’（）()\[\]{}<>《》,.!?;:\"'`~\-_/\\|]", "", text)
    return text


def tokenize_for_metric(text: str) -> List[str]:
    text = remove_card_ids_for_metric(text)
    try:
        import jieba
        return [x for x in jieba.lcut(text) if x.strip()]
    except Exception:
        return list(normalize_text(text))


def lcs_len(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0

    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                curr.append(prev[j - 1] + 1)
            else:
                curr.append(max(prev[j], curr[-1]))
        prev = curr
    return prev[-1]


def rouge_l(pred: str, ref: str) -> Optional[float]:
    p = tokenize_for_metric(pred)
    r = tokenize_for_metric(ref)
    if not p or not r:
        return None

    lcs = lcs_len(p, r)
    precision = lcs / len(p)
    recall = lcs / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4(pred: str, ref: str) -> Optional[float]:
    p = tokenize_for_metric(pred)
    r = tokenize_for_metric(ref)
    if not p or not r:
        return None

    precisions = []
    for n in range(1, 5):
        p_ngrams = ngrams(p, n)
        r_ngrams = ngrams(r, n)
        total = sum(p_ngrams.values())
        if total == 0:
            precisions.append(1e-9)
            continue
        overlap = sum(min(count, r_ngrams[gram]) for gram, count in p_ngrams.items())
        precisions.append(max(overlap / total, 1e-9))

    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / max(len(p), 1))
    return bp * math.exp(sum(math.log(x) for x in precisions) / 4)


def extract_card_ids(text: str) -> List[str]:
    ids = CARD_ID_RE.findall(text or "")
    result = []
    for cid in ids:
        if cid not in result:
            result.append(cid)
    return result


def remove_card_ids_for_metric(text: str) -> str:
    text = CARD_ID_RE.sub("", text or "")
    text = re.sub(r"引用来源[:：].*$", "", text, flags=re.DOTALL)
    return text.strip()


def get_gold_evidence_ids(item: Dict) -> Set[str]:
    ids = set()
    reference = get_reference(item)
    ids.update(extract_card_ids(reference))

    for key in ("citations", "gold_card_ids", "seed_card_ids"):
        vals = item.get(key, [])
        if isinstance(vals, list):
            ids.update(str(x) for x in vals if x)
        elif isinstance(vals, str):
            ids.update(extract_card_ids(vals))
            if CARD_ID_RE.fullmatch(vals):
                ids.add(vals)

    return ids


def compute_hit_recall_at_k(
    retrieved_ids: List[str],
    gold_ids: Set[str],
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    if not gold_ids:
        return None, None

    top_ids = set(retrieved_ids[:k])
    hit = 1.0 if top_ids & gold_ids else 0.0
    recall = len(top_ids & gold_ids) / len(gold_ids)
    return hit, recall


def load_knowledge_cards(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}

    if path.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.values() if isinstance(data, dict) else data

    card_map = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        cid = item.get("card_id")
        if cid:
            card_map[cid] = (
                item.get("content")
                or item.get("refined_result")
                or item.get("citation_text")
                or item.get("title")
                or ""
            )

    print(f"已加载知识库卡片: {len(card_map)} 条")
    return card_map


KNOWLEDGE_CARD_MAP = {}  # no-RAG setting: do not load knowledge cards by default


def compute_citation_validity(
    answer: str,
    retrieved_cards: List[Tuple[Dict, Dict]],
) -> Dict:
    cited_ids = extract_card_ids(answer)
    retrieved_map = {
        card.get("card_id", ""): get_card_content(card)
        for card, _ in retrieved_cards
        if card.get("card_id")
    }
    retrieved_ids = set(retrieved_map.keys())
    existence_map = KNOWLEDGE_CARD_MAP if KNOWLEDGE_CARD_MAP else retrieved_map

    if not cited_ids:
        return {
            "cited_ids": [],
            "citation_coverage": 0.0,
            "citation_existence_rate": None,
            "citation_from_retrieval_rate": None,
            "citation_validity": None,
            "non_existing_cited_ids": [],
            "retrieved_cited_ids": [],
        }

    existing_ids = [cid for cid in cited_ids if cid in existence_map]
    from_retrieval_ids = [cid for cid in cited_ids if cid in retrieved_ids]
    non_existing_ids = [cid for cid in cited_ids if cid not in existence_map]

    return {
        "cited_ids": cited_ids,
        "citation_coverage": 1.0,
        "citation_existence_rate": len(existing_ids) / len(cited_ids),
        "citation_from_retrieval_rate": len(from_retrieval_ids) / len(cited_ids),
        "citation_validity": 1.0 if len(non_existing_ids) == 0 and len(from_retrieval_ids) == len(cited_ids) else 0.0,
        "non_existing_cited_ids": non_existing_ids,
        "retrieved_cited_ids": from_retrieval_ids,
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

        for pred, ref in zip(preds, refs):
            pred = remove_card_ids_for_metric(pred)
            ref = remove_card_ids_for_metric(ref)

            pred_ids = bert_tokenizer.encode(pred, add_special_tokens=False)
            ref_ids = bert_tokenizer.encode(ref, add_special_tokens=False)

            if len(pred_ids) > BERTSCORE_MAX_TOKENS:
                pred_trunc_count += 1
            if len(ref_ids) > BERTSCORE_MAX_TOKENS:
                ref_trunc_count += 1

            clean_preds.append(token_truncate(pred, bert_tokenizer, BERTSCORE_MAX_TOKENS))
            clean_refs.append(token_truncate(ref, bert_tokenizer, BERTSCORE_MAX_TOKENS))

        print(f"BERTScore prediction 截断数量: {pred_trunc_count}")
        print(f"BERTScore reference 截断数量: {ref_trunc_count}")

        _, _, f1 = score(
            clean_preds,
            clean_refs,
            model_type=BERTSCORE_MODEL_TYPE,
            num_layers=BERTSCORE_NUM_LAYERS,
            verbose=True,
            rescale_with_baseline=False,
            batch_size=BERTSCORE_BATCH_SIZE,
        )

        return [float(x) for x in f1.cpu().tolist()], pred_trunc_count, ref_trunc_count
    except Exception as e:
        print(f"BERTScore 计算失败，跳过。error={e}")
        return [None] * len(preds), 0, 0


def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_results(results: List[Dict]) -> Dict:
    source_dist = Counter(r.get("answer_source", "unknown") for r in results)
    return {
        "total": len(results),
        "rouge_l": round(safe_mean([r.get("rouge_l") for r in results]) or 0.0, 4),
        "bleu4": round(safe_mean([r.get("bleu4") for r in results]) or 0.0, 4),
        "bertscore_f1": _round_or_none(safe_mean([r.get("bertscore_f1") for r in results])),
        "hit6": _round_or_none(safe_mean([r.get("hit6") for r in results])),
        "recall6": _round_or_none(safe_mean([r.get("recall6") for r in results])),
        "hit10": _round_or_none(safe_mean([r.get("hit10") for r in results])),
        "recall10": _round_or_none(safe_mean([r.get("recall10") for r in results])),
        "hit_all": _round_or_none(safe_mean([r.get("hit_all") for r in results])),
        "recall_all": _round_or_none(safe_mean([r.get("recall_all") for r in results])),
        "citation_coverage": _round_or_none(safe_mean([r.get("citation_coverage") for r in results])),
        "citation_existence_rate": _round_or_none(safe_mean([r.get("citation_existence_rate") for r in results])),
        "citation_from_retrieval_rate": _round_or_none(safe_mean([r.get("citation_from_retrieval_rate") for r in results])),
        "citation_validity": _round_or_none(safe_mean([r.get("citation_validity") for r in results])),
        "gold_citation_recall": _round_or_none(safe_mean([r.get("gold_citation_recall") for r in results])),
        "answer_source_dist": dict(source_dist),
    }


def _round_or_none(value: Optional[float]) -> Optional[float]:
    return round(value, 4) if value is not None else None


# ============================================================
# Main
# ============================================================

def main():
    dataset = load_dataset(DATA_PATH)
    print(f"QA 样本数: {len(dataset)}")
    print(f"Mode: {MODE_NAME}")
    print(f"Model: {MODEL_NAME}")
    print(f"Base URL: {BASE_URL}")
    print("RAG: disabled\n")

    all_results, done_ids = load_checkpoint()

    for idx, item in enumerate(tqdm(dataset, desc="QA no-RAG eval")):
        sample_id = get_sample_id(item, idx)
        if sample_id in done_ids:
            continue

        question = get_question(item)
        reference = get_reference(item)
        question_type = get_question_type(item)

        pred = ""
        answer_source = "no_rag"
        error = None

        try:
            prompt = build_no_rag_prompt(
                question=question,
                question_type=question_type,
            )
            pred = query_model(prompt).strip()
        except Exception as e:
            error = repr(e)
            answer_source = "exception"
            tqdm.write(f"\n[ERROR] sample_id={sample_id} failed: {error}")

        gold_ids = get_gold_evidence_ids(item)

        # No retrieval is performed. Keep these fields as None so the output
        # schema stays close to the RAG script without reporting misleading 0s.
        hit6, recall6 = None, None
        hit10, recall10 = None, None
        hit_all, recall_all = None, None
        gold_citation_recall = None
        citation_info = build_no_rag_citation_info(pred)

        row = {
            "sample_id": sample_id,
            "question": question,
            "question_type": question_type,
            "reference": reference,
            "prediction": pred,
            "answer_source": answer_source,
            "error": error,
            "retrieval_query": "",
            "retrieved_card_ids": [],
            "retrieved_cards": [],
            "gold_card_ids": sorted(gold_ids),
            "rouge_l": rouge_l(pred, reference),
            "bleu4": bleu4(pred, reference),
            "hit6": hit6,
            "recall6": recall6,
            "hit10": hit10,
            "recall10": recall10,
            "hit_all": hit_all,
            "recall_all": recall_all,
            "gold_citation_recall": gold_citation_recall,
            **citation_info,
        }
        all_results.append(row)

        if len(all_results) <= DEBUG_SAMPLES:
            tqdm.write("\n" + "=" * 80)
            tqdm.write(f"sample_id={sample_id}")
            tqdm.write(f"Q: {question[:160]}")
            tqdm.write(f"Pred: {pred[:300]}")
            tqdm.write(f"Ref: {reference[:300]}")
            tqdm.write("RAG: disabled")
            tqdm.write(f"ROUGE-L={row['rouge_l']} BLEU4={row['bleu4']}")

        if len(all_results) % SAVE_EVERY == 0:
            save_checkpoint(all_results, len(dataset))
            tqdm.write(f"[断点] 已保存 {len(all_results)}/{len(dataset)}")

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    print("\n开始计算 BERTScore F1...")
    bertscore_f1, pred_trunc_count, ref_trunc_count = compute_bertscore_batch(
        [r.get("prediction", "") for r in all_results],
        [r.get("reference", "") for r in all_results],
    )
    for row, score_value in zip(all_results, bertscore_f1):
        row["bertscore_f1"] = score_value

    summary = summarize_results(all_results)

    output = {
        "mode": MODE_NAME,
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "data_path": DATA_PATH,
        "timestamp": timestamp,
        "config": {
            "single_rag": False,
            "no_rag": True,
            "retrieve_top_k": None,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "use_bertscore": USE_BERTSCORE,
            "bertscore_model_type": BERTSCORE_MODEL_TYPE if USE_BERTSCORE else None,
            "bertscore_num_layers": BERTSCORE_NUM_LAYERS if USE_BERTSCORE else None,
            "bertscore_pred_trunc_count": pred_trunc_count,
            "bertscore_ref_trunc_count": ref_trunc_count,
        },
        "summary": summary,
        "results": all_results,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n结果已保存到: {RESULT_PATH}")


if __name__ == "__main__":
    main()
