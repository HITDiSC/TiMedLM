# -*- coding: utf-8 -*-

import json
import os
import pickle
from typing import List

import jieba
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

KB_DIR = os.environ.get("TIMEDLM_KB_DIR", "data/atomic_cards")
EMBEDDING_MODEL_PATH = os.environ.get("TIMEDLM_EMBEDDING_MODEL", "BAAI/bge-m3")
EMB_CACHE = os.environ.get("TIMEDLM_EMB_CACHE", "cache/atoms_all_bge.pkl")
BM25_CACHE = os.environ.get("TIMEDLM_BM25_CACHE", "cache/atoms_bm25.pkl")

CARD_FILES_IN_ORDER = [
    "atoms_lll.jsonl",
    "atoms_sbyd.jsonl",
    "atoms_ywyz.jsonl",
    "atoms_jzbc.jsonl",
    "atoms_zyyx.jsonl",
    "diag_manual.jsonl",
]

STOPWORDS = set()
TERM_MAP = {}


def _read_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_all_cards() -> List[dict]:
    cards = []
    for fname in CARD_FILES_IN_ORDER:
        fpath = os.path.join(KB_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing card file: {fpath}")
        cards.extend(_read_jsonl(fpath))
    return cards


def _load_all_embeddings() -> np.ndarray:
    if not os.path.exists(EMB_CACHE):
        raise FileNotFoundError(
            f"Missing embedding file: {EMB_CACHE}\nPlease run build_embeddings.py first."
        )
    with open(EMB_CACHE, "rb") as f:
        arr = pickle.load(f)
    return np.asarray(arr, dtype=np.float32)


def tokenize(text: str) -> List[str]:
    """Tokenize text with the same logic used by build_embeddings.py."""
    return [
        tok for tok in jieba.cut(text)
        if tok.strip() and tok not in STOPWORDS and len(tok) > 1
    ]


def normalize_query(query: str) -> str:
    """Normalize query text and apply optional term mappings."""
    query = (query or "").strip()
    for key, value in TERM_MAP.items():
        query = query.replace(key, value)
    return query


def build_text(card: dict) -> str:
    """Build the text representation used for BM25 indexing."""
    re_info = card.get("retrieval_enhancement", {})
    sf = card.get("structured_fields", {})
    parts = [
        card.get("title", ""),
        card.get("content", ""),
        " ".join(re_info.get("keywords", [])),
        " ".join(sf.get("attribute_value", [])),
    ]
    return " ".join(filter(None, parts))


print("Loading BGE-M3 model...")
_bge_model = BGEM3FlagModel(EMBEDDING_MODEL_PATH, use_fp16=True)
print("BGE-M3 model loaded.")

ALL_CARDS = _load_all_cards()
ALL_EMBEDDINGS = _load_all_embeddings()
if ALL_EMBEDDINGS.shape[0] != len(ALL_CARDS):
    raise ValueError(
        f"Embedding count {ALL_EMBEDDINGS.shape[0]} does not match "
        f"card count {len(ALL_CARDS)}. Check JSONL order and embedding cache."
    )

print(f"Loaded {len(ALL_CARDS)} cards with embedding dimension {ALL_EMBEDDINGS.shape[1]}.")


def _load_or_build_bm25(cards: List[dict]) -> BM25Okapi:
    if os.path.exists(BM25_CACHE):
        with open(BM25_CACHE, "rb") as f:
            data = pickle.load(f)
        return data["bm25"] if isinstance(data, dict) else data

    print("BM25 cache not found; rebuilding index...")
    os.makedirs(os.path.dirname(BM25_CACHE), exist_ok=True)
    texts = [build_text(c) for c in cards]
    tokenized = [tokenize(t) for t in texts]
    index = BM25Okapi(tokenized)
    with open(BM25_CACHE, "wb") as f:
        pickle.dump({"bm25": index, "tokenized_corpus": tokenized}, f)
    print(f"BM25 saved to: {BM25_CACHE}")
    return index


BM25_INDEX = _load_or_build_bm25(ALL_CARDS)


def _get_query_embedding(query: str) -> np.ndarray:
    """Encode and L2-normalize a query."""
    result = _bge_model.encode([query], batch_size=1, max_length=512)
    vec = np.asarray(result["dense_vecs"][0], dtype=np.float32)
    return vec / (np.linalg.norm(vec) + 1e-9)


def bm25_search(query: str, top_k: int = 50) -> List[tuple]:
    tokens = tokenize(normalize_query(query))
    scores = BM25_INDEX.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(ALL_CARDS[i], float(scores[i])) for i in top_idx if scores[i] > 0]


def dense_search(query: str, top_k: int = 50) -> List[tuple]:
    query_emb = _get_query_embedding(normalize_query(query))
    scores = ALL_EMBEDDINGS @ query_emb
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [(ALL_CARDS[i], float(scores[i])) for i in top_idx]


def fusion_score(
    bm25_results: List[tuple],
    dense_results: List[tuple],
    bm25_weight: float = 0.45,
    dense_weight: float = 0.45,
    both_bonus: float = 0.10,
) -> List[tuple]:
    id_to_card = {c.get("card_id"): c for c in ALL_CARDS}
    bm25_map = {c.get("card_id"): s for c, s in bm25_results}
    dense_map = {c.get("card_id"): s for c, s in dense_results}

    bm25_max = max(bm25_map.values(), default=1e-9)
    dense_max = max(dense_map.values(), default=1e-9)

    all_ids = set(bm25_map) | set(dense_map)
    scored = []
    for cid in all_ids:
        b = bm25_map.get(cid, 0.0) / bm25_max
        d = dense_map.get(cid, 0.0) / dense_max
        bonus = both_bonus if (cid in bm25_map and cid in dense_map) else 0.0
        score = bm25_weight * b + dense_weight * d + bonus
        if cid in id_to_card:
            scored.append((id_to_card[cid], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:100]


def diversify(scored_cards: List[tuple], top_k: int = 6) -> List[dict]:
    """Diversify retrieved cards by card type, then fill by fusion score."""
    buckets: dict = {"fact": [], "case": [], "diagnosis": [], "other": []}
    for card, score in scored_cards:
        ct = card.get("card_type", "other")
        key = ct if ct in buckets else "other"
        buckets[key].append((card, score))

    result = []
    seen_ids = set()

    for key in ["diagnosis", "fact", "case", "other"]:
        for card, _ in buckets[key]:
            cid = card.get("card_id")
            if cid not in seen_ids:
                result.append(card)
                seen_ids.add(cid)
                break

    for card, _ in scored_cards:
        if len(result) >= top_k:
            break
        cid = card.get("card_id")
        if cid not in seen_ids:
            result.append(card)
            seen_ids.add(cid)

    return result[:top_k]


def retrieve(query: str, top_k: int = 6) -> List[dict]:
    bm25_results = bm25_search(query, top_k=50)
    dense_results = dense_search(query, top_k=50)
    scored_cards = fusion_score(bm25_results, dense_results)
    return diversify(scored_cards, top_k=top_k)


def retrieve_with_scores(query: str, top_k: int = 6) -> List[tuple]:
    """Return [(card, fusion_score), ...] for callers that need scores."""
    bm25_results = bm25_search(query, top_k=50)
    dense_results = dense_search(query, top_k=50)
    scored_cards = fusion_score(bm25_results, dense_results)

    buckets: dict = {"fact": [], "case": [], "diagnosis": [], "other": []}
    for card, score in scored_cards:
        ct = card.get("card_type", "other")
        key = ct if ct in buckets else "other"
        buckets[key].append((card, score))

    result = []
    seen_ids = set()

    for key in ["diagnosis", "fact", "case", "other"]:
        for card, score in buckets[key]:
            cid = card.get("card_id")
            if cid not in seen_ids:
                result.append((card, score))
                seen_ids.add(cid)
                break

    for card, score in scored_cards:
        if len(result) >= top_k:
            break
        cid = card.get("card_id")
        if cid not in seen_ids:
            result.append((card, score))
            seen_ids.add(cid)

    return result[:top_k]


__all__ = ["retrieve", "retrieve_with_scores"]
