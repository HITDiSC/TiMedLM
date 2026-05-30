#!/usr/bin/env bash
set -euo pipefail

python src/timedlm/retrieval/build_embeddings.py \
  --cards_dir "${TIMEDLM_KB_DIR:-data/atomic_cards}" \
  --embedding_model "${TIMEDLM_EMBEDDING_MODEL:-BAAI/bge-m3}" \
  --embedding_cache "${TIMEDLM_EMB_CACHE:-cache/atoms_all_bge.pkl}" \
  --bm25_cache "${TIMEDLM_BM25_CACHE:-cache/atoms_bm25.pkl}"
