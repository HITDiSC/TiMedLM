# TiMedLM

This repository contains the code release for TiMedLM, including knowledge-base construction, instruction-data construction, retrieval, training, and evaluation scripts for Tibetan medicine language-model experiments.

The repository only includes source code and synthetic sample files. Full training data, evaluation data, source books, generated trajectories, DPO data, model checkpoints, LoRA adapters, vector indexes, and experiment outputs are intentionally excluded from GitHub.

## Quick Links

- [Overview](#overview)
- [Directory](#directory)
- [Requirements](#requirements)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Experiments](#experiments)
- [Data and Model Release](#data-and-model-release)
- [Citation](#citation)
- [License](#license)

## Overview

TiMedLM uses a retrieval-augmented training and evaluation pipeline:

1. Extract atomic knowledge cards from source material.
2. Build retrieval indexes over chunk-level or card-level knowledge.
3. Construct SFT, DPO, and GRPO training examples.
4. Train the base model and alignment adapters.
5. Evaluate MCQ, open-ended QA, IFEval, and retrieval-ablation settings.

## Directory

- `construction/knowledge_base/`: extract atomic knowledge cards from source books.
- `construction/sft/`: build SFT data from MCQ, open-ended QA, and general instruction data.
- `construction/dpo/`: build DPO preference data for RAG decision learning.
- `construction/grpo/`: build GRPO training data.
- `src/timedlm/retrieval/`: build chunk/card embedding indexes and run hybrid retrieval.
- `src/timedlm/training/`: SFT, DPO, and GRPO training scripts.
- `evaluation/mcq/`: multiple-choice evaluation scripts.
- `evaluation/qa/`: open-ended QA evaluation scripts.
- `evaluation/ifeval/`: IFEval instruction-following evaluation script.
- `data/samples/`: synthetic schema examples.
- `prompts/`: prompts used for atomic-card extraction.
- `docs/`: data-format and training-stage notes.
- `scripts/`: runnable example commands.

## Requirements

Create an environment and install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies include `torch`, `transformers`, `peft`, `trl`, `datasets`, `FlagEmbedding`, `rank-bm25`, `openai`, `bert-score`, and common scientific Python packages.

## Data Preparation

### Knowledge Cards

Prepare source books under `data/books/` locally. This directory is ignored by Git.

```bash
python construction/knowledge_base/extract_atomic_cards.py \
  --books_dir data/books \
  --output_dir data/atomic_cards \
  --prompt_path prompts/atomic_card_extraction_prompt.json
```

If you use DashScope-compatible APIs for extraction, set:

```bash
export DASHSCOPE_API_KEY=your_api_key
```

### Retrieval Index

Build card embeddings and BM25 cache:

```bash
python src/timedlm/retrieval/build_embeddings.py \
  --cards_dir data/atomic_cards \
  --embedding_model BAAI/bge-m3 \
  --embedding_cache cache/atoms_all_bge.pkl \
  --bm25_cache cache/atoms_bm25.pkl
```

### Training Data

SFT data can be constructed from MCQ, open-ended QA, and general instruction sources:

```bash
python construction/sft/build_sft_mcq_data.py
python construction/sft/build_sft_oqa_data.py
python construction/sft/build_general_instruction_data.py
python construction/sft/merge_sft_data.py
```

DPO data is not released in this repository. To rebuild compatible DPO data from your own intermediate trajectories:

```bash
python construction/dpo/build_dpo_rag_decision.py
```

GRPO data construction:

```bash
python construction/grpo/build_grpo_data.py
```

## Training

### SFT

```bash
python src/timedlm/training/sft_train.py \
  --model_path Qwen/Qwen3-8B \
  --data_path data/samples/sft_trajectory_sample.json \
  --output_dir outputs/sft
```

### DPO

```bash
python src/timedlm/training/train_dpo.py \
  --base_model /path/to/Qwen3-8B \
  --sft_lora /path/to/sft-lora \
  --dpo_data /path/to/dpo_rag_decision.jsonl \
  --output_dir outputs/dpo
```

### GRPO

```bash
python src/timedlm/training/grpo/train_mcq_grpo_fivestep.py
python src/timedlm/training/grpo/train_qa_grpo_after_mcq.py
```

## Experiments

The scripts below correspond to the major evaluation settings used in the paper. Replace sample paths with the private or released evaluation sets before reproducing reported numbers.

| Setting | Script |
|---|---|
| MCQ no-RAG baseline | `evaluation/mcq/qwen8b_no_rag_mcq.py` |
| MCQ single-RAG baseline | `evaluation/mcq/qwen8b_single_rag_mcq.py` |
| MCQ TiMedLM multi-RAG | `evaluation/mcq/timedlm_multi_rag_mcq.py` |
| MCQ retrieval ablation | `evaluation/mcq/mcq_retrieval_ablation.py` |
| Closed-model MCQ single-RAG | `evaluation/mcq/closed_model_single_rag_mcq.py` |
| Closed-model MCQ multi-RAG | `evaluation/mcq/closed_model_multi_rag_mcq.py` |
| QA no-RAG baseline | `evaluation/qa/qwen8b_no_rag_qa.py` |
| QA single-RAG baseline | `evaluation/qa/qwen8b_single_rag_qa.py` |
| QA TiMedLM multi-RAG | `evaluation/qa/timedlm_multi_rag_qa.py` |
| Closed-model QA no-RAG | `evaluation/qa/closed_model_no_rag_qa.py` |
| Closed-model QA single-RAG | `evaluation/qa/closed_model_single_rag_qa.py` |
| IFEval | `evaluation/ifeval/timedlm_ifeval.py` |

Example MCQ multi-RAG evaluation:

```bash
TIMEDLM_MODEL_PATH=models/timedlm-sft \
TIMEDLM_LORA_PATH=models/timedlm-lora \
TIMEDLM_MCQ_TEST_PATH=data/samples/mcq_eval_sample.json \
python evaluation/mcq/timedlm_multi_rag_mcq.py
```

Example QA multi-RAG evaluation:

```bash
TIMEDLM_MODEL_PATH=models/timedlm-sft \
TIMEDLM_LORA_PATH=models/timedlm-lora \
TIMEDLM_QA_TEST_PATH=data/samples/oqa_eval_sample.json \
python evaluation/qa/timedlm_multi_rag_qa.py
```

## Common Environment Variables

- `TIMEDLM_MODEL_PATH`: path to the TiMedLM base or merged model.
- `TIMEDLM_LORA_PATH`: path to LoRA adapter weights.
- `TIMEDLM_BASE_MODEL_PATH`: path or Hugging Face id for the base Qwen model.
- `TIMEDLM_MCQ_TEST_PATH`: MCQ evaluation set path.
- `TIMEDLM_QA_TEST_PATH`: open-ended QA evaluation set path.
- `TIMEDLM_MCQ_RESULT_DIR`: MCQ result directory.
- `TIMEDLM_QA_RESULT_DIR`: QA result directory.
- `TIMEDLM_KB_DIR`: knowledge-card directory used by retrieval.
- `TIMEDLM_EMBEDDING_MODEL`: embedding model path or id, default `BAAI/bge-m3`.
- `DASHSCOPE_API_KEY`: API key used for atomic-card extraction.

## Data and Model Release

Recommended external storage:

- model weights and LoRA adapters: Hugging Face model repository
- released datasets or sampled datasets: Hugging Face dataset repository
- copyrighted exam/book material: do not commit to GitHub; document the source and access conditions instead

GitHub excludes generated outputs, checkpoints, caches, source books, DPO data, and intermediate data through `.gitignore`.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@inproceedings{timedlm2026,
  title     = {TiMedLM: Retrieval-Augmented Language Modeling for Tibetan Medicine},
  author    = {TiMedLM Contributors},
  booktitle = {Proceedings of the ACM Hypertext Conference},
  year      = {2026}
}
```

Please update the citation metadata after the paper is formally published.

## License

The code is released under the MIT License. Data and model weights may have separate licenses depending on their source and release location.
