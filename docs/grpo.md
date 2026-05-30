# GRPO Data and Training

This directory contains the GRPO training code and data construction entry point used for the TiMedLM five-step retrieval reasoning stage.

## Files

- `src/timedlm/training/grpo/train_mcq_grpo_fivestep.py`: MCQ-GRPO training entry point.
- `src/timedlm/training/grpo/mcq_fivestep_rollout.py`: five-step MCQ rollout and reward logic used by MCQ-GRPO.
- `src/timedlm/training/grpo/train_qa_grpo_after_mcq.py`: QA-GRPO training entry point initialized from the MCQ-GRPO LoRA.
- `construction/grpo/build_grpo_data.py`: rebuilds GRPO training files from local MCQ and QA source pools.
- `data/samples/grpo_mcq_sample.jsonl`: small MCQ-GRPO format sample.
- `data/samples/grpo_qa_sample.jsonl`: small QA-GRPO format sample.

The full GRPO training data is not stored in this repository. Rebuild it from local source pools before training.

## External Dependencies

The GRPO rollout code expects a retrieval module that exposes:

```python
retrieve_with_scores(query: str, top_k: int) -> list[dict]
```

The retrieval implementation and evaluation scripts are intentionally not included in this GRPO release folder.

## Training

First rebuild the local GRPO data:

```bash
python construction/grpo/build_grpo_data.py \
  --mcq_source /path/to/mcq_source_pool.jsonl \
  --qa_source /path/to/qa_source_pool.jsonl \
  --out_dir data/grpo
```

Then set the base model path with `TIMEDLM_BASE_MODEL`. For QA-GRPO, set `TIMEDLM_MCQ_GRPO_LORA` to the MCQ-GRPO LoRA checkpoint if it is not stored at the default output path.

```bash
TIMEDLM_BASE_MODEL=/path/to/qwen3-8b-sft-merged \
python src/timedlm/training/grpo/train_mcq_grpo_fivestep.py
```

```bash
TIMEDLM_BASE_MODEL=/path/to/qwen3-8b-sft-merged \
TIMEDLM_MCQ_GRPO_LORA=outputs/grpo/qwen3-8b-mcq-grpo-lora \
python src/timedlm/training/grpo/train_qa_grpo_after_mcq.py
```

Default outputs are written under `outputs/grpo/`, which is ignored by git.
