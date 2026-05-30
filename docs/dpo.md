# DPO Data and Training

This project uses DPO preference data for RAG decision alignment. The full DPO dataset is not released in this GitHub repository.

## Files

- `construction/dpo/build_dpo_rag_decision.py`: script for rebuilding DPO preference pairs from intermediate QA and MCQ trajectory files.
- `src/timedlm/training/train_dpo.py`: DPO training entry point.

## Data Format

Each JSONL row should contain:

```json
{
  "prompt": "...",
  "chosen": "...",
  "rejected": "...",
  "type": "...",
  "meta": {},
  "id": "..."
}
```

The DPO data used in the paper is not committed because it may contain derived evaluation and trajectory content. To reproduce training with your own data, prepare a compatible JSONL file and pass it with `--dpo_data`.

## Training

```bash
python src/timedlm/training/train_dpo.py \
  --base_model /path/to/Qwen3-8B \
  --sft_lora /path/to/sft-lora \
  --dpo_data /path/to/dpo_rag_decision.jsonl \
  --output_dir outputs/qwen3-8b-tibetan-dpo
```

Default hyperparameters:

- `max_length`: 1536
- `max_prompt_length`: 1024
- `beta`: 0.03
- `learning_rate`: 2e-6
- `epochs`: 1
- `gradient_accumulation_steps`: 8

## Rebuilding Data

If you have the intermediate source files, place them under:

```text
data/interim/dpo_sources/
  qa_train_dedup_selected.jsonl
  mcq_dedup_selected_v2.jsonl
```

Then run:

```bash
python construction/dpo/build_dpo_rag_decision.py
```

The rebuilt dataset and report will be written to `data/dpo/`. This output directory is ignored by Git and should not be committed unless you intentionally release the data.
