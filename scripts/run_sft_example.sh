#!/usr/bin/env bash
set -euo pipefail

python src/timedlm/training/sft_train.py \
  --model_path "${TIMEDLM_BASE_MODEL_PATH:-Qwen/Qwen3-8B}" \
  --data_path "${TIMEDLM_SFT_DATA_PATH:-data/samples/sft_trajectory_sample.json}" \
  --output_dir "${TIMEDLM_SFT_OUTPUT_DIR:-outputs/sft}"
