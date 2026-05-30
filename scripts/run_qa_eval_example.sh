#!/usr/bin/env bash
set -euo pipefail

TIMEDLM_QA_TEST_PATH="${TIMEDLM_QA_TEST_PATH:-data/samples/oqa_eval_sample.json}" \
TIMEDLM_QA_RESULT_DIR="${TIMEDLM_QA_RESULT_DIR:-results/qa/timedlm_multi_rag}" \
python evaluation/qa/timedlm_multi_rag_qa.py
