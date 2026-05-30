# Evaluation

This directory keeps the evaluation code needed to reproduce the paper
experiments. Historical trial scripts and timestamped intermediate outputs are
left outside this cleaned list unless they correspond directly to a reported
paper table.

## MCQ

- `mcq/timedlm_multi_rag_mcq.py`: TiMedLM multi-round RAG evaluation.
- `mcq/qwen8b_no_rag_mcq.py`: Qwen3-8B MCQ baseline without RAG.
- `mcq/qwen8b_single_rag_mcq.py`: Qwen3-8B MCQ baseline with single-round RAG.
- `mcq/closed_model_single_rag_mcq.py`: closed/open model MCQ single-RAG baseline.
- `mcq/closed_model_multi_rag_mcq.py`: closed/open model MCQ multi-RAG baseline.
- `mcq/mcq_retrieval_ablation.py`: retrieval ablation for MCQ.

## QA

- `qa/timedlm_multi_rag_qa.py`: TiMedLM multi-round RAG QA evaluation.
- `qa/qwen8b_no_rag_qa.py`: Qwen3-8B QA baseline without RAG.
- `qa/qwen8b_single_rag_qa.py`: Qwen3-8B QA baseline with single-round RAG.
- `qa/closed_model_no_rag_qa.py`: closed/open model QA baseline without RAG.
- `qa/closed_model_single_rag_qa.py`: closed/open model QA single-RAG baseline.

## IFEval

- `ifeval/timedlm_ifeval.py`: TiMedLM IFEval evaluation.

## Shared Code

- `retrieval.py`: shared retrieval implementation used by RAG evaluations.

## Notes

- Set local paths such as model, data, output, and cache locations before
  running the scripts. Several scripts still keep the original local defaults
  used for the paper experiments.
- Set API keys through environment variables such as `GR_API_KEY`; no API key
  should be committed in these evaluation scripts.
- Keep only final result files that map to reported paper tables. Do not add
  debug checkpoints, old timestamped runs, or training checkpoints here.
