# -*- coding: utf-8 -*-
# This file builds GRPO training data from retrieval and answer trajectories.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
"""Build TiMedLM GRPO training files from local source pools.

The repository keeps only small samples. Use this script with your private
source JSONL files to reconstruct the MCQ-GRPO and QA-GRPO training sets.
"""

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_OUT_DIR = Path("data/grpo")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"JSON parse failed at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def score(row: dict) -> float:
    value = row.get("target_score")
    if isinstance(value, (int, float)):
        return float(value)

    reason = row.get("target_reason") or {}
    fact_count = len(row.get("gold_fact_ids") or row.get("silver_fact_ids") or [])
    query_count = len(row.get("source_queries") or [])
    round_count = int(row.get("rounds_min") or reason.get("rounds_min") or 1)
    return float(fact_count * 2 + query_count * 3 + round_count)


def build_subset(rows: list[dict], limit: int | None, min_score: float | None) -> list[dict]:
    if min_score is not None:
        rows = [row for row in rows if score(row) >= min_score]
    rows = sorted(rows, key=lambda row: (-score(row), str(row.get("question_id", ""))))
    return rows[:limit] if limit else rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TiMedLM GRPO data from local source pools.")
    parser.add_argument("--mcq_source", type=Path, help="MCQ source JSONL with question/options/gold facts.")
    parser.add_argument("--qa_source", type=Path, help="QA source JSONL with question/reference/gold facts.")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mcq_limit", type=int, default=280)
    parser.add_argument("--mcq_smoke_limit", type=int, default=50)
    parser.add_argument("--qa_limit", type=int, default=50)
    parser.add_argument("--mcq_min_score", type=float, default=None)
    parser.add_argument("--qa_min_score", type=float, default=None)
    args = parser.parse_args()

    report = {}

    if args.mcq_source:
        mcq_rows = load_jsonl(args.mcq_source)
        mcq_subset = build_subset(mcq_rows, args.mcq_limit, args.mcq_min_score)
        mcq_smoke_subset = mcq_subset[: args.mcq_smoke_limit]
        mcq_path = args.out_dir / "mcq" / "mcq_grpo_train_prompts_targeted280.jsonl"
        mcq_smoke_path = args.out_dir / "mcq" / "mcq_grpo_train_prompts_targeted50.jsonl"
        write_jsonl(mcq_subset, mcq_path)
        write_jsonl(mcq_smoke_subset, mcq_smoke_path)
        report["mcq"] = {
            "source_rows": len(mcq_rows),
            "output_rows": len(mcq_subset),
            "output": str(mcq_path),
            "smoke_output_rows": len(mcq_smoke_subset),
            "smoke_output": str(mcq_smoke_path),
        }

    if args.qa_source:
        qa_rows = load_jsonl(args.qa_source)
        qa_subset = build_subset(qa_rows, args.qa_limit, args.qa_min_score)
        qa_path = args.out_dir / "qa" / "qa_grpo_train_targeted50.jsonl"
        write_jsonl(qa_subset, qa_path)
        report["qa"] = {"source_rows": len(qa_rows), "output_rows": len(qa_subset), "output": str(qa_path)}

    report_path = args.out_dir / "grpo_data_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
