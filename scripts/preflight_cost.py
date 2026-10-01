#!/usr/bin/env python3
"""Conservatively estimate embedding, evaluation, storage, and serving costs."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/indexes/chunks.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("reports/preflight-cost.json"))
    parser.add_argument("--embedding-batch-price-per-million", type=float, default=0.12)
    parser.add_argument("--generation-input-price-per-million", type=float, default=0.75)
    parser.add_argument("--generation-output-price-per-million", type=float, default=3.75)
    parser.add_argument("--one-time-gate", type=float, default=25.0)
    parser.add_argument("--monthly-gate", type=float, default=20.0)
    args = parser.parse_args()

    db = sqlite3.connect(args.index)
    try:
        chunks, characters = db.execute("SELECT COUNT(*), SUM(length(text)) FROM chunks").fetchone()
    finally:
        db.close()
    # One token per three characters plus 15% safety is deliberately conservative for English.
    conservative_tokens = int((characters / 3) * 1.15)
    embedding_cost = conservative_tokens * args.embedding_batch_price_per_million / 1_000_000
    evaluation_answers = 300
    evaluation_input_tokens = evaluation_answers * 3_200
    evaluation_output_tokens = evaluation_answers * 300
    answer_cost = (
        evaluation_input_tokens * args.generation_input_price_per_million
        + evaluation_output_tokens * args.generation_output_price_per_million
    ) / 1_000_000
    judge_allowance = 5.0
    storage_allowance = 1.0
    total = embedding_cost + answer_cost + judge_allowance + storage_allowance
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "basis": "conservative preflight; replace with measured token counts after successful batch",
        "chunks": chunks,
        "characters": characters,
        "conservative_embedding_tokens": conservative_tokens,
        "estimated": {
            "embedding_batch_usd": round(embedding_cost, 2),
            "benchmark_answers_usd": round(answer_cost, 2),
            "judge_allowance_usd": judge_allowance,
            "storage_and_jobs_allowance_usd": storage_allowance,
            "one_time_total_usd": round(total, 2),
            "public_demo_target_usd_per_month": 15.0,
        },
        "gates": {"one_time_usd": args.one_time_gate, "monthly_usd": args.monthly_gate},
        "approval_required": total > args.one_time_gate or 15.0 > args.monthly_gate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["approval_required"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
