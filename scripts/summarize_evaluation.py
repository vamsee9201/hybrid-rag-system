#!/usr/bin/env python3
"""Summarize answer metrics, judge agreement, and paired bootstrap intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random
import re


TOKEN = re.compile(r"\w+")
MODES = ["bm25", "dense", "hybrid"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def token_f1(answer: str, reference: str) -> float:
    a = TOKEN.findall(answer.casefold())
    b = TOKEN.findall(reference.casefold())
    overlap = sum((min(a.count(token), b.count(token)) for token in set(a)))
    if not a or not b or not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(b)
    return 2 * precision * recall / (precision + recall)


def bootstrap_difference(left: dict[str, float], right: dict[str, float], samples: int = 10_000):
    ids = sorted(set(left) & set(right))
    rng = random.Random(20260914)
    values = []
    for _ in range(samples):
        sampled = [rng.choice(ids) for _ in ids]
        values.append(sum(left[item] - right[item] for item in sampled) / len(sampled))
    values.sort()
    observed = sum(left[item] - right[item] for item in ids) / len(ids)
    tail = min(sum(value <= 0 for value in values), sum(value >= 0 for value in values))
    return {"difference": observed, "ci95": [values[249], values[9749]],
            "p_value_two_sided": min(1.0, 2 * tail / samples)}


def holm_adjust(comparisons: dict[str, dict]) -> None:
    ordered = sorted(comparisons, key=lambda key: comparisons[key]["p_value_two_sided"])
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * comparisons[key]["p_value_two_sided"])
        running = max(running, adjusted)
        comparisons[key]["p_value_holm"] = running


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/generated/summary.json"))
    parser.add_argument("--human-review", type=Path, default=Path("results/generated/human-review.csv"))
    args = parser.parse_args()
    questions = {row["question_id"]: row for path in args.questions for row in read_jsonl(path)}
    answers = read_jsonl(args.answers)
    judgments = read_jsonl(args.judgments)
    judge_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in judgments:
        judge_scores[(row["question_id"], row["mode"])].append(
            (row["correctness"] + row["groundedness"] + row["completeness"]) / 6
        )
    per_mode: dict[str, list[dict]] = defaultdict(list)
    for answer in answers:
        if answer.get("status") != "ok":
            continue
        question = questions[answer["question_id"]]
        scores = judge_scores[(answer["question_id"], answer["mode"])]
        per_mode[answer["mode"]].append({
            "question_id": answer["question_id"],
            "judge_score": sum(scores) / len(scores),
            "token_f1": token_f1(answer["answer"], question.get("reference_answer", "Insufficient evidence.")),
            "abstention_correct": ("insufficient evidence" in answer["answer"].casefold()) == (not question.get("answerable", True)),
            "citation_valid": not answer.get("invalid_citations"),
            "cost": answer.get("estimated_cost_usd", 0),
            "generation_ms": answer.get("generation_ms", 0),
        })
    summary = {"modes": {}, "by_category": {}, "paired": {}, "judge_stability": {}}
    for mode, rows in per_mode.items():
        summary["modes"][mode] = {
            "answers": len(rows),
            "judge_score": sum(row["judge_score"] for row in rows) / len(rows),
            "token_f1": sum(row["token_f1"] for row in rows) / len(rows),
            "abstention_accuracy": sum(row["abstention_correct"] for row in rows) / len(rows),
            "valid_citation_rate": sum(row["citation_valid"] for row in rows) / len(rows),
            "mean_generation_ms": sum(row["generation_ms"] for row in rows) / len(rows),
            "total_cost_usd": sum(row["cost"] for row in rows),
        }
    score_maps = {mode: {row["question_id"]: row["judge_score"] for row in rows} for mode, rows in per_mode.items()}
    for left, right in (("hybrid", "bm25"), ("hybrid", "dense"), ("dense", "bm25")):
        summary["paired"][f"{left}_minus_{right}"] = bootstrap_difference(score_maps[left], score_maps[right])
    holm_adjust(summary["paired"])
    for category in sorted({questions[row["question_id"]]["category"] for rows in per_mode.values() for row in rows}):
        summary["by_category"][category] = {}
        for mode, rows in per_mode.items():
            selected = [row for row in rows if questions[row["question_id"]]["category"] == category]
            summary["by_category"][category][mode] = {
                "answers": len(selected),
                "judge_score": sum(row["judge_score"] for row in selected) / len(selected),
                "token_f1": sum(row["token_f1"] for row in selected) / len(selected),
            }
    for mode in MODES:
        pairs = [scores for (question_id, answer_mode), scores in judge_scores.items()
                 if answer_mode == mode and len(scores) == 2]
        summary["judge_stability"][mode] = {
            "pairs": len(pairs),
            "exact_agreement": sum(pair[0] == pair[1] for pair in pairs) / len(pairs),
            "mean_absolute_difference": sum(abs(pair[0] - pair[1]) for pair in pairs) / len(pairs),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")

    chosen_ids = random.Random(20260914).sample(sorted(questions), 20)
    answer_map = {(row["question_id"], row["mode"]): row.get("answer", "") for row in answers}
    with args.human_review.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["question_id", "question", "reference", "answer_a", "answer_b", "answer_c",
                         "label_key", "best_answer", "notes"])
        for question_id in chosen_ids:
            question = questions[question_id]
            labels = list(MODES)
            random.Random(f"20260914:{question_id}").shuffle(labels)
            writer.writerow([question_id, question["question"], question.get("reference_answer", ""),
                             *(answer_map.get((question_id, mode), "") for mode in labels),
                             f"A={labels[0]};B={labels[1]};C={labels[2]}", "", ""])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
