#!/usr/bin/env python3
"""Run and score the three retrievers on a frozen benchmark."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path

from google import genai
from google.genai import types
from google.oauth2 import service_account
import numpy as np

from app.models import RagMode
from app.retrieval import RetrievalEngine, normalize


SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def client(credentials_path: Path, project: str):
    credentials = service_account.Credentials.from_service_account_file(credentials_path, scopes=[SCOPE])
    return genai.Client(vertexai=True, project=project, location="us-central1", credentials=credentials,
                        http_options=types.HttpOptions(api_version="v1"))


def expected_pairs(question: dict) -> set[tuple[str, int]]:
    documents = question.get("gold_documents", [])
    pages = question.get("gold_pages", [])
    if len(documents) == len(pages):
        return set(zip(documents, map(int, pages), strict=True))
    return {(document, int(page)) for document in documents for page in pages}


def score(question: dict, passages: list[dict]) -> dict:
    expected = expected_pairs(question)
    if not question.get("answerable", True) or not expected:
        return {
            "all_gold_recall_at_5": None, "all_gold_recall_at_10": None,
            "passage_recall_at_5": None, "passage_recall_at_10": None,
            "mrr": None, "ndcg_at_5": None,
        }
    retrieved = [(p["citation"]["document_id"], p["citation"]["page"]) for p in passages]
    hits = [1 if pair in expected else 0 for pair in retrieved]
    found_5 = expected & set(retrieved[:5])
    found_10 = expected & set(retrieved[:10])
    first = next((rank for rank, hit in enumerate(hits, 1) if hit), None)
    dcg = sum(hit / math.log2(rank + 1) for rank, hit in enumerate(hits[:5], 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(expected), 5) + 1))
    return {
        "all_gold_recall_at_5": float(found_5 == expected),
        "all_gold_recall_at_10": float(found_10 == expected),
        "passage_recall_at_5": len(found_5) / len(expected),
        "passage_recall_at_10": len(found_10) / len(expected),
        "mrr": 0 if first is None else 1 / first,
        "ndcg_at_5": 0 if not ideal else dcg / ideal,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--questions", type=Path, nargs="+", required=True)
    parser.add_argument("--credentials", type=Path, default=Path("ai-lab-fasa.json"))
    parser.add_argument("--project", default="ai-lab-502500")
    parser.add_argument("--output", type=Path, default=Path("results/generated/retrieval.jsonl"))
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    questions = [row for path in args.questions for row in read_jsonl(path)]
    engine = RetrievalEngine(args.artifact)
    vertex = client(args.credentials, args.project)

    def embed(row: dict):
        response = vertex.models.embed_content(
            model="gemini-embedding-001", contents=row["question"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768, auto_truncate=False),
        )
        return row, normalize(np.asarray(response.embeddings[0].values, dtype=np.float32))

    records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(embed, row) for row in questions]
        for future in as_completed(futures):
            question, vector = future.result()
            for mode, result in engine.retrieve_all(question["question"], vector, list(RagMode), top_k=10).items():
                passages = [passage.model_dump(mode="json") for passage in result.passages]
                records.append({
                    "question_id": question["question_id"], "category": question["category"],
                    "mode": mode.value, "passages": passages, "retrieval_ms": result.elapsed_ms,
                    **score(question, passages),
                })
            print(f"{len(records)}/{len(questions) * 3}", flush=True)
    records.sort(key=lambda row: (row["question_id"], row["mode"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        grouped[row["mode"]].append(row)
    summary = {"overall": {}, "by_category": {}}
    metrics = (
        "all_gold_recall_at_5", "all_gold_recall_at_10",
        "passage_recall_at_5", "passage_recall_at_10", "mrr", "ndcg_at_5",
    )

    def aggregate(values: list[dict]) -> dict:
        result = {}
        for metric in metrics:
            observed = [row[metric] for row in values if row[metric] is not None]
            result[metric] = sum(observed) / len(observed) if observed else None
        result["mean_retrieval_ms"] = sum(row["retrieval_ms"] for row in values) / len(values)
        result["questions"] = len(values)
        return result

    for mode, values in grouped.items():
        summary["overall"][mode] = aggregate(values)
        for category in sorted({row["category"] for row in values}):
            summary["by_category"].setdefault(category, {})[mode] = aggregate(
                [row for row in values if row["category"] == category]
            )
    summary_path = args.output.with_name("retrieval-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
