#!/usr/bin/env python3
"""Select a deterministic document-disjoint 50-question confirmatory benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TARGETS = {
    "direct": 20,
    "numeric_date": 10,
    "multi_passage": 10,
    "cross_document": 5,
    "unanswerable": 5,
}
ALIASES = {
    "single_document_multi_passage": "multi_passage",
    "cross_document_comparison": "cross_document",
}


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=Path("../lean-rag/data/benchmark/questions.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("../lean-rag/data/confirmatory/benchmark/questions.jsonl"))
    parser.add_argument("--corpus-documents", type=Path, default=Path("data/processed/corpus500/documents.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/generated/benchmark"))
    args = parser.parse_args()
    legacy = rows(args.legacy)
    corpus_ids = {row["document_id"] for row in rows(args.corpus_documents)}
    legacy_gold = {doc for row in legacy for doc in row.get("gold_documents", [])}
    selected: list[dict] = []
    counts = {key: 0 for key in TARGETS}
    for row in sorted(rows(args.candidates), key=lambda item: item["question_id"]):
        category = ALIASES.get(row["category"], row["category"])
        gold = set(row.get("gold_documents", []))
        if category not in TARGETS or counts[category] >= TARGETS[category]:
            continue
        if gold & legacy_gold or not gold.issubset(corpus_ids):
            continue
        selected.append({**row, "category": category})
        counts[category] += 1
    if counts != TARGETS:
        raise ValueError(f"Candidate pool cannot satisfy target distribution: {counts}")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = {"legacy": args.output / "legacy.jsonl", "confirmatory": args.output / "confirmatory.jsonl"}
    for path, values in ((paths["legacy"], legacy), (paths["confirmatory"], selected)):
        with path.open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "sealed": True,
        "questions": 100,
        "legacy_source_sha256": digest(args.legacy),
        "confirmatory_source_sha256": digest(args.candidates),
        "legacy_sha256": digest(paths["legacy"]),
        "confirmatory_sha256": digest(paths["confirmatory"]),
        "confirmatory_distribution": counts,
        "document_disjoint_from_legacy": True,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
