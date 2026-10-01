#!/usr/bin/env python3
"""Create a frozen 500-searchable-document corpus from Lean RAG artifacts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def existing_or_tmp(directory: Path, name: str) -> Path:
    final = directory / name
    temporary = final.with_suffix(final.suffix + ".tmp")
    if final.exists():
        return final
    if temporary.exists():
        return temporary
    raise FileNotFoundError(f"Neither {final} nor {temporary} exists")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-rag-root", type=Path, default=Path("../lean-rag"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/corpus500"))
    parser.add_argument("--minimum-characters", type=int, default=1_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    primary = args.lean_rag_root / "data/processed/corpus500"
    candidates = args.lean_rag_root / "data/processed/corpus1000"
    manifest_path = args.lean_rag_root / "data/manifest.csv"
    outputs = [args.output / name for name in ("documents.jsonl", "pages.jsonl", "selection.json", "summary.json")]
    if any(path.exists() for path in outputs) and not args.force:
        raise SystemExit(f"Output already exists under {args.output}; pass --force to replace it")

    primary_docs = jsonl(existing_or_tmp(primary, "documents.jsonl"))
    candidate_docs = jsonl(existing_or_tmp(candidates, "documents.jsonl"))
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = {row["package_id"]: row for row in csv.DictReader(stream)}

    kept = [doc for doc in primary_docs if doc.get("character_count", 0) >= args.minimum_characters]
    excluded = [doc for doc in primary_docs if doc.get("character_count", 0) < args.minimum_characters]
    if len(kept) + len(excluded) != 500:
        raise ValueError("Primary corpus must contain exactly 500 documents")
    used = {doc["document_id"] for doc in primary_docs}
    pool = [
        doc for doc in candidate_docs
        if doc["document_id"] not in used
        and doc.get("character_count", 0) >= args.minimum_characters
        and doc.get("status") == "ok"
    ]
    by_collection: dict[str, list[dict]] = defaultdict(list)
    for doc in pool:
        by_collection[doc["collection"]].append(doc)
    for docs in by_collection.values():
        docs.sort(key=lambda row: (row.get("expected_pages", 0), row["document_id"]))

    replacements: list[dict] = []
    available = list(pool)
    replacement_counts: Counter[str] = Counter()
    for old in sorted(excluded, key=lambda row: (row["collection"], row["document_id"])):
        same_collection = by_collection.get(old["collection"], [])
        replacement = next((doc for doc in same_collection if doc in available), None)
        if replacement is None:
            # If an entire collection is scan-only, spread replacements evenly across
            # the remaining collections instead of accidentally biasing toward one.
            minimum_added = min(replacement_counts[doc["collection"]] for doc in available)
            balanced = [doc for doc in available if replacement_counts[doc["collection"]] == minimum_added]
            replacement = min(balanced, key=lambda doc: (
                abs(doc.get("expected_pages", 0) - old.get("expected_pages", 0)),
                doc["collection"], doc["document_id"],
            ))
        available.remove(replacement)
        replacement_counts[replacement["collection"]] += 1
        replacements.append({**replacement, "replaces_document_id": old["document_id"]})

    selected = kept + replacements
    if len(selected) != 500 or len({doc["document_id"] for doc in selected}) != 500:
        raise ValueError("Replacement selection did not produce 500 unique searchable documents")
    selected_ids = {doc["document_id"] for doc in selected}
    args.output.mkdir(parents=True, exist_ok=True)

    pages_tmp = args.output / "pages.jsonl.tmp"
    found_pages: set[str] = set()
    with pages_tmp.open("w", encoding="utf-8") as destination:
        source_selections = (
            (primary, {doc["document_id"] for doc in kept}),
            (candidates, {doc["document_id"] for doc in replacements}),
        )
        for source_dir, source_ids in source_selections:
            with existing_or_tmp(source_dir, "pages.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    document_id = row["document_id"]
                    if document_id in source_ids:
                        destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                        found_pages.add(document_id)
    if found_pages != selected_ids:
        raise ValueError(f"Missing extracted pages for {len(selected_ids - found_pages)} documents")
    pages_tmp.replace(args.output / "pages.jsonl")

    replacements_by_id = {doc["document_id"]: doc["replaces_document_id"] for doc in replacements}
    with (args.output / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for doc in sorted(selected, key=lambda row: row["document_id"]):
            source = source_rows.get(doc["document_id"], {})
            record = {
                **{key: value for key, value in doc.items() if key != "replaces_document_id"},
                "source_url": source.get("source_url"),
                "selection_reason": "scan_replacement" if doc["document_id"] in replacements_by_id else "original_searchable",
                "replaces_document_id": replacements_by_id.get(doc["document_id"]),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    selection = {
        "documents": 500,
        "minimum_characters": args.minimum_characters,
        "original_searchable": len(kept),
        "replacements": [
            {"removed": old["document_id"], "added": new["document_id"], "collection": new["collection"]}
            for old, new in zip(sorted(excluded, key=lambda row: (row["collection"], row["document_id"])), replacements, strict=True)
        ],
        "document_ids": sorted(selected_ids),
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    summary = {
        "documents": len(selected),
        "pages": sum(int(doc["extracted_pages"]) for doc in selected),
        "characters": sum(int(doc["character_count"]) for doc in selected),
        "collections": dict(sorted(Counter(doc["collection"] for doc in selected).items())),
        "scan_only_documents": 0,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
