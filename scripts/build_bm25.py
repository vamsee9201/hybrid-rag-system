#!/usr/bin/env python3
"""Build the shared chunk store and SQLite FTS5 index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3


def chunks_for_page(page: dict, chunk_words: int, overlap_words: int):
    words = page["text"].split()
    if not words:
        return
    step = chunk_words - overlap_words
    if step < 1:
        raise ValueError("overlap must be smaller than chunk size")
    for number, start in enumerate(range(0, len(words), step)):
        words_slice = words[start : start + chunk_words]
        if not words_slice:
            continue
        yield {
            "chunk_id": f"{page['document_id']}:p{page['page']}:c{number}",
            "document_id": page["document_id"],
            "title": page["title"],
            "collection": page["collection"],
            "page_start": page["page"],
            "page_end": page["page"],
            "text": " ".join(words_slice),
        }
        if start + chunk_words >= len(words):
            break


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/processed/corpus500"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes/chunks.sqlite3"))
    parser.add_argument("--chunk-words", type=int, default=350)
    parser.add_argument("--overlap-words", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to replace it")

    documents = {}
    with (args.corpus / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            documents[row["document_id"]] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".sqlite3.tmp")
    temporary.unlink(missing_ok=True)
    db = sqlite3.connect(temporary)
    seen_hashes: set[str] = set()
    document_ids: set[str] = set()
    chunk_count = 0
    try:
        db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE chunks(
                chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, title TEXT NOT NULL,
                collection TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
                source_url TEXT, text TEXT NOT NULL
            );
            CREATE INDEX chunks_document_idx ON chunks(document_id);
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED, title UNINDEXED, text,
                tokenize='unicode61 remove_diacritics 2'
            );
            CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        with (args.corpus / "pages.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                page = json.loads(line)
                document = documents[page["document_id"]]
                for chunk in chunks_for_page(page, args.chunk_words, args.overlap_words):
                    digest = hashlib.sha256(chunk["text"].casefold().encode()).hexdigest()
                    if digest in seen_hashes:
                        continue
                    seen_hashes.add(digest)
                    values = (
                        chunk["chunk_id"], chunk["document_id"], chunk["title"], chunk["collection"],
                        chunk["page_start"], chunk["page_end"], document.get("source_url"), chunk["text"],
                    )
                    db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", values)
                    db.execute("INSERT INTO chunks_fts VALUES(?,?,?)", (chunk["chunk_id"], chunk["title"], chunk["text"]))
                    document_ids.add(chunk["document_id"])
                    chunk_count += 1
        metadata = {
            "chunk_words": str(args.chunk_words),
            "overlap_words": str(args.overlap_words),
            "documents": str(len(document_ids)),
            "chunks": str(chunk_count),
        }
        db.executemany("INSERT INTO metadata VALUES(?,?)", metadata.items())
        db.commit()
    finally:
        db.close()
    temporary.replace(args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
