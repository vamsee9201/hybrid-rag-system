from __future__ import annotations

import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    directory = tmp_path / "artifact"
    directory.mkdir()
    db = sqlite3.connect(directory / "chunks.sqlite3")
    db.executescript("""
        CREATE TABLE chunks(
            chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, title TEXT NOT NULL,
            collection TEXT NOT NULL, page_start INTEGER NOT NULL, page_end INTEGER NOT NULL,
            source_url TEXT, text TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, title UNINDEXED, text);
    """)
    rows = [
        ("doc-a:p1:c0", "doc-a", "Budget Report", "BUDGET", 1, 1, "https://www.govinfo.gov/a", "The program received five million dollars."),
        ("doc-b:p2:c0", "doc-b", "Agency Report", "GAO", 2, 2, "https://www.govinfo.gov/b", "The agency completed four audits."),
        ("doc-a:p2:c0", "doc-a", "Budget Report", "BUDGET", 2, 2, "https://www.govinfo.gov/a", "The project began in 2024."),
    ]
    db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?)", rows)
    db.executemany("INSERT INTO chunks_fts VALUES(?,?,?)", [(row[0], row[2], row[7]) for row in rows])
    db.commit()
    db.close()
    vectors = np.eye(3, 768, dtype=np.float32)
    np.save(directory / "embeddings.npy", vectors)
    (directory / "chunk_ids.json").write_text(json.dumps([row[0] for row in rows]))
    (directory / "manifest.json").write_text(json.dumps({
        "complete": True, "version": "test-v1", "dimensions": 768, "documents": 2, "chunks": 3,
    }))
    return directory
