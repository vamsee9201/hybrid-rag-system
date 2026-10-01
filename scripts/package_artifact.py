#!/usr/bin/env python3
"""Assemble and checksum an immutable runtime artifact."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tarfile


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bm25", type=Path, default=Path("data/indexes/chunks.sqlite3"))
    parser.add_argument("--dense", type=Path, default=Path("data/indexes/dense"))
    parser.add_argument("--corpus-summary", type=Path, default=Path("data/processed/corpus500/summary.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/artifacts"))
    parser.add_argument("--version")
    args = parser.parse_args()
    version = args.version or datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    output = args.output_root / version
    if output.exists():
        raise SystemExit(f"Immutable artifact version already exists: {output}")
    output.mkdir(parents=True)
    shutil.copy2(args.bm25, output / "chunks.sqlite3")
    shutil.copy2(args.dense / "embeddings.npy", output / "embeddings.npy")
    shutil.copy2(args.dense / "chunk_ids.json", output / "chunk_ids.json")
    dense_manifest = json.loads((args.dense / "manifest.json").read_text())
    corpus = json.loads(args.corpus_summary.read_text())
    with sqlite3.connect(args.bm25) as db:
        chunks = int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    manifest = {
        "complete": True,
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "documents": corpus["documents"],
        "chunks": chunks,
        "embedding_model": dense_manifest["model"],
        "dimensions": dense_manifest["dimensions"],
        "dtype": "float32",
        "generation_model": "gemini-3.8-flash",
        "files": {
            name: digest(output / name)
            for name in ("chunks.sqlite3", "embeddings.npy", "chunk_ids.json")
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = args.output_root / f"{version}.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for path in sorted(output.iterdir()):
            bundle.add(path, arcname=path.name)
    print(json.dumps({"directory": str(output), "archive": str(archive), "sha256": digest(archive), **manifest}, indent=2))


if __name__ == "__main__":
    main()
