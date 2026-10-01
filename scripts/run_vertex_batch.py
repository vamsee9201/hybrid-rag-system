#!/usr/bin/env python3
"""Run the audited Lean RAG Vertex batch embedder against this project's chunk database."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-rag-root", type=Path, default=Path("../lean-rag"))
    parser.add_argument("--index", type=Path, default=Path("data/indexes/chunks.sqlite3"))
    parser.add_argument("--output", type=Path, default=Path("data/indexes/dense"))
    parser.add_argument("--credentials", type=Path, default=Path("ai-lab-fasa.json"))
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--execute", action="store_true", help="Required acknowledgement for the paid Vertex job")
    args = parser.parse_args()
    preflight = Path("reports/preflight-cost.json")
    if not preflight.exists():
        raise SystemExit("Run scripts/preflight_cost.py first")
    if not args.execute:
        raise SystemExit("Paid operation blocked. Review reports/preflight-cost.json, then pass --execute")
    script = args.lean_rag_root / "scripts/build_vertex_dense_batch.py"
    command = [
        sys.executable, str(script), "--split", "test", "--index", str(args.index),
        "--output-dir", str(args.output), "--credentials", str(args.credentials),
        "--bucket", args.bucket, "--location", "us-central1", "--model", "gemini-embedding-001",
        "--dimensions", "768", "--cleanup-gcs", "--cleanup-raw",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
