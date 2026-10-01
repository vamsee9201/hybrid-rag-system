from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import sqlite3
import time
from typing import Iterable

import numpy as np

from app.models import Citation, RagMode, RetrievedPassage


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "did", "do", "does", "for", "from", "how", "if", "in", "is", "it", "of", "on",
    "or", "than", "that", "the", "then", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "why", "with", "according", "document", "page",
    "report", "specific", "text", "exact",
}


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Embedding has zero norm")
    return vector / norm


def fts_query(text: str) -> str:
    terms = re.findall(r"[\w]+", text, flags=re.UNICODE)
    if not terms:
        raise ValueError("Query contains no searchable terms")
    content = [term for term in terms if len(term) > 2 and term.casefold() not in STOPWORDS]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in (content or terms))


def cap_documents(rows: Iterable[dict], top_k: int, document_cap: int = 2) -> list[dict]:
    selected: list[dict] = []
    counts: dict[str, int] = {}
    for row in rows:
        document_id = row["document_id"]
        if counts.get(document_id, 0) >= document_cap:
            continue
        counts[document_id] = counts.get(document_id, 0) + 1
        selected.append(row)
        if len(selected) == top_k:
            break
    return selected


@dataclass
class RetrievalResult:
    mode: RagMode
    passages: list[RetrievedPassage]
    elapsed_ms: float


class RetrievalEngine:
    def __init__(self, artifact_dir: Path):
        self.directory = artifact_dir
        self.manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
        self.chunk_ids: list[str] = json.loads(
            (artifact_dir / "chunk_ids.json").read_text(encoding="utf-8")
        )
        if len(self.chunk_ids) != len(set(self.chunk_ids)):
            raise ValueError("Dense index contains duplicate chunk IDs")
        self.embeddings = np.load(artifact_dir / "embeddings.npy", mmap_mode="r")
        expected = (len(self.chunk_ids), int(self.manifest["dimensions"]))
        if self.embeddings.shape != expected:
            raise ValueError(f"Dense index shape {self.embeddings.shape} != {expected}")
        if self.embeddings.dtype != np.float32:
            raise ValueError("Dense index must use float32 vectors")
        self.db_path = artifact_dir / "chunks.sqlite3"

    @property
    def version(self) -> str:
        return str(self.manifest.get("version", "unknown"))

    def _connection(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.resolve()}?mode=ro"
        db = sqlite3.connect(uri, uri=True)
        db.row_factory = sqlite3.Row
        return db

    def bm25_candidates(self, query: str, depth: int = 30) -> list[dict]:
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT c.*, bm25(chunks_fts, 0.0, 0.0, 1.0) AS score
                FROM chunks_fts JOIN chunks c USING(chunk_id)
                WHERE chunks_fts MATCH ?
                ORDER BY score, c.chunk_id LIMIT ?
                """,
                (fts_query(query), depth),
            ).fetchall()
        return [dict(row) for row in rows]

    def dense_candidates(self, query_vector: np.ndarray, depth: int = 30) -> list[dict]:
        query = normalize(query_vector)
        if query.shape != (self.embeddings.shape[1],):
            raise ValueError("Query embedding dimensions do not match the document index")
        scores = np.asarray(self.embeddings @ query, dtype=np.float32)
        depth = min(depth, len(scores))
        if depth == 0:
            return []
        candidates = np.argpartition(scores, -depth)[-depth:]
        ordered = sorted(candidates, key=lambda i: (-float(scores[i]), self.chunk_ids[i]))
        ids = [self.chunk_ids[i] for i in ordered]
        metadata = self.chunk_metadata(ids)
        return [
            {**metadata[chunk_id], "score": float(scores[index])}
            for index, chunk_id in zip(ordered, ids, strict=True)
        ]

    def chunk_metadata(self, chunk_ids: list[str]) -> dict[str, dict]:
        if not chunk_ids:
            return {}
        marks = ",".join("?" for _ in chunk_ids)
        with self._connection() as db:
            rows = db.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", chunk_ids).fetchall()
        values = {row["chunk_id"]: dict(row) for row in rows}
        if missing := set(chunk_ids) - set(values):
            raise ValueError(f"Dense index references missing chunks: {sorted(missing)[:3]}")
        return values

    @staticmethod
    def hybrid_candidates(bm25: list[dict], dense: list[dict], rrf_k: int = 60) -> list[dict]:
        merged: dict[str, dict] = {}
        for rank, row in enumerate(bm25, 1):
            item = merged.setdefault(row["chunk_id"], {"chunk": row})
            item.update(bm25_rank=rank, bm25_score=float(row["score"]))
        for rank, row in enumerate(dense, 1):
            item = merged.setdefault(row["chunk_id"], {"chunk": row})
            item.update(dense_rank=rank, dense_score=float(row["score"]))
        output = []
        for chunk_id, item in merged.items():
            score = 0.0
            if item.get("bm25_rank"):
                score += 1 / (rrf_k + item["bm25_rank"])
            if item.get("dense_rank"):
                score += 1 / (rrf_k + item["dense_rank"])
            output.append({
                **item["chunk"],
                "chunk_id": chunk_id,
                "score": score,
                "bm25_rank": item.get("bm25_rank"),
                "dense_rank": item.get("dense_rank"),
            })
        return sorted(output, key=lambda row: (-row["score"], row["chunk_id"]))

    @staticmethod
    def to_passages(rows: list[dict], top_k: int = 5) -> list[RetrievedPassage]:
        rows = cap_documents(rows, top_k=top_k, document_cap=2)
        return [
            RetrievedPassage(
                citation=Citation(
                    document_id=row["document_id"],
                    title=row["title"],
                    page=int(row["page_start"]),
                    source_url=row.get("source_url") or None,
                    chunk_id=row["chunk_id"],
                ),
                text=row["text"],
                rank=rank,
                score=float(row["score"]) if row.get("score") is not None else None,
                bm25_rank=row.get("bm25_rank"),
                dense_rank=row.get("dense_rank"),
            )
            for rank, row in enumerate(rows, 1)
        ]

    def retrieve_all(
        self, query: str, query_vector: np.ndarray, modes: list[RagMode], top_k: int = 5
    ) -> dict[RagMode, RetrievalResult]:
        started = time.perf_counter()
        bm25 = self.bm25_candidates(query) if RagMode.BM25 in modes or RagMode.HYBRID in modes else []
        after_bm25 = time.perf_counter()
        dense = self.dense_candidates(query_vector) if RagMode.DENSE in modes or RagMode.HYBRID in modes else []
        after_dense = time.perf_counter()
        results: dict[RagMode, RetrievalResult] = {}
        if RagMode.BM25 in modes:
            results[RagMode.BM25] = RetrievalResult(
                RagMode.BM25, self.to_passages(bm25, top_k), (after_bm25 - started) * 1_000
            )
        if RagMode.DENSE in modes:
            results[RagMode.DENSE] = RetrievalResult(
                RagMode.DENSE, self.to_passages(dense, top_k), (after_dense - after_bm25) * 1_000
            )
        if RagMode.HYBRID in modes:
            fused = self.hybrid_candidates(bm25, dense)
            results[RagMode.HYBRID] = RetrievalResult(
                RagMode.HYBRID, self.to_passages(fused, top_k), (time.perf_counter() - started) * 1_000
            )
        return results
