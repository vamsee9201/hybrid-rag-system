from __future__ import annotations

import numpy as np

from app.models import RagMode
from app.retrieval import RetrievalEngine, cap_documents, fts_query, normalize


def test_normalize_rejects_zero_vector():
    try:
        normalize(np.zeros(3))
    except ValueError as exc:
        assert "zero norm" in str(exc)
    else:
        raise AssertionError("zero vector was accepted")


def test_fts_query_removes_stopwords():
    assert fts_query("What is the annual budget?") == '"annual" OR "budget"'


def test_document_cap():
    rows = [{"document_id": "a"}, {"document_id": "a"}, {"document_id": "a"}, {"document_id": "b"}]
    assert cap_documents(rows, top_k=4, document_cap=2) == rows[:2] + rows[3:]


def test_three_retrieval_modes(artifact):
    engine = RetrievalEngine(artifact)
    vector = np.eye(1, 768, dtype=np.float32)[0]
    results = engine.retrieve_all("program dollars", vector, list(RagMode))
    assert set(results) == set(RagMode)
    assert results[RagMode.BM25].passages[0].citation.document_id == "doc-a"
    assert results[RagMode.DENSE].passages[0].citation.chunk_id == "doc-a:p1:c0"
    assert results[RagMode.HYBRID].passages
