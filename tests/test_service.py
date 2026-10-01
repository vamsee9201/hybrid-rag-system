from __future__ import annotations

from app.models import Citation, RagMode, RetrievedPassage
from app.retrieval import RetrievalResult
from app.service import estimate_usage_cost, validate_citations
from app.config import Settings


def test_citations_must_come_from_supplied_passages():
    citation = Citation(document_id="doc-a", title="A", page=4, chunk_id="a:4")
    retrieval = RetrievalResult(
        RagMode.BM25,
        [RetrievedPassage(citation=citation, text="Evidence", rank=1)],
        1.0,
    )
    found = validate_citations("Supported [doc-a, p. 4]. Fake [doc-z, p. 2].", retrieval)
    assert found == [citation]


def test_cost_calculation():
    settings = Settings(input_price_per_million=1, output_price_per_million=2)
    assert estimate_usage_cost(settings, 1_000_000, 500_000) == 2
