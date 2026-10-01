from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import logging
import re
import time
from typing import AsyncIterator
from uuid import uuid4

import numpy as np

from app.config import Settings
from app.models import ChatRequest, Citation, RagAnswer, RagMode, Usage
from app.prompts import evidence_prompt
from app.retrieval import RetrievalEngine, RetrievalResult
from app.vertex import VertexGateway


LOGGER = logging.getLogger("hybrid_rag")
CITATION_PATTERN = re.compile(r"\[([^\],]+),\s*p\.\s*(\d+)\]")


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def validate_citations(answer: str, retrieval: RetrievalResult) -> list[Citation]:
    available = {
        (passage.citation.document_id, passage.citation.page): passage.citation
        for passage in retrieval.passages
    }
    found: list[Citation] = []
    seen: set[tuple[str, int]] = set()
    for document_id, page_text in CITATION_PATTERN.findall(answer):
        key = (document_id.strip(), int(page_text))
        if key in available and key not in seen:
            found.append(available[key])
            seen.add(key)
    return found


def estimate_usage_cost(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * settings.input_price_per_million
        + output_tokens * settings.output_price_per_million
    ) / 1_000_000


class RagService:
    def __init__(self, settings: Settings, retrieval: RetrievalEngine, vertex: VertexGateway):
        self.settings = settings
        self.retrieval = retrieval
        self.vertex = vertex

    async def stream_chat(self, request: ChatRequest) -> AsyncIterator[str]:
        run_id = uuid4().hex
        created_at = datetime.now(UTC)
        mode_queries = {mode: request.message for mode in request.modes}
        unique_queries = set(mode_queries.values())
        query_vectors: dict[str, np.ndarray] = {}
        if RagMode.DENSE in request.modes or RagMode.HYBRID in request.modes:
            query_vectors = {
                query: vector
                for query, vector in zip(
                    unique_queries,
                    await asyncio.gather(*(self.vertex.embed_query(query) for query in unique_queries)),
                    strict=True,
                )
            }

        retrievals: dict[RagMode, RetrievalResult] = {}
        for query in unique_queries:
            modes = [mode for mode, mode_query in mode_queries.items() if mode_query == query]
            vector = query_vectors.get(query)
            if vector is None:
                vector = np.ones(self.settings.embedding_dimensions, dtype=np.float32)
            results = await asyncio.to_thread(self.retrieval.retrieve_all, query, vector, modes)
            retrievals.update(results)

        yield sse("run", {"run_id": run_id, "created_at": created_at.isoformat()})
        for mode in request.modes:
            result = retrievals[mode]
            yield sse("retrieval", {
                "run_id": run_id,
                "mode": mode.value,
                "retrieval_ms": round(result.elapsed_ms, 2),
                "passages": [passage.model_dump(mode="json") for passage in result.passages],
            })

        queue: asyncio.Queue[tuple[str, RagMode, dict]] = asyncio.Queue()

        async def generate(mode: RagMode) -> None:
            started = time.perf_counter()
            retrieval = retrievals[mode]
            prompt = evidence_prompt(
                request.message,
                retrieval.passages,
            )
            answer_parts: list[str] = []
            usage_data = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            try:
                async for delta, usage in self.vertex.generate_stream(prompt):
                    if delta:
                        answer_parts.append(delta)
                        await queue.put(("delta", mode, {"text": delta}))
                    if usage is not None:
                        usage_data = usage
                answer = "".join(answer_parts).strip()
                if not answer:
                    raise ValueError("Gemini returned an empty answer")
                citations = validate_citations(answer, retrieval)
                usage = Usage(
                    **usage_data,
                    estimated_cost_usd=estimate_usage_cost(
                        self.settings,
                        usage_data["input_tokens"],
                        usage_data["output_tokens"],
                    ),
                )
                result = RagAnswer(
                    mode=mode,
                    status="ok",
                    answer=answer,
                    citations=citations,
                    passages=retrieval.passages,
                    retrieval_ms=retrieval.elapsed_ms,
                    generation_ms=(time.perf_counter() - started) * 1_000,
                    usage=usage,
                    model=self.settings.generation_model,
                    index_version=self.retrieval.version,
                )
                await queue.put(("answer", mode, result.model_dump(mode="json")))
                LOGGER.info(json.dumps({
                    "event": "answer_complete",
                    "run_id": run_id,
                    "mode": mode.value,
                    "retrieval_ms": round(result.retrieval_ms, 2),
                    "generation_ms": round(result.generation_ms, 2),
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "estimated_cost_usd": usage.estimated_cost_usd,
                    "model": result.model,
                    "index_version": result.index_version,
                }))
            except Exception as exc:
                LOGGER.exception("generation failed for run=%s mode=%s", run_id, mode.value)
                await queue.put(("error", mode, {
                    "message": "This RAG mode could not complete the answer.",
                    "error_type": type(exc).__name__,
                }))
            finally:
                await queue.put(("done", mode, {}))

        tasks = [asyncio.create_task(generate(mode)) for mode in request.modes]
        completed = 0
        while completed < len(tasks):
            event, mode, payload = await queue.get()
            if event == "done":
                completed += 1
                continue
            yield sse(event, {"run_id": run_id, "mode": mode.value, **payload})
        await asyncio.gather(*tasks)
        yield sse("complete", {"run_id": run_id})
