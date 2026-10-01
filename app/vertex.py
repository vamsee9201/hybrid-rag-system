from __future__ import annotations

import asyncio
import hashlib
from typing import AsyncIterator

import numpy as np
from google import genai
from google.genai import types

from app.config import Settings
from app.prompts import SYSTEM_PROMPT
from app.retrieval import normalize


class VertexGateway:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None if settings.mock_vertex else genai.Client(
            vertexai=True,
            project=settings.gcp_project_id,
            location=settings.gemini_location,
            http_options=types.HttpOptions(api_version="v1"),
        )

    async def embed_query(self, text: str) -> np.ndarray:
        if self.settings.mock_vertex:
            # Stable development-only vector. Production never uses this path.
            seed = hashlib.sha256(text.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
            return normalize(rng.standard_normal(self.settings.embedding_dimensions, dtype=np.float32))
        for attempt in range(3):
            try:
                response = await self.client.aio.models.embed_content(
                    model=self.settings.embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_QUERY",
                        output_dimensionality=self.settings.embedding_dimensions,
                        auto_truncate=False,
                    ),
                )
                break
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        if len(response.embeddings or []) != 1:
            raise ValueError("Vertex returned an unexpected embedding count")
        embedding = response.embeddings[0]
        if getattr(embedding.statistics, "truncated", False):
            raise ValueError("Vertex truncated the query embedding input")
        return normalize(np.asarray(embedding.values, dtype=np.float32))

    async def generate_stream(self, prompt: str) -> AsyncIterator[tuple[str, dict | None]]:
        if self.settings.mock_vertex:
            answer = "Insufficient evidence."
            for word in answer.split(" "):
                yield word + " ", None
            yield "", {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            return

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0,
            max_output_tokens=self.settings.max_output_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0, include_thoughts=False),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        usage = None
        for attempt in range(3):
            emitted = False
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=self.settings.generation_model,
                    contents=prompt,
                    config=config,
                )
                async for chunk in stream:
                    text = chunk.text or ""
                    if text:
                        emitted = True
                        yield text, None
                    if chunk.usage_metadata:
                        metadata = chunk.usage_metadata
                        usage = {
                            "input_tokens": metadata.prompt_token_count or 0,
                            "output_tokens": metadata.candidates_token_count or 0,
                            "total_tokens": metadata.total_token_count or 0,
                        }
                break
            except Exception:
                if emitted or attempt == 2:
                    raise
                await asyncio.sleep(2 ** attempt)
        yield "", usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
