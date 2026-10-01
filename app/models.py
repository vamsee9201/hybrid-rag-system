from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class RagMode(StrEnum):
    BM25 = "bm25"
    DENSE = "dense"
    HYBRID = "hybrid"


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=2_000)
    modes: list[RagMode] = Field(default_factory=lambda: list(RagMode), min_length=1, max_length=3)

    @field_validator("message")
    @classmethod
    def non_whitespace_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message cannot be blank")
        return value

    @field_validator("modes")
    @classmethod
    def unique_modes(cls, value: list[RagMode]) -> list[RagMode]:
        if len(set(value)) != len(value):
            raise ValueError("modes must be unique")
        return value


class Citation(BaseModel):
    document_id: str
    title: str
    page: int = Field(ge=1)
    source_url: HttpUrl | None = None
    chunk_id: str


class RetrievedPassage(BaseModel):
    citation: Citation
    text: str
    rank: int = Field(ge=1)
    score: float | None = None
    bm25_rank: int | None = None
    dense_rank: int | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class RagAnswer(BaseModel):
    mode: RagMode
    status: Literal["ok", "error"]
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    passages: list[RetrievedPassage] = Field(default_factory=list)
    retrieval_ms: float = 0
    generation_ms: float = 0
    usage: Usage = Field(default_factory=Usage)
    model: str
    index_version: str
    error: str | None = None


class ComparisonRun(BaseModel):
    run_id: str
    question: str
    answers: dict[RagMode, RagAnswer]
    created_at: datetime


class FeedbackRequest(BaseModel):
    run_id: str = Field(min_length=8, max_length=64)
    preferred_mode: RagMode | None = None
    reason_tags: list[Literal["correct", "grounded", "clear", "complete", "fast"]] = Field(
        default_factory=list, max_length=5
    )
    comment: str | None = Field(default=None, max_length=500)
