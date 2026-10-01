from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gcp_project_id: str = "ai-lab-502500"
    gcp_region: str = "us-central1"
    gemini_location: str = "global"
    generation_model: str = "gemini-3.8-flash"
    generation_fallback_model: str = "gemini-3.7-flash"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    max_output_tokens: int = 300
    artifact_dir: Path = Path("data/artifacts/current")
    artifact_gcs_uri: str | None = None
    artifact_sha256: str | None = None
    web_dist_dir: Path = Path("web/dist")
    ip_hash_salt: str = "local-development-only"
    daily_budget_usd: float = 0.50
    per_ip_daily_answers: int = 20
    per_ip_requests_per_minute: int = 5
    input_price_per_million: float = 0.75
    output_price_per_million: float = 3.75
    mock_vertex: bool = False
    allow_local_quota_fallback: bool = True

    @field_validator("embedding_dimensions")
    @classmethod
    def supported_dimensions(cls, value: int) -> int:
        if value not in {768, 1536, 3072}:
            raise ValueError("embedding_dimensions must be 768, 1536, or 3072")
        return value

    @field_validator("ip_hash_salt")
    @classmethod
    def reject_default_salt_in_cloud(cls, value: str) -> str:
        # A deployment check enforces a real secret; local development stays frictionless.
        return value

    @property
    def estimated_cost_per_answer_cap(self) -> float:
        return (8_000 * self.input_price_per_million + 300 * self.output_price_per_million) / 1_000_000


@lru_cache
def get_settings() -> Settings:
    return Settings()
