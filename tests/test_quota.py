from __future__ import annotations

import pytest

from app.config import Settings
from app.quota import LocalQuotaStore, QuotaExceeded


def test_daily_answer_quota_is_shared_by_client_hash():
    store = LocalQuotaStore(Settings(per_ip_daily_answers=3, daily_budget_usd=10))
    snapshot = store.reserve("client-a", 2)
    assert snapshot.remaining_ip_answers == 1
    with pytest.raises(QuotaExceeded, match="Daily answer limit"):
        store.reserve("client-a", 2)


def test_global_budget_blocks_generation():
    settings = Settings(
        daily_budget_usd=0.01,
        input_price_per_million=1,
        output_price_per_million=1,
    )
    store = LocalQuotaStore(settings)
    with pytest.raises(QuotaExceeded, match="daily model budget"):
        store.reserve("client-a", 2)


def test_per_minute_rate_limit():
    store = LocalQuotaStore(Settings(per_ip_requests_per_minute=1, daily_budget_usd=10))
    store.reserve("client-a", 1)
    with pytest.raises(QuotaExceeded, match="Per-minute"):
        store.reserve("client-a", 1)
