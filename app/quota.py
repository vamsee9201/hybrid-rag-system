from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import threading
import time

from google.cloud import firestore

from app.config import Settings


class QuotaExceeded(Exception):
    pass


@dataclass
class QuotaSnapshot:
    remaining_daily_usd: float
    remaining_ip_answers: int


class LocalQuotaStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.lock = threading.Lock()
        self.day = ""
        self.global_cost = 0.0
        self.ip_answers: dict[str, int] = {}
        self.minute_requests: dict[tuple[str, int], int] = {}

    def reserve(self, ip_hash: str, answers: int) -> QuotaSnapshot:
        now = datetime.now(UTC)
        day = now.date().isoformat()
        minute = int(time.time() // 60)
        cost = answers * self.settings.estimated_cost_per_answer_cap
        with self.lock:
            if day != self.day:
                self.day, self.global_cost, self.ip_answers = day, 0.0, {}
            minute_key = (ip_hash, minute)
            requests = self.minute_requests.get(minute_key, 0)
            if requests >= self.settings.per_ip_requests_per_minute:
                raise QuotaExceeded("Per-minute request limit reached")
            used = self.ip_answers.get(ip_hash, 0)
            if used + answers > self.settings.per_ip_daily_answers:
                raise QuotaExceeded("Daily answer limit reached for this client")
            if self.global_cost + cost > self.settings.daily_budget_usd:
                raise QuotaExceeded("The public demo has reached its daily model budget")
            self.minute_requests[minute_key] = requests + 1
            self.ip_answers[ip_hash] = used + answers
            self.global_cost += cost
            return QuotaSnapshot(
                remaining_daily_usd=max(0, self.settings.daily_budget_usd - self.global_cost),
                remaining_ip_answers=self.settings.per_ip_daily_answers - self.ip_answers[ip_hash],
            )


class QuotaManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.local = LocalQuotaStore(settings)
        try:
            self.firestore = None if settings.mock_vertex else firestore.Client(project=settings.gcp_project_id)
        except Exception:
            if not settings.allow_local_quota_fallback:
                raise
            self.firestore = None

    def hash_ip(self, ip: str) -> str:
        return hashlib.sha256(f"{self.settings.ip_hash_salt}:{ip}".encode()).hexdigest()

    async def reserve(self, ip: str, answers: int) -> QuotaSnapshot:
        ip_hash = self.hash_ip(ip)
        if self.firestore is None:
            return self.local.reserve(ip_hash, answers)
        return await asyncio.to_thread(self._reserve_firestore, ip_hash, answers)

    def _reserve_firestore(self, ip_hash: str, answers: int) -> QuotaSnapshot:
        client = self.firestore
        now = datetime.now(UTC)
        day = now.date().isoformat()
        minute = now.strftime("%Y%m%d%H%M")
        expiry = now + timedelta(days=2)
        global_ref = client.collection("rag_quotas").document(f"global-{day}")
        ip_ref = client.collection("rag_quotas").document(f"ip-{day}-{ip_hash}")
        minute_ref = client.collection("rag_quotas").document(f"minute-{minute}-{ip_hash}")
        cost = answers * self.settings.estimated_cost_per_answer_cap
        transaction = client.transaction()

        @firestore.transactional
        def reserve(transaction):
            global_doc, ip_doc, minute_doc = transaction.get_all([global_ref, ip_ref, minute_ref])
            global_used = (global_doc.to_dict() or {}).get("reserved_usd", 0.0)
            ip_used = (ip_doc.to_dict() or {}).get("answers", 0)
            minute_used = (minute_doc.to_dict() or {}).get("requests", 0)
            if minute_used >= self.settings.per_ip_requests_per_minute:
                raise QuotaExceeded("Per-minute request limit reached")
            if ip_used + answers > self.settings.per_ip_daily_answers:
                raise QuotaExceeded("Daily answer limit reached for this client")
            if global_used + cost > self.settings.daily_budget_usd:
                raise QuotaExceeded("The public demo has reached its daily model budget")
            transaction.set(global_ref, {"reserved_usd": global_used + cost, "expires_at": expiry})
            transaction.set(ip_ref, {"answers": ip_used + answers, "expires_at": expiry})
            transaction.set(minute_ref, {"requests": minute_used + 1, "expires_at": expiry})
            return QuotaSnapshot(
                remaining_daily_usd=self.settings.daily_budget_usd - global_used - cost,
                remaining_ip_answers=self.settings.per_ip_daily_answers - ip_used - answers,
            )

        return reserve(transaction)
