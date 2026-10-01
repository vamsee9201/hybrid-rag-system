from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.artifacts import ensure_artifact, load_manifest
from app.config import Settings, get_settings
from app.models import ChatRequest, FeedbackRequest, RagMode
from app.quota import QuotaExceeded, QuotaManager
from app.retrieval import RetrievalEngine
from app.service import RagService
from app.vertex import VertexGateway


logging.basicConfig(level=logging.INFO, format="%(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        artifact_dir = ensure_artifact(
            settings.artifact_dir,
            settings.artifact_gcs_uri,
            settings.artifact_sha256,
        )
        app.state.settings = settings
        app.state.manifest = load_manifest(artifact_dir)
        app.state.retrieval = None
        app.state.startup_error = None
        try:
            app.state.retrieval = RetrievalEngine(artifact_dir)
        except Exception as exc:
            app.state.startup_error = f"{type(exc).__name__}: {exc}"
        app.state.vertex = VertexGateway(settings)
        app.state.quota = QuotaManager(settings)
        yield

    app = FastAPI(title="Hybrid GovInfo RAG", version="0.1.0", lifespan=lifespan)

    @app.get("/api/health")
    async def health(request: Request):
        ready = request.app.state.retrieval is not None
        payload = {
            "status": "ready" if ready else "not_ready",
            "model": settings.generation_model,
            "embedding_model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "index_version": request.app.state.manifest.get("version", "unavailable"),
            "mock_vertex": settings.mock_vertex,
        }
        if not ready:
            payload["reason"] = request.app.state.startup_error
            return JSONResponse(payload, status_code=503)
        return payload

    @app.get("/api/config")
    async def config(request: Request):
        manifest = request.app.state.manifest
        return {
            "modes": [mode.value for mode in RagMode],
            "default_modes": [mode.value for mode in RagMode],
            "model": settings.generation_model,
            "embedding_model": settings.embedding_model,
            "dimensions": settings.embedding_dimensions,
            "corpus": {
                "documents": manifest.get("documents"),
                "chunks": manifest.get("chunks"),
                "version": manifest.get("version", "unavailable"),
            },
            "limits": {
                "message_characters": 2_000,
                "interaction": "single_turn",
                "per_ip_daily_answers": settings.per_ip_daily_answers,
                "daily_budget_usd": settings.daily_budget_usd,
            },
        }

    @app.post("/api/chat")
    async def chat(body: ChatRequest, request: Request):
        if request.app.state.retrieval is None:
            raise HTTPException(503, "The retrieval index is not ready")
        client_ip = request.headers.get("x-forwarded-for", "").split(",", 1)[0].strip()
        if not client_ip and request.client:
            client_ip = request.client.host
        try:
            await request.app.state.quota.reserve(client_ip or "unknown", len(body.modes))
        except QuotaExceeded as exc:
            raise HTTPException(429, str(exc)) from exc
        service = RagService(settings, request.app.state.retrieval, request.app.state.vertex)
        return StreamingResponse(
            service.stream_chat(body),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/feedback", status_code=202)
    async def feedback(body: FeedbackRequest, request: Request):
        client = request.app.state.quota.firestore
        if client is not None:
            data = body.model_dump(mode="json")
            data["created_at"] = datetime.now(UTC)
            data["expires_at"] = datetime.now(UTC) + timedelta(days=90)
            client.collection("rag_feedback").document().set(data)
        return {"accepted": True}

    web_dir = settings.web_dist_dir
    if web_dir.exists():
        assets = web_dir / "assets"
        if assets.exists():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            candidate = (web_dir / path).resolve()
            if path and web_dir.resolve() in candidate.parents and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(web_dir / "index.html")

    return app


app = create_app()
