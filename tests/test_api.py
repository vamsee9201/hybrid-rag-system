from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_health_and_streaming_chat(artifact, tmp_path):
    settings = Settings(
        artifact_dir=artifact,
        web_dist_dir=tmp_path / "missing-web",
        mock_vertex=True,
    )
    with TestClient(create_app(settings)) as client:
        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["index_version"] == "test-v1"
        response = client.post("/api/chat", json={"message": "What was funded?", "modes": ["bm25"]})
        assert response.status_code == 200
        assert "event: retrieval" in response.text
        assert "event: answer" in response.text


def test_request_validation(artifact, tmp_path):
    settings = Settings(artifact_dir=artifact, web_dist_dir=tmp_path / "missing", mock_vertex=True)
    with TestClient(create_app(settings)) as client:
        response = client.post("/api/chat", json={"message": " ", "modes": ["bm25"]})
        assert response.status_code == 422


def test_chat_rejects_conversation_history(artifact, tmp_path):
    settings = Settings(artifact_dir=artifact, web_dist_dir=tmp_path / "missing", mock_vertex=True)
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/chat",
            json={
                "message": "What was funded?",
                "modes": ["bm25"],
                "histories": {"bm25": [{"role": "user", "content": "Earlier question"}]},
            },
        )
        assert response.status_code == 422
