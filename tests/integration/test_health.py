"""Integration tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_returns_ok(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["vector_store_count"] == 0
    assert body["embedding_model"] == "text-embedding-3-small"
    assert body["llm_model"] == "gpt-4o-mini"
    assert body["prompt_template_version"] == "v1"


def test_health_count_increases_after_ingest(client: TestClient) -> None:
    pre = client.get("/health").json()["vector_store_count"]
    r = client.post(
        "/ingest",
        json={"text": "Some content to ingest. " * 10, "source": "doc.txt"},
    )
    assert r.status_code == 201
    post = client.get("/health").json()["vector_store_count"]
    assert post > pre
