"""Integration tests for the /ingest endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import FakeOpenAI


def test_ingest_returns_chunk_ids(client: TestClient) -> None:
    r = client.post(
        "/ingest",
        json={
            "text": "Decorators in Python wrap functions. " * 30,
            "source": "python_decorators.txt",
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["source"] == "python_decorators.txt"
    assert body["num_chunks"] >= 1
    assert len(body["chunk_ids"]) == body["num_chunks"]
    # IDs are stable across re-ingests (deterministic by content hash).
    r2 = client.post(
        "/ingest",
        json={
            "text": "Decorators in Python wrap functions. " * 30,
            "source": "python_decorators.txt",
        },
    )
    assert r2.json()["chunk_ids"] == body["chunk_ids"]


def test_ingest_requires_non_empty_text(client: TestClient) -> None:
    r = client.post("/ingest", json={"text": "", "source": "x"})
    assert r.status_code == 422  # Pydantic validation


def test_ingest_requires_source(client: TestClient) -> None:
    r = client.post("/ingest", json={"text": "hello", "source": ""})
    assert r.status_code == 422


def test_ingest_propagates_metadata(client: TestClient) -> None:
    r = client.post(
        "/ingest",
        json={
            "text": "some text " * 20,
            "source": "x",
            "metadata": {"lang": "en", "tier": "premium"},
        },
    )
    assert r.status_code == 201
    # Verify the metadata round-trips through retrieval. The actual
    # check uses /query in the next file; here we just ensure 201.


def test_ingest_calls_embedder_once_per_request(
    client: TestClient,
    fake_openai: FakeOpenAI,
) -> None:
    client.post(
        "/ingest",
        json={"text": "alpha beta gamma " * 50, "source": "doc"},
    )
    # The pipeline batches all chunks into a single embed call.
    assert len(fake_openai.embeddings.calls) == 1
