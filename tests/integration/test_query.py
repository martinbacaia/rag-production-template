"""Integration tests for the /query endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import FakeOpenAI


def _ingest_corpus(client: TestClient) -> None:
    client.post(
        "/ingest",
        json={
            "text": (
                "Python decorators wrap functions to add behavior without "
                "changing the original function definition. They are "
                "applied with the @ syntax above the function signature."
            ),
            "source": "python_decorators.txt",
        },
    )
    client.post(
        "/ingest",
        json={
            "text": (
                "List comprehensions in Python provide a concise way to "
                "create lists. They are written as expressions enclosed "
                "in square brackets."
            ),
            "source": "python_list_comprehensions.txt",
        },
    )


def test_query_returns_answer_with_citations(
    client: TestClient,
    fake_openai: FakeOpenAI,
) -> None:
    _ingest_corpus(client)
    fake_openai.chat.completions.response = "Decorators wrap functions [1]."

    r = client.post("/query", json={"question": "what are decorators?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Decorators wrap functions [1]."
    assert body["retrieved_count"] >= 1
    assert body["prompt_version"] == "v1"
    assert body["model"] == "gpt-4o-mini"
    # Every citation has the required shape.
    for c in body["citations"]:
        assert set(c.keys()) == {"chunk_id", "source", "text", "score"}
        assert 0.0 <= c["score"] <= 1.0


def test_query_short_circuits_on_empty_index(
    client: TestClient,
    fake_openai: FakeOpenAI,
) -> None:
    """No ingest → no context → fixed fallback, no LLM call."""
    r = client.post("/query", json={"question": "anything"})
    assert r.status_code == 200
    body = r.json()
    assert "do not have enough information" in body["answer"].lower()
    assert body["citations"] == []
    assert body["retrieved_count"] == 0
    # The chat client must not have been invoked.
    assert fake_openai.chat.completions.calls == []


def test_query_validates_top_k(client: TestClient) -> None:
    r = client.post("/query", json={"question": "q", "top_k": 0})
    assert r.status_code == 422
    r = client.post("/query", json={"question": "q", "top_k": 1000})
    assert r.status_code == 422


def test_query_validates_score_threshold(client: TestClient) -> None:
    r = client.post(
        "/query",
        json={"question": "q", "score_threshold": 1.5},
    )
    assert r.status_code == 422


def test_query_calls_chat_with_configured_template(
    client: TestClient,
    fake_openai: FakeOpenAI,
) -> None:
    _ingest_corpus(client)
    client.post("/query", json={"question": "what are decorators?"})

    assert len(fake_openai.chat.completions.calls) == 1
    call = fake_openai.chat.completions.calls[0]
    # The system message comes from the v1 template (default).
    sys_msg = call["messages"][0]
    assert sys_msg["role"] == "system"
    assert "context" in sys_msg["content"].lower()
    # The user message contains the question and at least one [N] citation marker.
    user_msg = call["messages"][1]
    assert user_msg["role"] == "user"
    assert "decorators" in user_msg["content"].lower()
    assert "[1]" in user_msg["content"]


def test_query_threshold_filters_results(
    client: TestClient,
    fake_openai: FakeOpenAI,
) -> None:
    """A threshold above any possible score returns the no-context answer."""
    _ingest_corpus(client)
    r = client.post(
        "/query",
        json={"question": "anything", "score_threshold": 0.99},
    )
    body = r.json()
    assert body["retrieved_count"] == 0
    assert body["citations"] == []
    # LLM must not have been called for empty-context responses.
    assert fake_openai.chat.completions.calls == []
