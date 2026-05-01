"""Fixtures shared by integration tests.

The strategy:

* Build a fresh FastAPI app per test (no global state).
* Override the OpenAI client with a fake whose embed/chat behavior is
  controllable per test.
* Override the vector store with an ephemeral, per-test ChromaDB.

This means integration tests exercise real wiring: routing, dependency
injection, Pydantic validation, the ingestion pipeline, the retriever,
prompt rendering — but they never hit the network. Each test runs in
sub-second time and is fully deterministic.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag.api.dependencies import (
    get_openai_client,
    get_vector_store,
)
from rag.api.main import create_app
from rag.vectorstore.chroma import ChromaVectorStore

# -- Fake OpenAI surface ------------------------------------------------------


@dataclass
class _EmbItem:
    embedding: list[float]


@dataclass
class _EmbResponse:
    data: list[_EmbItem]


@dataclass
class _Msg:
    content: str | None


@dataclass
class _Choice:
    message: _Msg
    finish_reason: str = "stop"


@dataclass
class _ChatResponse:
    choices: list[_Choice]


class FakeEmbeddings:
    """Deterministic embedder: hashes text into a fixed-dimension vector."""

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []

    def create(self, *, model: str, input: Sequence[str]) -> _EmbResponse:
        self.calls.append(list(input))
        return _EmbResponse(
            data=[_EmbItem(embedding=self._encode(t)) for t in input]
        )

    def _encode(self, text: str) -> list[float]:
        # Tiny content-aware embedding: bag-of-character-buckets over the
        # alphabet, normalized. Two texts with shared characters get
        # similar vectors — close enough for retrieval tests, and 100%
        # deterministic.
        vec = [0.0] * self.dim
        for ch in text.lower():
            if ch.isalnum():
                vec[ord(ch) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec


class FakeChat:
    """Returns a configurable chat response, optionally per-call."""

    def __init__(self, response: str = "stub answer [1]") -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> _ChatResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return _ChatResponse(
            choices=[_Choice(message=_Msg(self.response))]
        )


@dataclass
class _ChatNamespace:
    completions: FakeChat


@dataclass
class FakeOpenAI:
    """Just enough surface to satisfy our wrappers."""

    embeddings: FakeEmbeddings
    chat: _ChatNamespace


# -- Pytest fixtures ----------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tenacity-driven retries call time.sleep — make them noops."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


@pytest.fixture
def fake_openai() -> FakeOpenAI:
    return FakeOpenAI(
        embeddings=FakeEmbeddings(),
        chat=_ChatNamespace(completions=FakeChat()),
    )


@pytest.fixture
def chroma_store() -> ChromaVectorStore:
    """Ephemeral, per-test Chroma collection."""
    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    return ChromaVectorStore(
        collection_name=f"int_{uuid.uuid4().hex}",
        client=client,
    )


@pytest.fixture
def app(fake_openai: FakeOpenAI, chroma_store: ChromaVectorStore) -> Iterator[FastAPI]:
    """Build a fresh app and wire it to the per-test fakes."""
    app = create_app()
    app.dependency_overrides[get_openai_client] = lambda: fake_openai
    app.dependency_overrides[get_vector_store] = lambda: chroma_store
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)
