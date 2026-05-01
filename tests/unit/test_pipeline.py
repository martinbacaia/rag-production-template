"""Unit tests for :mod:`rag.ingestion.pipeline`.

We use lightweight fakes for the embedder and vector store. Hitting the real
ones is the job of integration tests later in the suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag.ingestion.chunker import ChunkingConfig
from rag.ingestion.pipeline import IngestionPipeline


class FakeEmbedder:
    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.calls: list[Sequence[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        # Deterministic dummy vectors, content-aware so different texts get
        # different vectors. Real embedders are tested against the OpenAI
        # mock at the integration layer.
        return [[float(len(t) % 10)] * self.dim for t in texts]


class FakeStore:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        for i, t, e, m in zip(ids, texts, embeddings, metadatas, strict=True):
            self.records.append({"id": i, "text": t, "embedding": list(e), "metadata": m})


@pytest.fixture
def pipeline() -> tuple[IngestionPipeline, FakeEmbedder, FakeStore]:
    embedder = FakeEmbedder()
    store = FakeStore()
    pipe = IngestionPipeline(
        embedder=embedder,
        vector_store=store,
        chunking_config=ChunkingConfig(chunk_size=80, chunk_overlap=10),
    )
    return pipe, embedder, store


def test_ingest_text_round_trip(
    pipeline: tuple[IngestionPipeline, FakeEmbedder, FakeStore],
) -> None:
    pipe, embedder, store = pipeline
    text = "The first sentence is here. " * 10
    result = pipe.ingest_text(text, source="my-doc")

    assert result.source == "my-doc"
    assert result.num_chunks > 0
    assert len(result.chunk_ids) == result.num_chunks
    # ids are unique
    assert len(set(result.chunk_ids)) == len(result.chunk_ids)
    # store received exactly the same number of records
    assert len(store.records) == result.num_chunks
    # embedder was called once with all texts
    assert len(embedder.calls) == 1
    assert len(embedder.calls[0]) == result.num_chunks
    # Standard metadata is attached to every record
    for rec in store.records:
        assert "source" in rec["metadata"]
        assert "chunk_index" in rec["metadata"]
        assert "start_char" in rec["metadata"]
        assert "end_char" in rec["metadata"]


def test_ingest_text_propagates_extra_metadata(
    pipeline: tuple[IngestionPipeline, FakeEmbedder, FakeStore],
) -> None:
    pipe, _, store = pipeline
    pipe.ingest_text("Some content here. " * 5, source="d", extra_metadata={"lang": "en"})
    assert all(rec["metadata"]["lang"] == "en" for rec in store.records)


def test_ingest_empty_text_is_a_noop(
    pipeline: tuple[IngestionPipeline, FakeEmbedder, FakeStore],
) -> None:
    pipe, embedder, store = pipeline
    result = pipe.ingest_text("   ", source="d")
    assert result.num_chunks == 0
    assert store.records == []
    assert embedder.calls == []


def test_ingest_file(
    tmp_path: Path,
    pipeline: tuple[IngestionPipeline, FakeEmbedder, FakeStore],
) -> None:
    pipe, _, store = pipeline
    f = tmp_path / "doc.txt"
    f.write_text("Some words to ingest. " * 20, encoding="utf-8")
    result = pipe.ingest_file(f)
    assert result.source == str(f)
    assert result.num_chunks > 0
    assert all(rec["metadata"]["file_type"] == "txt" for rec in store.records)


def test_chunk_ids_are_deterministic(
    pipeline: tuple[IngestionPipeline, FakeEmbedder, FakeStore],
) -> None:
    pipe1, _, _ = pipeline
    embedder2, store2 = FakeEmbedder(), FakeStore()
    pipe2 = IngestionPipeline(
        embedder=embedder2,
        vector_store=store2,
        chunking_config=ChunkingConfig(chunk_size=80, chunk_overlap=10),
    )
    text = "Same content. " * 10
    r1 = pipe1.ingest_text(text, source="doc")
    r2 = pipe2.ingest_text(text, source="doc")
    assert r1.chunk_ids == r2.chunk_ids


def test_embedder_length_mismatch_raises(
    pipeline: tuple[IngestionPipeline, FakeEmbedder, FakeStore],
) -> None:
    pipe, _, _ = pipeline

    class BrokenEmbedder:
        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.0]]  # always returns one vector regardless of input

    pipe._embedder = BrokenEmbedder()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="vectors for"):
        pipe.ingest_text("a b c d e f g h i j " * 10, source="d")
