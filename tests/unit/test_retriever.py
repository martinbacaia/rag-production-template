"""Tests for :mod:`rag.retrieval.retriever`."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import Any

import pytest

from rag.retrieval.reranker import KeywordOverlapReranker, NoOpReranker
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import SearchResult, VectorStore


class FakeEmbedder:
    """Records the texts it was asked to embed and returns deterministic vectors."""

    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t))] for t in texts]


class FakeStore(VectorStore):
    """In-memory store that records search calls and returns canned results."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.searches: list[dict[str, Any]] = []

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        raise NotImplementedError("not exercised in retriever tests")

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        self.searches.append(
            {
                "query_embedding": list(query_embedding),
                "top_k": top_k,
                "score_threshold": score_threshold,
                "where": where,
            }
        )
        # Honor top_k so the caller-side oversampling is observable.
        return self._results[:top_k]

    def delete(self, ids: Sequence[str]) -> None: ...
    def count(self) -> int:
        return len(self._results)

    def reset(self) -> None: ...


def _hit(id_: str, text: str, score: float) -> SearchResult:
    return SearchResult(id=id_, text=text, score=score, metadata={})


def test_retrieve_embeds_query_and_searches_store() -> None:
    embedder = FakeEmbedder()
    store = FakeStore([_hit("a", "alpha", 0.9), _hit("b", "beta", 0.5)])
    retriever = Retriever(embedder=embedder, vector_store=store)

    out = retriever.retrieve("hello world", top_k=2)

    assert embedder.calls == [["hello world"]]
    assert len(store.searches) == 1
    assert store.searches[0]["top_k"] == 2  # NoOp reranker → no oversample
    assert [r.id for r in out] == ["a", "b"]


def test_empty_query_short_circuits() -> None:
    embedder = FakeEmbedder()
    store = FakeStore([_hit("a", "alpha", 0.9)])
    retriever = Retriever(embedder=embedder, vector_store=store)
    assert retriever.retrieve("   ", top_k=3) == []
    # No embed/search should have happened.
    assert embedder.calls == []
    assert store.searches == []


def test_invalid_top_k_raises() -> None:
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=FakeStore([]))
    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("q", top_k=0)


def test_invalid_oversample_factor_raises() -> None:
    with pytest.raises(ValueError, match="oversample_factor"):
        Retriever(
            embedder=FakeEmbedder(),
            vector_store=FakeStore([]),
            oversample_factor=0,
        )


def test_oversample_when_reranker_is_active() -> None:
    """A real reranker should receive ``top_k * factor`` candidates."""
    pool = [_hit(f"id{i}", f"doc {i}", 0.9 - i * 0.05) for i in range(20)]
    store = FakeStore(pool)
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        reranker=KeywordOverlapReranker(alpha=0.7),
        oversample_factor=4,
    )
    retriever.retrieve("query", top_k=3)
    assert store.searches[0]["top_k"] == 12  # 3 * 4


def test_score_threshold_and_where_are_passed_through() -> None:
    store = FakeStore([_hit("a", "alpha", 0.9)])
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=store)
    retriever.retrieve(
        "query",
        top_k=2,
        score_threshold=0.5,
        where={"lang": "en"},
    )
    call = store.searches[0]
    assert call["score_threshold"] == 0.5
    assert call["where"] == {"lang": "en"}


def test_reranker_changes_order() -> None:
    """A reranker that prefers lexical overlap should reorder dense results."""
    pool = [
        _hit("dense_winner", "completely off topic content", 0.95),
        _hit("lexical_winner", "this mentions the python query exactly", 0.50),
    ]
    store = FakeStore(pool)
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        reranker=KeywordOverlapReranker(alpha=0.0),  # pure lexical
        oversample_factor=2,
    )
    out = retriever.retrieve("python query", top_k=1)
    assert out[0].id == "lexical_winner"


def test_returns_top_k_after_reranking() -> None:
    pool = [_hit(f"id{i}", f"doc {i}", 0.9 - i * 0.05) for i in range(10)]
    store = FakeStore(pool)
    retriever = Retriever(
        embedder=FakeEmbedder(),
        vector_store=store,
        reranker=KeywordOverlapReranker(alpha=0.7),
    )
    out = retriever.retrieve("query", top_k=3)
    assert len(out) == 3


class _CustomReranker:
    """Verifies that custom Reranker implementations work via the Protocol."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, results: Iterable[SearchResult]) -> list[SearchResult]:
        results_list = list(results)
        self.calls.append((query, len(results_list)))
        return [replace(r, score=r.score + 1.0) for r in results_list]


def test_custom_reranker_via_protocol() -> None:
    pool = [_hit("a", "alpha", 0.5), _hit("b", "beta", 0.4)]
    store = FakeStore(pool)
    reranker = _CustomReranker()
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=store, reranker=reranker)

    out = retriever.retrieve("q", top_k=2)
    assert reranker.calls == [("q", 2)]
    assert all(r.score > 1.0 for r in out)


def test_default_reranker_is_noop() -> None:
    retriever = Retriever(embedder=FakeEmbedder(), vector_store=FakeStore([]))
    assert isinstance(retriever._reranker, NoOpReranker)
