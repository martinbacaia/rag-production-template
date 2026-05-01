"""Tests for the Chroma-backed :class:`VectorStore`.

These run against a real, in-memory Chroma client (``EphemeralClient``).
Mocking ChromaDB would test our wiring against our own assumptions, which
is exactly the failure mode that bites in production. An in-memory client
is fast enough (sub-second per test) to use directly.
"""

from __future__ import annotations

import math
import uuid

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from rag.vectorstore.base import VectorStore
from rag.vectorstore.chroma import ChromaVectorStore


@pytest.fixture
def store() -> ChromaVectorStore:
    # ChromaDB's ``EphemeralClient`` reuses the same in-memory backend across
    # instances in the same process, so we isolate tests by using a unique
    # collection name per test.
    client = chromadb.EphemeralClient(settings=ChromaSettings(anonymized_telemetry=False))
    return ChromaVectorStore(
        collection_name=f"test_{uuid.uuid4().hex}",
        client=client,
    )


def _unit_vec(angle: float, dim: int = 4) -> list[float]:
    """Build a deterministic unit vector for similarity tests.

    A 4-D vector with the first two components on the unit circle at the
    given angle and zeros elsewhere. This lets tests reason about expected
    cosine similarity in closed form: ``cos(a - b)``.
    """
    v = [math.cos(angle), math.sin(angle)] + [0.0] * (dim - 2)
    return v


def test_implements_vector_store_interface(store: ChromaVectorStore) -> None:
    assert isinstance(store, VectorStore)


def test_add_and_count(store: ChromaVectorStore) -> None:
    assert store.count() == 0
    store.add(
        ids=["a", "b"],
        texts=["alpha", "beta"],
        embeddings=[_unit_vec(0.0), _unit_vec(1.0)],
        metadatas=[{"src": "x"}, {"src": "y"}],
    )
    assert store.count() == 2


def test_add_with_mismatched_lengths_raises(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="same length"):
        store.add(
            ids=["a"],
            texts=["alpha", "beta"],
            embeddings=[_unit_vec(0.0)],
            metadatas=[{}],
        )


def test_add_empty_input_is_noop(store: ChromaVectorStore) -> None:
    store.add(ids=[], texts=[], embeddings=[], metadatas=[])
    assert store.count() == 0


def test_search_returns_top_k_sorted(store: ChromaVectorStore) -> None:
    store.add(
        ids=["a", "b", "c"],
        texts=["close", "medium", "far"],
        embeddings=[_unit_vec(0.0), _unit_vec(0.5), _unit_vec(2.0)],
        metadatas=[{}, {}, {}],
    )
    results = store.search(query_embedding=_unit_vec(0.0), top_k=3)
    assert [r.id for r in results] == ["a", "b", "c"]
    # Descending score
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    # Identical vector pair should be at the top with similarity ~1.0
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_search_top_k_limits_results(store: ChromaVectorStore) -> None:
    store.add(
        ids=[f"id-{i}" for i in range(5)],
        texts=[f"doc {i}" for i in range(5)],
        embeddings=[_unit_vec(i * 0.3) for i in range(5)],
        metadatas=[{}] * 5,
    )
    results = store.search(query_embedding=_unit_vec(0.0), top_k=2)
    assert len(results) == 2


def test_score_threshold_filters_low_quality_hits(store: ChromaVectorStore) -> None:
    store.add(
        ids=["close", "far"],
        texts=["c", "f"],
        embeddings=[_unit_vec(0.0), _unit_vec(math.pi)],
        metadatas=[{}, {}],
    )
    # The "far" vector is at angle pi, so similarity ≈ 0.0. A 0.5 threshold
    # must drop it.
    results = store.search(
        query_embedding=_unit_vec(0.0),
        top_k=10,
        score_threshold=0.5,
    )
    assert [r.id for r in results] == ["close"]


def test_invalid_top_k_raises(store: ChromaVectorStore) -> None:
    with pytest.raises(ValueError, match="top_k"):
        store.search(query_embedding=_unit_vec(0.0), top_k=0)


def test_metadata_filter(store: ChromaVectorStore) -> None:
    store.add(
        ids=["en1", "en2", "es1"],
        texts=["english one", "english two", "spanish one"],
        embeddings=[_unit_vec(0.0), _unit_vec(0.1), _unit_vec(0.2)],
        metadatas=[{"lang": "en"}, {"lang": "en"}, {"lang": "es"}],
    )
    results = store.search(
        query_embedding=_unit_vec(0.0),
        top_k=10,
        where={"lang": "en"},
    )
    assert {r.id for r in results} == {"en1", "en2"}
    assert all(r.metadata["lang"] == "en" for r in results)


def test_upsert_overwrites_same_id(store: ChromaVectorStore) -> None:
    store.add(
        ids=["x"],
        texts=["original"],
        embeddings=[_unit_vec(0.0)],
        metadatas=[{"v": 1}],
    )
    store.add(
        ids=["x"],
        texts=["updated"],
        embeddings=[_unit_vec(0.0)],
        metadatas=[{"v": 2}],
    )
    assert store.count() == 1
    results = store.search(_unit_vec(0.0), top_k=1)
    assert results[0].text == "updated"
    assert results[0].metadata["v"] == 2


def test_delete(store: ChromaVectorStore) -> None:
    store.add(
        ids=["a", "b"],
        texts=["x", "y"],
        embeddings=[_unit_vec(0.0), _unit_vec(0.1)],
        metadatas=[{}, {}],
    )
    store.delete(["a"])
    assert store.count() == 1
    results = store.search(_unit_vec(0.0), top_k=10)
    assert [r.id for r in results] == ["b"]


def test_delete_missing_id_is_noop(store: ChromaVectorStore) -> None:
    store.add(
        ids=["a"],
        texts=["x"],
        embeddings=[_unit_vec(0.0)],
        metadatas=[{}],
    )
    store.delete(["does-not-exist"])
    assert store.count() == 1


def test_delete_empty_input_is_noop(store: ChromaVectorStore) -> None:
    store.delete([])  # must not raise


def test_reset_clears_collection(store: ChromaVectorStore) -> None:
    store.add(
        ids=["a", "b"],
        texts=["x", "y"],
        embeddings=[_unit_vec(0.0), _unit_vec(0.1)],
        metadatas=[{}, {}],
    )
    assert store.count() == 2
    store.reset()
    assert store.count() == 0
    # Still usable after reset
    store.add(ids=["c"], texts=["z"], embeddings=[_unit_vec(0.0)], metadatas=[{}])
    assert store.count() == 1
