"""Qdrant-backed :class:`VectorStore` (stub).

This module is deliberately a stub. It exists to document the extension
point: swapping vector stores is a single class away. A full implementation
would:

* Open a ``QdrantClient`` (local on-disk, or remote HTTP/gRPC).
* ``upsert`` ``PointStruct`` records with ids, vectors, and payloads.
* ``search`` with ``query_vector`` and a ``Filter`` for metadata.
* Translate Qdrant ``score`` (cosine similarity, already in ``[-1, 1]``)
  into our ``[0, 1]`` :class:`SearchResult.score` via ``(score + 1) / 2``.

The stub is intentionally raised rather than silently no-oping so that
selecting the ``qdrant`` backend in :mod:`rag.config` without a real
implementation fails loudly at construction time.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rag.vectorstore.base import SearchResult, VectorStore


class QdrantVectorStore(VectorStore):
    """Placeholder. Replace with a real ``qdrant-client`` integration."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Qdrant backend is not implemented in this template. "
            "Add ``qdrant-client`` to requirements and implement the five "
            "VectorStore methods to enable it."
        )

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError

    def delete(self, ids: Sequence[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
