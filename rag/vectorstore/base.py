"""Abstract vector store interface.

The interface is intentionally small (six methods) and synchronous. Async
support belongs at the API layer, not here — most vector store SDKs already
batch internally, and an async wrapper can be added without changing this
contract.

Backends implement :class:`VectorStore`. The narrower :class:`Embedder` and
``VectorStore`` ``Protocol`` exposed in :mod:`rag.ingestion.pipeline` is a
subset of this interface so that the ingestion layer can stay decoupled from
the full search-side surface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single similarity-search hit.

    ``score`` is normalized to ``[0.0, 1.0]`` where 1.0 is identical. Backends
    that natively return distances are responsible for the conversion before
    handing results out.
    """

    id: str
    text: str
    score: float
    metadata: dict[str, Any]


class VectorStore(ABC):
    """Backend-agnostic vector store contract.

    The contract is:

    * ``add`` upserts records keyed by ``id``. Re-adding the same id overwrites.
    * ``search`` runs nearest-neighbor lookup against a query embedding,
      optionally filtered by metadata and a minimum score.
    * ``delete`` removes records by id; missing ids are silently skipped.
    * ``count`` returns the current number of records.
    * ``reset`` wipes the collection (intended for tests and eval setup).
    """

    @abstractmethod
    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        """Upsert ``len(ids)`` records into the store."""

    @abstractmethod
    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return up to ``top_k`` results sorted by descending ``score``.

        ``score_threshold`` filters out hits below the threshold *after*
        retrieval, so the caller can request a generous ``top_k`` and still
        rely on the threshold to prune low-quality matches.

        ``where`` is a backend-agnostic metadata filter. Each implementation
        documents the filter syntax it supports; for ChromaDB the dict is
        passed directly to its ``where`` parameter.
        """

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None:
        """Remove records matching the given ids. Missing ids are ignored."""

    @abstractmethod
    def count(self) -> int:
        """Return the current number of stored records."""

    @abstractmethod
    def reset(self) -> None:
        """Drop all records. Used by tests and the eval harness."""
