"""Query-time retrieval.

The :class:`Retriever` is the single entry point used by the generation
layer. Its job is small but well-defined:

1. Embed the query with the configured embedder.
2. Search the vector store for ``oversample_k`` candidates (more than the
   final ``top_k`` so the reranker has options).
3. Apply the reranker.
4. Return the top ``top_k`` results.

The retriever depends on :class:`Embedder` and :class:`VectorStore` through
narrow interfaces so it can be tested in isolation with fakes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from rag.logging import get_logger
from rag.retrieval.reranker import NoOpReranker, Reranker
from rag.vectorstore.base import SearchResult, VectorStore

logger = get_logger(__name__)


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class Retriever:
    """Embed → search → rerank → return."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        reranker: Reranker | None = None,
        oversample_factor: int = 3,
    ) -> None:
        if oversample_factor < 1:
            raise ValueError("oversample_factor must be >= 1")
        self._embedder = embedder
        self._store = vector_store
        self._reranker: Reranker = reranker if reranker is not None else NoOpReranker()
        self._oversample_factor = oversample_factor

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return up to ``top_k`` reranked hits for ``query``.

        ``score_threshold`` is applied at the vector-store layer (against the
        raw dense score) before reranking, so very weak matches never reach
        the reranker. Callers that want post-rerank thresholding can do it
        themselves on the returned list.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not query.strip():
            logger.warning("retrieval.empty_query")
            return []

        query_embedding = self._embedder.embed([query])[0]

        # Oversample so the reranker has a meaningful pool to choose from;
        # cap at the final top_k when the reranker is a NoOp to avoid wasted
        # work.
        oversample_k = (
            top_k if isinstance(self._reranker, NoOpReranker) else top_k * self._oversample_factor
        )

        candidates = self._store.search(
            query_embedding=query_embedding,
            top_k=oversample_k,
            score_threshold=score_threshold,
            where=where,
        )
        logger.debug(
            "retrieval.candidates",
            query_len=len(query),
            candidates=len(candidates),
            oversample_k=oversample_k,
        )

        reranked = self._reranker.rerank(query, candidates)
        return reranked[:top_k]
