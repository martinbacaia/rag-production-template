"""Reranking layer.

Initial dense retrieval scores chunks by embedding similarity. Reranking
re-scores those candidates with a complementary signal (lexical overlap, a
cross-encoder, etc.) so that the top-k passed to the LLM is higher quality
than what dense retrieval alone returns.

This module provides:

* :class:`Reranker` — the protocol the retriever depends on.
* :class:`NoOpReranker` — pass-through, useful as a default and for ablation
  studies in the eval harness.
* :class:`KeywordOverlapReranker` — a tiny hybrid reranker that blends dense
  similarity with Jaccard overlap between query and document tokens.

Why a tiny lexical reranker rather than a cross-encoder? Three reasons:

1. **No extra model dependency.** A cross-encoder ships hundreds of MB of
   weights; a token-overlap step is ~30 lines of Python.
2. **It actually moves the needle.** Dense retrieval misses exact-term
   matches (numbers, IDs, code snippets); lexical scoring catches them.
   This is the same intuition behind hybrid BM25 + dense pipelines.
3. **It's debuggable.** A reviewer can read this file and see exactly why
   a result moved up or down.

A cross-encoder slot exists conceptually — a real implementation would
implement :class:`Reranker` and live alongside this file.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from rag.vectorstore.base import SearchResult

# Conservative tokenization: word characters only, lowercased. Splitting on
# punctuation avoids matching "model." against "model," but keeps things
# language-neutral enough for our test corpora (English Python docs).
_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


class Reranker(Protocol):
    """Anything that can re-order a list of :class:`SearchResult`."""

    def rerank(self, query: str, results: Iterable[SearchResult]) -> list[SearchResult]: ...


class NoOpReranker:
    """Pass-through reranker. Used as the default and in ablations."""

    def rerank(self, query: str, results: Iterable[SearchResult]) -> list[SearchResult]:
        return list(results)


class KeywordOverlapReranker:
    """Hybrid reranker: ``alpha * vector_score + (1 - alpha) * jaccard``.

    Parameters
    ----------
    alpha:
        Weight given to the dense score. ``1.0`` falls back to dense-only
        ranking (effectively a NoOp), ``0.0`` is pure lexical, ``0.7`` is a
        sensible default that biases toward dense while letting strong
        keyword matches break ties.
    """

    def __init__(self, alpha: float = 0.7) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0.0, 1.0]")
        self._alpha = alpha

    def rerank(self, query: str, results: Iterable[SearchResult]) -> list[SearchResult]:
        query_tokens = _tokenize(query)
        rescored: list[SearchResult] = []
        for r in results:
            overlap = _jaccard(query_tokens, _tokenize(r.text))
            blended = self._alpha * r.score + (1.0 - self._alpha) * overlap
            rescored.append(replace(r, score=blended))
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored
