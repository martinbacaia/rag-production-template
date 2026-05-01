"""Tests for :mod:`rag.retrieval.reranker`."""

from __future__ import annotations

import pytest

from rag.retrieval.reranker import (
    KeywordOverlapReranker,
    NoOpReranker,
    _jaccard,
    _tokenize,
)
from rag.vectorstore.base import SearchResult


def _hit(id_: str, text: str, score: float) -> SearchResult:
    return SearchResult(id=id_, text=text, score=score, metadata={})


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert _tokenize("The Quick, brown FOX!") == {"the", "quick", "brown", "fox"}


def test_tokenize_empty_input() -> None:
    assert _tokenize("") == set()
    assert _tokenize("...!!!") == set()


def test_jaccard_known_values() -> None:
    a = {"apple", "banana", "cherry"}
    b = {"banana", "cherry", "date"}
    # |∩| = 2, |∪| = 4
    assert _jaccard(a, b) == 0.5
    assert _jaccard(a, a) == 1.0
    assert _jaccard(a, set()) == 0.0
    assert _jaccard(set(), set()) == 0.0


def test_noop_reranker_preserves_order() -> None:
    hits = [_hit("a", "x", 0.9), _hit("b", "y", 0.5), _hit("c", "z", 0.1)]
    out = NoOpReranker().rerank("query text", hits)
    assert [r.id for r in out] == ["a", "b", "c"]
    # NoOp must not mutate scores
    assert [r.score for r in out] == [0.9, 0.5, 0.1]


def test_keyword_overlap_reranker_alpha_validation() -> None:
    with pytest.raises(ValueError):
        KeywordOverlapReranker(alpha=-0.1)
    with pytest.raises(ValueError):
        KeywordOverlapReranker(alpha=1.1)


def test_keyword_overlap_pure_dense_when_alpha_is_one() -> None:
    """alpha=1.0 ⇒ keyword signal contributes 0 ⇒ ordering follows dense scores."""
    reranker = KeywordOverlapReranker(alpha=1.0)
    hits = [
        _hit("a", "completely unrelated content", 0.9),
        _hit("b", "matching python query terms", 0.5),
    ]
    out = reranker.rerank("python query", hits)
    assert [r.id for r in out] == ["a", "b"]


def test_keyword_overlap_pure_lexical_when_alpha_is_zero() -> None:
    """alpha=0.0 ⇒ ordering follows token overlap with the query."""
    reranker = KeywordOverlapReranker(alpha=0.0)
    hits = [
        _hit("a", "completely unrelated content", 0.95),  # no keyword overlap
        _hit("b", "matching python query terms", 0.10),  # full overlap
    ]
    out = reranker.rerank("python query", hits)
    assert out[0].id == "b"


def test_keyword_overlap_breaks_ties() -> None:
    """With identical dense scores, the result containing the query keywords wins."""
    reranker = KeywordOverlapReranker(alpha=0.7)
    hits = [
        _hit("a", "no relevance here whatsoever", 0.7),
        _hit("b", "this paragraph mentions the python query directly", 0.7),
    ]
    out = reranker.rerank("python query", hits)
    assert out[0].id == "b"
    # Final score must be in [0, 1] since both inputs were
    assert 0.0 <= out[0].score <= 1.0


def test_keyword_overlap_handles_empty_input() -> None:
    reranker = KeywordOverlapReranker(alpha=0.7)
    assert reranker.rerank("anything", []) == []


def test_keyword_overlap_does_not_mutate_input() -> None:
    """Reranker must return new SearchResult instances — the input list
    should not be modified, since SearchResult is frozen anyway, but the
    caller may reuse the original list."""
    reranker = KeywordOverlapReranker(alpha=0.5)
    original = [
        _hit("a", "alpha", 0.9),
        _hit("b", "beta", 0.5),
    ]
    snapshot = [(r.id, r.score) for r in original]
    reranker.rerank("alpha", original)
    assert [(r.id, r.score) for r in original] == snapshot
