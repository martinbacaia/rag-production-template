"""Unit tests for :mod:`rag.eval.metrics`."""

from __future__ import annotations

import pytest

from rag.eval.metrics import (
    answer_relevancy,
    answer_substring_recall,
    context_precision,
)
from rag.vectorstore.base import SearchResult


def _hit(text: str) -> SearchResult:
    return SearchResult(id="x", text=text, score=1.0, metadata={})


# -- context_precision -------------------------------------------------------


def test_context_precision_perfect_match() -> None:
    chunks = [_hit("decorators wrap functions"), _hit("decorators are syntactic sugar")]
    assert context_precision(chunks, ["decorator"]) == 1.0


def test_context_precision_no_relevant_chunks() -> None:
    chunks = [_hit("unrelated content"), _hit("also unrelated")]
    assert context_precision(chunks, ["decorator"]) == 0.0


def test_context_precision_partial() -> None:
    chunks = [
        _hit("decorators wrap functions"),
        _hit("unrelated"),
        _hit("decorators use @ syntax"),
        _hit("totally off-topic"),
    ]
    assert context_precision(chunks, ["decorator"]) == 0.5


def test_context_precision_empty_expected_returns_one() -> None:
    """No expected substrings = vacuously precise (used for unanswerable items)."""
    assert context_precision([_hit("anything")], []) == 1.0


def test_context_precision_no_chunks_returns_zero() -> None:
    assert context_precision([], ["something"]) == 0.0


def test_context_precision_case_insensitive() -> None:
    chunks = [_hit("DECORATORS in caps")]
    assert context_precision(chunks, ["decorator"]) == 1.0


def test_context_precision_any_substring_counts() -> None:
    chunks = [_hit("alpha"), _hit("beta")]
    # Each chunk only needs to match ONE expected substring.
    assert context_precision(chunks, ["alpha", "beta", "gamma"]) == 1.0


# -- answer_substring_recall ------------------------------------------------


def test_answer_substring_recall_all_present() -> None:
    answer = "Decorators wrap functions to add behavior."
    assert answer_substring_recall(answer, ["decorator", "wrap", "behavior"]) == 1.0


def test_answer_substring_recall_partial() -> None:
    answer = "Decorators wrap functions."
    assert answer_substring_recall(answer, ["decorator", "wrap", "missing-word"]) == pytest.approx(
        2 / 3
    )


def test_answer_substring_recall_none_present() -> None:
    answer = "completely unrelated answer"
    assert answer_substring_recall(answer, ["decorator", "wrap"]) == 0.0


def test_answer_substring_recall_empty_expected_returns_one() -> None:
    assert answer_substring_recall("any answer", []) == 1.0


# -- answer_relevancy -------------------------------------------------------


def test_answer_relevancy_strong_overlap() -> None:
    score = answer_relevancy(
        "What does the property decorator do?",
        "The property decorator turns a method into a getter.",
    )
    assert score > 0.2  # at least some shared tokens


def test_answer_relevancy_zero_overlap() -> None:
    assert answer_relevancy("What is X?", "Completely unrelated content here.") == 0.0


def test_answer_relevancy_empty_inputs() -> None:
    assert answer_relevancy("", "answer") == 0.0
    assert answer_relevancy("question", "") == 0.0


def test_answer_relevancy_identical_returns_one() -> None:
    """Question repeated as answer has maximum Jaccard."""
    assert answer_relevancy("alpha beta gamma", "alpha beta gamma") == 1.0
