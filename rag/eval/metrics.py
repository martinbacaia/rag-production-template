"""Evaluation metrics.

These metrics are intentionally heuristic — they are deterministic
(reproducible across runs), fast (no extra LLM calls), and good enough
to catch regressions. They are *not* a substitute for a full LLM-as-
judge framework like RAGAS for absolute scoring; see the README for
the production trade-offs.

Three metrics are exposed:

* :func:`context_precision` — fraction of retrieved chunks that contain
  at least one expected substring. Measures whether retrieval is
  surfacing relevant material.
* :func:`answer_substring_recall` — fraction of expected substrings
  that appear in the answer. Measures whether the generator is using
  the relevant facts. Acts as a proxy for answer correctness.
* :func:`answer_relevancy` — Jaccard token overlap between question
  and answer. Cheap proxy for "does the answer address the question".

A higher-fidelity faithfulness check would parse claims with an LLM
and verify each against the context. We trade that fidelity for speed
and determinism in CI.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from rag.vectorstore.base import SearchResult

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text)}


def _contains_any_substring(text: str, substrings: Iterable[str]) -> bool:
    text_l = text.lower()
    return any(s.lower() in text_l for s in substrings)


def context_precision(
    retrieved: Sequence[SearchResult],
    expected_substrings_in_context: Sequence[str],
) -> float:
    """Fraction of retrieved chunks that contain at least one expected substring.

    A score of 1.0 means every retrieved chunk is on-topic; 0.0 means
    none of them are. The metric is undefined when the question has no
    expected context (e.g. unanswerable control questions) — in that
    case we return 1.0 to avoid penalizing the system for retrieving
    irrelevant chunks for a question that has no right answer anyway.
    """
    if not expected_substrings_in_context:
        return 1.0
    if not retrieved:
        return 0.0
    relevant = sum(
        1
        for chunk in retrieved
        if _contains_any_substring(chunk.text, expected_substrings_in_context)
    )
    return relevant / len(retrieved)


def answer_substring_recall(
    answer: str,
    expected_substrings_in_answer: Sequence[str],
) -> float:
    """Fraction of expected substrings that appear in the answer.

    With no expected substrings, returns 1.0 (vacuously satisfied). This
    is the metric the eval harness reports as ``answer_correctness``.
    """
    if not expected_substrings_in_answer:
        return 1.0
    answer_l = answer.lower()
    found = sum(1 for s in expected_substrings_in_answer if s.lower() in answer_l)
    return found / len(expected_substrings_in_answer)


def answer_relevancy(question: str, answer: str) -> float:
    """Jaccard overlap between question and answer tokens.

    Cheap proxy for "the answer addresses the question". A near-zero
    score is a strong signal of off-topic generation; a near-one score
    is *not* by itself proof of correctness (the answer could echo the
    question without saying anything useful).
    """
    q_tokens = _tokenize(question)
    a_tokens = _tokenize(answer)
    if not q_tokens or not a_tokens:
        return 0.0
    return len(q_tokens & a_tokens) / len(q_tokens | a_tokens)
