"""Tests for :mod:`rag.generation.prompt_templates`."""

from __future__ import annotations

import pytest

from rag.generation.prompt_templates import (
    TEMPLATES,
    RenderedPrompt,
    render_prompt,
)
from rag.vectorstore.base import SearchResult


def _hit(idx: int, text: str, source: str = "doc.txt") -> SearchResult:
    return SearchResult(
        id=f"chunk-{idx}",
        text=text,
        score=0.9 - idx * 0.1,
        metadata={"source": source, "chunk_index": idx},
    )


def test_registry_has_v1_and_v2() -> None:
    assert set(TEMPLATES) == {"v1", "v2"}


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_render_returns_pair_with_correct_version(version: str) -> None:
    out = render_prompt(version, "what is X?", [_hit(0, "X is a thing.")])
    assert isinstance(out, RenderedPrompt)
    assert out.version == version
    assert out.system
    assert out.user


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_question_appears_in_user_message(version: str) -> None:
    out = render_prompt(version, "What does decorators wrap?", [_hit(0, "Decorators wrap functions.")])
    assert "decorators wrap" in out.user.lower()


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_context_chunks_are_numbered(version: str) -> None:
    chunks = [_hit(0, "alpha"), _hit(1, "beta"), _hit(2, "gamma")]
    out = render_prompt(version, "anything", chunks)
    # Every chunk must appear with its [N] marker, in order.
    for i, c in enumerate(chunks, start=1):
        assert f"[{i}]" in out.user
        assert c.text in out.user
    # Order must match the input order.
    assert out.user.index("[1]") < out.user.index("[2]") < out.user.index("[3]")


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_source_metadata_is_surfaced(version: str) -> None:
    out = render_prompt(version, "q", [_hit(0, "text", source="python_docs.txt")])
    assert "python_docs.txt" in out.user


def test_v2_includes_anti_hallucination_rules() -> None:
    out = render_prompt("v2", "q", [_hit(0, "context")])
    sys = out.system.lower()
    # Must mention NOT inventing facts and the explicit fallback string.
    assert "do not invent" in sys or "outside knowledge" in sys
    assert "i do not have enough information" in out.system.lower()


def test_v1_keeps_system_short() -> None:
    """v1 is the minimal baseline. If it grows, bump the version instead."""
    out = render_prompt("v1", "q", [_hit(0, "x")])
    assert len(out.system) < 300  # generous bound; current is ~130 chars


def test_unknown_version_raises() -> None:
    with pytest.raises(KeyError):
        render_prompt("v99", "q", [_hit(0, "x")])  # type: ignore[arg-type]


def test_empty_results_renders_without_crashing() -> None:
    out = render_prompt("v1", "lonely question", [])
    assert "lonely question" in out.user
    # Context block is empty but the prompt is still well-formed.
    assert "Context:" in out.user


def test_chunk_id_used_when_source_metadata_missing() -> None:
    chunk = SearchResult(id="orphan-chunk", text="text", score=1.0, metadata={})
    out = render_prompt("v1", "q", [chunk])
    assert "orphan-chunk" in out.user
