"""Unit tests for :mod:`rag.ingestion.chunker`."""

from __future__ import annotations

import pytest

from rag.ingestion.chunker import ChunkingConfig, chunk_text


def test_empty_input_returns_no_chunks() -> None:
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    assert chunk_text("", source="x", config=cfg) == []
    assert chunk_text("   \n\n  ", source="x", config=cfg) == []


def test_short_text_yields_single_chunk() -> None:
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    chunks = chunk_text("Hello world.", source="doc", config=cfg)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."
    assert chunks[0].chunk_index == 0
    assert chunks[0].source == "doc"
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len("Hello world.")


def test_long_text_is_split_with_overlap() -> None:
    cfg = ChunkingConfig(chunk_size=50, chunk_overlap=10, boundary_tolerance=0.0)
    text = "A" * 200
    chunks = chunk_text(text, source="doc", config=cfg)
    assert len(chunks) >= 4
    # chunk_index is monotonic and starts at 0
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # Every chunk except possibly the last is exactly chunk_size long
    for c in chunks[:-1]:
        assert c.end_char - c.start_char == cfg.chunk_size
    # Consecutive windows overlap by chunk_overlap (when boundary tolerance is 0)
    for prev, curr in zip(chunks, chunks[1:], strict=False):
        assert prev.end_char - curr.start_char == cfg.chunk_overlap


def test_boundary_snapping_prefers_sentence_end() -> None:
    cfg = ChunkingConfig(chunk_size=60, chunk_overlap=10, boundary_tolerance=0.3)
    # Two sentences. The first ends at char 24. With chunk_size 60 and a 30%
    # tolerance window (18 chars: 42..60), the cut should snap to the
    # following sentence end if reachable, otherwise stay at target_end.
    text = (
        "The quick brown fox jumps. "  # 27
        "Then a slow turtle waddles by. "  # +31 = 58
        "And finally a parrot squawks."  # +30 = 88
    )
    chunks = chunk_text(text, source="doc", config=cfg)
    # First chunk should not break a sentence — it should end with "by." or "jumps."
    assert chunks[0].text.rstrip().endswith((".", "!"))
    assert "fox jumps." in chunks[0].text


def test_chunks_are_stripped() -> None:
    cfg = ChunkingConfig(chunk_size=20, chunk_overlap=2, boundary_tolerance=0.0)
    text = "   leading and trailing whitespace inside chunks   "
    chunks = chunk_text(text, source="doc", config=cfg)
    for c in chunks:
        assert c.text == c.text.strip()
        assert c.text != ""


def test_extra_metadata_is_propagated() -> None:
    cfg = ChunkingConfig(chunk_size=100, chunk_overlap=10)
    chunks = chunk_text(
        "Some text content here.",
        source="doc",
        config=cfg,
        extra_metadata={"file_type": "txt", "author": "ada"},
    )
    assert chunks[0].metadata == {"file_type": "txt", "author": "ada"}


@pytest.mark.parametrize(
    ("size", "overlap", "tolerance"),
    [
        (0, 10, 0.1),  # size must be > 0
        (100, -1, 0.1),  # overlap must be >= 0
        (100, 100, 0.1),  # overlap must be < size
        (100, 110, 0.1),  # overlap must be < size
        (100, 10, -0.1),  # tolerance must be in [0, 0.5]
        (100, 10, 0.6),
    ],
)
def test_invalid_config_rejected(size: int, overlap: int, tolerance: float) -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=size, chunk_overlap=overlap, boundary_tolerance=tolerance)


def test_makes_progress_on_pathological_overlap() -> None:
    """Regression: with very large overlap relative to a small advance band,
    the loop must still terminate."""
    cfg = ChunkingConfig(chunk_size=50, chunk_overlap=49, boundary_tolerance=0.0)
    text = "x" * 200
    chunks = chunk_text(text, source="doc", config=cfg)
    # We don't care about the exact count, only that we terminate and cover
    # the input.
    assert chunks[-1].end_char == len(text)
    assert len(chunks) < len(text)  # not one-chunk-per-character
