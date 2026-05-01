"""Text chunking.

The strategy is a *boundary-aware sliding window* over characters:

1. Walk the text in windows of ``chunk_size`` characters with ``chunk_overlap``
   characters of overlap between consecutive windows.
2. When a window would split mid-sentence, snap the cut to the nearest
   sentence/paragraph boundary within a tolerance band. This avoids the
   classic "cut a sentence in half" problem without requiring a tokenizer.
3. Emit empty-stripped chunks only.

The implementation is deliberately library-free: no LangChain, no NLTK. This
keeps the dependency surface small and makes the behavior easy to reason
about — useful for a recruiter reading the code, and for debugging eval
regressions later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Sentence terminators we treat as good cut points. The list is intentionally
# conservative: aggressive boundary detection costs recall on documents that
# don't end sentences with punctuation (e.g. headings, bullet lists).
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A chunk of text plus the metadata needed to cite it later."""

    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Configuration for :func:`chunk_text`.

    ``boundary_tolerance`` controls how far we are willing to drift from the
    target window end to land on a sentence boundary. Expressed as a fraction
    of ``chunk_size``; 0.0 disables boundary-snapping entirely.
    """

    chunk_size: int
    chunk_overlap: int
    boundary_tolerance: float = 0.15

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if not 0.0 <= self.boundary_tolerance <= 0.5:
            raise ValueError("boundary_tolerance must be in [0.0, 0.5]")


def _find_boundary(text: str, target_end: int, tolerance: int) -> int:
    """Return the best cut position within ``[target_end - tolerance, target_end]``.

    "Best" = the rightmost sentence terminator inside the window. Falls back
    to ``target_end`` if no terminator is found.
    """
    if tolerance == 0 or target_end >= len(text):
        return min(target_end, len(text))

    window_start = max(target_end - tolerance, 0)
    window = text[window_start:target_end]
    matches = list(_SENTENCE_END.finditer(window))
    if not matches:
        return target_end
    # Prefer the latest boundary so we keep chunks as close to chunk_size as possible.
    return window_start + matches[-1].end()


def chunk_text(
    text: str,
    source: str,
    config: ChunkingConfig,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Split ``text`` into a list of :class:`Chunk` objects.

    Returns an empty list for empty / whitespace-only input.
    """
    if not text or not text.strip():
        return []

    tolerance = int(config.chunk_size * config.boundary_tolerance)
    chunks: list[Chunk] = []

    start = 0
    chunk_index = 0
    text_len = len(text)
    extra = extra_metadata or {}

    while start < text_len:
        target_end = min(start + config.chunk_size, text_len)
        end = _find_boundary(text, target_end, tolerance)

        # Boundary snapping must always make forward progress.
        if end <= start:
            end = target_end

        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    chunk_index=chunk_index,
                    start_char=start,
                    end_char=end,
                    metadata=dict(extra),
                )
            )
            chunk_index += 1

        if end >= text_len:
            break

        # Slide the window forward, applying overlap. The ``max`` guard
        # prevents pathological inputs from creating a zero-advance loop.
        next_start = end - config.chunk_overlap
        start = max(next_start, start + 1)

    return chunks
