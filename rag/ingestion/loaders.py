"""Document loaders that turn a file on disk into raw text.

Each loader exposes a single :func:`load` function returning a
:class:`LoadedDocument`. The loader registry dispatches by file suffix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """A document after loading but before chunking.

    Parameters
    ----------
    source:
        Stable identifier for the document, typically the file path.
    text:
        Full document text. May contain newlines but should not contain
        binary data.
    metadata:
        Arbitrary key/value metadata attached to the document. Values must be
        JSON-serializable to be stored in the vector store.
    """

    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_text(path: Path) -> LoadedDocument:
    """Load a UTF-8 plain-text file."""
    text = path.read_text(encoding="utf-8")
    return LoadedDocument(
        source=str(path),
        text=text,
        metadata={"file_type": "txt", "file_name": path.name},
    )


def load_pdf(path: Path) -> LoadedDocument:
    """Extract text from a PDF, page by page.

    Pages are joined with a form-feed character (``\\f``) so that downstream
    consumers can recover page boundaries if needed without relying on
    metadata.
    """
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\f".join(pages)
    return LoadedDocument(
        source=str(path),
        text=text,
        metadata={
            "file_type": "pdf",
            "file_name": path.name,
            "page_count": len(pages),
        },
    )


_LOADERS: dict[str, Callable[[Path], LoadedDocument]] = {
    ".txt": load_text,
    ".md": load_text,
    ".pdf": load_pdf,
}


class UnsupportedFileTypeError(ValueError):
    """Raised when no loader is registered for a given file suffix."""


def load_document(path: Path | str) -> LoadedDocument:
    """Dispatch to the correct loader based on file suffix."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    loader = _LOADERS.get(p.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(
            f"No loader registered for suffix {p.suffix!r}; "
            f"supported: {sorted(_LOADERS)}"
        )
    return loader(p)


def supported_suffixes() -> tuple[str, ...]:
    """Return the file suffixes for which a loader is registered."""
    return tuple(sorted(_LOADERS))
