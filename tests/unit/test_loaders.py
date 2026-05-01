"""Unit tests for :mod:`rag.ingestion.loaders`."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.ingestion.loaders import (
    UnsupportedFileTypeError,
    load_document,
    supported_suffixes,
)


def test_load_text_file(tmp_path: Path) -> None:
    f = tmp_path / "doc.txt"
    f.write_text("Hello, world.", encoding="utf-8")
    doc = load_document(f)
    assert doc.text == "Hello, world."
    assert doc.source == str(f)
    assert doc.metadata["file_type"] == "txt"
    assert doc.metadata["file_name"] == "doc.txt"


def test_load_markdown_file(tmp_path: Path) -> None:
    f = tmp_path / "doc.md"
    f.write_text("# heading\n\nbody", encoding="utf-8")
    doc = load_document(f)
    assert "heading" in doc.text
    # .md is dispatched to the text loader, so file_type is "txt".
    assert doc.metadata["file_type"] == "txt"


def test_unsupported_suffix_raises(tmp_path: Path) -> None:
    f = tmp_path / "doc.docx"
    f.write_bytes(b"binary")
    with pytest.raises(UnsupportedFileTypeError):
        load_document(f)


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nope.txt")


def test_supported_suffixes_includes_txt_and_pdf() -> None:
    assert ".txt" in supported_suffixes()
    assert ".pdf" in supported_suffixes()


def test_load_pdf_returns_text(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pdf_path = _write_minimal_pdf(tmp_path / "doc.pdf", "Hello PDF world")
    doc = load_document(pdf_path)
    assert "Hello" in doc.text
    assert doc.metadata["file_type"] == "pdf"
    assert doc.metadata["page_count"] == 1


def _write_minimal_pdf(path: Path, body: str) -> Path:
    """Create a tiny but valid one-page PDF using pypdf.

    Avoids depending on an external fixture file so tests stay self-contained.
    """
    from pypdf import PdfWriter
    from pypdf.generic import (
        ArrayObject,
        DecodedStreamObject,
        DictionaryObject,
        FloatObject,
        NameObject,
        NumberObject,
    )

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    content = DecodedStreamObject()
    content.set_data(
        f"BT /F1 12 Tf 50 250 Td ({body}) Tj ET".encode("latin-1")
    )
    page[NameObject("/Contents")] = content
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    page[NameObject("/MediaBox")] = ArrayObject(
        [NumberObject(0), NumberObject(0), FloatObject(300), FloatObject(300)]
    )
    with path.open("wb") as fh:
        writer.write(fh)
    return path
