"""Tests for :mod:`rag.vectorstore.factory`."""

from __future__ import annotations

import pytest

from rag.config import Settings
from rag.vectorstore.chroma import ChromaVectorStore
from rag.vectorstore.factory import build_vector_store


def test_factory_builds_chroma(tmp_path) -> None:
    settings = Settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        vector_store_backend="chroma",
    )
    store = build_vector_store(settings)
    assert isinstance(store, ChromaVectorStore)


def test_factory_qdrant_raises_not_implemented(tmp_path) -> None:
    settings = Settings(
        chroma_persist_dir=str(tmp_path / "chroma"),
        vector_store_backend="qdrant",
    )
    with pytest.raises(NotImplementedError):
        build_vector_store(settings)
