"""Vector store factory.

Reads :class:`rag.config.Settings` and returns the concrete
:class:`VectorStore` implementation for the configured backend.
"""

from __future__ import annotations

from rag.config import Settings
from rag.vectorstore.base import VectorStore
from rag.vectorstore.chroma import ChromaVectorStore
from rag.vectorstore.qdrant import QdrantVectorStore


def build_vector_store(settings: Settings) -> VectorStore:
    """Construct a :class:`VectorStore` based on ``settings.vector_store_backend``."""
    backend = settings.vector_store_backend
    if backend == "chroma":
        return ChromaVectorStore(
            collection_name=settings.chroma_collection,
            persist_dir=settings.chroma_persist_dir,
        )
    if backend == "qdrant":
        return QdrantVectorStore()
    # The Literal type on Settings makes this branch unreachable at runtime,
    # but mypy still wants a fallback and an explicit error is friendlier
    # than a silent ``None`` if someone widens the type later.
    raise ValueError(f"Unknown vector_store_backend: {backend!r}")
