"""Ingestion pipeline: orchestrates loading, chunking, embedding, and storage.

The pipeline depends on the vector store and embedding client through narrow
``Protocol`` interfaces so unit tests can inject lightweight fakes without
spinning up Chroma or hitting OpenAI.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from rag.ingestion.chunker import Chunk, ChunkingConfig, chunk_text
from rag.ingestion.loaders import load_document
from rag.logging import get_logger

logger = get_logger(__name__)


class Embedder(Protocol):
    """Anything that can turn a list of strings into a list of vectors."""

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class VectorStore(Protocol):
    """Minimal vector store contract used during ingestion.

    The full interface (with search) lives in :mod:`rag.vectorstore.base`;
    this Protocol mirrors only the methods the ingestion pipeline calls so
    the modules stay loosely coupled.
    """

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Summary of an ingestion run."""

    source: str
    num_chunks: int
    chunk_ids: list[str]


def _chunk_id(source: str, chunk_index: int, text: str) -> str:
    """Deterministic chunk id: ``{source-hash}:{chunk_index}:{text-hash}``.

    Including a content hash makes the id stable across re-ingests of the
    same content, but it changes if the text changes — so re-ingesting an
    edited document inserts new ids rather than silently overwriting.
    """
    src_h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    text_h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    return f"{src_h}:{chunk_index}:{text_h}"


class IngestionPipeline:
    """End-to-end ingestion: file → chunks → embeddings → store."""

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunking_config: ChunkingConfig,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._chunking = chunking_config

    def ingest_file(self, path: Path | str) -> IngestionResult:
        """Load, chunk, embed, and store a single file."""
        document = load_document(path)
        chunks = chunk_text(
            document.text,
            source=document.source,
            config=self._chunking,
            extra_metadata=document.metadata,
        )
        return self._store_chunks(chunks)

    def ingest_text(
        self,
        text: str,
        source: str,
        extra_metadata: dict[str, Any] | None = None,
    ) -> IngestionResult:
        """Ingest raw text under an arbitrary ``source`` identifier.

        Useful for tests, evals, and ingesting content from non-file origins
        (e.g. scraped documents passed via API).
        """
        chunks = chunk_text(
            text,
            source=source,
            config=self._chunking,
            extra_metadata=extra_metadata,
        )
        return self._store_chunks(chunks)

    def ingest_paths(self, paths: Iterable[Path | str]) -> list[IngestionResult]:
        """Ingest each path independently, returning per-file results."""
        return [self.ingest_file(p) for p in paths]

    def _store_chunks(self, chunks: list[Chunk]) -> IngestionResult:
        if not chunks:
            source = "<empty>"
            logger.warning("ingestion.empty_chunks", source=source)
            return IngestionResult(source=source, num_chunks=0, chunk_ids=[])

        ids = [_chunk_id(c.source, c.chunk_index, c.text) for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [
            {
                **c.metadata,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
            }
            for c in chunks
        ]

        embeddings = self._embedder.embed(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError(
                f"Embedder returned {len(embeddings)} vectors for {len(texts)} chunks"
            )

        self._store.add(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
        logger.info(
            "ingestion.stored",
            source=chunks[0].source,
            num_chunks=len(chunks),
        )
        return IngestionResult(
            source=chunks[0].source,
            num_chunks=len(chunks),
            chunk_ids=ids,
        )
