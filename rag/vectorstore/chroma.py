"""ChromaDB-backed :class:`VectorStore` implementation.

Two modes are supported:

* **Persistent** (default): a ``PersistentClient`` writes to ``persist_dir``.
  Suitable for development and small production deployments.
* **Ephemeral**: an in-memory ``EphemeralClient`` that vanishes with the
  process. Used by the test suite to keep test runs fast and stateless.

The class is intentionally thin: ChromaDB already does the heavy lifting
(HNSW index, metadata filtering, persistence). Our job is to translate
between the project's neutral types (:class:`SearchResult`, plain dicts)
and Chroma's API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings as ChromaSettings

from rag.logging import get_logger
from rag.vectorstore.base import SearchResult, VectorStore

logger = get_logger(__name__)

# ChromaDB rejects empty metadata dicts (it requires at least one key).
# We inject this sentinel on write and strip it on read so callers can
# pass ``{}`` freely without leaking the implementation detail.
_EMPTY_META_SENTINEL = "__rag_empty__"


def _to_chroma_metadata(m: dict[str, Any]) -> dict[str, Any]:
    return m if m else {_EMPTY_META_SENTINEL: True}


def _from_chroma_metadata(m: Mapping[str, Any] | None) -> dict[str, Any]:
    if not m:
        return {}
    return {k: v for k, v in m.items() if k != _EMPTY_META_SENTINEL}


class ChromaVectorStore(VectorStore):
    """:class:`VectorStore` implementation backed by ChromaDB."""

    def __init__(
        self,
        collection_name: str,
        persist_dir: str | None = None,
        *,
        client: ClientAPI | None = None,
    ) -> None:
        """Construct the store.

        Parameters
        ----------
        collection_name:
            Name of the Chroma collection. Created if it does not exist.
        persist_dir:
            On-disk directory for the persistent client. Ignored when
            ``client`` is supplied explicitly. Pass ``None`` along with an
            ephemeral ``client`` to run in-memory.
        client:
            Optional pre-configured Chroma client. Useful for tests (inject
            an ``EphemeralClient``) and for advanced setups (remote server,
            custom telemetry).
        """
        if client is not None:
            self._client = client
        elif persist_dir is not None:
            self._client = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.EphemeralClient(
                settings=ChromaSettings(anonymized_telemetry=False),
            )

        self._collection_name = collection_name
        # Cosine distance is the right default for OpenAI embeddings, which
        # are L2-normalized — cosine and dot product give the same ordering
        # but cosine maps cleanly into the [0, 1] similarity range used by
        # SearchResult.
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "ids, texts, embeddings, and metadatas must have the same length; "
                f"got {len(ids)}, {len(texts)}, {len(embeddings)}, {len(metadatas)}"
            )
        if len(ids) == 0:
            return

        # ``upsert`` (vs ``add``) lets us re-ingest the same chunk ids without
        # raising — which is exactly what deterministic ids in the ingestion
        # pipeline are designed to take advantage of.
        # Chroma's typing accepts ``list[Sequence[float]]`` but mypy does not
        # widen ``list[list[float]]`` automatically; cast keeps the runtime
        # behavior unchanged while satisfying the strict signature.
        self._collection.upsert(
            ids=list(ids),
            documents=list(texts),
            embeddings=cast("list[Sequence[float]]", [list(e) for e in embeddings]),
            metadatas=[_to_chroma_metadata(dict(m)) for m in metadatas],
        )
        logger.debug("vectorstore.upsert", count=len(ids))

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        # ``query`` returns parallel lists, one per query. We always send a
        # single query so we read index ``[0]`` everywhere.
        result = self._collection.query(
            query_embeddings=cast("list[Sequence[float]]", [list(query_embedding)]),
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchResult] = []
        for hit_id, doc, meta, dist in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            # Chroma cosine distance is in [0, 2]. Map to a similarity in
            # [0, 1] via ``1 - dist/2``. Clamp to handle floating-point drift
            # at the boundaries.
            similarity = max(0.0, min(1.0, 1.0 - (float(dist) / 2.0)))
            if similarity < score_threshold:
                continue
            hits.append(
                SearchResult(
                    id=hit_id,
                    text=doc or "",
                    score=similarity,
                    metadata=_from_chroma_metadata(meta),
                )
            )
        return hits

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=list(ids))
        logger.debug("vectorstore.delete", count=len(ids))

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        # Recreate the collection rather than deleting individual records:
        # cheaper, and side-steps Chroma quirks around bulk deletes.
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("vectorstore.reset", collection=self._collection_name)
