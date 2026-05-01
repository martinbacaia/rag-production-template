"""FastAPI dependency providers.

Each ``get_*`` function is a FastAPI dependency. Tests override them via
``app.dependency_overrides`` to inject fakes (a fake OpenAI client, an
in-memory vector store), without touching production code.

We expose ``Annotated[T, Depends(...)]`` aliases (``SettingsDep``,
``OpenAIClientDep``, ...) so route signatures read as plain typed
parameters. This is the modern FastAPI idiom; the older
``param: T = Depends(...)`` form trips on ruff's B008 rule.

The dependency chain is::

    Settings
      ├── OpenAI client  ──── OpenAIEmbedder ────┐
      │                                          ├── Retriever ── Generator
      │                                          │
      │                  ── ChromaVectorStore ───┘
      └── ChunkingConfig ── IngestionPipeline (uses Embedder + VectorStore)

All providers cache their result via ``functools.lru_cache``: an instance
built on the first request is reused for the lifetime of the worker.
Workers re-build them on cold start, the right granularity for vector-
store handles and HTTP clients.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from openai import OpenAI

from rag.config import Settings, get_settings
from rag.generation.generator import Generator
from rag.generation.llm_client import OpenAIChatClient, OpenAIEmbedder
from rag.ingestion.chunker import ChunkingConfig
from rag.ingestion.pipeline import IngestionPipeline
from rag.retrieval.reranker import KeywordOverlapReranker
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import VectorStore
from rag.vectorstore.factory import build_vector_store

SettingsDep = Annotated[Settings, Depends(get_settings)]


@lru_cache(maxsize=1)
def _cached_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def get_openai_client(settings: SettingsDep) -> OpenAI:
    return _cached_openai_client(settings.openai_api_key)


@lru_cache(maxsize=1)
def _cached_vector_store(settings: Settings) -> VectorStore:
    return build_vector_store(settings)


def get_vector_store(settings: SettingsDep) -> VectorStore:
    return _cached_vector_store(settings)


OpenAIClientDep = Annotated[OpenAI, Depends(get_openai_client)]
VectorStoreDep = Annotated[VectorStore, Depends(get_vector_store)]


def get_embedder(client: OpenAIClientDep, settings: SettingsDep) -> OpenAIEmbedder:
    return OpenAIEmbedder(client=client, model=settings.openai_embedding_model)


def get_chat_client(
    client: OpenAIClientDep, settings: SettingsDep
) -> OpenAIChatClient:
    return OpenAIChatClient(
        client=client,
        model=settings.openai_llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


def get_chunking_config(settings: SettingsDep) -> ChunkingConfig:
    return ChunkingConfig(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


EmbedderDep = Annotated[OpenAIEmbedder, Depends(get_embedder)]
ChatClientDep = Annotated[OpenAIChatClient, Depends(get_chat_client)]
ChunkingConfigDep = Annotated[ChunkingConfig, Depends(get_chunking_config)]


def get_ingestion_pipeline(
    embedder: EmbedderDep,
    vector_store: VectorStoreDep,
    chunking_config: ChunkingConfigDep,
) -> IngestionPipeline:
    return IngestionPipeline(
        embedder=embedder,
        vector_store=vector_store,
        chunking_config=chunking_config,
    )


def get_retriever(
    embedder: EmbedderDep, vector_store: VectorStoreDep
) -> Retriever:
    # Hybrid reranker is the production default; ablations can use NoOp via
    # an override in the eval harness.
    return Retriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=KeywordOverlapReranker(alpha=0.7),
    )


RetrieverDep = Annotated[Retriever, Depends(get_retriever)]


def get_generator(
    retriever: RetrieverDep,
    chat_client: ChatClientDep,
    settings: SettingsDep,
) -> Generator:
    return Generator(
        retriever=retriever,
        llm_client=chat_client,
        model_name=settings.openai_llm_model,
        prompt_version=settings.prompt_template_version,
    )


IngestionPipelineDep = Annotated[IngestionPipeline, Depends(get_ingestion_pipeline)]
GeneratorDep = Annotated[Generator, Depends(get_generator)]
