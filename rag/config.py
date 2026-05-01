"""Application configuration loaded from environment variables.

Settings are grouped by concern (OpenAI, vector store, ingestion, retrieval,
generation, API). All values can be overridden via environment variables or a
``.env`` file at the project root.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

VectorStoreBackend = Literal["chroma", "qdrant"]
PromptTemplateVersion = Literal["v1", "v2"]


class Settings(BaseSettings):
    """Strongly-typed runtime configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # OpenAI
    openai_api_key: str = Field(default="sk-test", description="OpenAI API key.")
    openai_llm_model: str = Field(default="gpt-4o-mini")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    # Vector store
    vector_store_backend: VectorStoreBackend = Field(default="chroma")
    chroma_persist_dir: str = Field(default="./chroma_data")
    chroma_collection: str = Field(default="rag_documents")
    # When ``chroma_host`` is set, the factory builds an ``HttpClient`` and
    # ignores ``chroma_persist_dir``. This is the docker-compose mode where
    # ChromaDB runs as a separate service.
    chroma_host: str | None = Field(default=None)
    chroma_port: int = Field(default=8000, ge=1, le=65535)

    # Ingestion
    chunk_size: int = Field(default=800, ge=64, le=8000)
    chunk_overlap: int = Field(default=120, ge=0, le=2000)

    # Retrieval
    retrieval_top_k: int = Field(default=4, ge=1, le=50)
    retrieval_score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)

    # Generation
    prompt_template_version: PromptTemplateVersion = Field(default="v1")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1, le=4096)

    # API
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    The cache is process-wide so configuration is loaded exactly once. Tests
    that need to override values should call ``get_settings.cache_clear()``.
    """
    return Settings()
