"""FastAPI application factory.

The app is constructed via a factory rather than instantiated at module
import. That decoupling buys us:

* Tests can build an isolated app per test (or session) and override
  dependencies cleanly.
* A single ``rag.api.main:create_app`` entry point works for both
  ``uvicorn rag.api.main:app`` (production) and direct programmatic use.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag.api.routers import eval as eval_router
from rag.api.routers import health, ingest, query
from rag.config import get_settings
from rag.logging import configure_logging, get_logger


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Configure logging once on startup."""
    settings = get_settings()
    configure_logging(level=settings.log_level)
    logger = get_logger(__name__)
    logger.info(
        "api.startup",
        embedding_model=settings.openai_embedding_model,
        llm_model=settings.openai_llm_model,
        prompt_version=settings.prompt_template_version,
    )
    yield
    logger.info("api.shutdown")


def create_app() -> FastAPI:
    """Build a fresh FastAPI app with all routers registered."""
    app = FastAPI(
        title="RAG Production Template",
        description=(
            "Retrieval-Augmented Generation reference API. See /docs for the interactive schema."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(query.router)
    app.include_router(eval_router.router)
    return app


# Module-level instance for ``uvicorn rag.api.main:app``.
app = create_app()
