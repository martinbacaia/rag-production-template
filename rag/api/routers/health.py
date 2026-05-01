"""Health endpoint.

Returns a small JSON document with process state plus key configuration
echoes (models, prompt version) so an operator can confirm at a glance
which model is serving traffic.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag.api.dependencies import SettingsDep, VectorStoreDep
from rag.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: SettingsDep, vector_store: VectorStoreDep) -> HealthResponse:
    return HealthResponse(
        status="ok",
        vector_store_count=vector_store.count(),
        embedding_model=settings.openai_embedding_model,
        llm_model=settings.openai_llm_model,
        prompt_template_version=settings.prompt_template_version,
    )
