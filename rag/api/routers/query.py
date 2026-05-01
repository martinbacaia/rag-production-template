"""Query endpoint.

POST /query runs the full retrieval + generation pipeline and returns
the answer with source citations.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag.api.dependencies import GeneratorDep
from rag.api.schemas import CitationOut, QueryRequest, QueryResponse

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, generator: GeneratorDep) -> QueryResponse:
    result = generator.generate(
        query=request.question,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )
    return QueryResponse(
        answer=result.answer,
        citations=[
            CitationOut(
                chunk_id=c.chunk_id,
                source=c.source,
                text=c.text,
                score=c.score,
            )
            for c in result.citations
        ],
        prompt_version=result.prompt_version,
        model=result.model,
        retrieved_count=result.retrieved_count,
    )
