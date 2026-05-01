"""Ingestion endpoint.

POST /ingest accepts a text body + source identifier and returns the
chunk ids that were upserted into the vector store. The endpoint is
synchronous on purpose: clients can plumb retries themselves, and
async ingestion belongs in a queue (Celery / SQS), not in the same
process serving query traffic.
"""

from __future__ import annotations

from fastapi import APIRouter

from rag.api.dependencies import IngestionPipelineDep
from rag.api.schemas import IngestRequest, IngestResponse

router = APIRouter(tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse, status_code=201)
def ingest(request: IngestRequest, pipeline: IngestionPipelineDep) -> IngestResponse:
    result = pipeline.ingest_text(
        text=request.text,
        source=request.source,
        extra_metadata=request.metadata,
    )
    return IngestResponse(
        source=result.source,
        num_chunks=result.num_chunks,
        chunk_ids=result.chunk_ids,
    )
