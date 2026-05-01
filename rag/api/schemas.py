"""Pydantic request/response models for the HTTP API.

These types are the public contract — clients (CLI, frontend, other
services) generate code from this schema. Keep them stable: every
breaking change should bump the route version (``/v2/query`` etc.).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# -- Health ------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = Field(description="Always 'ok' if the process is responding.")
    vector_store_count: int = Field(description="Total number of chunks currently indexed.")
    embedding_model: str
    llm_model: str
    prompt_template_version: str


# -- Ingestion ---------------------------------------------------------------


class IngestRequest(BaseModel):
    """Ingest raw text under a caller-specified source identifier.

    File upload is intentionally out of scope: the recommended pattern is
    that the client reads the file and posts the text. Keeps the API
    stateless and avoids multipart parsing in the hot path.
    """

    text: str = Field(min_length=1, description="Document body to ingest.")
    source: str = Field(
        min_length=1,
        description="Stable identifier for the document (path, URL, slug).",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional extra metadata propagated to every chunk.",
    )


class IngestResponse(BaseModel):
    source: str
    num_chunks: int
    chunk_ids: list[str]


# -- Query -------------------------------------------------------------------


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class CitationOut(BaseModel):
    chunk_id: str
    source: str
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    prompt_version: str
    model: str
    retrieved_count: int


# -- Eval --------------------------------------------------------------------


class EvalRequest(BaseModel):
    """Trigger an evaluation run.

    The dataset path defaults to the bundled golden set; ``prompt_version``
    overrides the configured template so callers can A/B test versions
    against the same questions.
    """

    dataset_path: str | None = Field(
        default=None,
        description="Path to a JSON golden dataset; defaults to the bundled one.",
    )
    prompt_version: str | None = Field(
        default=None,
        description="Override the configured prompt template (v1, v2, ...).",
    )


class EvalMetricRow(BaseModel):
    question: str
    answer: str
    expected_substrings: list[str]
    faithfulness: float
    answer_relevancy: float
    context_precision: float


class EvalResponse(BaseModel):
    num_questions: int
    aggregate: dict[str, float]
    rows: list[EvalMetricRow]
