"""Evaluation endpoint (stub).

The full implementation lands with the eval harness in
:mod:`rag.eval`. The route is exposed now so the OpenAPI schema is
stable from day one — clients (and CI smoke tests) can rely on its
presence.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from rag.api.schemas import EvalRequest, EvalResponse

router = APIRouter(tags=["eval"])


@router.post(
    "/eval",
    response_model=EvalResponse,
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
)
def run_eval(request: EvalRequest) -> EvalResponse:
    # Wired in module 7. Keeping the route documented in the OpenAPI
    # schema here lets the API surface stay stable across releases.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Eval endpoint not yet wired. Run via `python -m rag.eval` "
        "or wait for the next release.",
    )
