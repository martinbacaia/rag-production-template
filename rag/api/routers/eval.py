"""Evaluation endpoint.

POST /eval runs the bundled golden dataset (or a caller-supplied path)
against the live system and returns per-question scores plus aggregate
metrics. The endpoint shares the same harness as ``python -m rag.eval``
so the API and the CLI cannot diverge.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

from rag.api.dependencies import GeneratorDep, RetrieverDep
from rag.api.schemas import EvalMetricRow, EvalRequest, EvalResponse
from rag.config import PromptTemplateVersion
from rag.eval.harness import EvalHarness, load_golden_dataset

router = APIRouter(tags=["eval"])

_DEFAULT_DATASET = Path(__file__).resolve().parents[3] / "evals" / "golden_dataset.json"
_VALID_PROMPT_VERSIONS = {"v1", "v2"}


@router.post("/eval", response_model=EvalResponse)
def run_eval(
    request: EvalRequest,
    generator: GeneratorDep,
    retriever: RetrieverDep,
) -> EvalResponse:
    dataset_path = Path(request.dataset_path) if request.dataset_path else _DEFAULT_DATASET
    if not dataset_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found: {dataset_path}",
        )

    prompt_override: PromptTemplateVersion | None = None
    if request.prompt_version is not None:
        if request.prompt_version not in _VALID_PROMPT_VERSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unknown prompt_version: {request.prompt_version!r}",
            )
        prompt_override = request.prompt_version  # type: ignore[assignment]

    harness = EvalHarness(
        generator=generator,
        retrieve_fn=lambda q, k: retriever.retrieve(q, top_k=k),
        prompt_version=prompt_override,
    )
    dataset = load_golden_dataset(dataset_path)
    report = harness.run(dataset)
    return EvalResponse(
        num_questions=len(report.rows),
        aggregate=report.aggregate,
        rows=[
            EvalMetricRow(
                question=r.question,
                answer=r.answer,
                expected_substrings=[],
                faithfulness=r.context_precision,
                answer_relevancy=r.answer_relevancy,
                context_precision=r.context_precision,
            )
            for r in report.rows
        ],
    )
