"""Evaluation harness.

Loads a golden dataset, runs each question through the configured
:class:`Generator`, and scores the output against the per-question
expectations using the metrics in :mod:`rag.eval.metrics`.

The harness is intentionally simple to read: a list comprehension over
the dataset, a small per-row score record, and one aggregation pass.
There is no parallelism here — eval runs are bound by the LLM API and
the parallelism that matters lives inside the OpenAI client.

The output is structured (a list of :class:`EvalRow` plus an
aggregate dict) so the same data can drive a CLI table, a JSON file,
the API response, and the README.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag.config import PromptTemplateVersion
from rag.eval.metrics import (
    answer_relevancy,
    answer_substring_recall,
    context_precision,
)
from rag.generation.generator import GeneratedAnswer, Generator
from rag.logging import get_logger
from rag.vectorstore.base import SearchResult

logger = get_logger(__name__)

RetrieveFn = Callable[[str, int], list[SearchResult]]


@dataclass(frozen=True, slots=True)
class GoldenItem:
    """One question in the golden dataset."""

    id: str
    question: str
    expected_substrings_in_answer: list[str]
    expected_substrings_in_context: list[str]
    must_be_answerable: bool


@dataclass(frozen=True, slots=True)
class EvalRow:
    """Per-question eval result."""

    id: str
    question: str
    answer: str
    context_precision: float
    answer_correctness: float
    answer_relevancy: float
    retrieved_count: int


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Full eval output: rows + aggregate scores + run metadata."""

    rows: list[EvalRow]
    aggregate: dict[str, float]
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def load_golden_dataset(path: Path | str) -> list[GoldenItem]:
    """Read a golden dataset JSON file from disk."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        GoldenItem(
            id=item["id"],
            question=item["question"],
            expected_substrings_in_answer=item["expected_substrings_in_answer"],
            expected_substrings_in_context=item["expected_substrings_in_context"],
            must_be_answerable=item["must_be_answerable"],
        )
        for item in raw["items"]
    ]


def _score_row(
    item: GoldenItem,
    generated: GeneratedAnswer,
    retrieved: list[SearchResult],
) -> EvalRow:
    return EvalRow(
        id=item.id,
        question=item.question,
        answer=generated.answer,
        context_precision=context_precision(retrieved, item.expected_substrings_in_context),
        answer_correctness=answer_substring_recall(
            generated.answer, item.expected_substrings_in_answer
        ),
        answer_relevancy=answer_relevancy(item.question, generated.answer),
        retrieved_count=generated.retrieved_count,
    )


def _aggregate(rows: Iterable[EvalRow]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {
            "context_precision": 0.0,
            "answer_correctness": 0.0,
            "answer_relevancy": 0.0,
        }
    return {
        "context_precision": statistics.mean(r.context_precision for r in rows),
        "answer_correctness": statistics.mean(r.answer_correctness for r in rows),
        "answer_relevancy": statistics.mean(r.answer_relevancy for r in rows),
    }


class EvalHarness:
    """Run a golden dataset through a :class:`Generator` and score the output.

    The harness depends on a ``Retriever`` callable to capture the
    retrieved chunks separately. The :class:`Generator` does its own
    retrieval internally; we re-run retrieval here so the metrics can
    inspect the retrieved set without monkeypatching the generator.
    The cost of one extra retrieval call per question is negligible
    versus the LLM call.
    """

    def __init__(
        self,
        generator: Generator,
        retrieve_fn: RetrieveFn,
        top_k: int = 4,
        prompt_version: PromptTemplateVersion | None = None,
    ) -> None:
        self._generator = generator
        self._retrieve_fn = retrieve_fn
        self._top_k = top_k
        self._prompt_version = prompt_version

    def run(self, dataset: list[GoldenItem]) -> EvalReport:
        rows: list[EvalRow] = []
        for item in dataset:
            retrieved = self._retrieve_fn(item.question, self._top_k)
            generated = self._generator.generate(
                item.question,
                top_k=self._top_k,
                prompt_version=self._prompt_version,
            )
            row = _score_row(item, generated, retrieved)
            rows.append(row)
            logger.info(
                "eval.row",
                id=item.id,
                context_precision=round(row.context_precision, 3),
                answer_correctness=round(row.answer_correctness, 3),
                answer_relevancy=round(row.answer_relevancy, 3),
            )

        report = EvalReport(rows=rows, aggregate=_aggregate(rows))
        logger.info("eval.aggregate", **report.aggregate)
        return report
