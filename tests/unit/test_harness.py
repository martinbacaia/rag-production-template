"""Unit tests for :mod:`rag.eval.harness`."""

from __future__ import annotations

import json
from pathlib import Path

from rag.eval.harness import (
    EvalHarness,
    GoldenItem,
    load_golden_dataset,
)
from rag.generation.generator import GeneratedAnswer
from rag.vectorstore.base import SearchResult


class _FakeGenerator:
    """Returns a configurable answer; exposes call history for assertions."""

    def __init__(self, answers: dict[str, str], model: str = "fake-model") -> None:
        self._answers = answers
        self.model = model
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        prompt_version: str | None = None,
    ) -> GeneratedAnswer:
        self.calls.append(
            {"query": query, "top_k": top_k, "prompt_version": prompt_version}
        )
        text = self._answers.get(query, "")
        return GeneratedAnswer(
            answer=text,
            citations=[],
            prompt_version=(prompt_version or "v1"),  # type: ignore[arg-type]
            model=self.model,
            retrieved_count=2,
        )


def _hit(text: str) -> SearchResult:
    return SearchResult(id="x", text=text, score=0.9, metadata={})


def _golden(
    id_: str,
    question: str,
    expect_in_answer: list[str],
    expect_in_context: list[str],
    answerable: bool = True,
) -> GoldenItem:
    return GoldenItem(
        id=id_,
        question=question,
        expected_substrings_in_answer=expect_in_answer,
        expected_substrings_in_context=expect_in_context,
        must_be_answerable=answerable,
    )


# -- load_golden_dataset -----------------------------------------------------


def test_load_golden_dataset(tmp_path: Path) -> None:
    payload = {
        "version": "v1",
        "items": [
            {
                "id": "q1",
                "question": "what?",
                "expected_substrings_in_answer": ["alpha"],
                "expected_substrings_in_context": ["beta"],
                "must_be_answerable": True,
            }
        ],
    }
    p = tmp_path / "golden.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    items = load_golden_dataset(p)
    assert len(items) == 1
    assert items[0].id == "q1"
    assert items[0].expected_substrings_in_answer == ["alpha"]


def test_bundled_golden_dataset_loads() -> None:
    """The shipped dataset must always parse — guards the README/CI promise."""
    repo_root = Path(__file__).resolve().parents[2]
    items = load_golden_dataset(repo_root / "evals" / "golden_dataset.json")
    assert len(items) >= 15  # we ship 20; guard against accidental truncation
    # Every item has the required fields.
    for it in items:
        assert it.question
        assert isinstance(it.expected_substrings_in_answer, list)
        assert isinstance(it.expected_substrings_in_context, list)


# -- EvalHarness -------------------------------------------------------------


def test_harness_scores_perfect_run() -> None:
    """Every answer contains the expected substrings, every retrieved chunk
    contains the expected context substring → all metrics 1.0."""
    dataset = [
        _golden("q1", "what is alpha?", ["alpha"], ["alpha"]),
        _golden("q2", "describe beta", ["beta"], ["beta"]),
    ]
    gen = _FakeGenerator(
        {"what is alpha?": "alpha is the first letter", "describe beta": "beta follows alpha"}
    )

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return [_hit("alpha and beta together")]

    harness = EvalHarness(generator=gen, retrieve_fn=retrieve)  # type: ignore[arg-type]
    report = harness.run(dataset)

    assert len(report.rows) == 2
    assert report.aggregate["context_precision"] == 1.0
    assert report.aggregate["answer_correctness"] == 1.0
    # Relevancy is non-trivial but not necessarily 1.0
    assert report.aggregate["answer_relevancy"] > 0.0


def test_harness_zero_when_answer_misses_substrings() -> None:
    dataset = [_golden("q1", "what is alpha?", ["alpha"], ["alpha"])]
    gen = _FakeGenerator({"what is alpha?": "I have no idea about anything"})

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return [_hit("alpha is the first letter")]

    harness = EvalHarness(generator=gen, retrieve_fn=retrieve)  # type: ignore[arg-type]
    report = harness.run(dataset)
    assert report.rows[0].context_precision == 1.0
    assert report.rows[0].answer_correctness == 0.0


def test_harness_top_k_propagates() -> None:
    dataset = [_golden("q1", "anything", ["x"], [])]
    gen = _FakeGenerator({"anything": "stub"})
    captured: list[int] = []

    def retrieve(q: str, k: int) -> list[SearchResult]:
        captured.append(k)
        return []

    harness = EvalHarness(generator=gen, retrieve_fn=retrieve, top_k=7)  # type: ignore[arg-type]
    harness.run(dataset)
    assert captured == [7]
    assert gen.calls[0]["top_k"] == 7


def test_harness_prompt_version_override() -> None:
    dataset = [_golden("q1", "anything", ["x"], [])]
    gen = _FakeGenerator({"anything": "x"})

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return []

    harness = EvalHarness(
        generator=gen,  # type: ignore[arg-type]
        retrieve_fn=retrieve,
        prompt_version="v2",
    )
    harness.run(dataset)
    assert gen.calls[0]["prompt_version"] == "v2"


def test_harness_aggregate_is_average_of_rows() -> None:
    """Aggregate = arithmetic mean across rows, per metric."""
    dataset = [
        _golden("q1", "a", ["a"], ["x"]),
        _golden("q2", "b", ["b"], ["y"]),
    ]
    # First answer hits both (1.0); second misses (0.0)
    gen = _FakeGenerator({"a": "a is true", "b": "completely off topic"})

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return [_hit("x and y")]  # contains both expected ctx substrings

    harness = EvalHarness(generator=gen, retrieve_fn=retrieve)  # type: ignore[arg-type]
    report = harness.run(dataset)
    assert report.aggregate["answer_correctness"] == 0.5
    # Both expected_substrings_in_context match the single retrieved chunk.
    assert report.aggregate["context_precision"] == 1.0


def test_harness_writes_json(tmp_path: Path) -> None:
    dataset = [_golden("q1", "anything", ["x"], [])]
    gen = _FakeGenerator({"anything": "x"})

    def retrieve(q: str, k: int) -> list[SearchResult]:
        return []

    harness = EvalHarness(generator=gen, retrieve_fn=retrieve)  # type: ignore[arg-type]
    report = harness.run(dataset)
    out = tmp_path / "report.json"
    report.write_json(out)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "rows" in payload
    assert "aggregate" in payload
    assert payload["rows"][0]["id"] == "q1"
