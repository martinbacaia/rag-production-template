"""Tests for :mod:`rag.generation.generator`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from rag.generation.generator import NO_CONTEXT_ANSWER, Generator
from rag.generation.prompt_templates import RenderedPrompt
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import SearchResult, VectorStore


class _FakeEmbedder:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]


class _FakeStore(VectorStore):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def add(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None: ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int,
        score_threshold: float = 0.0,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        return [r for r in self._results[:top_k] if r.score >= score_threshold]

    def delete(self, ids: Sequence[str]) -> None: ...
    def count(self) -> int:
        return len(self._results)

    def reset(self) -> None: ...


class _FakeLLMClient:
    def __init__(self, response: str = "the answer") -> None:
        self.response = response
        self.prompts: list[RenderedPrompt] = []

    def complete(self, prompt: RenderedPrompt) -> str:
        self.prompts.append(prompt)
        return self.response


def _hit(idx: int, text: str, score: float, source: str = "doc.txt") -> SearchResult:
    return SearchResult(
        id=f"chunk-{idx}",
        text=text,
        score=score,
        metadata={"source": source, "chunk_index": idx},
    )


def _build_generator(
    *,
    results: list[SearchResult],
    llm: _FakeLLMClient | None = None,
    prompt_version: str = "v1",
) -> tuple[Generator, _FakeLLMClient]:
    retriever = Retriever(
        embedder=_FakeEmbedder(),
        vector_store=_FakeStore(results),
    )
    llm = llm or _FakeLLMClient()
    gen = Generator(
        retriever=retriever,
        llm_client=llm,
        model_name="gpt-4o-mini-test",
        prompt_version=prompt_version,  # type: ignore[arg-type]
    )
    return gen, llm


def test_generate_happy_path() -> None:
    gen, llm = _build_generator(
        results=[
            _hit(0, "Decorators wrap functions.", 0.9),
            _hit(1, "They are syntactic sugar.", 0.8),
        ],
        llm=_FakeLLMClient("Decorators wrap functions [1]."),
    )
    out = gen.generate("what are decorators?", top_k=2)

    assert out.answer == "Decorators wrap functions [1]."
    assert out.retrieved_count == 2
    assert out.prompt_version == "v1"
    assert out.model == "gpt-4o-mini-test"

    # Citations carry chunk id, source, text, and score.
    ids = [c.chunk_id for c in out.citations]
    assert ids == ["chunk-0", "chunk-1"]
    assert all(c.source == "doc.txt" for c in out.citations)
    assert out.citations[0].score == 0.9

    # The LLM saw the configured template version.
    assert llm.prompts[0].version == "v1"


def test_generate_no_results_short_circuits_llm() -> None:
    gen, llm = _build_generator(results=[])
    out = gen.generate("query with no matches")
    assert out.answer == NO_CONTEXT_ANSWER
    assert out.citations == []
    assert out.retrieved_count == 0
    # The LLM was not called.
    assert llm.prompts == []


def test_generate_empty_query_short_circuits() -> None:
    gen, llm = _build_generator(results=[_hit(0, "x", 0.9)])
    out = gen.generate("   ")
    assert out.answer == NO_CONTEXT_ANSWER
    assert llm.prompts == []


def test_generate_passes_through_threshold() -> None:
    gen, llm = _build_generator(
        results=[_hit(0, "weak", 0.2), _hit(1, "strong", 0.95)],
    )
    out = gen.generate("q", top_k=5, score_threshold=0.5)
    # Only the strong hit survives; LLM is invoked with that single citation.
    assert out.retrieved_count == 1
    assert out.citations[0].chunk_id == "chunk-1"
    assert llm.prompts[0].version == "v1"


def test_v2_template_is_used_when_configured() -> None:
    gen, llm = _build_generator(
        results=[_hit(0, "context", 0.9)],
        prompt_version="v2",
    )
    gen.generate("q")
    assert llm.prompts[0].version == "v2"


def test_citation_falls_back_to_chunk_id_when_source_missing() -> None:
    chunk = SearchResult(id="orphan", text="x", score=1.0, metadata={})
    gen, _ = _build_generator(results=[chunk])
    out = gen.generate("q")
    assert out.citations[0].source == "orphan"


def test_invalid_top_k_propagates_from_retriever() -> None:
    gen, _ = _build_generator(results=[_hit(0, "x", 0.9)])
    with pytest.raises(ValueError):
        gen.generate("q", top_k=0)
