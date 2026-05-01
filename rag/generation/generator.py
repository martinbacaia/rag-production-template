"""End-to-end answer generation.

The :class:`Generator` is the entry point used by the ``/query`` endpoint
and the eval harness. Its single responsibility:

    query --[retriever]--> chunks --[template]--> prompt --[LLM]--> answer

It returns a :class:`GeneratedAnswer` that bundles the answer text with
the citations that produced it, plus the prompt version and model used.
Including those last two fields is what makes evals reproducible: a row
in the eval results table is enough to re-run the same configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rag.generation.llm_client import LLMClient
from rag.generation.prompt_templates import PromptVersion, render_prompt
from rag.logging import get_logger
from rag.retrieval.retriever import Retriever
from rag.vectorstore.base import SearchResult

logger = get_logger(__name__)

NO_CONTEXT_ANSWER = "I do not have enough information to answer."


@dataclass(frozen=True, slots=True)
class Citation:
    """One source attribution attached to a generated answer."""

    chunk_id: str
    source: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class GeneratedAnswer:
    """Result of :meth:`Generator.generate`."""

    answer: str
    citations: list[Citation]
    prompt_version: PromptVersion
    model: str
    retrieved_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


def _to_citation(result: SearchResult) -> Citation:
    return Citation(
        chunk_id=result.id,
        source=str(result.metadata.get("source", result.id)),
        text=result.text,
        score=result.score,
    )


class Generator:
    """Orchestrates retrieval + prompt rendering + LLM completion."""

    def __init__(
        self,
        retriever: Retriever,
        llm_client: LLMClient,
        model_name: str,
        prompt_version: PromptVersion = "v1",
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client
        self._model = model_name
        self._prompt_version = prompt_version

    def generate(
        self,
        query: str,
        top_k: int = 4,
        score_threshold: float = 0.0,
        prompt_version: PromptVersion | None = None,
    ) -> GeneratedAnswer:
        """Answer ``query`` using retrieved context.

        ``prompt_version`` overrides the configured template for this call
        only — useful for A/B comparisons in the eval harness without
        rebuilding the generator. If ``None``, the configured default is
        used.

        If retrieval returns nothing (empty index, threshold too high, or
        empty/whitespace query), short-circuit with a stable "I don't
        know" sentinel rather than calling the LLM. This saves a token
        round-trip and gives the eval harness a deterministic signal for
        unanswerable questions.
        """
        version = prompt_version or self._prompt_version

        results = self._retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)
        if not results:
            logger.info("generation.no_context", query_len=len(query))
            return GeneratedAnswer(
                answer=NO_CONTEXT_ANSWER,
                citations=[],
                prompt_version=version,
                model=self._model,
                retrieved_count=0,
            )

        prompt = render_prompt(version, query, results)
        answer_text = self._llm.complete(prompt)

        return GeneratedAnswer(
            answer=answer_text,
            citations=[_to_citation(r) for r in results],
            prompt_version=version,
            model=self._model,
            retrieved_count=len(results),
        )
