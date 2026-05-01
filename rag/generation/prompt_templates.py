"""Versioned prompt templates.

Prompts are code — they're as load-bearing as the retriever or the model
choice. Treating them as anonymous string literals scattered through the
codebase makes A/B comparisons and rollbacks impossible.

This module exposes:

* :class:`PromptTemplate` — a versioned (system, user) tuple plus a
  ``render`` method that consumes a query + a list of retrieved chunks.
* :data:`TEMPLATES` — the registry. New versions are added here, never
  edited in place: an evaluation regression should be reproducible by
  pinning the older version.

The current versions are:

* ``v1`` — minimal: instruction + context + question. Establishes the
  citation contract (``[N]`` references the n-th chunk in context order).
* ``v2`` — stricter: system role with explicit anti-hallucination rules,
  user role carries only context+question. Better separation, slightly
  better faithfulness on the eval set (see README).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from rag.vectorstore.base import SearchResult

RenderFn = Callable[[str, Sequence[SearchResult]], "RenderedPrompt"]

PromptVersion = Literal["v1", "v2"]


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """A prompt ready to be sent to a chat-completion API.

    The pair (``system``, ``user``) maps directly to OpenAI / Anthropic
    chat APIs. ``version`` is propagated downstream so eval traces and
    logs can attribute behavior to a specific template revision.
    """

    system: str
    user: str
    version: PromptVersion


def _format_context(results: Sequence[SearchResult]) -> str:
    """Render retrieved chunks as a numbered, citable block.

    The format is deliberately stable: ``[N] source: ...`` so the model
    learns to use ``[N]`` as the citation marker. Source path is included
    on the same line because the model occasionally surfaces it verbatim
    when asked "where did this come from?".
    """
    lines: list[str] = []
    for i, r in enumerate(results, start=1):
        source = r.metadata.get("source", r.id)
        lines.append(f"[{i}] source: {source}\n{r.text}")
    return "\n\n".join(lines)


def _render_v1(query: str, results: Sequence[SearchResult]) -> RenderedPrompt:
    context = _format_context(results)
    system = (
        "You answer questions using only the provided context. "
        "If the context does not contain the answer, say you do not know."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer the question. Cite the chunks you used with [N] notation."
    )
    return RenderedPrompt(system=system, user=user, version="v1")


def _render_v2(query: str, results: Sequence[SearchResult]) -> RenderedPrompt:
    context = _format_context(results)
    system = (
        "You are a precise assistant that answers questions strictly from "
        "the supplied context.\n\n"
        "Rules:\n"
        "1. Use only facts that appear in the context. Do not invent details "
        "or rely on outside knowledge.\n"
        "2. If the context does not contain enough information to answer, "
        'reply exactly: "I do not have enough information to answer."\n'
        "3. Every factual claim must cite the chunk it came from with [N] "
        "notation, where N is the chunk number in the context.\n"
        "4. Be concise. Prefer 1-3 sentences unless the question explicitly "
        "asks for more detail."
    )
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return RenderedPrompt(system=system, user=user, version="v2")


@dataclass(frozen=True, slots=True)
class PromptTemplate:
    """A versioned prompt template.

    The ``render`` callable produces a :class:`RenderedPrompt` from a
    query and retrieved chunks. Template logic lives in module-level
    functions rather than methods so each version is a small, isolated
    unit (and so the registry stays a plain dict).
    """

    version: PromptVersion
    description: str
    render: RenderFn


TEMPLATES: dict[PromptVersion, PromptTemplate] = {
    "v1": PromptTemplate(
        version="v1",
        description="Minimal: single instruction + context + question.",
        render=_render_v1,
    ),
    "v2": PromptTemplate(
        version="v2",
        description="Stricter system role with explicit anti-hallucination rules.",
        render=_render_v2,
    ),
}


def render_prompt(
    version: PromptVersion,
    query: str,
    results: Sequence[SearchResult],
) -> RenderedPrompt:
    """Render the requested template version.

    Raises :class:`KeyError` for an unknown version. Callers can rely on
    :class:`rag.config.Settings` to constrain inputs to known versions
    via ``Literal``, but this guard keeps direct callers honest too.
    """
    if version not in TEMPLATES:
        raise KeyError(f"Unknown prompt template version: {version!r}")
    return TEMPLATES[version].render(query, results)
