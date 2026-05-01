"""Tests for :mod:`rag.generation.llm_client`.

We never hit the real OpenAI API in unit tests. Instead we build a
minimal fake of the ``openai.OpenAI`` client surface — only the two
methods our wrappers call (``embeddings.create`` and
``chat.completions.create``).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

from rag.generation.llm_client import OpenAIChatClient, OpenAIEmbedder
from rag.generation.prompt_templates import RenderedPrompt


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tenacity uses ``time.sleep`` for backoff. Tests must not actually
    wait — patch it so retry-bearing tests run in milliseconds."""
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)


# -- Fake OpenAI client -------------------------------------------------------


@dataclass
class _EmbItem:
    embedding: list[float]


@dataclass
class _EmbResponse:
    data: list[_EmbItem]


@dataclass
class _Msg:
    content: str | None


@dataclass
class _Choice:
    message: _Msg
    finish_reason: str = "stop"


@dataclass
class _ChatResponse:
    choices: list[_Choice]


class _FakeEmbeddings:
    def __init__(self, response: _EmbResponse | Exception | list[Any]) -> None:
        # ``response`` may be a single response, an exception to raise, or a
        # list of responses/exceptions consumed in order across calls — the
        # latter lets retry tests cycle through "fail, fail, succeed".
        self._scenario = response
        self.calls: list[dict[str, Any]] = []

    def create(self, *, model: str, input: Sequence[str]) -> _EmbResponse:
        self.calls.append({"model": model, "input": list(input)})
        return _consume_scenario(self._scenario, len(self.calls) - 1)


class _FakeChat:
    def __init__(self, response: _ChatResponse | Exception | list[Any]) -> None:
        self._scenario = response
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> _ChatResponse:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return _consume_scenario(self._scenario, len(self.calls) - 1)


@dataclass
class _ChatNamespace:
    completions: _FakeChat


@dataclass
class _FakeOpenAI:
    embeddings: _FakeEmbeddings
    chat: _ChatNamespace


def _consume_scenario(scenario: Any, idx: int) -> Any:
    """Either raise, return, or pop the n-th element of a list scenario."""
    item = scenario[idx] if isinstance(scenario, list) else scenario
    if isinstance(item, Exception):
        raise item
    return item


def _fake_response(rate_limit: bool = False) -> httpx.Response:
    return httpx.Response(
        status_code=429 if rate_limit else 500,
        request=httpx.Request("POST", "https://api.openai.com/v1/x"),
    )


# -- Embedder tests -----------------------------------------------------------


def test_embedder_returns_vectors_in_order() -> None:
    fake_embed = _FakeEmbeddings(
        _EmbResponse(data=[_EmbItem([0.1, 0.2]), _EmbItem([0.3, 0.4])])
    )
    fake_chat = _FakeChat(_ChatResponse(choices=[]))
    client = _FakeOpenAI(
        embeddings=fake_embed,
        chat=_ChatNamespace(completions=fake_chat),
    )
    embedder = OpenAIEmbedder(client=client, model="text-embedding-3-small")  # type: ignore[arg-type]
    out = embedder.embed(["hello", "world"])
    assert out == [[0.1, 0.2], [0.3, 0.4]]
    assert fake_embed.calls[0]["input"] == ["hello", "world"]
    assert fake_embed.calls[0]["model"] == "text-embedding-3-small"


def test_embedder_empty_input_skips_api_call() -> None:
    fake_embed = _FakeEmbeddings(_EmbResponse(data=[]))
    fake_chat = _FakeChat(_ChatResponse(choices=[]))
    client = _FakeOpenAI(
        embeddings=fake_embed,
        chat=_ChatNamespace(completions=fake_chat),
    )
    embedder = OpenAIEmbedder(client=client, model="m")  # type: ignore[arg-type]
    assert embedder.embed([]) == []
    assert fake_embed.calls == []


def test_embedder_retries_on_rate_limit_then_succeeds() -> None:
    err = RateLimitError(
        message="rate limited",
        response=_fake_response(rate_limit=True),
        body=None,
    )
    success = _EmbResponse(data=[_EmbItem([0.5, 0.5])])
    fake_embed = _FakeEmbeddings([err, err, success])
    fake_chat = _FakeChat(_ChatResponse(choices=[]))
    client = _FakeOpenAI(
        embeddings=fake_embed,
        chat=_ChatNamespace(completions=fake_chat),
    )
    embedder = OpenAIEmbedder(client=client, model="m")  # type: ignore[arg-type]
    out = embedder.embed(["x"])
    assert out == [[0.5, 0.5]]
    assert len(fake_embed.calls) == 3


def test_embedder_does_not_retry_on_4xx_status() -> None:
    """A 400 (bad request) is the caller's fault — no point retrying."""
    err = APIStatusError(
        message="bad request",
        response=httpx.Response(
            status_code=400,
            request=httpx.Request("POST", "https://api.openai.com/v1/x"),
        ),
        body=None,
    )
    fake_embed = _FakeEmbeddings(err)
    fake_chat = _FakeChat(_ChatResponse(choices=[]))
    client = _FakeOpenAI(
        embeddings=fake_embed,
        chat=_ChatNamespace(completions=fake_chat),
    )
    embedder = OpenAIEmbedder(client=client, model="m")  # type: ignore[arg-type]
    with pytest.raises(APIStatusError):
        embedder.embed(["x"])
    # Single attempt — retry policy fired but the 4xx short-circuited it.
    assert len(fake_embed.calls) == 1


# -- Chat client tests --------------------------------------------------------


def _prompt() -> RenderedPrompt:
    return RenderedPrompt(system="sys", user="usr", version="v1")


def test_chat_complete_returns_message_text() -> None:
    fake_chat = _FakeChat(
        _ChatResponse(choices=[_Choice(message=_Msg("the answer is 42"))])
    )
    fake_embed = _FakeEmbeddings(_EmbResponse(data=[]))
    client = _FakeOpenAI(
        embeddings=fake_embed,
        chat=_ChatNamespace(completions=fake_chat),
    )
    chat = OpenAIChatClient(client=client, model="gpt-4o-mini", temperature=0.0)  # type: ignore[arg-type]
    assert chat.complete(_prompt()) == "the answer is 42"
    call = fake_chat.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["temperature"] == 0.0
    assert call["messages"][0] == {"role": "system", "content": "sys"}
    assert call["messages"][1] == {"role": "user", "content": "usr"}


def test_chat_complete_strips_whitespace() -> None:
    fake_chat = _FakeChat(
        _ChatResponse(choices=[_Choice(message=_Msg("  answer  \n"))])
    )
    client = _FakeOpenAI(
        embeddings=_FakeEmbeddings(_EmbResponse(data=[])),
        chat=_ChatNamespace(completions=fake_chat),
    )
    chat = OpenAIChatClient(client=client, model="m")  # type: ignore[arg-type]
    assert chat.complete(_prompt()) == "answer"


def test_chat_complete_handles_none_content() -> None:
    """``finish_reason='length'`` can leave content as None — must not crash."""
    fake_chat = _FakeChat(
        _ChatResponse(
            choices=[_Choice(message=_Msg(None), finish_reason="length")]
        )
    )
    client = _FakeOpenAI(
        embeddings=_FakeEmbeddings(_EmbResponse(data=[])),
        chat=_ChatNamespace(completions=fake_chat),
    )
    chat = OpenAIChatClient(client=client, model="m")  # type: ignore[arg-type]
    assert chat.complete(_prompt()) == ""


def test_chat_retries_on_connection_error() -> None:
    err = APIConnectionError(
        request=httpx.Request("POST", "https://api.openai.com/v1/chat"),
    )
    success = _ChatResponse(choices=[_Choice(message=_Msg("ok"))])
    fake_chat = _FakeChat([err, success])
    client = _FakeOpenAI(
        embeddings=_FakeEmbeddings(_EmbResponse(data=[])),
        chat=_ChatNamespace(completions=fake_chat),
    )
    chat = OpenAIChatClient(client=client, model="m")  # type: ignore[arg-type]
    assert chat.complete(_prompt()) == "ok"
    assert len(fake_chat.calls) == 2
