"""OpenAI client wrappers: embeddings and chat completion.

Both wrappers:

* Take an ``openai.OpenAI`` client by injection so tests can pass a fake.
* Apply exponential-backoff retries on transient errors (rate limits,
  timeouts, connection errors, server-side 5xx). Authentication and
  bad-request errors are *not* retried — they are the caller's fault.
* Emit structured log events on retry and on permanent failure.

The two classes intentionally implement the narrow ``Embedder`` and
``LLMClient`` contracts used elsewhere in the package. Anyone wanting
Cohere / Anthropic / Groq writes a sibling class with the same shape;
the rest of the system does not care which provider is behind it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from rag.generation.prompt_templates import RenderedPrompt
from rag.logging import get_logger

logger = get_logger(__name__)


def _is_retryable_status(exc: BaseException) -> bool:
    """Retry on transient failures only.

    The OpenAI SDK exception hierarchy is::

        APIError
        ├── APIConnectionError
        │   └── APITimeoutError
        └── APIStatusError      (any non-2xx)
            ├── RateLimitError              (429)
            ├── BadRequestError             (400)
            ├── AuthenticationError         (401)
            ├── ...                         (other 4xx)
            └── InternalServerError         (5xx)

    We retry rate limits, timeouts, connection errors, and 5xx. Other
    4xx (auth, bad request, 404) are caller bugs — retrying them wastes
    money and masks the real problem.
    """
    if isinstance(exc, RateLimitError | APITimeoutError | APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return 500 <= exc.status_code < 600
    return False


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "openai.retry",
        attempt=retry_state.attempt_number,
        wait_seconds=getattr(retry_state.next_action, "sleep", None),
        error=type(exc).__name__ if exc else None,
    )


_retry_policy = retry(
    retry=retry_if_exception(_is_retryable_status),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
    before_sleep=_log_retry,
)


class LLMClient(Protocol):
    """Anything that turns a :class:`RenderedPrompt` into a string answer."""

    def complete(self, prompt: RenderedPrompt) -> str: ...


class OpenAIEmbedder:
    """Calls OpenAI's embeddings endpoint, returning plain ``list[list[float]]``."""

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    @_retry_policy
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model,
            input=list(texts),
        )
        # OpenAI returns items in input order — preserve it.
        return [item.embedding for item in response.data]


class OpenAIChatClient:
    """Calls OpenAI's chat-completion endpoint and returns the message text."""

    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    @_retry_policy
    def complete(self, prompt: RenderedPrompt) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": prompt.system},
                {"role": "user", "content": prompt.user},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        choice = response.choices[0]
        content = choice.message.content
        if content is None:
            # Empty content with finish_reason="length" or "content_filter" is
            # a real production case — bubble up empty string rather than
            # crashing on ``.strip()``.
            logger.warning(
                "openai.empty_completion",
                finish_reason=choice.finish_reason,
            )
            return ""
        return content.strip()
