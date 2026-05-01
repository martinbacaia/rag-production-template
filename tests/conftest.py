"""Shared pytest fixtures.

Module-specific fixtures live next to the tests that consume them; only
project-wide fixtures (settings overrides, deterministic env, in-memory
vector store, OpenAI client mocks) belong here.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from rag.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Iterator[None]:
    """Isolate every test from the developer's real ``.env`` file.

    We force a deterministic config: a stub OpenAI key (real calls are mocked
    in tests that need them) and a per-test Chroma directory so persistent
    backends never share state.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    # Ensure cached settings are rebuilt after the monkeypatch is in place.
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(_isolated_env: None) -> Settings:
    """Return a fresh, isolated :class:`Settings` instance for the test."""
    return get_settings()
