"""Smoke tests that prove configuration loads and validates."""

from __future__ import annotations

import pytest

from rag.config import Settings, get_settings


def test_defaults_are_sane(settings: Settings) -> None:
    assert settings.openai_llm_model == "gpt-4o-mini"
    assert settings.openai_embedding_model == "text-embedding-3-small"
    assert settings.vector_store_backend == "chroma"
    assert 0 < settings.chunk_size <= 8000
    assert settings.chunk_overlap < settings.chunk_size, (
        "overlap must be strictly smaller than chunk size"
    )
    assert settings.retrieval_top_k >= 1


def test_env_overrides_take_effect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "256")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "10")
    get_settings.cache_clear()
    s = get_settings()
    assert s.chunk_size == 256
    assert s.retrieval_top_k == 10


def test_invalid_chunk_size_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "10")  # below ge=64
    get_settings.cache_clear()
    with pytest.raises(ValueError):
        get_settings()


def test_invalid_temperature_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TEMPERATURE", "5.0")  # above le=2.0
    get_settings.cache_clear()
    with pytest.raises(ValueError):
        get_settings()


def test_get_settings_is_cached() -> None:
    a = get_settings()
    b = get_settings()
    assert a is b
