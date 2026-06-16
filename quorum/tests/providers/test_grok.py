"""Smoke tests for the xAI Grok provider (no real API calls)."""

from __future__ import annotations

import pytest

from quorum.providers.grok import GrokProvider, grok_3, grok_4, grok_4_mini


def test_default_model_is_grok_4_0709():
    p = GrokProvider()
    assert p.model == "grok-4-0709"
    assert p.name == "grok-4"


def test_factory_grok_4_uses_correct_model():
    p = grok_4(api_key="test-key")
    assert isinstance(p, GrokProvider)
    assert p.model == "grok-4-0709"
    assert p.api_key == "test-key"


def test_factory_grok_4_mini_uses_correct_model():
    p = grok_4_mini(api_key="test-key")
    assert isinstance(p, GrokProvider)
    assert p.model == "grok-4-mini"


def test_factory_grok_3_uses_correct_model():
    p = grok_3(api_key="test-key")
    assert isinstance(p, GrokProvider)
    assert p.model == "grok-3"


def test_api_key_read_from_env(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "env-key-123")
    p = GrokProvider()
    assert p.api_key == "env-key-123"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "env-key")
    p = GrokProvider(api_key="explicit-key")
    assert p.api_key == "explicit-key"


@pytest.mark.asyncio
async def test_complete_returns_error_when_no_api_key(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    p = GrokProvider()
    result = await p.complete("hello")
    assert result.error == "no_api_key"
    assert result.response == ""
    assert result.name == "grok-4"
