"""Smoke tests for the xAI Grok provider (no real API calls).

The test suite covers:
  * Default model selection — what does `GrokProvider()` pick when
    no kwargs are passed? (Important: callers rely on this default
    via auto-discovery in `quorum.providers.registry`.)
  * Factory parity — every public `grok_*` factory must return a
    `GrokProvider` whose `.model` matches its name. Drift here
    silently routes prod traffic to the wrong xAI model.
  * Env-var resolution — `XAI_API_KEY` must be read on construction,
    and explicit kwargs must override it.
  * Graceful failure — when no key is configured the provider must
    return a structured `ModelResponse(error="no_api_key")`, NOT
    raise. The consensus fan-out depends on this contract: any
    raise would surface as the user's whole consensus call dying,
    not a single provider being skipped.

Historical note: an earlier revision exported `grok_3` and
`grok_4_mini` factories. Both were removed when xAI deprecated
those tiers. Re-test against the current public surface
(`grok_4`, `grok_4_20_chat`).
"""

from __future__ import annotations

import pytest

from quorum.providers.grok import GrokProvider, grok_4, grok_4_20_chat


def test_default_model_is_grok_4_0709():
    """The bare constructor picks grok-4-0709 — current xAI flagship
    that the registry's auto-discovery relies on."""
    p = GrokProvider()
    assert p.model == "grok-4-0709"
    assert p.name == "grok-4-0709"


def test_factory_grok_4_uses_correct_model():
    """`grok_4()` is the canonical factory; the model id has to
    match `grok-4-0709` for the pricing table lookup to succeed."""
    p = grok_4(api_key="test-key")
    assert isinstance(p, GrokProvider)
    assert p.model == "grok-4-0709"
    assert p.api_key == "test-key"


def test_factory_grok_4_20_chat_uses_correct_model():
    """`grok_4_20_chat()` exposes the non-reasoning variant. If this
    drifts, callers wanting cheap completions end up paying for the
    reasoning SKU (or vice versa)."""
    p = grok_4_20_chat(api_key="test-key")
    assert isinstance(p, GrokProvider)
    assert p.model == "grok-4.20-0309-non-reasoning"


def test_api_key_read_from_env(monkeypatch):
    """Env-var fallback — production typically sets `XAI_API_KEY`
    rather than passing it through `__init__`."""
    monkeypatch.setenv("XAI_API_KEY", "env-key-123")
    p = GrokProvider()
    assert p.api_key == "env-key-123"


def test_explicit_api_key_overrides_env(monkeypatch):
    """Explicit kwarg must win over env so test fixtures and
    multi-tenant setups can scope keys per-call."""
    monkeypatch.setenv("XAI_API_KEY", "env-key")
    p = GrokProvider(api_key="explicit-key")
    assert p.api_key == "explicit-key"


@pytest.mark.asyncio
async def test_complete_returns_error_when_no_api_key(monkeypatch):
    """No key configured → structured error, never a raise. This is
    the contract the consensus fan-out relies on to keep the round
    going when a single provider is mis-configured."""
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    p = GrokProvider()
    result = await p.complete("hello")
    assert result.error == "no_api_key"
    assert result.response == ""
    # name == model, so default constructor → grok-4-0709
    assert result.name == "grok-4-0709"
