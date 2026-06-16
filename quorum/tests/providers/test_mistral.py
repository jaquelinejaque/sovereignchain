"""Smoke tests for the Mistral provider (no real API calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from quorum.providers.base import ModelResponse
from quorum.providers.mistral import (
    MistralProvider,
    codestral,
    mistral_large,
    mistral_small,
)


def test_default_model_and_name():
    p = MistralProvider(api_key="dummy")
    assert p.model == "mistral-large-latest"
    assert p.name == "mistral-mistral-large-latest"


def test_factory_mistral_large():
    p = mistral_large()
    assert isinstance(p, MistralProvider)
    assert p.model == "mistral-large-latest"
    assert p.name == "mistral-mistral-large-latest"


def test_factory_codestral():
    p = codestral()
    assert isinstance(p, MistralProvider)
    assert p.model == "codestral-latest"
    assert p.name == "mistral-codestral-latest"


def test_factory_mistral_small():
    p = mistral_small()
    assert isinstance(p, MistralProvider)
    assert p.model == "mistral-small-latest"
    assert p.name == "mistral-mistral-small-latest"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    p = MistralProvider(api_key="explicit")
    assert p.api_key == "explicit"


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "env-secret")
    p = MistralProvider()
    assert p.api_key == "env-secret"


def test_empty_api_key_when_env_missing(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    p = MistralProvider()
    assert p.api_key == ""


@pytest.mark.asyncio
async def test_no_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    p = MistralProvider()
    resp = await p.complete("hello")
    assert isinstance(resp, ModelResponse)
    assert resp.error == "no_api_key"
    assert resp.response == ""


def _mock_async_client(response_mock: MagicMock):
    """Build a fake httpx.AsyncClient context manager that returns response_mock on post()."""
    client_instance = MagicMock()
    client_instance.post = AsyncMock(return_value=response_mock)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_instance)
    cm.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock(return_value=cm)
    return factory, client_instance


@pytest.mark.asyncio
async def test_complete_success_parses_response_and_cost():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "hello world"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50},
    }

    factory, client = _mock_async_client(fake_response)
    with patch("quorum.providers.mistral.httpx.AsyncClient", factory):
        p = MistralProvider(api_key="test-key")
        resp = await p.complete("hi", max_tokens=42)

    assert resp.error == ""
    assert resp.response == "hello world"
    assert resp.tokens_in == 100
    assert resp.tokens_out == 50
    # mistral-large-latest pricing: (2.0, 6.0) per 1M
    expected = (100 * 2.0 + 50 * 6.0) / 1_000_000
    assert abs(resp.cost_usd - expected) < 1e-9

    # Verify endpoint + bearer auth used
    call_args = client.post.call_args
    assert call_args.args[0] == "https://api.mistral.ai/v1/chat/completions"
    headers = call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer test-key"
    payload = call_args.kwargs["json"]
    assert payload["model"] == "mistral-large-latest"
    assert payload["max_tokens"] == 42
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_complete_http_error_wraps_status_and_body():
    fake_response = MagicMock()
    fake_response.status_code = 401
    fake_response.content = b"unauthorized\nleak\rinjection"

    factory, _ = _mock_async_client(fake_response)
    with patch("quorum.providers.mistral.httpx.AsyncClient", factory):
        p = MistralProvider(api_key="bad")
        resp = await p.complete("hi")

    assert resp.response == ""
    assert resp.error.startswith("http_401:")
    # CR/LF stripped to prevent log injection
    assert "\n" not in resp.error
    assert "\r" not in resp.error


@pytest.mark.asyncio
async def test_complete_utf8_safe_decoding_on_error_body():
    fake_response = MagicMock()
    fake_response.status_code = 500
    # Invalid UTF-8 byte (0xff) — must not raise
    fake_response.content = b"err\xffmsg"

    factory, _ = _mock_async_client(fake_response)
    with patch("quorum.providers.mistral.httpx.AsyncClient", factory):
        p = MistralProvider(api_key="x")
        resp = await p.complete("hi")

    assert resp.error.startswith("http_500:")


@pytest.mark.asyncio
async def test_complete_parse_error_on_bad_json_shape():
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {"unexpected": "shape"}

    factory, _ = _mock_async_client(fake_response)
    with patch("quorum.providers.mistral.httpx.AsyncClient", factory):
        p = MistralProvider(api_key="x")
        resp = await p.complete("hi")

    assert resp.error == "parse_error"
    assert resp.response == ""


@pytest.mark.asyncio
async def test_complete_internal_error_on_unexpected_exception():
    factory = MagicMock(side_effect=RuntimeError("boom"))
    with patch("quorum.providers.mistral.httpx.AsyncClient", factory):
        p = MistralProvider(api_key="x")
        resp = await p.complete("hi")

    assert resp.error == "internal_error"
    assert resp.response == ""
