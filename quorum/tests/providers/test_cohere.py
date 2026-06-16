"""Smoke tests for the Cohere provider (no real API calls)."""

from __future__ import annotations

import pytest

from quorum.providers.base import ModelResponse
from quorum.providers.cohere import (
    CohereProvider,
    command_a,
    command_r,
    command_r_plus,
)


def test_default_model_and_name():
    p = CohereProvider(api_key="dummy")
    assert p.model == "command-r-plus-08-2024"
    assert p.name == "cohere-command-r-plus-08-2024"


def test_factory_command_r_plus():
    p = command_r_plus()
    assert isinstance(p, CohereProvider)
    assert p.model == "command-r-plus-08-2024"
    assert p.name == "cohere-command-r-plus-08-2024"


def test_factory_command_r():
    p = command_r()
    assert isinstance(p, CohereProvider)
    assert p.model == "command-r-08-2024"
    assert p.name == "cohere-command-r-08-2024"


def test_factory_command_a():
    p = command_a()
    assert isinstance(p, CohereProvider)
    assert p.model == "command-a-03-2025"
    assert p.name == "cohere-command-a-03-2025"


def test_explicit_api_key_overrides_env(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "from-env")
    p = CohereProvider(api_key="explicit")
    assert p.api_key == "explicit"


def test_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "env-secret")
    p = CohereProvider()
    assert p.api_key == "env-secret"


def test_empty_api_key_when_env_missing(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    p = CohereProvider()
    assert p.api_key == ""


@pytest.mark.asyncio
async def test_no_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    p = CohereProvider()
    resp = await p.complete("hello")
    assert isinstance(resp, ModelResponse)
    assert resp.error == "no_api_key"
    assert resp.response == ""
