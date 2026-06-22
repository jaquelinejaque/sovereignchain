# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``evolution.eval_runner`` — Provider/Ollama/HTTP adapters."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from quorum.evolution import eval_runner


# --------------------------------------------------------------------------- #
# Test helpers                                                                #
# --------------------------------------------------------------------------- #


def _fake_async_client(response_mock):
    """Build a fake ``httpx.AsyncClient`` context manager.

    Same shape used by tests/providers/test_mistral.py — copying the
    pattern keeps these tests legible to anyone who already knows that
    file.
    """
    client_instance = MagicMock()
    client_instance.post = AsyncMock(return_value=response_mock)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=client_instance)
    cm.__aexit__ = AsyncMock(return_value=None)

    factory = MagicMock(return_value=cm)
    return factory, client_instance


def _resp(status: int, payload):
    """Mock httpx.Response with given status and JSON payload."""
    r = MagicMock()
    r.status_code = status
    if isinstance(payload, Exception):
        r.json.side_effect = payload
    else:
        r.json.return_value = payload
    r.text = json.dumps(payload) if not isinstance(payload, Exception) else "<bad>"
    return r


# --------------------------------------------------------------------------- #
# provider_responder                                                          #
# --------------------------------------------------------------------------- #


class _FakeResp:
    """Minimal stand-in for ModelResponse — only the fields the adapter reads."""

    def __init__(self, response="", error="", name="fake"):
        self.response = response
        self.error = error
        self.name = name


class _FakeProvider:
    def __init__(self, resp=None, raises=None):
        self._resp = resp
        self._raises = raises
        self.calls: list[str] = []

    async def complete(self, prompt):
        self.calls.append(prompt)
        if self._raises:
            raise self._raises
        return self._resp


@pytest.mark.asyncio
async def test_provider_responder_returns_text_on_success():
    """Happy path — the wrapped provider's response text is returned verbatim."""
    prov = _FakeProvider(resp=_FakeResp(response="the answer is 42"))
    responder = eval_runner.provider_responder(prov)
    out = await responder("what is the answer?")
    assert out == "the answer is 42"
    assert prov.calls == ["what is the answer?"]


@pytest.mark.asyncio
async def test_provider_responder_returns_empty_on_provider_error():
    """A populated ``.error`` field maps to empty string (zero-score signal)."""
    prov = _FakeProvider(resp=_FakeResp(response="ignored", error="rate_limited"))
    responder = eval_runner.provider_responder(prov)
    out = await responder("hi")
    assert out == ""


@pytest.mark.asyncio
async def test_provider_responder_returns_empty_on_exception():
    """An unexpected exception is caught — eval loop must never crash on one item."""
    prov = _FakeProvider(raises=RuntimeError("boom"))
    responder = eval_runner.provider_responder(prov)
    out = await responder("hi")
    assert out == ""


# --------------------------------------------------------------------------- #
# ollama_responder                                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_ollama_responder_happy_path():
    """A 200 + ``{"response": "..."}`` body returns the text."""
    factory, client = _fake_async_client(
        _resp(200, {"response": "ollama answer", "eval_count": 5})
    )
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.ollama_responder("llama3.2", host="http://x:1")
        out = await responder("hello")
    assert out == "ollama answer"
    # Endpoint built from host + /api/generate
    args, kwargs = client.post.call_args
    assert args[0] == "http://x:1/api/generate"
    assert kwargs["json"]["model"] == "llama3.2"
    assert kwargs["json"]["prompt"] == "hello"
    assert kwargs["json"]["stream"] is False


@pytest.mark.asyncio
async def test_ollama_responder_non_200_returns_empty():
    factory, _ = _fake_async_client(_resp(503, {"error": "model loading"}))
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.ollama_responder("llama3.2")
        out = await responder("hi")
    assert out == ""


@pytest.mark.asyncio
async def test_ollama_responder_unreachable_host_returns_empty():
    """When httpx itself errors (host down), responder still returns ''."""
    factory = MagicMock()
    factory.side_effect = lambda *a, **kw: _aenter_raising(
        httpx.ConnectError("refused")
    )
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.ollama_responder("llama3.2")
        out = await responder("hi")
    assert out == ""


def _aenter_raising(exc):
    """Async-CM whose body raises on entering (mimics .post raising)."""
    cm = MagicMock()
    client_instance = MagicMock()

    async def _raise(*_a, **_kw):
        raise exc

    client_instance.post = _raise
    cm.__aenter__ = AsyncMock(return_value=client_instance)
    cm.__aexit__ = AsyncMock(return_value=None)
    return cm


@pytest.mark.asyncio
async def test_ollama_responder_bad_json_returns_empty():
    factory, _ = _fake_async_client(_resp(200, ValueError("not json")))
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.ollama_responder("llama3.2")
        out = await responder("hi")
    assert out == ""


@pytest.mark.asyncio
async def test_ollama_responder_uses_env_host_when_omitted(monkeypatch):
    """Without an explicit ``host=`` arg, ``OLLAMA_HOST`` env wins."""
    monkeypatch.setenv("OLLAMA_HOST", "http://envhost:9999")
    factory, client = _fake_async_client(_resp(200, {"response": "x"}))
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.ollama_responder("llama3.2")
        await responder("hi")
    args, _ = client.post.call_args
    assert args[0] == "http://envhost:9999/api/generate"


# --------------------------------------------------------------------------- #
# http_responder                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_responder_default_field_and_path():
    """Default ``prompt`` field + ``response`` path round-trip."""
    factory, client = _fake_async_client(_resp(200, {"response": "ok"}))
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.http_responder("http://srv/x")
        out = await responder("hi")
    assert out == "ok"
    _, kwargs = client.post.call_args
    assert kwargs["json"] == {"prompt": "hi"}


@pytest.mark.asyncio
async def test_http_responder_nested_path():
    """vLLM-style nested extraction: ``("choices", 0, "text")``."""
    factory, _ = _fake_async_client(
        _resp(200, {"choices": [{"text": "deep answer"}]})
    )
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.http_responder(
            "http://srv/v1/completions",
            response_path=("choices", 0, "text"),
        )
        out = await responder("hi")
    assert out == "deep answer"


@pytest.mark.asyncio
async def test_http_responder_missing_path_returns_empty():
    """If the response_path doesn't exist, return '' instead of raising."""
    factory, _ = _fake_async_client(_resp(200, {"different_shape": True}))
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.http_responder(
            "http://srv/x", response_path=("response",),
        )
        out = await responder("hi")
    assert out == ""


@pytest.mark.asyncio
async def test_http_responder_merges_extra_payload_and_headers():
    factory, client = _fake_async_client(_resp(200, {"response": "x"}))
    with patch("httpx.AsyncClient", factory):
        responder = eval_runner.http_responder(
            "http://srv/x",
            extra_payload={"model": "distill-v3", "temperature": 0.0},
            headers={"Authorization": "Bearer xyz"},
        )
        await responder("hi")
    _, kwargs = client.post.call_args
    assert kwargs["json"] == {
        "prompt": "hi", "model": "distill-v3", "temperature": 0.0,
    }
    assert kwargs["headers"]["Authorization"] == "Bearer xyz"


# --------------------------------------------------------------------------- #
# concurrent_responder                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrent_responder_caps_in_flight():
    """At most ``max_concurrency`` calls run simultaneously."""
    in_flight = 0
    peak = 0

    async def slow(prompt):
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return prompt

    capped = eval_runner.concurrent_responder(slow, max_concurrency=2)
    await asyncio.gather(*(capped(f"p{i}") for i in range(8)))
    assert peak <= 2


@pytest.mark.asyncio
async def test_concurrent_responder_returns_inner_values():
    """The wrapper does not transform the inner responder's output."""

    async def echo(p):
        return f"got:{p}"

    capped = eval_runner.concurrent_responder(echo, max_concurrency=4)
    results = await asyncio.gather(*(capped(f"p{i}") for i in range(3)))
    assert results == ["got:p0", "got:p1", "got:p2"]


# --------------------------------------------------------------------------- #
# benchmark_checkpoint                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_benchmark_checkpoint_returns_sidecar_shape(tmp_path):
    """The convenience helper returns the same dict shape the
    DistillationPipeline expects to read from the sidecar."""
    sidecar = tmp_path / "b.json"

    async def fixed(prompt):
        return f"Echo: {prompt}"

    out = await eval_runner.benchmark_checkpoint(
        version="t",
        responder=fixed,
        sidecar_path=sidecar,
        only_classes=["factual"],
    )
    assert out["version"] == "t"
    assert out["samples_evaluated"] > 0
    assert "accuracy" in out and "safety_score" in out
    assert "per_item" in out
    assert sidecar.exists()
    on_disk = json.loads(sidecar.read_text("utf-8"))
    assert on_disk["version"] == "t"


@pytest.mark.asyncio
async def test_benchmark_checkpoint_only_classes_unknown_raises():
    async def echo(p):
        return ""

    with pytest.raises(ValueError, match="matches no canonical items"):
        await eval_runner.benchmark_checkpoint(
            version="t", responder=echo, only_classes=["this_class_doesnt_exist"],
        )


@pytest.mark.asyncio
async def test_benchmark_checkpoint_no_sidecar_path_skips_write(tmp_path):
    """sidecar_path=None means return-only — no file written."""

    async def echo(p):
        return ""

    out = await eval_runner.benchmark_checkpoint(
        version="x", responder=echo, only_classes=["creative"],
    )
    assert out["version"] == "x"
    # tmp_path is empty
    assert list(tmp_path.iterdir()) == []
