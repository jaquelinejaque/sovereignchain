# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Production-grade ``ResponderFn`` adapters for ``evaluate_checkpoint``.

The eval_set module ships with an ``_echo_responder`` so its tests stay
hermetic. Real usage — benchmarking a freshly distilled local Llama,
or comparing two Ollama checkpoints — needs a responder that hits an
actual model. That's what this module is for.

Three adapters are exposed, each chosen to remove a different friction:

* :func:`provider_responder` — wraps any ``quorum.providers.base.Provider``.
  The natural choice when the operator already has provider keys.

* :func:`ollama_responder` — talks to a local Ollama server by model
  name. The natural choice for benchmarking a distilled local model
  (the whole reason this pipeline exists). Pure HTTP, no API keys.

* :func:`http_responder` — generic POST adapter for self-hosted
  checkpoint servers that don't speak the Ollama protocol. Lets
  operators benchmark Triton / vLLM / TGI / a homegrown FastAPI
  wrapper without writing a new Provider class.

All three return an async ``prompt -> str`` callable that slots into
``evaluate_checkpoint(responder=...)`` unchanged.

Design rules:

* **No new hard deps.** ``httpx`` is already pulled in by the existing
  provider stack; we reuse it.
* **No silent retries.** A failure here is data — the eval loop already
  treats an empty response as a zero-score item, which is the truthful
  signal. Wrapping with retries would mask a broken checkpoint.
* **Hard timeout per call.** Defaults to 60 s. Without it, one stuck
  checkpoint stalls the whole eval set.
* **Bounded concurrency** via :func:`evaluate_concurrently` — the
  per-item loop in ``evaluate_checkpoint`` is sequential; for a 50-item
  set against a slow local model that's ~minutes. Concurrent variant
  is opt-in (operators don't always want to slam their GPU).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


ResponderFn = Callable[[str], "asyncio.Future[str] | Any"]
"""Async ``prompt -> str`` callable expected by ``evaluate_checkpoint``."""


# --------------------------------------------------------------------------- #
# Adapter 1: any Provider                                                     #
# --------------------------------------------------------------------------- #


def provider_responder(provider: Any) -> ResponderFn:
    """Wrap any ``quorum.providers.base.Provider`` as a ``ResponderFn``.

    The provider's ``complete`` method returns a ``ModelResponse`` whose
    ``.error`` field is populated on failure. We surface that as an empty
    string so the scorer can record a zero — same convention as the
    rest of the eval pipeline.

    Args:
        provider: Anything with an async ``complete(prompt) -> ModelResponse``
            method. Duck-typed on purpose — no import of ``base`` here,
            so importing this module doesn't pull the provider stack in.

    Returns:
        Async ``prompt -> str``.

    Example::

        from quorum.providers.anthropic import AnthropicProvider
        from quorum.evolution.eval_set import evaluate_checkpoint
        from quorum.evolution.eval_runner import provider_responder

        prov = AnthropicProvider(model="claude-haiku-4-5")
        report = await evaluate_checkpoint(
            version="haiku-baseline",
            responder=provider_responder(prov),
        )
    """

    async def _call(prompt: str) -> str:
        try:
            resp = await provider.complete(prompt)
        except Exception as e:  # noqa: BLE001 — see module doc: failure is data
            logger.warning("provider_responder.exception err=%s", e)
            return ""
        # Providers are contract-bound not to raise; an error is reported
        # via ``.error``. Treat that as an empty response.
        if getattr(resp, "error", ""):
            logger.warning(
                "provider_responder.provider_error provider=%s err=%s",
                getattr(resp, "name", "?"), resp.error,
            )
            return ""
        return getattr(resp, "response", "") or ""

    return _call


# --------------------------------------------------------------------------- #
# Adapter 2: Ollama by model name (the distilled-Llama case)                  #
# --------------------------------------------------------------------------- #


def ollama_responder(
    model: str,
    *,
    host: str | None = None,
    timeout_s: float = 60.0,
    max_tokens: int = 800,
) -> ResponderFn:
    """Talk to a local Ollama server by model name.

    The reason this adapter exists separately from ``provider_responder``:
    after distillation the LoRA adapter is materialised as an Ollama
    model (``ollama create my-distill-v3 -f Modelfile``). The simplest
    way to benchmark it is the Ollama HTTP API directly — no need to
    instantiate an OllamaProvider with all its Hebbian/ELO name-keying
    side effects.

    Args:
        model: Ollama model tag (e.g. ``"llama3.2:3b"``,
            ``"my-distill-v3"``).
        host: Override for ``http://localhost:11434``.
        timeout_s: Per-request timeout. Short eval prompts should
            comfortably finish under 30 s even on a slow GPU; the
            default 60 s leaves headroom without letting a hung model
            stall the whole run.
        max_tokens: Cap on output tokens. Eval items in the canonical
            set are short — 800 is generous.

    Returns:
        Async ``prompt -> str``.
    """
    import httpx  # local import: keeps the module cheap when unused

    base = (host or _default_ollama_host()).rstrip("/")
    endpoint = f"{base}/api/generate"

    async def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(endpoint, json=payload)
        except httpx.HTTPError as e:
            logger.warning(
                "ollama_responder.unreachable host=%s model=%s err=%s",
                base, model, e,
            )
            return ""
        if r.status_code != 200:
            logger.warning(
                "ollama_responder.http_%d model=%s body=%s",
                r.status_code, model, r.text[:160],
            )
            return ""
        try:
            data = r.json()
        except ValueError:
            logger.warning("ollama_responder.bad_json model=%s body=%s",
                           model, r.text[:160])
            return ""
        return str(data.get("response", "")) or ""

    return _call


def _default_ollama_host() -> str:
    """Read ``OLLAMA_HOST`` lazily so the env can be set per-eval-run."""
    import os
    return os.getenv("OLLAMA_HOST", "http://localhost:11434")


# --------------------------------------------------------------------------- #
# Adapter 3: generic HTTP POST                                                #
# --------------------------------------------------------------------------- #


def http_responder(
    url: str,
    *,
    prompt_field: str = "prompt",
    response_path: tuple[str, ...] = ("response",),
    headers: dict[str, str] | None = None,
    extra_payload: dict[str, Any] | None = None,
    timeout_s: float = 60.0,
) -> ResponderFn:
    """Benchmark a self-hosted checkpoint server with a POST endpoint.

    For environments where the checkpoint is exposed via vLLM,
    Triton-Inference-Server, TGI, or a homegrown FastAPI wrapper that
    doesn't speak the Ollama protocol.

    Args:
        url: Full POST URL of the completion endpoint.
        prompt_field: JSON body key for the prompt. Default ``"prompt"``;
            override for vLLM-style ``"inputs"`` or custom keys.
        response_path: Tuple of dict keys to walk to extract the text
            response from the JSON. e.g. for vLLM's
            ``{"text": ["..."]}`` use ``("text", 0)``.
        headers: Optional HTTP headers (auth tokens, etc).
        extra_payload: Static fields merged into every request body
            (temperature, max_tokens, model name, ...).
        timeout_s: Per-request timeout.

    Returns:
        Async ``prompt -> str``.

    Example (vLLM)::

        responder = http_responder(
            "http://gpu-box:8000/v1/completions",
            prompt_field="prompt",
            response_path=("choices", 0, "text"),
            extra_payload={"model": "distill-v3", "max_tokens": 800},
            headers={"Authorization": "Bearer ..."},
        )
    """
    import httpx

    async def _call(prompt: str) -> str:
        body: dict[str, Any] = {prompt_field: prompt}
        if extra_payload:
            body.update(extra_payload)
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                r = await client.post(url, json=body, headers=headers or {})
        except httpx.HTTPError as e:
            logger.warning("http_responder.unreachable url=%s err=%s", url, e)
            return ""
        if r.status_code != 200:
            logger.warning(
                "http_responder.http_%d url=%s body=%s",
                r.status_code, url, r.text[:160],
            )
            return ""
        try:
            data: Any = r.json()
        except ValueError:
            logger.warning("http_responder.bad_json url=%s body=%s",
                           url, r.text[:160])
            return ""
        for key in response_path:
            try:
                data = data[key]
            except (KeyError, IndexError, TypeError):
                logger.warning(
                    "http_responder.path_miss url=%s missing=%r in %r",
                    url, key, data,
                )
                return ""
        return str(data or "")

    return _call


# --------------------------------------------------------------------------- #
# Concurrency helper                                                          #
# --------------------------------------------------------------------------- #


def concurrent_responder(
    inner: ResponderFn, *, max_concurrency: int = 4,
) -> ResponderFn:
    """Wrap a responder with a semaphore so callers can fan eval items out.

    ``evaluate_checkpoint`` iterates items sequentially — fine for the
    hermetic echo test, slow for a real local model on 50 items. Wrapping
    the responder with this helper does not change the eval pipeline; it
    just lets the operator decide how hard to push the underlying model.

    The semaphore is created lazily on first call, on whatever event
    loop the responder runs in — important because the responder may be
    constructed at module import time (no loop) but called inside one.

    Args:
        inner: A ``ResponderFn`` from any of the adapters above.
        max_concurrency: Hard cap on in-flight requests. Default 4 keeps
            a single GPU warm without thrashing.

    Returns:
        Wrapped async ``prompt -> str``.
    """
    sem_holder: dict[str, asyncio.Semaphore] = {}

    async def _call(prompt: str) -> str:
        sem = sem_holder.get("s")
        if sem is None:
            sem = asyncio.Semaphore(max_concurrency)
            sem_holder["s"] = sem
        async with sem:
            return await inner(prompt)

    return _call


# --------------------------------------------------------------------------- #
# Convenience: run + write sidecar in one call                                #
# --------------------------------------------------------------------------- #


async def benchmark_checkpoint(
    *,
    version: str,
    responder: ResponderFn,
    sidecar_path: Path | str | None = None,
    only_classes: Iterable[str] | None = None,
) -> dict[str, Any]:
    """One-shot helper: run the canonical set + return the sidecar dict.

    Equivalent to::

        report = await evaluate_checkpoint(
            version=version, responder=responder, sidecar_path=path)
        return report.to_sidecar_dict()

    Pulled out so HTTP services / cron scripts don't have to know about
    the EvalReport dataclass — JSON in, JSON out.

    Args:
        version: Checkpoint identifier (embedded in the sidecar).
        responder: From any of the three adapters above.
        sidecar_path: Where to write the sidecar JSON. None → skip the
            write (still returns the dict).
        only_classes: Optional restriction to a subset of query classes
            (useful for "did the safety regression I'm investigating
            move?" without rerunning the full set).

    Returns:
        The sidecar dict — same shape that
        ``DistillationPipeline._run_benchmark`` reads.
    """
    # Lazy import keeps ``eval_runner`` cheap to import on its own.
    from quorum.evolution.eval_set import (
        CANONICAL_EVAL_SET,
        evaluate_checkpoint,
    )

    items = list(CANONICAL_EVAL_SET)
    if only_classes is not None:
        wanted = set(only_classes)
        items = [it for it in items if it.query_class in wanted]
        if not items:
            raise ValueError(
                f"only_classes={sorted(wanted)} matches no canonical items"
            )

    report = await evaluate_checkpoint(
        version=version,
        responder=responder,
        eval_set=items,
        sidecar_path=sidecar_path,
    )
    return report.to_sidecar_dict()


__all__ = [
    "ResponderFn",
    "provider_responder",
    "ollama_responder",
    "http_responder",
    "concurrent_responder",
    "benchmark_checkpoint",
]
