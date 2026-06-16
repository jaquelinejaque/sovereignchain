"""Anthropic Claude provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

# Pricing per 1M tokens (USD, current as of 2026-06; per-model below).
_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-4-8": (15.0, 75.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5-20251001": (0.8, 4.0),
}


class AnthropicProvider(Provider):
    name = "claude-sonnet-4-6"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.name = model

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(name=self.name, response="", error="no_api_key")

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=payload,
            )

        if r.status_code != 200:
            # Sanitise upstream body: decode bytes safely (avoid mid-codepoint splits)
            # and strip CR/LF so an echoed prompt can't inject fake log lines.
            safe = r.content[:200].decode("utf-8", "replace").replace("\n", " ").replace("\r", " ")
            return ModelResponse(
                name=self.name, response="",
                error=f"http_{r.status_code}: {safe}",
            )

        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        usage = data.get("usage", {})
        ti, to = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        in_rate, out_rate = _PRICING.get(self.model, (3.0, 15.0))
        cost = (ti * in_rate + to * out_rate) / 1_000_000

        return ModelResponse(
            name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
        )


def claude_opus() -> AnthropicProvider:
    return AnthropicProvider(model="claude-opus-4-8")


def claude_sonnet() -> AnthropicProvider:
    return AnthropicProvider(model="claude-sonnet-4-6")


def claude_haiku() -> AnthropicProvider:
    return AnthropicProvider(model="claude-haiku-4-5-20251001")
