"""xAI Grok provider (OpenAI-compatible API)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_PRICING: dict[str, tuple[float, float]] = {
    "grok-4-0709": (3.0, 15.0),
    "grok-4.20-0309-non-reasoning": (3.0, 15.0),
    "grok-4.20-0309-reasoning": (3.0, 15.0),
}


class GrokProvider(Provider):
    name = "grok-4-0709"

    def __init__(self, model: str = "grok-4-0709", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("XAI_API_KEY", "")
        self.name = model

    async def complete(self, prompt: str, *, max_tokens: int = 800, **kwargs) -> ModelResponse:
        try:
            if not self.api_key:
                return ModelResponse(name=self.name, response="", error="no_api_key")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers, json=payload,
                )

            if r.status_code != 200:
                return ModelResponse(
                    name=self.name, response="",
                    error=f"http_{r.status_code}: {r.text[:120]}",
                )

            try:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                ti, to = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                in_price, out_price = _PRICING.get(self.model, (3.0, 15.0))
                cost = (ti * in_price + to * out_price) / 1_000_000
            except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
                return ModelResponse(name=self.name, response="", error="parse_error")

            return ModelResponse(
                name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
            )
        except Exception:
            return ModelResponse(name=self.name, response="", error="internal_error")


def grok_4(api_key: str | None = None) -> GrokProvider:
    return GrokProvider(model="grok-4-0709", api_key=api_key)


def grok_4_20_chat(api_key: str | None = None) -> GrokProvider:
    return GrokProvider(model="grok-4.20-0309-non-reasoning", api_key=api_key)
