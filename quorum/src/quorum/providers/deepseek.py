"""DeepSeek direct API provider (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_ENDPOINT = "https://api.deepseek.com/chat/completions"

# USD per 1M tokens (input, output). Source: platform.deepseek.com/pricing
_PRICING: dict[str, tuple[float, float]] = {
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
}


class DeepSeekProvider(Provider):
    name = "deepseek-chat"

    def __init__(self, model: str = "deepseek-chat", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.name = f"deepseek-{model.split('-', 1)[-1]}"

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> ModelResponse:
        try:
            if not self.api_key:
                return ModelResponse(name=self.name, response="", error="no_api_key")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.2,
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(_ENDPOINT, headers=headers, json=payload)

            if r.status_code != 200:
                body = r.content[:200].decode("utf-8", errors="replace").replace("\n", " ").replace("\r", " ")
                return ModelResponse(name=self.name, response="", error=f"http_{r.status_code}: {body}")

            try:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                ti = usage.get("prompt_tokens", 0)
                to = usage.get("completion_tokens", 0)
                in_rate, out_rate = _PRICING.get(self.model, (0.0, 0.0))
                cost = (ti * in_rate + to * out_rate) / 1_000_000
            except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
                return ModelResponse(name=self.name, response="", error="parse_error")

            return ModelResponse(
                name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
            )
        except Exception:
            return ModelResponse(name=self.name, response="", error="internal_error")


def deepseek_chat() -> DeepSeekProvider:
    return DeepSeekProvider(model="deepseek-chat")


def deepseek_reasoner() -> DeepSeekProvider:
    return DeepSeekProvider(model="deepseek-reasoner")
