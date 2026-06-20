"""Cohere provider — v2 chat completion API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_ENDPOINT = "https://api.cohere.com/v2/chat"

_PRICING: dict[str, tuple[float, float]] = {
    "command-r-plus-08-2024": (2.50, 10.0),
    "command-r-08-2024": (0.15, 0.60),
    "command-a-03-2025": (2.50, 10.0),
}


class CohereProvider(Provider):
    name = "cohere-command-r-plus"

    def __init__(self, model: str = "command-r-plus-08-2024", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY", "")
        self.name = f"cohere-{model}"

    async def complete(self, prompt: str, *, max_tokens: int = 800, **kwargs) -> ModelResponse:
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
                "p": 0.7,
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(_ENDPOINT, headers=headers, json=payload)

            if r.status_code != 200:
                body = r.content[:200].decode("utf-8", errors="replace").replace("\n", " ").replace("\r", " ")
                return ModelResponse(name=self.name, response="", error=f"http_{r.status_code}: {body}")

            try:
                data = r.json()
                # Cohere v2 shape: data['message']['content'][0]['text']
                text = data["message"]["content"][0]["text"]
                usage = data.get("usage", {})
                billed = usage.get("billed_units", {}) if isinstance(usage, dict) else {}
                ti = billed.get("input_tokens", 0)
                to = billed.get("output_tokens", 0)
                in_rate, out_rate = _PRICING.get(self.model, (0.0, 0.0))
                cost = (ti * in_rate + to * out_rate) / 1_000_000
            except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
                return ModelResponse(name=self.name, response="", error="parse_error")

            return ModelResponse(
                name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
            )
        except Exception:
            return ModelResponse(name=self.name, response="", error="internal_error")


def command_r_plus() -> CohereProvider:
    return CohereProvider(model="command-r-plus-08-2024")


def command_r() -> CohereProvider:
    return CohereProvider(model="command-r-08-2024")


def command_a() -> CohereProvider:
    return CohereProvider(model="command-a-03-2025")
