"""OpenAI GPT provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_INPUT_PER_1M = 2.50
_OUTPUT_PER_1M = 10.0


class OpenAIProvider(Provider):
    name = "gpt-5"

    def __init__(self, model: str = "gpt-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    async def complete(self, prompt: str, *, max_tokens: int = 800) -> ModelResponse:
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

        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers, json=payload,
            )

        if r.status_code != 200:
            return ModelResponse(
                name=self.name, response="",
                error=f"http_{r.status_code}: {r.text[:120]}",
            )

        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        ti, to = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        cost = (ti * _INPUT_PER_1M + to * _OUTPUT_PER_1M) / 1_000_000

        return ModelResponse(
            name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
        )
