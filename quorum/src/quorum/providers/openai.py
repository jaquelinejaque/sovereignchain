"""OpenAI GPT provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

# Pricing per 1M tokens (USD, 2026-06).
_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5": (5.0, 15.0),
    "gpt-5-mini": (0.4, 1.6),
    "gpt-4.1": (2.0, 8.0),
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
}


class OpenAIProvider(Provider):
    name = "gpt-5"

    def __init__(self, model: str = "gpt-5", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.name = model

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 800,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ModelResponse:
        try:
            if not self.api_key:
                return ModelResponse(name=self.name, response="", error="no_api_key")

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            images = kwargs.get("images", [])
            content = [{"type": "text", "text": prompt}]
            if images:
                for img in images:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}})

            # OpenAI Chat Completions: system prompt is a regular message with
            # role='system', prepended before the user turn.
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": content})

            tokens_field = "max_completion_tokens" if self.model.startswith("gpt-5") else "max_tokens"
            payload: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                tokens_field: max_tokens,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers, json=payload,
                )

            if r.status_code != 200:
                safe = r.content[:200].decode("utf-8", "replace").replace("\n", " ").replace("\r", " ")
                return ModelResponse(name=self.name, response="", error=f"http_{r.status_code}: {safe}")

            try:
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                ti, to = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                in_rate, out_rate = _PRICING.get(self.model, (2.5, 10.0))
                cost = (ti * in_rate + to * out_rate) / 1_000_000
            except (KeyError, IndexError, ValueError, TypeError, json.JSONDecodeError):
                return ModelResponse(name=self.name, response="", error="parse_error")

            return ModelResponse(
                name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
            )
        except Exception:
            return ModelResponse(name=self.name, response="", error="internal_error")


def gpt_5() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-5")


def gpt_5_mini() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-5-mini")


def gpt_4_1() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4.1")


def gpt_4o_mini() -> OpenAIProvider:
    return OpenAIProvider(model="gpt-4o-mini")
