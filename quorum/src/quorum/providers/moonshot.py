"""Moonshot AI provider — Kimi family via OpenAI-compatible chat completions.

Endpoint: https://api.moonshot.ai/v1/chat/completions (international)
        / https://api.moonshot.cn/v1/chat/completions (PRC)
Auth: Authorization: Bearer $MOONSHOT_API_KEY
Signup: https://platform.moonshot.ai (international)

Kimi K2.6 (released 2026-04-20) was the first open-weight model to beat
GPT-5.4 (xhigh) on SWE-Bench Pro. The thinking parameter controls whether
chain-of-thought reasoning is emitted. We default to disabled because
consensus benefits from many short calls more than from one long reasoning
trace per provider — flip it to enabled when accuracy beats latency.

Data residency note: same caveat as Zhipu — PRC-hosted infra; review GDPR /
sector guidance before routing production customer prompts through this
provider. OSS weights on Hugging Face avoid the data flow entirely.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_ENDPOINT = "https://api.moonshot.ai/v1/chat/completions"

# Per-million-token pricing in USD. Conservative defaults — refine when
# Moonshot publishes the international sheet.
_PRICING: dict[str, tuple[float, float]] = {
    "kimi-k2.6": (0.60, 2.50),
    "kimi-k2-turbo": (0.30, 1.20),
}


class MoonshotProvider(Provider):
    name = "moonshot-kimi-k2.6"

    def __init__(
        self,
        model: str = "kimi-k2.6",
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("MOONSHOT_API_KEY", "")
        self.thinking_enabled = thinking_enabled
        self.name = f"moonshot-{model}"

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
                "temperature": 1.0 if self.thinking_enabled else 0.6,
                "top_p": 0.95,
                "stream": False,
                "thinking": {"type": "enabled" if self.thinking_enabled else "disabled"},
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


def kimi_k2_6() -> MoonshotProvider:
    return MoonshotProvider(model="kimi-k2.6")


def kimi_k2_6_thinking() -> MoonshotProvider:
    return MoonshotProvider(model="kimi-k2.6", thinking_enabled=True)


def kimi_k2_turbo() -> MoonshotProvider:
    return MoonshotProvider(model="kimi-k2-turbo")
