"""NVIDIA AI Foundation provider — OpenAI-compatible chat completion API."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

_PRICING: dict[str, tuple[float, float]] = {
    "meta/llama-3.3-70b-instruct": (0.0, 0.0),
    "meta/llama-3.2-3b-instruct": (0.0, 0.0),
    "meta/llama-3.1-8b-instruct": (0.0, 0.0),
    "meta/llama-4-maverick-17b-128e-instruct": (0.0, 0.0),
    "deepseek-ai/deepseek-v4-flash": (0.0, 0.0),
    "abacusai/dracarys-llama-3.1-70b-instruct": (0.0, 0.0),
}


class NvidiaProvider(Provider):
    name = "nvidia-llama-3.3-70b"

    def __init__(self, model: str = "meta/llama-3.3-70b-instruct", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY", "")
        self.name = f"nvidia-{model.split('/')[-1]}"

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
                "top_p": 0.7,
                "stream": False,
            }

            async with httpx.AsyncClient(timeout=90.0) as client:
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


def llama_3_3_nvidia() -> NvidiaProvider:
    return NvidiaProvider(model="meta/llama-3.3-70b-instruct")


def llama_3_2_3b_nvidia() -> NvidiaProvider:
    return NvidiaProvider(model="meta/llama-3.2-3b-instruct")


def llama_3_1_8b_nvidia() -> NvidiaProvider:
    return NvidiaProvider(model="meta/llama-3.1-8b-instruct")


def llama_4_maverick_nvidia() -> NvidiaProvider:
    return NvidiaProvider(model="meta/llama-4-maverick-17b-128e-instruct")


def deepseek_v4_nvidia() -> NvidiaProvider:
    return NvidiaProvider(model="deepseek-ai/deepseek-v4-flash")


def dracarys_70b_nvidia() -> NvidiaProvider:
    return NvidiaProvider(model="abacusai/dracarys-llama-3.1-70b-instruct")
