"""Replicate provider — one key, dozens of open-source models.

Default models: Llama 3.3 70B, Mistral Large, DeepSeek V3, Qwen 2.5 72B, Phi-4.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

# Rough costs in USD per 1M tokens (Replicate billing varies).
# These are conservative estimates — Replicate bills by GPU-second on most models.
_DEFAULT_COST_PER_1M = 0.30


class ReplicateProvider(Provider):
    """Generic Replicate adapter. Specify model slug at construction time."""

    def __init__(
        self,
        model_slug: str,
        name: str | None = None,
        api_token: str | None = None,
    ):
        self.model_slug = model_slug  # e.g. "meta/llama-3.3-70b-instruct"
        self.name = name or model_slug.split("/")[-1]
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN", "")

    async def complete(self, prompt: str, *, max_tokens: int = 800, **kwargs) -> ModelResponse:
        if not self.api_token:
            return ModelResponse(name=self.name, response="", error="no_api_key")

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
            "Prefer": "wait",
        }
        payload: dict[str, Any] = {
            "input": {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": 0.4,
            }
        }

        url = f"https://api.replicate.com/v1/models/{self.model_slug}/predictions"

        async with httpx.AsyncClient(timeout=90.0) as client:
            r = await client.post(url, headers=headers, json=payload)

        if r.status_code not in (200, 201):
            return ModelResponse(
                name=self.name, response="",
                error=f"http_{r.status_code}: {r.text[:160]}",
            )

        data = r.json()
        # Replicate returns output as list of strings or a single string.
        raw_output = data.get("output", "")
        if isinstance(raw_output, list):
            text = "".join(str(x) for x in raw_output)
        else:
            text = str(raw_output)

        # Token counts: Replicate usually returns metrics in `metrics`.
        metrics = data.get("metrics", {}) or {}
        ti = int(metrics.get("input_token_count", 0))
        to = int(metrics.get("output_token_count", 0))
        cost = (ti + to) * _DEFAULT_COST_PER_1M / 1_000_000

        return ModelResponse(
            name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
        )


# Convenience factories for popular open-source models
def llama_3_3() -> ReplicateProvider:
    return ReplicateProvider("meta/llama-3.3-70b-instruct", name="llama-3.3-70b")


def mistral_large() -> ReplicateProvider:
    return ReplicateProvider("mistralai/mistral-large", name="mistral-large")


def deepseek_v3() -> ReplicateProvider:
    return ReplicateProvider("deepseek-ai/deepseek-v3", name="deepseek-v3")


def qwen_2_5() -> ReplicateProvider:
    return ReplicateProvider("alibaba/qwen-2.5-72b-instruct", name="qwen-2.5-72b")


def phi_4() -> ReplicateProvider:
    return ReplicateProvider("microsoft/phi-4", name="phi-4")


def hermes_3_70b() -> ReplicateProvider:
    return ReplicateProvider(
        "nousresearch/hermes-3-llama-3.1-70b",
        name="hermes-3-llama-3.1-70b",
    )


def hermes_3_405b() -> ReplicateProvider:
    return ReplicateProvider(
        "nousresearch/hermes-3-llama-3.1-405b",
        name="hermes-3-llama-3.1-405b",
    )
