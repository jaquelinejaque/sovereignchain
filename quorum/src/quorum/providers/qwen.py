"""Alibaba Qwen provider — DashScope international OpenAI-compatible endpoint.

Endpoint: https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions
        / https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions (PRC)
Auth: Authorization: Bearer $DASHSCOPE_API_KEY (alias: QWEN_API_KEY)
Signup: https://modelstudio.console.alibabacloud.com (international)

Qwen 3.7 Max (2026) leads agentic-coding benchmarks (SWE-Pro 60.6%,
Terminal-Bench 2.0 69.7%), ahead of DeepSeek V4 Pro. Qwen3 also supports an
optional `enable_thinking` flag; we default it off because consensus
benefits more from many short calls than from one deep reasoning trace per
provider — flip it on when accuracy beats latency.

We use the Singapore endpoint (`dashscope-intl`) so EU/UK customers stay
out of PRC data residency. A US endpoint also exists; switch via
DASHSCOPE_REGION env var if needed.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_ENDPOINTS = {
    "intl": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
    "us": "https://dashscope-us.aliyuncs.com/compatible-mode/v1/chat/completions",
    "cn": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
}

# Per-million-token pricing in USD. Conservative defaults — refine when
# Alibaba publishes 3.7 Max sheet officially.
_PRICING: dict[str, tuple[float, float]] = {
    "qwen3.7-max": (1.20, 6.00),
    "qwen3.7-plus": (0.40, 1.20),
    "qwen3-max": (1.20, 6.00),
    "qwen-max": (1.20, 6.00),
    "qwen-plus": (0.40, 1.20),
    "qwen-turbo": (0.05, 0.20),
    "qwen3-coder-plus": (1.20, 6.00),
}


class QwenProvider(Provider):
    name = "qwen-3-max"

    def __init__(
        self,
        model: str = "qwen3-max",
        api_key: str | None = None,
        thinking_enabled: bool = False,
    ):
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or ""
        )
        self.thinking_enabled = thinking_enabled
        # DASHSCOPE_BASE_URL takes precedence — lets workspace-dedicated
        # MaaS deployments (Alibaba PAI custom endpoints) be used in place
        # of the global DashScope service. Append `/chat/completions` to the
        # base, since MaaS endpoints expose the OpenAI-compat root.
        base_url = os.getenv("DASHSCOPE_BASE_URL", "").rstrip("/")
        if base_url:
            self.endpoint = f"{base_url}/chat/completions"
        else:
            region = os.getenv("DASHSCOPE_REGION", "intl").lower()
            self.endpoint = _ENDPOINTS.get(region, _ENDPOINTS["intl"])
        self.name = f"qwen-{model}"

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
                "top_p": 0.7,
                "stream": False,
                "enable_thinking": self.thinking_enabled,
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(self.endpoint, headers=headers, json=payload)

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


def qwen3_max() -> QwenProvider:
    return QwenProvider(model="qwen3-max")


def qwen3_7_max() -> QwenProvider:
    """Qwen 3.7 Max — agentic coding leader (SWE-Pro 60.6%). Available in PAI MaaS workspaces."""
    return QwenProvider(model="qwen3.7-max")


def qwen3_7_plus() -> QwenProvider:
    return QwenProvider(model="qwen3.7-plus")


def qwen3_coder_plus() -> QwenProvider:
    return QwenProvider(model="qwen3-coder-plus")


def qwen_max() -> QwenProvider:
    return QwenProvider(model="qwen-max")


def qwen_plus() -> QwenProvider:
    return QwenProvider(model="qwen-plus")


def qwen_turbo() -> QwenProvider:
    return QwenProvider(model="qwen-turbo")
