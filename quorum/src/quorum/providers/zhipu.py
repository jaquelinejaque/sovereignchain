"""Zhipu AI provider — GLM family via OpenAI-compatible chat completions.

Endpoint: https://api.z.ai/api/paas/v4/chat/completions
Auth: Authorization: Bearer $ZHIPU_API_KEY (alias: GLM_API_KEY)
Signup: https://open.bigmodel.cn (mainland) or https://z.ai (international)

GLM-5.2 is Zhipu's flagship released 2026-06-13: 744B MoE with 40B active
params, 1M context window, dual reasoning modes (High/Max), open weights
under MIT license. Particularly strong on coding (77.8% SWE-bench Verified).

Data residency note: API traffic terminates on PRC infrastructure. Sovereign
Chain Ltd UK should evaluate against GDPR transfer requirements and any
sector-specific guidance before routing customer prompts through this
provider in production. The OSS engine (self-hosted weights) avoids the
data flow concern entirely.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"

# Per-million-token pricing in USD. Conservative defaults — refine when
# Zhipu publishes the official sheet for 5.2.
_PRICING: dict[str, tuple[float, float]] = {
    "glm-5.2": (0.60, 2.20),
    "glm-5.2-air": (0.20, 0.60),
    "glm-4.6": (0.50, 1.50),
}


class ZhipuProvider(Provider):
    name = "zhipu-glm-5.2"

    def __init__(self, model: str = "glm-5.2", api_key: str | None = None):
        self.model = model
        self.api_key = (
            api_key
            or os.getenv("ZHIPU_API_KEY")
            or os.getenv("GLM_API_KEY")
            or ""
        )
        self.name = f"zhipu-{model}"

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


def glm_5_2() -> ZhipuProvider:
    return ZhipuProvider(model="glm-5.2")


def glm_5_2_air() -> ZhipuProvider:
    return ZhipuProvider(model="glm-5.2-air")


def glm_4_6() -> ZhipuProvider:
    return ZhipuProvider(model="glm-4.6")
