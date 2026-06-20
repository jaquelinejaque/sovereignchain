"""Google Gemini provider."""

from __future__ import annotations

import os
from typing import Any

import httpx

from quorum.providers.base import ModelResponse, Provider

_INPUT_PER_1M = 0.075
_OUTPUT_PER_1M = 0.30


class GeminiProvider(Provider):
    name = "gemini-flash"

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_AI_STUDIO_KEY")
            or ""
        )

    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 800,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ModelResponse:
        if not self.api_key:
            return ModelResponse(name=self.name, response="", error="no_api_key")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        images = kwargs.get("images", [])
        parts = [{"text": prompt}]
        if images:
            for img in images:
                parts.append({
                    "inlineData": {
                        "mimeType": "image/jpeg",
                        "data": img
                    }
                })

        payload: dict[str, Any] = {
            "contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": max_tokens},
        }
        # Gemini v1beta: systemInstruction is a top-level sibling of `contents`.
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                r = await client.post(url, json=payload, headers=headers)
        except httpx.HTTPError:
            return ModelResponse(name=self.name, response="", error="network_error")

        if r.status_code != 200:
            return ModelResponse(
                name=self.name, response="",
                error=f"http_{r.status_code}: {r.text[:120]}",
            )

        data = r.json()
        candidates = data.get("candidates") or []
        text = ""
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(p.get("text", "") for p in parts)

        usage = data.get("usageMetadata", {})
        ti = usage.get("promptTokenCount", 0)
        to = usage.get("candidatesTokenCount", 0)
        cost = (ti * _INPUT_PER_1M + to * _OUTPUT_PER_1M) / 1_000_000

        return ModelResponse(
            name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=cost,
        )
