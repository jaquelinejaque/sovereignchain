"""Ollama local provider (free, runs on your machine)."""

from __future__ import annotations

import os

import httpx

from quorum.providers.base import ModelResponse, Provider


class OllamaProvider(Provider):
    def __init__(
        self,
        model: str = "llama3.2",
        host: str | None = None,
        name: str | None = None,
    ):
        self.model = model
        self.host = (host or os.getenv("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        # Default name kept as "llama-local" for backwards compatibility with the
        # default llama3.2 instance (Hebbian/ELO history is keyed by this name).
        # Other models surface as ollama:<model> so they get their own evolution row.
        self.name = name or ("llama-local" if model.startswith("llama3.2") else f"ollama:{model}")

    async def complete(self, prompt: str, *, max_tokens: int = 800, **kwargs) -> ModelResponse:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{self.host}/api/generate", json=payload)
        except httpx.HTTPError as e:
            return ModelResponse(name=self.name, response="", error=f"ollama_unreachable: {e}")

        if r.status_code != 200:
            return ModelResponse(
                name=self.name, response="",
                error=f"http_{r.status_code}: {r.text[:120]}",
            )

        data = r.json()
        text = data.get("response", "")
        ti = data.get("prompt_eval_count", 0)
        to = data.get("eval_count", 0)
        # Cost is 0 — runs on user's machine.
        return ModelResponse(
            name=self.name, response=text, tokens_in=ti, tokens_out=to, cost_usd=0.0,
        )
