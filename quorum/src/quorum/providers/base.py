"""Base interfaces for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Output of one provider's call."""

    name: str
    response: str
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    weight: float = 0.0  # filled in by consensus engine
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "response": self.response,
            "latency_ms": round(self.latency_ms, 1),
            "cost_usd": round(self.cost_usd, 6),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "weight": round(self.weight, 4),
            "error": self.error,
        }


class Provider(ABC):
    """Abstract LLM provider — implements complete(prompt) -> ModelResponse."""

    name: str

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 800,
        system_prompt: str | None = None,
        **kwargs,
    ) -> ModelResponse:
        """Run a single completion. Must return a ModelResponse, never raise.

        Args:
            prompt: User-facing message content.
            max_tokens: Hard cap on output tokens.
            system_prompt: Optional system instruction (Layer 1 of the
                SelfPromptOptimizer wiring). Providers that natively support
                system prompts (Anthropic/OpenAI/Gemini) inject it via the
                appropriate provider-specific channel. Providers without
                native support silently ignore it via ``**kwargs`` to keep
                backwards compatibility — no caller is forced to know which
                providers honour the field.
        """
        raise NotImplementedError
