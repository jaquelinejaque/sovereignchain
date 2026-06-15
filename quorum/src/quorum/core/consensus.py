"""Consensus engine — N models in parallel, semantic agreement scoring."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Sequence

from quorum.providers.base import ModelResponse, Provider
from quorum.providers.registry import load_default_providers


@dataclass
class ConsensusResult:
    """Result of a multi-model consensus query."""

    answer: str
    """The synthesized consensus answer (top-weighted response by default)."""

    confidence: float
    """Semantic agreement score 0..1 (1.0 = all models said essentially the same thing)."""

    models: list[ModelResponse] = field(default_factory=list)
    """Per-model responses with latency, tokens, cost, and weight in the final answer."""

    disagreements: list[str] = field(default_factory=list)
    """Points where models materially diverged."""

    evolution_signals: dict[str, bool] = field(default_factory=dict)
    """Which evolution loops fired during this query (for downstream learning)."""

    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "models": [m.to_dict() for m in self.models],
            "disagreements": self.disagreements,
            "evolution_signals": self.evolution_signals,
            "total_cost_usd": self.total_cost_usd,
            "total_latency_ms": self.total_latency_ms,
        }


async def consensus(
    prompt: str,
    *,
    providers: Sequence[Provider] | None = None,
    max_concurrency: int = 8,
    timeout_s: float = 30.0,
) -> ConsensusResult:
    """Run a query through N LLMs in parallel and synthesize a consensus answer.

    Args:
        prompt: The user query.
        providers: Optional list of pre-configured Provider instances.
                   If None, loads all configured providers from environment.
        max_concurrency: Cap on parallel calls.
        timeout_s: Per-provider timeout.

    Returns:
        ConsensusResult with synthesized answer, confidence, and per-model details.
    """
    if providers is None:
        providers = load_default_providers()

    if not providers:
        raise RuntimeError(
            "No providers configured. Set at least one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_AI_STUDIO_KEY, "
            "REPLICATE_API_TOKEN, or run Ollama locally."
        )

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _call(p: Provider) -> ModelResponse:
        async with semaphore:
            t0 = time.perf_counter()
            try:
                resp = await asyncio.wait_for(p.complete(prompt), timeout=timeout_s)
            except asyncio.TimeoutError:
                return ModelResponse(
                    name=p.name, response="", error="timeout",
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    cost_usd=0.0, tokens_in=0, tokens_out=0,
                )
            except Exception as e:  # noqa: BLE001
                return ModelResponse(
                    name=p.name, response="", error=str(e)[:200],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    cost_usd=0.0, tokens_in=0, tokens_out=0,
                )
            resp.latency_ms = (time.perf_counter() - t0) * 1000
            return resp

    t_start = time.perf_counter()
    responses = await asyncio.gather(*(_call(p) for p in providers))
    valid = [r for r in responses if r.response and not r.error]

    if not valid:
        return ConsensusResult(
            answer="(all providers failed)",
            confidence=0.0,
            models=responses,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
        )

    # Semantic scoring — TODO v0.1: replace with embedding cosine similarity.
    # v0.0.1 uses simple length-normalized token overlap as placeholder.
    confidence, weights = _score_agreement(valid)

    # Pick highest-weighted response as the canonical answer.
    canonical = max(zip(valid, weights), key=lambda x: x[1])[0]

    # Apply weights back to each response (for transparency).
    for r, w in zip(valid, weights):
        r.weight = w

    return ConsensusResult(
        answer=canonical.response,
        confidence=confidence,
        models=responses,
        disagreements=_extract_disagreements(valid, weights),
        total_cost_usd=sum(r.cost_usd for r in responses),
        total_latency_ms=(time.perf_counter() - t_start) * 1000,
    )


def _score_agreement(responses: list[ModelResponse]) -> tuple[float, list[float]]:
    """Compute pairwise agreement and per-model weight.

    Placeholder: token-set overlap (Jaccard-ish). v0.1 replaces with embeddings.
    """
    if len(responses) == 1:
        return 1.0, [1.0]

    token_sets = [set(r.response.lower().split()) for r in responses]
    n = len(responses)
    sim_matrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i][j] = 1.0
            else:
                a, b = token_sets[i], token_sets[j]
                if not a or not b:
                    sim_matrix[i][j] = 0.0
                else:
                    sim_matrix[i][j] = len(a & b) / len(a | b)

    # Average similarity to others = per-model weight
    weights = [
        sum(sim_matrix[i][j] for j in range(n) if i != j) / (n - 1)
        for i in range(n)
    ]
    # Normalize
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]

    # Overall confidence = mean of upper triangle
    pairs = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    confidence = sum(pairs) / len(pairs) if pairs else 1.0

    return confidence, weights


def _extract_disagreements(
    responses: list[ModelResponse], weights: list[float]
) -> list[str]:
    """Surface points of low cross-model agreement.

    Placeholder: returns the names of low-weighted dissenting models.
    v0.1 will extract actual differing claims via embedding clustering.
    """
    threshold = 1.0 / (len(responses) * 1.5)  # below uniform weight = dissenter
    return [r.name for r, w in zip(responses, weights) if w < threshold]
