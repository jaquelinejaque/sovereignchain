# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# HSP (Hybrid Sovereign Protocol) attribution:
#   PCT/US26/11908 — the *self-evolution feedback loop* triggered from this
#   consensus engine (RLHF weights, Hebbian co-activation, MoE routing,
#   vector memory ingest) is patent-pending. The plain consensus call
#   (single round, no loops fired) is Apache 2.0 and unrestricted.
"""Consensus engine — semantic-agreement scoring + evolution loops.

This is the orchestration layer. The actual algorithms live in
``quorum.core.embeddings`` (semantic agreement) and ``quorum.evolution.*``
(router, RLHF, Hebbian, memory). We keep this file thin so the hot path is
auditable end-to-end in <300 lines.

Why each piece exists
---------------------
* **MoE router** picks WHICH providers to invoke. Fanning out to all 8 every
  time costs ~£0.02/call; a learned router cuts that ~3x.
* **Semantic agreement** scores the responses we *did* get — embedding cosine,
  not Jaccard, so paraphrases count as agreement.
* **RLHF weights** bias the canonical-answer pick toward models the user has
  historically up-voted on this query class.
* **Hebbian boost** rewards model *pairs* that historically vote together
  (compresses redundant fan-out over time).
* **Vector memory** ingests every (query, consensus_answer) pair so later
  calls get cross-session context injection.

All evolution imports are LAZY — a fresh clone with no env vars can still
``from quorum.core.consensus import consensus``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from quorum.providers.base import ModelResponse, Provider
from quorum.providers.registry import load_default_providers

logger = logging.getLogger(__name__)

# Hard caps to prevent adversarial prompt/response amplification:
# a single oversized input would otherwise fan out into N provider bills,
# 2N embeddings, and 1 permanent vector-memory write per consensus call.
MAX_PROMPT_BYTES = 32_000
MAX_RESPONSE_BYTES = 16_000


@dataclass
class ConsensusResult:
    """Result of a multi-model consensus query.

    Backwards-compatible with v0.0.1: the original fields keep their
    semantics and defaults; new fields default to neutral values so
    callers that ignore them see no behaviour change.
    """

    answer: str
    """Synthesized consensus answer (top-weighted response)."""

    confidence: float
    """Legacy agreement score 0..1 — kept for callers built against v0.0.1."""

    models: list[ModelResponse] = field(default_factory=list)
    """Per-model responses with latency, tokens, cost, and final weight."""

    disagreements: list[str] = field(default_factory=list)
    """Names of models that materially diverged from the rest."""

    evolution_signals: dict[str, bool] = field(default_factory=dict)
    """Which evolution loops fired during this query (for downstream learning)."""

    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    # --- v0.1 additions --------------------------------------------------
    embedding_confidence: float = 0.0
    """Semantic-agreement score from embedding cosine (replaces Jaccard)."""

    router_used: list[str] = field(default_factory=list)
    """Provider names the MoE router selected for this query."""

    hebbian_boost_applied: float = 1.0
    """Mean co-activation multiplier folded into model weights (1.0 = none)."""

    rlhf_weights_applied: dict[str, float] = field(default_factory=dict)
    """Per-model RLHF prior actually multiplied into the final weight."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "confidence": self.confidence,
            "embedding_confidence": self.embedding_confidence,
            "models": [m.to_dict() for m in self.models],
            "disagreements": self.disagreements,
            "evolution_signals": self.evolution_signals,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "router_used": self.router_used,
            "hebbian_boost_applied": round(self.hebbian_boost_applied, 4),
            "rlhf_weights_applied": {
                k: round(v, 4) for k, v in self.rlhf_weights_applied.items()
            },
        }


async def _route_providers(
    prompt: str,
    user_id: str | None,
    providers: Sequence[Provider],
    budget_usd: float,
) -> tuple[list[Provider], list[str], str]:
    """Ask the MoE router which providers to invoke; fall back to all.

    Returns ``(selected_providers, router_names, query_class)``. Lazy-imports
    so the router module being absent never breaks the consensus call.
    """
    try:
        from quorum.evolution.router import MoERouter, classify_query
    except Exception as e:  # noqa: BLE001
        logger.debug("MoERouter unavailable (%s); using all providers", e)
        return list(providers), [], "general"

    query_class = classify_query(prompt)
    if not user_id:
        # No user → can't apply RLHF; still let the router pick by global policy.
        user_id = "_anon_"
    try:
        router = MoERouter(
            candidate_models=[p.name for p in providers],
        )
        chosen_names = await router.route(
            prompt, user_id=user_id, budget_usd=budget_usd,
            available_models=[p.name for p in providers],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("router.route failed (%s); using all providers", e)
        return list(providers), [], query_class

    if not chosen_names:
        return list(providers), [], query_class
    by_name = {p.name: p for p in providers}
    selected = [by_name[n] for n in chosen_names if n in by_name]
    if len(selected) < 2:
        # Always keep at least 2 so we have something to compare.
        selected = list(providers)
    return selected, chosen_names, query_class


async def _score_semantic(
    valid: list[ModelResponse],
) -> tuple[float, list[float], list[tuple[int, int, float]]]:
    """Embed-and-score; fall back to the legacy Jaccard if embedder is absent."""
    try:
        from quorum.core.embeddings import (
            EmbeddingProvider,
            extract_disagreement_pairs,
            semantic_agreement,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("embeddings module unavailable (%s); falling back", e)
        return _jaccard_fallback(valid)

    try:
        embedder = EmbeddingProvider.from_env()
    except Exception as e:  # noqa: BLE001
        logger.warning("no embedding backend (%s); using Jaccard", e)
        return _jaccard_fallback(valid)

    try:
        # Truncate each response before embedding to bound cost/latency
        # against an adversarial provider that returns a megabyte of text.
        texts = [r.response[:MAX_RESPONSE_BYTES] for r in valid]
        confidence, weights = await semantic_agreement(texts, embedder)
        pairs = await extract_disagreement_pairs(texts, embedder)
    finally:
        await embedder.aclose()
    return confidence, weights, pairs


def _jaccard_fallback(
    valid: list[ModelResponse],
) -> tuple[float, list[float], list[tuple[int, int, float]]]:
    """Last-resort lexical scoring when embeddings are unreachable.

    Identical to the v0.0.1 logic. Worse than embeddings but never crashes.
    """
    if len(valid) == 1:
        return 1.0, [1.0], []
    sets = [set(r.response.lower().split()) for r in valid]
    n = len(valid)
    sims: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        sims[i][i] = 1.0
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            s = (len(a & b) / len(a | b)) if (a or b) else 0.0
            sims[i][j] = sims[j][i] = s
    weights = [sum(sims[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)]
    total = sum(weights) or 1.0
    weights = [w / total for w in weights]
    pairs_flat = [sims[i][j] for i in range(n) for j in range(i + 1, n)]
    confidence = sum(pairs_flat) / len(pairs_flat) if pairs_flat else 1.0
    disagree = [(i, j, sims[i][j]) for i in range(n) for j in range(i + 1, n) if sims[i][j] < 0.5]
    disagree.sort(key=lambda t: t[2])
    return confidence, weights, disagree


async def _apply_rlhf_and_hebbian(
    valid: list[ModelResponse],
    base_weights: list[float],
    user_id: str | None,
    query_class: str,
    confidence: float,
) -> tuple[list[float], dict[str, float], float]:
    """Multiply RLHF prior + Hebbian pair-boost into each model's weight.

    Returns ``(new_weights, rlhf_applied, mean_hebbian_boost)``. Any sub-loop
    that errors degrades to identity (no change to weights). Records the
    round in HebbianMatrix as a side effect so co-activation learns online.
    """
    rlhf_map: dict[str, float] = {}
    if user_id:
        try:
            from quorum.evolution.rlhf import RLHFTracker
            tracker = RLHFTracker()
            rlhf_map = await tracker.get_weights(user_id, query_class)
        except Exception as e:  # noqa: BLE001
            logger.debug("RLHF read failed (%s); skipping", e)

    # Hebbian: query pair boosts and update the matrix from this round.
    mean_boost = 1.0
    try:
        from quorum.evolution.hebbian import HebbianMatrix
        matrix = HebbianMatrix()
        names = [r.name for r in valid]
        pair_boosts: dict[tuple[str, str], float] = {}
        pair_sims: dict[tuple[str, str], float] = {}
        boost_vals: list[float] = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                b = await matrix.get_pair_boost(names[i], names[j])
                pair_boosts[(names[i], names[j])] = b
                # Re-use the embedding confidence as a per-pair similarity
                # proxy (cheap; the per-pair matrix already lives inside
                # extract_disagreement_pairs but we don't carry it here).
                pair_sims[(names[i], names[j])] = confidence
                boost_vals.append(b)
        if boost_vals:
            mean_boost = sum(boost_vals) / len(boost_vals)

        # Online learning: record the round so future calls benefit.
        await matrix.record_round(valid, pair_sims, reward=confidence)
    except Exception as e:  # noqa: BLE001
        logger.debug("Hebbian path skipped (%s)", e)
        pair_boosts = {}

    # Apply multipliers, then renormalise to a probability distribution.
    new: list[float] = []
    for i, r in enumerate(valid):
        rlhf_mult = rlhf_map.get(r.name, 1.0 / max(len(valid), 1))
        # Hebbian: per-model contribution = mean of its pair boosts.
        my_boosts = [
            v for (a, b), v in pair_boosts.items() if a == r.name or b == r.name
        ]
        heb_mult = sum(my_boosts) / len(my_boosts) if my_boosts else 1.0
        new.append(base_weights[i] * max(rlhf_mult, 1e-6) * heb_mult)

    total = sum(new) or 1.0
    new = [w / total for w in new]
    return new, rlhf_map, mean_boost


async def _ingest_memory(
    user_id: str | None,
    prompt: str,
    answer: str,
) -> bool:
    """Persist (query, answer) into per-user VectorMemory. Best-effort."""
    if not user_id:
        return False
    try:
        from quorum.core.embeddings import EmbeddingProvider
        from quorum.evolution.memory_loop import MemoryEvolution
        mem = MemoryEvolution()
        embedder = EmbeddingProvider.from_env()
        try:
            # Truncate before permanent storage so a single oversized round
            # cannot bloat the per-user VectorMemory.
            await mem.ingest(
                user_id,
                prompt[:MAX_PROMPT_BYTES],
                answer[:MAX_RESPONSE_BYTES],
                embedder,
            )
        finally:
            await embedder.aclose()
        return True
    except Exception as e:  # noqa: BLE001
        logger.debug("memory ingest skipped (%s)", e)
        return False


async def _maybe_ingest_synthetic(
    prompt: str,
    result: "ConsensusResult",
    user_id: str | None,
    opt_in: bool,
) -> None:
    """Fire-and-forget ingest into the synthetic training corpus.

    Kept off the response path: callers see the consensus result the
    moment it is synthesized; the corpus append happens in the background
    via ``asyncio.create_task`` (or skipped entirely on import failure).
    """
    if not opt_in:
        return
    try:
        from quorum.evolution.synthetic_data import SyntheticDatasetStore
        store = SyntheticDatasetStore()
        await store.maybe_ingest(prompt, result, user_id=user_id, opt_in=True)
    except Exception as e:  # noqa: BLE001
        logger.debug("synthetic ingest skipped (%s)", e)


async def consensus(
    prompt: str,
    *,
    providers: Sequence[Provider] | None = None,
    max_concurrency: int = 8,
    timeout_s: float = 30.0,
    user_id: str | None = None,
    budget_usd: float = 0.05,
    route: bool = True,
    opt_in_synthetic: bool = False,
) -> ConsensusResult:
    """Run N LLMs in parallel and synthesize a consensus answer.

    Backwards-compatible: if ``providers`` is None we load from env, exactly
    as v0.0.1 did. New evolution behaviour activates only when a ``user_id``
    is supplied (per-user RLHF + memory) or globally (Hebbian, router).

    Args:
        prompt: The user query.
        providers: Pre-configured Provider instances (None → env autodiscovery).
        max_concurrency: Cap on parallel provider calls.
        timeout_s: Per-provider wall-clock timeout.
        user_id: Stable id for RLHF/memory scoping. Omit for anonymous calls.
        budget_usd: Hard cap the router uses to drop expensive models.

    Size limits (anti-abuse):
        ``prompt`` is rejected with ``ValueError`` if it exceeds
        ``MAX_PROMPT_BYTES`` (32 000 chars). Each provider response is
        truncated to ``MAX_RESPONSE_BYTES`` (16 000 chars) before embedding
        and before vector-memory ingest, to bound fan-out cost and storage.

    Returns:
        ``ConsensusResult`` with the synthesized answer plus full audit trail.
    """
    if len(prompt) > MAX_PROMPT_BYTES:
        raise ValueError(
            f"prompt too large: {len(prompt)} chars exceeds "
            f"MAX_PROMPT_BYTES={MAX_PROMPT_BYTES}"
        )
    if providers is None:
        providers = load_default_providers()
    if not providers:
        raise RuntimeError(
            "No providers configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "GOOGLE_AI_STUDIO_KEY, REPLICATE_API_TOKEN, or run Ollama locally."
        )

    if route:
        selected, router_names, query_class = await _route_providers(
            prompt, user_id, providers, budget_usd
        )
    else:
        selected, router_names, query_class = list(providers), [], "general"

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
                )
            except Exception as e:  # noqa: BLE001
                return ModelResponse(
                    name=p.name, response="", error=str(e)[:200],
                    latency_ms=(time.perf_counter() - t0) * 1000,
                )
            resp.latency_ms = (time.perf_counter() - t0) * 1000
            return resp

    t_start = time.perf_counter()
    responses = await asyncio.gather(*(_call(p) for p in selected))
    valid = [r for r in responses if r.response and not r.error]

    if not valid:
        return ConsensusResult(
            answer="(all providers failed)",
            confidence=0.0,
            models=responses,
            router_used=router_names,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
        )

    confidence, base_weights, disagree_pairs = await _score_semantic(valid)
    new_weights, rlhf_applied, hebbian_mean = await _apply_rlhf_and_hebbian(
        valid, base_weights, user_id, query_class, confidence
    )

    for r, w in zip(valid, new_weights):
        r.weight = w
    canonical = max(zip(valid, new_weights), key=lambda x: x[1])[0]

    disagreements = [valid[i].name for i, _, _ in disagree_pairs] + [
        valid[j].name for _, j, _ in disagree_pairs
    ]
    # Dedup preserving order.
    seen: set[str] = set()
    disagreements = [n for n in disagreements if not (n in seen or seen.add(n))]

    memory_fired = await _ingest_memory(user_id, prompt, canonical.response)

    result = ConsensusResult(
        answer=canonical.response,
        confidence=confidence,
        embedding_confidence=confidence,
        models=responses,
        disagreements=disagreements,
        evolution_signals={
            "router": bool(router_names),
            "rlhf": bool(rlhf_applied),
            "hebbian": hebbian_mean != 1.0,
            "memory": memory_fired,
        },
        total_cost_usd=sum(r.cost_usd for r in responses),
        total_latency_ms=(time.perf_counter() - t_start) * 1000,
        router_used=router_names,
        hebbian_boost_applied=hebbian_mean,
        rlhf_weights_applied=rlhf_applied,
    )

    # Synthetic-data ingest (opt-in, fire-and-forget so the response is not
    # blocked by a JSONL disk write). Default is opt_in=False for privacy —
    # the caller has to explicitly request it per-query.
    if opt_in_synthetic:
        try:
            asyncio.create_task(
                _maybe_ingest_synthetic(prompt, result, user_id, True)
            )
        except RuntimeError:
            # No running event loop (extremely rare here, but defensive).
            await _maybe_ingest_synthetic(prompt, result, user_id, True)

    return result


async def consensus_ab(
    prompt_a: str,
    prompt_b: str,
    *,
    providers: Sequence[Provider] | None = None,
    max_concurrency: int = 8,
    timeout_s: float = 30.0,
    user_id: str | None = None,
    budget_usd: float = 0.05,
    route: bool = True,
    prompt_template_id: str | None = None,
    query_id: str | None = None,
    store: Any | None = None,
) -> tuple[ConsensusResult, ConsensusResult, str]:
    """Fan out two prompt variants through ``consensus()`` and record the A/B.

    Runs both prompts concurrently (the whole point of an A/B is *parallel*
    evaluation, not sequential — otherwise the second run pollutes its own
    rlhf/memory state with the first), then persists the experiment via
    ``ABTestStore`` so a later /v1/ab/feedback call can attach a winner.

    Returns ``(result_a, result_b, experiment_id)``. The store import is
    LAZY (mirroring the other evolution-loop imports in this module) so a
    fresh clone with no ``QUORUM_DATA_DIR`` still imports cleanly.

    Args:
        prompt_a: First candidate prompt.
        prompt_b: Second candidate prompt.
        prompt_template_id: Optional group key — ``get_active_winner`` ranks
            arms within a template, so two unrelated A/B's never bleed
            into each other's win rate.
        query_id: Optional upstream query id (e.g. from the server's
            /v1/consensus assignment) for cross-referencing in audit logs.
        store: Pre-built ``ABTestStore`` for tests; constructed lazily when
            absent so production callers don't have to thread it through.
    """
    if not prompt_a or not prompt_b:
        raise ValueError("prompt_a and prompt_b must be non-empty.")

    call_kwargs: dict[str, Any] = {
        "providers": providers,
        "max_concurrency": max_concurrency,
        "timeout_s": timeout_s,
        "user_id": user_id,
        "budget_usd": budget_usd,
        "route": route,
    }
    result_a, result_b = await asyncio.gather(
        consensus(prompt_a, **call_kwargs),
        consensus(prompt_b, **call_kwargs),
    )

    if store is None:
        try:
            from quorum.evolution.ab_testing import ABTestStore
            store = ABTestStore()
        except Exception as e:  # noqa: BLE001
            logger.warning("ABTestStore unavailable (%s); A/B not recorded", e)
            return result_a, result_b, ""

    try:
        experiment_id = await store.record_experiment(
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            result_a=result_a,
            result_b=result_b,
            prompt_template_id=prompt_template_id,
            query_id=query_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ab_store record_experiment failed (%s)", e)
        experiment_id = ""

    return result_a, result_b, experiment_id


__all__ = ["ConsensusResult", "consensus", "consensus_ab"]
