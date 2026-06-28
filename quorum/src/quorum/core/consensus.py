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
import os
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

    scoring_method: str = "embedding"
    """Which agreement scorer produced ``confidence``: 'embedding' = cosine
    over semantic vectors (cross-vocabulary, default); 'jaccard' = lexical
    token overlap fallback used only when no embedder is reachable. Callers
    that gate on semantic agreement MUST check this — Jaccard scores are
    NOT comparable across vocabularies and will mark unrelated paraphrases
    as low-agreement."""

    router_used: list[str] = field(default_factory=list)
    """Provider names the MoE router selected for this query."""

    hebbian_boost_applied: float = 1.0
    """Mean co-activation multiplier folded into model weights (1.0 = none)."""

    rlhf_weights_applied: dict[str, float] = field(default_factory=dict)
    """Per-model RLHF prior actually multiplied into the final weight."""

    # --- v0.2.5 additions ------------------------------------------------
    hallucination_risk: dict = field(default_factory=dict)
    """Convergent-hallucination assessment from
    :mod:`quorum.core.hallucination_risk`. Empty dict when no flags fired.
    When populated, holds ``risk_level`` (``low|elevated|high``),
    ``suggested_penalty`` (already applied to ``confidence`` if non-zero),
    and ``flags`` (list of category/evidence/detail records).

    Why surface it: a 78% consensus on a regulated UK domain where six
    sub-models agreed in a shared fictional world produced 5/5 fabricated
    facts on 2026-06-28. Hiding the risk silently and only downgrading
    the score would let callers that *don't* gate on the score still
    repeat the hallucination."""

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
            "scoring_method": self.scoring_method,
            "hallucination_risk": self.hallucination_risk,
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
) -> tuple[float, list[float], list[tuple[int, int, float]], str]:
    """Embed-and-score; fall back to the legacy Jaccard if embedder is absent.

    Returns ``(confidence, weights, pairs, scoring_method)`` where
    ``scoring_method`` is ``"embedding"`` on the happy path or ``"jaccard"``
    when any fallback triggers. Callers MUST propagate the method into
    ``ConsensusResult.scoring_method`` so downstream consumers can detect
    degraded scoring and react (e.g. raise an alarm, refuse to gate on the
    score, retry later).
    """
    try:
        from quorum.core.embeddings import (
            EmbeddingProvider,
            extract_disagreement_pairs,
            semantic_agreement,
        )
    except Exception as e:  # noqa: BLE001
        logger.error(
            "DEGRADED SCORING: embeddings module unavailable (%s); "
            "using Jaccard lexical fallback — agreement score is NOT "
            "semantic and should not be compared to embedding scores",
            e,
        )
        return _jaccard_fallback(valid)

    try:
        embedder = EmbeddingProvider.from_env()
    except Exception as e:  # noqa: BLE001
        logger.error(
            "DEGRADED SCORING: no embedding backend (%s); using Jaccard "
            "lexical fallback — set GOOGLE_API_KEY / OPENAI_API_KEY / "
            "OLLAMA_HOST to restore semantic scoring",
            e,
        )
        return _jaccard_fallback(valid)

    try:
        # Truncate each response before embedding to bound cost/latency
        # against an adversarial provider that returns a megabyte of text.
        texts = [r.response[:MAX_RESPONSE_BYTES] for r in valid]
        try:
            confidence, weights = await semantic_agreement(texts, embedder)
            pairs = await extract_disagreement_pairs(texts, embedder)
        except Exception as e:  # noqa: BLE001
            # Embedder backends can fail mid-call (HTTP 429 quota, 5xx, network).
            # Without this guard the entire consensus crashes; fall back to
            # Jaccard so the caller still gets a usable answer.
            logger.error(
                "DEGRADED SCORING: embedding call failed mid-flight (%s); "
                "using Jaccard lexical fallback for this query — score is "
                "NOT semantic, check embedder quota / health",
                e,
            )
            return _jaccard_fallback(valid)
    finally:
        # Release the embedder's httpx client. Without this, sustained
        # parallel calls (e.g. sell-quorum's 5 concurrent drafts) leak
        # one AsyncClient per call until the process hits EMFILE.
        await embedder.aclose()
    return confidence, weights, pairs, "embedding"


def _jaccard_fallback(
    valid: list[ModelResponse],
) -> tuple[float, list[float], list[tuple[int, int, float]], str]:
    """Last-resort lexical scoring when embeddings are unreachable.

    Returns the same shape as :func:`_score_semantic` with
    ``scoring_method="jaccard"`` so callers can distinguish a semantic
    score (cross-vocabulary, [0,1] cosine) from a lexical one (token
    overlap, fails on paraphrases). Identical to the v0.0.1 logic
    otherwise — worse than embeddings but never crashes.
    """
    if len(valid) == 1:
        return 1.0, [1.0], [], "jaccard"
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
    return confidence, weights, disagree, "jaccard"


async def _apply_rlhf_and_hebbian(
    valid: list[ModelResponse],
    base_weights: list[float],
    user_id: str | None,
    query_class: str,
    confidence: float,
    enabled_loops: set[str] | None = None,
) -> tuple[list[float], dict[str, float], float]:
    """Multiply RLHF prior + Hebbian pair-boost into each model's weight.

    Returns ``(new_weights, rlhf_applied, mean_hebbian_boost)``. Any sub-loop
    that errors degrades to identity (no change to weights). Records the
    round in HebbianMatrix as a side effect so co-activation learns online.

    ``enabled_loops`` is the meta-learner enforcement set; if a loop name
    is absent, that sub-section becomes a no-op (its multiplier stays at
    identity). ``None`` means "no enforcement — run everything" so existing
    callers and tests are unaffected. The adversarial-probe path is
    STRUCTURAL and always runs regardless of enforcement.
    """
    # Default: no enforcement — preserve historical behaviour for callers
    # that haven't been migrated yet (e.g. unit tests calling this helper
    # directly).
    if enabled_loops is None:
        enabled_loops = {"rlhf", "hebbian"}

    rlhf_map: dict[str, float] = {}
    if user_id and "rlhf" in enabled_loops:
        try:
            from quorum.evolution.rlhf import RLHFTracker
            tracker = RLHFTracker()
            rlhf_map = await tracker.get_weights(user_id, query_class)
        except Exception as e:  # noqa: BLE001
            logger.debug("RLHF read failed (%s); skipping", e)
    elif user_id:
        logger.info("meta: skipped rlhf for class=%s", query_class)

    # Hebbian: query pair boosts and update the matrix from this round.
    # We run *both* engines:
    #   * HebbianMatrix  — legacy global running-mean, used by the router.
    #   * HebbianStore   — per-(pair, query_class) EMA, multiplies the boost
    #                       once it has SAMPLE_THRESHOLD observations.
    # If either errors we degrade gracefully (boost = 1.0); the consensus
    # call is never broken by a Hebbian outage.
    mean_boost = 1.0
    pair_boosts: dict[tuple[str, str], float] = {}
    if "hebbian" in enabled_loops:
        try:
            from quorum.evolution.hebbian import HebbianMatrix
            matrix = HebbianMatrix()
            names = [r.name for r in valid]
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
            logger.debug("Hebbian matrix path skipped (%s)", e)

        # HebbianStore (per-class EMA) — multiplicatively layered on top of
        # HebbianMatrix.  observe() always runs (so the EMA learns from this
        # round) but boost() returns 1.0 until SAMPLE_THRESHOLD is reached, so
        # cold-start is safe.
        try:
            from quorum.evolution.hebbian import _get_default_store
            store = await _get_default_store()
            names = [r.name for r in valid]
            store_vals: list[float] = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sb = await store.boost(names[i], names[j], query_class)
                    key = (names[i], names[j])
                    # Compose multiplicatively with the existing matrix boost.
                    pair_boosts[key] = pair_boosts.get(key, 1.0) * sb
                    store_vals.append(sb)
            if store_vals:
                store_mean = sum(store_vals) / len(store_vals)
                mean_boost = mean_boost * store_mean
            # Observe AFTER reading boost so the value we just applied was a
            # genuine prior, not contaminated by the round we're scoring.
            await store.observe(query_class, valid)
        except Exception as e:  # noqa: BLE001
            logger.debug("HebbianStore path skipped (%s)", e)
    else:
        logger.info("meta: skipped hebbian for class=%s", query_class)

    # Adversarial-probe penalty: drop the weight of models that have
    # demonstrably fallen for known attacks recently. Best-effort, fully
    # additive (default 1.0 if the loop is unavailable / never run) so
    # callers without an adversarial DB see identical behaviour.
    adv_penalties: dict[str, float] = {}
    try:
        from quorum.evolution.adversarial import AdversarialProbe
        probe = AdversarialProbe()
        for r in valid:
            adv_penalties[r.name] = await probe.penalty_multiplier(
                r.name, query_class
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("adversarial penalty path skipped (%s)", e)

    # Apply multipliers, then renormalise to a probability distribution.
    new: list[float] = []
    for i, r in enumerate(valid):
        rlhf_mult = rlhf_map.get(r.name, 1.0 / max(len(valid), 1))
        # Hebbian: per-model contribution = mean of its pair boosts.
        my_boosts = [
            v for (a, b), v in pair_boosts.items() if a == r.name or b == r.name
        ]
        heb_mult = sum(my_boosts) / len(my_boosts) if my_boosts else 1.0
        adv_mult = adv_penalties.get(r.name, 1.0)
        new.append(
            base_weights[i]
            * max(rlhf_mult, 1e-6)
            * heb_mult
            * adv_mult
        )

    total = sum(new) or 1.0
    new = [w / total for w in new]
    return new, rlhf_map, mean_boost


async def _record_competition(
    valid: list[ModelResponse],
    canonical_answer: str,
    query_class: str,
) -> bool:
    """Feed the natural pairwise battles into the ELO leaderboard.

    Best-effort: any failure here MUST NOT block the consensus answer
    returning. The store derives (winner, loser) pairs from each query's
    own responses against the canonical answer — zero extra LLM calls,
    pure post-processing on what we already paid for above.
    """
    if not canonical_answer or len(valid) < 2:
        return False
    try:
        from quorum.evolution.competition import CompetitionStore
        store = CompetitionStore()
        applied = await store.record_query(query_class, valid, canonical_answer)
        return applied > 0
    except Exception as e:  # noqa: BLE001
        logger.debug("competition store record_query skipped (%s)", e)
        return False


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
    timeout_s: float = 60.0,
    user_id: str | None = None,
    budget_usd: float = 0.05,
    route: bool = True,
    opt_in_synthetic: bool = False,
    enable_self_prompt: bool = True,
    self_prompt_threshold: float | None = None,
    self_prompt_rewriter: Any | None = None,
    images: list[str] | None = None,
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

    if user_id:
        embedder = None
        try:
            from quorum.core.embeddings import EmbeddingProvider
            from quorum.evolution.memory_loop import MemoryEvolution
            mem = MemoryEvolution()
            embedder = EmbeddingProvider.from_env()
            retrieved_context = await mem.retrieve_context(user_id, prompt, embedder)
            if retrieved_context:
                prompt = f"Previous Context:\n{retrieved_context}\n\nCurrent Request:\n{prompt}"
        except Exception as e:
            logger.debug("Memory retrieval skipped (%s)", e)
        finally:
            # Mirror the ingest path (line ~443): release the embedder's
            # httpx client so memory retrieval doesn't leak FDs per query.
            if embedder is not None:
                try:
                    await embedder.aclose()
                except Exception:
                    pass

    if route:
        selected, router_names, query_class = await _route_providers(
            prompt, user_id, providers, budget_usd
        )
    else:
        selected, router_names, query_class = list(providers), [], "general"

    # Loop 6 — Meta-learning ENFORCEMENT (was logged-only — BUG #2 fix).
    # ``recommend_loops_async`` returns the subset of OPTIONAL loops that
    # the learner currently judges worth firing for this ``query_class``.
    # ``router`` is structural and never gated. ``rlhf``, ``hebbian``,
    # ``memory`` and ``self_prompt`` are opt-out based on learned per-class
    # effectiveness. Cold-start safe: empty/missing history defaults to
    # all loops on; any exception falls back to all loops on.
    #
    # NOTE on ``memory``: the *retrieve* path (lines ~468-478) runs BEFORE
    # ``query_class`` is known, so it cannot be gated here. Only the
    # *ingest* path (post-consensus) is enforced. This is intentional —
    # retrieval is cheap and helps cold-class bootstrap.
    _OPTIONAL_LOOPS = ("rlhf", "hebbian", "memory", "self_prompt")
    _meta_enabled: set[str] = set(_OPTIONAL_LOOPS)  # cold-start default: all on
    try:
        from quorum.evolution.meta import MetaLearner

        recommended = await MetaLearner().recommend_loops_async(
            query_class,
            candidate_loops=list(_OPTIONAL_LOOPS),
        )
        # Only narrow if the learner returned something non-empty — an
        # empty list means "no opinion", not "disable everything".
        if recommended:
            _meta_enabled = set(recommended) & set(_OPTIONAL_LOOPS)
        logger.info(  # promoted from .debug so enforcement is visible
            "meta.enforce class=%s enabled=%s skipped=%s",
            query_class,
            sorted(_meta_enabled),
            sorted(set(_OPTIONAL_LOOPS) - _meta_enabled),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(
            "meta.enforce skipped (%s); defaulting to all loops on", e
        )

    semaphore = asyncio.Semaphore(max_concurrency)

    # Camada 1 — SelfPromptOptimizer wiring (BEFORE fan-out).
    # For each provider in the selected ensemble, look up the current
    # bandit-champion template via `get_current_prompt(model_name=p.name)`
    # and inject it as the `system_prompt` of that provider's call. This
    # is the "system prompt evolves per-model" path; the retry loop below
    # (which uses `champion.prompt_template` as a USER-prompt rewrite) is
    # the orthogonal "rewrite user prompt on disagreement" path.
    #
    # Failure mode: any error during lookup must NOT break the consensus
    # call. We fall back to no system prompt for the affected provider.
    # Cost: one cheap sqlite read per provider, off the event loop via
    # `asyncio.to_thread` inside `get_current_prompt`.
    system_prompts: dict[str, str] = {}
    if enable_self_prompt and "self_prompt" in _meta_enabled:
        try:
            from quorum.evolution.self_prompt import SelfPromptOptimizer
            _optimizer = SelfPromptOptimizer()
            for _p in selected:
                try:
                    tpl = await _optimizer.get_current_prompt(_p.name)
                    if tpl:
                        system_prompts[_p.name] = tpl
                except Exception as _e:  # noqa: BLE001
                    logger.debug(
                        "self_prompt.layer1: get_current_prompt failed "
                        "for %s (%s); no system prompt injected", _p.name, _e,
                    )
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "self_prompt.layer1: SelfPromptOptimizer init failed (%s); "
                "fan-out will run without per-provider system prompts", e,
            )

    # Camada 1.5 — Quorum persona layer (opt-in via QUORUM_PERSONA=1).
    # Prepends a persona system prompt to every sub-model call so that no
    # sub-model introduces itself by its own name (Qwen/Claude/GPT/etc) —
    # Quorum is the responder, sub-models are sub-processes. The hard-coded
    # honesty clause (in quorum.core.identity.HONESTY_CLAUSE) prevents
    # consciousness/sentience claims regardless of YAML edits.
    if os.environ.get("QUORUM_PERSONA", "").lower() in ("1", "true", "yes"):
        try:
            from quorum.core.identity import sub_model_system_prompt
            _persona = sub_model_system_prompt()
            for _p in selected:
                existing = system_prompts.get(_p.name, "")
                system_prompts[_p.name] = (
                    _persona + "\n\n" + existing if existing else _persona
                )
            logger.info("quorum.persona: enabled, injected into %d sub-models", len(selected))
        except Exception as e:  # noqa: BLE001
            logger.warning("quorum.persona: failed to load (%s); fan-out runs raw", e)

    async def _call(p: Provider) -> ModelResponse:
        async with semaphore:
            t0 = time.perf_counter()
            try:
                call_kwargs: dict[str, Any] = {}
                if images:
                    call_kwargs["images"] = images
                sp = system_prompts.get(p.name)
                if sp:
                    call_kwargs["system_prompt"] = sp
                resp = await asyncio.wait_for(
                    p.complete(prompt, **call_kwargs), timeout=timeout_s
                )
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
            if not resp.response.strip() and not resp.error:
                resp.error = "Empty response (Missing key or rate limit?)"
                logger.warning("Provider %s returned empty response without error (latency: %.1fms). Missing key or rate limited?", p.name, resp.latency_ms)
            return resp

    t_start = time.perf_counter()
    responses = await asyncio.gather(*(_call(p) for p in selected))
    valid = [r for r in responses if r.response and not r.error]

    # Refusal filter — exclude models that refused to answer from the consensus
    # scoring. A refusal ("I can't help with that") is NOT data; it inflates the
    # apparent agreement (all refusals look alike) and biases the canonical pick.
    # The remaining models — those that chose to answer of their own accord —
    # form the genuine consensus. This is NOT a jailbreak: each model retains
    # its own safety floor; Quorum simply routes around refusers.
    if valid:
        try:
            from quorum.core.refusal_filter import partition_refusals
            answered, refused = partition_refusals(valid)
            if refused:
                logger.info(
                    "refusal_filter: excluded %d refuser(s) from consensus: %s",
                    len(refused),
                    [r.name for r, _ in refused],
                )
                # If everyone refused, keep `valid` as-is so we still produce
                # a transparent answer (even if it's just the refusal itself).
                # Otherwise, narrow `valid` to those who genuinely answered.
                if answered:
                    valid = answered
        except Exception as _e:  # noqa: BLE001
            logger.debug("refusal_filter skipped (%s); using all valid responses", _e)

    if not valid:
        return ConsensusResult(
            answer="(all providers failed)",
            confidence=0.0,
            models=responses,
            router_used=router_names,
            total_latency_ms=(time.perf_counter() - t_start) * 1000,
        )

    confidence, base_weights, disagree_pairs, scoring_method = await _score_semantic(valid)

    # Loop 11 — Self-prompting retry. If the first pass came back with
    # low semantic agreement we ask the strongest available model to
    # clarify+decompose the original query, then re-run a single round
    # of providers on the rewritten prompt. Quorum-validated design:
    # at most one rewrite by default (cost vs quality), append-as-context
    # rather than replace, and never overwrite the original responses
    # — we replace `valid` only if the new pass returned anything usable.
    self_prompt_fired = False
    if enable_self_prompt and "self_prompt" not in _meta_enabled:
        logger.info("meta: skipped self_prompt for class=%s", query_class)
    if enable_self_prompt and "self_prompt" in _meta_enabled:
        try:
            from quorum.evolution.self_prompt import (
                DEFAULT_REWRITE_CONFIDENCE_THRESHOLD,
                PromptRewriter,
                SelfPromptOptimizer,
            )
            threshold = (
                self_prompt_threshold
                if self_prompt_threshold is not None
                else DEFAULT_REWRITE_CONFIDENCE_THRESHOLD
            )
            if confidence < threshold:
                # BUG #1 FIX — Hierarchical fallback:
                # Camada 1) SelfPromptOptimizer (bandit Bayesiano persistente):
                #   se existir variant ATIVA pra `query_class` (cold-start
                #   detectado via list_variants — get_current_prompt sozinho
                #   auto-seeda e mascararia o cold-start), usa o template
                #   campeão como rewrite. Memória cross-query, evolução real.
                # Camada 2) PromptRewriter (clarification reativa, original):
                #   fallback se optimizer não tem variants OU se qualquer
                #   coisa falhar. Caminho seguro/testado preservado.
                # Camada 3) record_outcome após o retry — fecha o loop bandit.
                rewritten_prompt = None
                optimizer_variant_id: str | None = None
                try:
                    optimizer = SelfPromptOptimizer()
                    existing_variants = await optimizer.list_variants(
                        query_class
                    )
                    if existing_variants:
                        champion = existing_variants[0]
                        rewritten_prompt = champion.prompt_template
                        optimizer_variant_id = champion.id
                        logger.debug(
                            "self_prompt: using optimizer champion "
                            "model=%s variant_id=%s avg_reward=%.3f",
                            query_class, champion.id, champion.avg_reward,
                        )
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "self_prompt: optimizer path skipped (%s); "
                        "falling back to PromptRewriter", e,
                    )
                    rewritten_prompt = None
                    optimizer_variant_id = None

                rewriter = self_prompt_rewriter or PromptRewriter(
                    confidence_threshold=threshold
                )
                if rewritten_prompt is None:
                    rewritten_prompt = await rewriter.rewrite(
                        prompt, confidence, query_class
                    )
                if rewritten_prompt:
                    # Re-run the same selected providers on the rewritten
                    # prompt. We do NOT re-route — the same ensemble that
                    # produced the original disagreement is the fair
                    # apples-to-apples comparison.
                    retry_responses = await asyncio.gather(
                        *(_call(p) for p in selected)
                    )
                    retry_valid = [
                        r for r in retry_responses if r.response and not r.error
                    ]
                    if retry_valid:
                        new_confidence, new_base_weights, new_disagree_pairs, new_scoring_method = (
                            await _score_semantic(retry_valid)
                        )
                        # Persist the delta unconditionally so the meta-
                        # learner sees BOTH wins and losses (negative
                        # deltas tell it to disable rewriting for that
                        # query class).
                        try:
                            await rewriter.log_rewrite(
                                original=prompt,
                                rewritten=rewritten_prompt,
                                original_confidence=confidence,
                                new_confidence=new_confidence,
                                query_class=query_class,
                                rewriter_name=getattr(
                                    rewriter.rewriter_provider, "name", ""
                                ) or "auto",
                            )
                        except Exception as e:  # noqa: BLE001
                            logger.debug(
                                "self_prompt: log_rewrite skipped (%s)", e
                            )
                        # Camada 3 — close the bandit loop. Only if the
                        # rewrite came from the optimizer (not from the
                        # PromptRewriter fallback). Reward = delta de
                        # confidence; positivo aumenta avg_reward do
                        # champion, negativo demove-o pra próximo retry.
                        if optimizer_variant_id is not None:
                            try:
                                reward = float(new_confidence - confidence)
                                await optimizer.record_outcome(
                                    optimizer_variant_id, reward=reward
                                )
                            except Exception as e:  # noqa: BLE001
                                logger.debug(
                                    "self_prompt: record_outcome skipped "
                                    "(%s)", e,
                                )
                        # Only adopt the rewrite if it actually improved
                        # confidence — otherwise the original ensemble
                        # response is the safer bet.
                        if new_confidence > confidence:
                            responses = retry_responses
                            valid = retry_valid
                            confidence = new_confidence
                            base_weights = new_base_weights
                            disagree_pairs = new_disagree_pairs
                            scoring_method = new_scoring_method
                            self_prompt_fired = True
        except Exception as e:  # noqa: BLE001
            logger.debug("self_prompt loop skipped (%s)", e)

    new_weights, rlhf_applied, hebbian_mean = await _apply_rlhf_and_hebbian(
        valid,
        base_weights,
        user_id,
        query_class,
        confidence,
        enabled_loops=_meta_enabled,
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

    if "memory" in _meta_enabled:
        memory_fired = await _ingest_memory(user_id, prompt, canonical.response)
    else:
        logger.info("meta: skipped memory for class=%s", query_class)
        memory_fired = False
    competition_fired = await _record_competition(
        valid, canonical.response, query_class
    )

    evolution_signals = {
        "router": bool(router_names),
        "rlhf": bool(rlhf_applied),
        "hebbian": hebbian_mean != 1.0,
        "memory": memory_fired,
        "competition": competition_fired,
        "self_prompt": self_prompt_fired,
    }

    # Loop 6 — Meta-learning: observe which loops fired and the resulting
    # final confidence so future calls can disable consistently-unhelpful
    # loops per query class. Best-effort: never let a logging failure
    # poison the response path.
    try:
        from quorum.evolution.meta import MetaLearner

        await MetaLearner().observe_async(
            query_class=query_class,
            loops_fired=evolution_signals,
            final_confidence=confidence,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("meta.observe skipped (%s)", e)

    # SelfPromptOptimizer always-on bandit signal capture (SIMPLE path).
    # Camada 3 above only fires inside the self-prompt retry block (i.e.
    # only when confidence < threshold AND the optimizer's champion was
    # actually used as the rewrite). That means the bandit never sees the
    # common case — high-confidence calls where no retry was needed — so
    # it has no way to learn that the current champion template is
    # *staying* good. This block closes that hole: on EVERY consensus
    # call, ask the optimizer for the active champion of this query_class
    # (auto-seeds the row on first read, so cold-start is safe) and feed
    # the final consensus confidence in as the reward signal. No per-
    # provider variant_id threading, no ModelResponse mutation, no extra
    # LLM call (get_current_prompt is a single SQLite read). When the
    # self-prompt retry already recorded an outcome on the same variant
    # the two signals stack via the incremental-mean update in
    # record_outcome — that is the desired behaviour (more samples =
    # tighter posterior). Best-effort: any failure here MUST NOT block
    # the response path.
    #
    # Meta-enforcement: gated on "self_prompt" alongside the other two
    # self-prompt call sites (system-prompt injection and rewriter retry).
    # If the meta-learner decided this query_class doesn't benefit from
    # the self-prompt loop, we skip the bandit signal capture too —
    # otherwise we'd be feeding rewards into a champion the consensus
    # call never actually used, polluting the bandit's posterior.
    if "self_prompt" in _meta_enabled:
        try:
            from quorum.evolution.self_prompt import SelfPromptOptimizer

            _opt = SelfPromptOptimizer()
            _variants = await _opt.list_variants(query_class)
            if _variants:
                # list_variants is sorted champion-first (avg_reward DESC,
                # samples DESC) so [0] is the same variant get_current_prompt
                # would have returned — no extra DB round-trip needed.
                await _opt.record_outcome(_variants[0].id, reward=float(confidence))
            else:
                # No row yet for this class — seed via get_current_prompt so
                # the NEXT call's list_variants finds something to score.
                # We don't record on the freshly seeded row because we have
                # no evidence it caused this round's confidence (the round
                # ran on the user's literal prompt, not on any template).
                await _opt.get_current_prompt(query_class)
        except Exception as e:  # noqa: BLE001
            logger.debug("self_prompt.record_outcome (always-on) skipped (%s)", e)
    else:
        logger.info(
            "meta: skipped self_prompt (always-on bandit) for class=%s",
            query_class,
        )

    result = ConsensusResult(
        answer=canonical.response,
        confidence=confidence,
        embedding_confidence=confidence if scoring_method == "embedding" else 0.0,
        scoring_method=scoring_method,
        models=responses,
        disagreements=disagreements,
        evolution_signals=evolution_signals,
        total_cost_usd=sum(r.cost_usd for r in responses),
        total_latency_ms=(time.perf_counter() - t_start) * 1000,
        router_used=router_names,
        hebbian_boost_applied=hebbian_mean,
        rlhf_weights_applied=rlhf_applied,
    )

    # Convergent-hallucination guard. See quorum.core.hallucination_risk
    # for the failure pattern this defends against (2026-06-28: six sub-
    # models agreed in a shared fictional world, producing 5/5 fabricated
    # facts at 78% confidence). Disabled if QUORUM_HALLUCINATION_GUARD=0
    # so a tight loop benchmarking pure agreement can opt out.
    if os.environ.get("QUORUM_HALLUCINATION_GUARD", "1") != "0":
        try:
            from quorum.core.hallucination_risk import (
                apply_risk_penalty,
                assess_hallucination_risk,
            )
            risk = assess_hallucination_risk(
                prompt, result.answer, confidence=result.confidence,
            )
            if risk.flags:
                # Always record the flags, even at low risk level — callers
                # may want to surface the warning to the user regardless of
                # whether we downgraded the score.
                result.hallucination_risk = risk.to_dict()
                if risk.suggested_penalty > 0:
                    new_conf = apply_risk_penalty(result.confidence, risk)
                    logger.warning(
                        "hallucination_guard: %s risk on %d flag(s); "
                        "confidence %.3f → %.3f",
                        risk.risk_level, len(risk.flags),
                        result.confidence, new_conf,
                    )
                    result.confidence = new_conf
        except Exception as e:  # noqa: BLE001
            # The guard MUST NEVER break a consensus call. It's a safety
            # net, not load-bearing — log and continue.
            logger.debug("hallucination_guard skipped (%s)", e)

    # Camada de re-síntese Quorum (opt-in via QUORUM_PERSONA=1).
    # Reescreve `result.answer` em voz unificada de Quorum, sintetizando
    # as N respostas dos sub-models. NÃO substitui canonical.response em
    # nenhum outro campo — modelos/disagreements/audit ficam inalterados
    # para auditabilidade. Failure mode: qualquer erro mantém answer original.
    if os.environ.get("QUORUM_PERSONA", "").lower() in ("1", "true", "yes"):
        try:
            from quorum.core.identity import synthesis_prompt
            valid_responses = [r for r in responses if r.response and not r.error]

            # Skip re-síntese se: (a) só 1 sub-model respondeu (nada a sintetizar);
            # (b) confidence muito alta (>= 0.92, todos os modelos concordam,
            #     re-rewrite só adiciona latência sem ganho); (c) resposta canônica
            #     já curta (< 200 chars, provavelmente saudação ou ack — re-rewrite
            #     produz meta-comentário confuso).
            canonical_len = len(canonical.response or "")
            should_skip = (
                len(valid_responses) < 2
                or confidence >= 0.92
                or canonical_len < 200
            )

            if should_skip:
                logger.info(
                    "quorum.persona: skipped re-synthesis (n=%d, conf=%.2f, "
                    "canonical_len=%d) — using canonical directly",
                    len(valid_responses), confidence, canonical_len,
                )
            else:
                labelled = [
                    (chr(ord("A") + i), r.response)
                    for i, r in enumerate(valid_responses)
                ]
                synth_prompt_text = synthesis_prompt(prompt, labelled)
                # Reusa o provider de menor latência observada nesta rodada
                fastest = min(valid_responses, key=lambda r: r.latency_ms)
                synth_provider = next(
                    (p for p in selected if p.name == fastest.name), selected[0]
                )
                # Budget de tempo para re-síntese: max 10s (não dobrar latency).
                synth_resp = await asyncio.wait_for(
                    synth_provider.complete(synth_prompt_text),
                    timeout=min(timeout_s, 10.0),
                )
                synth_text = (synth_resp.response or "").strip()
                if synth_text:
                    result.answer = synth_text
                    logger.info(
                        "quorum.persona: re-synthesised via %s (%d chars)",
                        synth_provider.name, len(synth_text),
                    )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "quorum.persona: re-synthesis failed (%s); keeping canonical answer", e,
            )

    # Loop 14 — HSP Black Box audit append. Tamper-evident chain for
    # EU AI Act Article 14 / SOC2 audit log compliance. Best-effort:
    # never poison the response path. Persists query hash (not text)
    # to avoid logging PII; full text stays in client memory only.
    try:
        from quorum.hsp.black_box import append as _audit_append
        import hashlib as _hashlib
        _audit_append({
            "event": "consensus",
            "query_hash": _hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "query_class": query_class,
            "confidence": confidence,
            "scoring_method": scoring_method,
            "models_invoked": [r.name for r in valid],
            "evolution_signals": evolution_signals,
            "total_cost_usd": result.total_cost_usd,
        })
    except Exception as e:  # noqa: BLE001
        logger.debug("hsp.black_box append skipped (%s)", e)

    # Raw response logging — opt-in via QUORUM_LOG_RESPONSES=1. Persists
    # each (query_hash, model, response_text) tuple so the dataset is
    # re-analysable later in a different embedding space without
    # paying for the LLM calls again. No-op when the env flag is off,
    # so a fresh clone sees zero behaviour change. Same fail-safe
    # contract as the audit chain above: never block the response.
    try:
        from quorum.evolution.response_log import (
            is_enabled as _resp_log_enabled,
            record_consensus_round as _resp_log_record,
        )
        if _resp_log_enabled():
            model_rows = [
                {
                    "model": r.name,
                    "response_text": r.response,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd,
                    "weight": r.weight,
                }
                for r in valid
            ]
            # Fire-and-forget so the disk write never delays the
            # consensus answer the caller is waiting on.
            asyncio.create_task(
                _resp_log_record(
                    prompt=prompt,
                    query_class=query_class,
                    model_responses=model_rows,
                    canonical_model=canonical.name,
                )
            )
    except Exception as e:  # noqa: BLE001
        logger.debug("response_log skipped (%s)", e)

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
    timeout_s: float = 60.0,
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
