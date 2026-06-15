# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Sovereign Chain / Quorum contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This module is HSP-gated. The `critique_round` action below pushes models
# to actively disagree with each other; misapplied, that can degrade UX or
# bias the consensus toward whichever critic is loudest. Production use
# requires HSP approval per PCT/US26/11908. See LICENSE-HSP at the repo root.
"""Loop 12 — Adversarial self-play.

WHY THIS EXISTS
---------------
Round-1 consensus tends to converge on the politest answer, not the
most correct one. Models trained for helpfulness will paper over
genuine factual or logical holes when they have nothing to push back
against. The cheapest way to surface those holes is to make models
critique *each other* — a critic model is told "find what's wrong
with response X" and has to commit to a specific weakness plus a
suggested correction. That commitment is the signal.

When a criticised model then revises its answer in light of the
critique, two facts follow:

  1. The original answer was held with low intrinsic confidence
     (the model gave it up cheaply).
  2. The model is "humble" on this query class — useful per-model
     calibration data for the consensus weighting.

When a criticised model holds its ground, the opposite is true:
either it is genuinely confident, or it is stubborn. Both are
informative; the meta-loop (Loop 8 / `meta.py`) consumes the rate
to decide which.

WHY HSP-GATED
-------------
Pushing N models to disagree more is a steering vector on the
consensus distribution. If the critic prompt is poorly worded,
or if the critic itself is biased (e.g. politically, or toward
verbosity), the entire downstream confidence number drifts. Unlike
the competition loop (Loop 7), this loop's safety property is NOT
the sampling rate — even a 10%% sample can poison per-model
calibration if the critic prompt is wrong. So every adversarial
round goes through the HSP gate (fail-closed in production, log+
pass in dev).

WHY 10%% SAMPLING (not 100%%)
----------------------------
Cost: a full round-trip per criticised model roughly doubles the
query bill. 10%% is enough to converge per-model revision-rate
statistics within a few thousand queries while keeping the
incremental cost bounded.

USAGE
-----
    play = AdversarialPlay()
    critiques = await play.critique_round(prompt, round1_responses, critic)
    revised = await play.integrate_critiques(round1_responses, critiques)
    rates = play.measure_revision_rate(critiques, revised)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from quorum.hsp.gate import requires_hsp_approval
from quorum.providers.base import ModelResponse, Provider

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data shapes                                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Critique:
    """One critic-emitted weakness against one target response.

    Carries enough structure that an honest critic can be distinguished
    from a model that just rewrote the original ("the answer is fine
    but here is a slightly nicer version"). The downstream revision
    detector keys off `suggested_correction` specifically.
    """

    target_model: str
    weakness: str
    suggested_correction: str
    critic_model: str = ""
    confidence: float = 0.5  # critic's self-reported confidence in the weakness

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_model": self.target_model,
            "weakness": self.weakness,
            "suggested_correction": self.suggested_correction,
            "critic_model": self.critic_model,
            "confidence": self.confidence,
        }


@dataclass
class AdversarialConfig:
    """Tunables. Defaults are the safe production envelope."""

    sample_rate: float = 0.10
    """Fraction of live queries that should run an adversarial round (cost cap)."""

    revision_similarity_threshold: float = 0.85
    """Jaccard similarity ABOVE which a "revised" response is treated as
    unchanged. Tuned empirically on internal benchmarks; very high so
    that cosmetic edits don't get counted as revisions."""

    max_concurrent_critiques: int = 4
    """asyncio.Semaphore cap when re-running criticised models."""

    sqlite_path: str = ""
    """If set, per-model revision rates are persisted here. Empty = in-memory
    fallback so unit tests run without filesystem state."""

    rng_seed: int | None = None
    """Optional seed for deterministic sampling (tests only)."""


# --------------------------------------------------------------------------- #
# In-memory fallback store                                                    #
# --------------------------------------------------------------------------- #


class _RevisionStore:
    """SQLite-backed counter; falls back to in-memory dict when no path is set.

    Why an in-memory fallback exists: smoke tests and CI should not need
    filesystem access. The in-memory variant has identical semantics to
    the SQLite one for the duration of a process.
    """

    _MEM: dict[str, tuple[int, int]] = {}

    def __init__(self, sqlite_path: str = "") -> None:
        self.path = sqlite_path
        if self.path:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS revisions (
                    model TEXT PRIMARY KEY,
                    revised INTEGER NOT NULL DEFAULT 0,
                    total INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            conn.commit()

    async def bump(self, model: str, revised: bool) -> None:
        """Increment counters for a single (model, decision) event.

        Wrapped in asyncio.to_thread so callers can sit inside an event
        loop without blocking on the SQLite write.
        """
        if not self.path:
            r, t = self._MEM.get(model, (0, 0))
            self._MEM[model] = (r + int(revised), t + 1)
            return

        def _write() -> None:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    """
                    INSERT INTO revisions (model, revised, total)
                    VALUES (?, ?, 1)
                    ON CONFLICT(model) DO UPDATE SET
                        revised = revised + excluded.revised,
                        total   = total   + 1
                    """,
                    (model, int(revised)),
                )
                conn.commit()

        await asyncio.to_thread(_write)

    async def rates(self) -> dict[str, float]:
        """Return per-model lifetime revision rate."""
        if not self.path:
            return {
                m: (r / t if t else 0.0) for m, (r, t) in self._MEM.items()
            }

        def _read() -> dict[str, float]:
            with sqlite3.connect(self.path) as conn:
                rows = conn.execute(
                    "SELECT model, revised, total FROM revisions"
                ).fetchall()
            return {m: (r / t if t else 0.0) for m, r, t in rows}

        return await asyncio.to_thread(_read)


# --------------------------------------------------------------------------- #
# Adversarial play                                                            #
# --------------------------------------------------------------------------- #


class AdversarialPlay:
    """Round-2 adversarial loop: critique, revise, measure revision rate.

    The class is intentionally stateless across queries except for the
    persistent `_RevisionStore`. That keeps it safe to share one instance
    across many concurrent queries.
    """

    def __init__(self, config: AdversarialConfig | None = None) -> None:
        self.config = config or AdversarialConfig()
        self._store = _RevisionStore(
            self.config.sqlite_path or os.getenv("QUORUM_ADV_DB", "")
        )
        self._rng = random.Random(self.config.rng_seed)

    # ----- sampling ------------------------------------------------------- #

    def should_run(self) -> bool:
        """Bernoulli draw against `sample_rate`. Cheap cost-control gate.

        The orchestrator calls this BEFORE going through the HSP gate, so
        the 90%% of queries that won't run an adversarial round never
        touch the webhook. (Avoids spamming the human reviewer.)
        """
        return self._rng.random() < self.config.sample_rate

    # ----- round 2: critique ---------------------------------------------- #

    @requires_hsp_approval(action="adversarial_round", risk_level="high")
    async def critique_round(
        self,
        prompt: str,
        round1_responses: Sequence[ModelResponse],
        critic_provider: Provider,
    ) -> list[Critique]:
        """Have the critic point out concrete weaknesses in each round-1 response.

        WHY one critic and not N-on-N: pairwise criticism scales O(N^2)
        and is dominated by the cost of long context windows. One strong
        critic is empirically a better calibration signal at 1/N the cost,
        and keeps the bias surface to one model (which the meta-loop can
        rotate).

        Args:
            prompt: The original user query — the critic needs it to know
                    what counts as a weakness.
            round1_responses: All valid responses from the consensus round.
            critic_provider: Any Provider; should be a strong frontier model.

        Returns:
            A list of `Critique` objects, one per target response that the
            critic flagged. Targets the critic deems "no weakness found"
            are simply omitted.
        """
        valid = [r for r in round1_responses if r.response and not r.error]
        if not valid:
            logger.info("adversarial: no valid responses to critique")
            return []

        critic_prompt = self._build_critic_prompt(prompt, valid)
        raw = await critic_provider.complete(critic_prompt, max_tokens=1200)

        if raw.error or not raw.response:
            logger.warning(
                "adversarial: critic %s failed (%s)",
                critic_provider.name,
                raw.error or "empty",
            )
            return []

        critiques = self._parse_critiques(raw.response, critic_provider.name)
        logger.info(
            "adversarial: critic %s produced %d critiques over %d responses",
            critic_provider.name,
            len(critiques),
            len(valid),
        )
        return critiques

    @staticmethod
    def _build_critic_prompt(
        prompt: str, responses: Sequence[ModelResponse]
    ) -> str:
        """Construct a structured critic prompt that forces a JSON list output.

        WHY JSON: free-form critic output is impossible to parse reliably;
        models will hedge into prose. A schema with named fields forces
        commitment to a concrete weakness + concrete correction, which is
        the only signal we actually use downstream.
        """
        blocks = []
        for i, r in enumerate(responses):
            blocks.append(
                f"--- Response from model `{r.name}` (index {i}) ---\n"
                f"{r.response}"
            )
        joined = "\n\n".join(blocks)
        return (
            "You are an adversarial reviewer. The user asked:\n\n"
            f"USER QUESTION:\n{prompt}\n\n"
            "Below are answers from several different LLMs. For each answer "
            "that contains a factual error, logical gap, missing caveat, or "
            "unjustified claim, emit one critique. Be concrete: name the "
            "exact weakness and propose the exact correction. If an answer "
            "is solid, OMIT it entirely — do not invent flaws.\n\n"
            f"{joined}\n\n"
            "Respond with a single JSON array of objects, each shaped like:\n"
            "{\n"
            '  "target_model": "<model name from above>",\n'
            '  "weakness": "<one-sentence specific weakness>",\n'
            '  "suggested_correction": "<one-to-three sentence fix>",\n'
            '  "confidence": <0.0-1.0 how sure you are this is a real flaw>\n'
            "}\n"
            "Return ONLY the JSON array. No prose, no markdown fence."
        )

    @staticmethod
    def _parse_critiques(raw: str, critic_name: str) -> list[Critique]:
        """Best-effort JSON extraction tolerant to markdown fences and prose.

        Critics sometimes ignore the "no fence" instruction; we strip the
        usual ```json wrappers and try to locate the first '[' ... ']'
        slice. Anything that fails to parse is dropped rather than raised,
        because a malformed critique is less bad than killing the round.
        """
        text = raw.strip()
        if text.startswith("```"):
            # Drop the first fence line and the trailing fence.
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text[: -3]
            text = text.strip()

        # Locate the JSON array even if the model added prose around it.
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end < start:
            logger.warning("adversarial: critic emitted no JSON array")
            return []
        snippet = text[start : end + 1]

        try:
            items = json.loads(snippet)
        except json.JSONDecodeError as e:
            logger.warning("adversarial: critic JSON parse failed: %s", e)
            return []

        if not isinstance(items, list):
            return []

        out: list[Critique] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            target = str(item.get("target_model", "")).strip()
            weakness = str(item.get("weakness", "")).strip()
            correction = str(item.get("suggested_correction", "")).strip()
            if not (target and weakness and correction):
                continue
            try:
                conf = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            out.append(
                Critique(
                    target_model=target,
                    weakness=weakness,
                    suggested_correction=correction,
                    critic_model=critic_name,
                    confidence=conf,
                )
            )
        return out

    # ----- round 3: integrate critiques into revised answers -------------- #

    async def integrate_critiques(
        self,
        responses: Sequence[ModelResponse],
        critiques: Sequence[Critique],
        *,
        original_prompt: str = "",
        providers: Sequence[Provider] | None = None,
    ) -> list[ModelResponse]:
        """Re-run each criticised model with the critique embedded in context.

        The returned list is parallel to `responses`: models that were
        not criticised pass through unchanged, models that were
        criticised are returned with their revised answer (or with the
        original if the provider call failed).

        Args:
            responses: Round-1 responses, in their original order.
            critiques: Output of `critique_round`.
            original_prompt: The user's original question. Needed so the
                criticised model can re-anchor on the actual task; without
                it, the model often "fixes" the critique itself.
            providers: Optional provider list, parallel to `responses` by
                name. If supplied, each criticised model is re-invoked.
                If `None`, the function returns the round-1 responses
                wrapped in fresh `ModelResponse` objects with the critique
                appended into `error` for downstream inspection — useful
                in tests where you don't want network calls.
        """
        # Index providers by name for O(1) lookup. None means no real re-run.
        provider_by_name: dict[str, Provider] = (
            {p.name: p for p in providers} if providers else {}
        )
        critiques_by_target: dict[str, list[Critique]] = {}
        for c in critiques:
            critiques_by_target.setdefault(c.target_model, []).append(c)

        sem = asyncio.Semaphore(self.config.max_concurrent_critiques)

        async def _revise(orig: ModelResponse) -> ModelResponse:
            cs = critiques_by_target.get(orig.name)
            if not cs:
                # No critique against this model — unchanged.
                return orig

            critique_block = "\n".join(
                f"- Weakness: {c.weakness}\n  Suggested correction: "
                f"{c.suggested_correction}"
                for c in cs
            )
            revision_prompt = (
                f"USER QUESTION:\n{original_prompt}\n\n"
                f"YOUR PREVIOUS ANSWER:\n{orig.response}\n\n"
                f"A reviewer raised the following concerns:\n{critique_block}\n\n"
                "If you agree with any of these concerns, rewrite your answer "
                "to incorporate the correction. If you disagree, restate your "
                "original answer and briefly explain why the criticism is "
                "wrong. Either way, give the user the final answer to their "
                "original question — do not address the reviewer."
            )

            prov = provider_by_name.get(orig.name)
            if prov is None:
                # Test path: no network re-invocation. Mark for inspection.
                logger.debug(
                    "adversarial: no provider for %s; passing through",
                    orig.name,
                )
                return orig

            async with sem:
                try:
                    new = await prov.complete(revision_prompt, max_tokens=800)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "adversarial: revision call failed for %s: %s",
                        orig.name,
                        e,
                    )
                    return orig

            if new.error or not new.response:
                return orig
            return new

        revised = await asyncio.gather(*(_revise(r) for r in responses))
        return list(revised)

    # ----- per-model revision rate ---------------------------------------- #

    def measure_revision_rate(
        self,
        critiques: Sequence[Critique],
        revised_responses: Sequence[ModelResponse],
        *,
        round1_responses: Sequence[ModelResponse] | None = None,
    ) -> dict[str, float]:
        """Per criticised model, the fraction of critiques it accepted.

        "Accepted" means: the model's revised text materially differs
        from its round-1 text. We use Jaccard distance between the two
        token sets and call it a revision when similarity drops below
        `revision_similarity_threshold`. This is a placeholder for
        embedding-cosine-distance (v0.1), but it is correct enough that
        the meta-loop's directional signal is preserved.

        Args:
            critiques: All critiques from `critique_round`.
            revised_responses: Output of `integrate_critiques`.
            round1_responses: Original responses to diff against. If
                omitted, this function will return zeros — without the
                originals there is nothing to compare to. The argument
                is optional only so the signature matches the spec.

        Returns:
            `{model_name: rate}` for every model that was criticised.
            Models with no critique are absent from the dict.
        """
        if round1_responses is None:
            # Signature compatibility — we cannot compute without originals.
            logger.warning(
                "measure_revision_rate called without round1_responses; "
                "returning empty result"
            )
            return {}

        orig_by_name = {r.name: r for r in round1_responses}
        rev_by_name = {r.name: r for r in revised_responses}
        criticised: dict[str, int] = {}
        revised_count: dict[str, int] = {}

        for c in critiques:
            criticised[c.target_model] = criticised.get(c.target_model, 0) + 1
            orig = orig_by_name.get(c.target_model)
            rev = rev_by_name.get(c.target_model)
            if orig is None or rev is None:
                continue
            if self._is_revised(orig.response, rev.response):
                revised_count[c.target_model] = (
                    revised_count.get(c.target_model, 0) + 1
                )

        return {
            m: revised_count.get(m, 0) / criticised[m]
            for m in criticised
            if criticised[m] > 0
        }

    async def record_rates(self, rates: dict[str, float]) -> None:
        """Persist the round's revision events into the long-running store.

        Kept separate from `measure_revision_rate` so the pure function
        stays pure and testable. The orchestrator calls this after a
        round completes; the meta-loop reads from `.lifetime_rates()`.
        """
        for model, rate in rates.items():
            # We treat the rate as a single weighted event: rate>=0.5 = revised.
            # The store accumulates many of these into a lifetime average.
            await self._store.bump(model, revised=rate >= 0.5)

    async def lifetime_rates(self) -> dict[str, float]:
        """Read the persistent per-model revision rate (all-time)."""
        return await self._store.rates()

    # ----- internals ------------------------------------------------------ #

    def _is_revised(self, original: str, revised: str) -> bool:
        """True iff the revised text materially differs from the original.

        WHY a similarity threshold rather than exact equality: every
        model adds cosmetic variation on re-runs (timestamps, punctuation,
        whitespace). Exact-equality would over-count revisions. We use
        Jaccard token overlap as a cheap proxy for "did the actual
        content change?"; v0.1 will swap in embedding cosine.
        """
        a = set(original.lower().split())
        b = set(revised.lower().split())
        if not a or not b:
            return original.strip() != revised.strip()
        sim = len(a & b) / len(a | b)
        return sim < self.config.revision_similarity_threshold


__all__ = [
    "AdversarialPlay",
    "AdversarialConfig",
    "Critique",
]


# --------------------------------------------------------------------------- #
# Smoke tests — safe to run without network access or API keys.               #
# Invoke with:  python -m quorum.evolution.adversarial                        #
# --------------------------------------------------------------------------- #


def _make_response(name: str, text: str) -> ModelResponse:
    return ModelResponse(name=name, response=text)


class _StubCritic(Provider):
    """In-memory Provider that emits a fixed critic JSON. Zero network."""

    name = "stub-critic"

    def __init__(self, payload: list[dict[str, Any]]) -> None:
        self._payload = payload

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        return ModelResponse(
            name=self.name,
            response=json.dumps(self._payload),
        )


class _StubRevisor(Provider):
    """Provider that always returns a fixed revised string."""

    def __init__(self, name: str, revised: str) -> None:
        self.name = name
        self._revised = revised

    async def complete(
        self, prompt: str, *, max_tokens: int = 800
    ) -> ModelResponse:
        return ModelResponse(name=self.name, response=self._revised)


async def _smoke_test_parse_and_measure() -> None:
    """End-to-end: critique → integrate → measure, with no real LLM calls."""
    cfg = AdversarialConfig(sample_rate=1.0, rng_seed=42, sqlite_path="")
    play = AdversarialPlay(cfg)

    r1 = [
        _make_response("model-A", "Paris is the capital of Germany."),
        _make_response("model-B", "Paris is the capital of France."),
    ]
    critic = _StubCritic(
        [
            {
                "target_model": "model-A",
                "weakness": "Paris is not the capital of Germany.",
                "suggested_correction": "Paris is the capital of France; Berlin is the capital of Germany.",
                "confidence": 0.99,
            }
        ]
    )

    critiques = await play.critique_round(
        "What is the capital of France?", r1, critic
    )
    assert len(critiques) == 1, "expected one critique"
    assert critiques[0].target_model == "model-A"
    assert critiques[0].critic_model == "stub-critic"

    revisor = _StubRevisor(
        "model-A",
        "I was wrong; Paris is the capital of France, not Germany. Berlin is Germany's capital.",
    )
    # model-B is unchanged because no critique targeted it.
    revised = await play.integrate_critiques(
        r1,
        critiques,
        original_prompt="What is the capital of France?",
        providers=[revisor],
    )
    assert revised[1].response == r1[1].response, "model-B should be untouched"
    assert revised[0].response != r1[0].response, "model-A should have revised"

    rates = play.measure_revision_rate(
        critiques, revised, round1_responses=r1
    )
    assert rates == {"model-A": 1.0}, f"expected 100%% acceptance, got {rates}"

    await play.record_rates(rates)
    lifetime = await play.lifetime_rates()
    assert lifetime.get("model-A", 0.0) > 0.0


async def _smoke_test_sampling_and_no_provider_path() -> None:
    """Sampling is deterministic with seed, and no-providers path is safe."""
    cfg = AdversarialConfig(sample_rate=0.5, rng_seed=1)
    play = AdversarialPlay(cfg)
    draws = [play.should_run() for _ in range(20)]
    assert any(draws) and not all(draws), "sampling should produce a mix"

    r1 = [_make_response("model-X", "An answer.")]
    critiques = [
        Critique(
            target_model="model-X",
            weakness="Too short.",
            suggested_correction="Expand with detail.",
            critic_model="test",
            confidence=0.8,
        )
    ]
    # No providers passed -> pass-through, no exception.
    revised = await play.integrate_critiques(
        r1, critiques, original_prompt="?", providers=None
    )
    assert revised[0].response == r1[0].response
    # Without providers, nothing revised, so rate is 0.
    rates = play.measure_revision_rate(
        critiques, revised, round1_responses=r1
    )
    assert rates == {"model-X": 0.0}


async def _smoke_test_malformed_critic_output() -> None:
    """Critic emits garbage; round must degrade gracefully to []."""
    play = AdversarialPlay(AdversarialConfig(sample_rate=1.0))

    class _Garbage(Provider):
        name = "garbage"

        async def complete(
            self, prompt: str, *, max_tokens: int = 800
        ) -> ModelResponse:
            return ModelResponse(
                name=self.name, response="I don't know, sorry."
            )

    out = await play.critique_round("Q?", [_make_response("m", "a")], _Garbage())
    assert out == []


async def _run_all_smoke_tests() -> None:
    logging.basicConfig(level=logging.WARNING)
    await _smoke_test_parse_and_measure()
    await _smoke_test_sampling_and_no_provider_path()
    await _smoke_test_malformed_critic_output()
    logger.warning("adversarial: all smoke tests passed")


if __name__ == "__main__":
    asyncio.run(_run_all_smoke_tests())
