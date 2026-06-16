# Copyright 2026 Jaqueline Martins / Sovereign Chain Ltd.
# SPDX-License-Identifier: Apache-2.0 WITH HSP-Commercial-Restrictions
"""Tests for ``HebbianStore`` — the per-(pair, query_class) EMA store.

WHY a separate test module from the legacy ``HebbianMatrix`` smoke tests:
HebbianStore has a different invariant (EMA convergence + sample threshold)
and a different storage table. Co-locating would obscure regressions; a
named module keeps each loop's failure modes attributable.

Test strategy
-------------
We synthesise 100 rounds of three models — A, B, C — where:
    * A and B agree perfectly every round (identical responses).
    * C consistently dissents (disjoint vocabulary).

After all 100 observations we assert that:
    * ``boost(A, B, "general")`` > ``boost(A, C, "general")`` strictly.
    * The advantage exceeds the Wilson lower bound on the difference,
      so the result isn't an artefact of the small sample size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from quorum.evolution.hebbian import (
    EMA_RATE,
    SAMPLE_THRESHOLD,
    STORE_MAX_BOOST,
    STORE_MIN_BOOST,
    HebbianStore,
)


@dataclass
class _FakeResponse:
    """Duck-typed ``ModelResponse`` — only the fields HebbianStore reads.

    WHY a fake instead of importing ``quorum.providers.base.ModelResponse``:
    keeps the test hermetic from provider auto-discovery and bills, and
    documents the *narrow* surface the store actually depends on.
    """

    name: str
    response: str
    error: str = ""


def _wilson_lower_bound(successes: int, trials: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial proportion.

    Standard 95% (z=1.96) two-sided. We use this to check that the
    advantage of pair (A,B) over (A,C) is statistically meaningful given
    the sample count, not just a point estimate fluke.
    """
    if trials == 0:
        return 0.0
    phat = successes / trials
    denom = 1.0 + z * z / trials
    centre = phat + z * z / (2.0 * trials)
    margin = z * math.sqrt(phat * (1.0 - phat) / trials + z * z / (4.0 * trials * trials))
    return max(0.0, (centre - margin) / denom)


@pytest.fixture
def store(tmp_path: Path) -> HebbianStore:
    """Fresh on-disk store per test under tmp_path/hebbian_store.db.

    Each test gets its own SQLite file so parallel pytest runs don't share
    state — important now that the store keys are per-(pair, class).
    """
    return HebbianStore(db_path=tmp_path / "hebbian_store.db")


@pytest.mark.asyncio
async def test_boost_returns_neutral_below_sample_threshold(store: HebbianStore) -> None:
    """Cold-start safety: until SAMPLE_THRESHOLD obs, boost is always 1.0.

    WHY the threshold exists: new providers shouldn't be penalised for
    lacking history, and a single lucky alignment shouldn't move weights.
    """
    responses = [
        _FakeResponse("alpha", "the quick brown fox"),
        _FakeResponse("beta", "the quick brown fox"),
    ]
    # Below threshold — observe a few rounds.
    for _ in range(SAMPLE_THRESHOLD - 1):
        await store.observe("general", responses)

    boost = await store.boost("alpha", "beta", "general")
    assert boost == STORE_MIN_BOOST, (
        f"boost {boost} should still be neutral with "
        f"{SAMPLE_THRESHOLD - 1} samples (< threshold {SAMPLE_THRESHOLD})"
    )


@pytest.mark.asyncio
async def test_observe_and_boost_distinguish_agreers_from_dissenter(
    store: HebbianStore,
) -> None:
    """Core invariant: A+B agree 100 rounds, C dissents → boost(A,B) > boost(A,C).

    The test does not rely on the exact EMA value (that's a tunable); it
    asserts the *ordering* and the Wilson-bounded statistical strength of
    the gap.
    """
    n_rounds = 100
    successes_ab_gt_ac = 0

    for _ in range(n_rounds):
        responses = [
            # A and B emit identical token sets → Jaccard = 1.0.
            _FakeResponse("alpha", "yes the answer is forty two"),
            _FakeResponse("beta",  "yes the answer is forty two"),
            # C uses a fully disjoint vocabulary → Jaccard = 0.0 vs A and B.
            _FakeResponse("gamma", "no completely different output entirely"),
        ]
        # Observe BEFORE checking — we want the post-observation ordering.
        await store.observe("general", responses)

        # After each round we sample the ordering as a Bernoulli trial:
        # 1 if boost(A,B) > boost(A,C), else 0. Wilson over this sequence
        # measures how reliably the EMA prefers the true co-firing pair.
        b_ab = await store.boost("alpha", "beta", "general")
        b_ac = await store.boost("alpha", "gamma", "general")
        if b_ab > b_ac:
            successes_ab_gt_ac += 1

    # Final point estimates after all 100 rounds.
    final_ab = await store.boost("alpha", "beta", "general")
    final_ac = await store.boost("alpha", "gamma", "general")

    # Sanity bounds: all boosts live in [MIN, MAX].
    assert STORE_MIN_BOOST <= final_ab <= STORE_MAX_BOOST
    assert STORE_MIN_BOOST <= final_ac <= STORE_MAX_BOOST

    # Core assertion — the agreeing pair must out-boost the dissenting one.
    assert final_ab > final_ac, (
        f"agreeing pair boost ({final_ab:.3f}) should exceed dissenting "
        f"pair boost ({final_ac:.3f})"
    )

    # The agreeing pair should converge near the ceiling (linear map of
    # EMA score 1.0 → STORE_MAX_BOOST). EMA rate 0.1 with seed=1.0 stays
    # at 1.0 forever (every obs = 1.0). Be tolerant: just require it has
    # actually moved above the neutral floor by a clear margin.
    assert final_ab >= STORE_MIN_BOOST + 0.4 * (STORE_MAX_BOOST - STORE_MIN_BOOST), (
        f"agreeing pair boost {final_ab:.3f} suspiciously low after 100 rounds"
    )

    # The dissenting pair must stay at the floor — its score never rises.
    assert math.isclose(final_ac, STORE_MIN_BOOST, abs_tol=1e-9), (
        f"dissenting pair boost {final_ac:.3f} should equal MIN ({STORE_MIN_BOOST})"
    )

    # Wilson lower bound on "ordered correctly" across the trials. We only
    # count rounds AFTER the sample threshold was met (otherwise both sides
    # tied at 1.0 and the result is uninformative).
    counted_trials = n_rounds - (SAMPLE_THRESHOLD - 1)
    counted_successes = min(successes_ab_gt_ac, counted_trials)
    wlb = _wilson_lower_bound(counted_successes, counted_trials)
    # We require at least 90% confidence that the EMA prefers the right
    # pair — anything lower would mean the loop isn't usable in production.
    assert wlb >= 0.9, (
        f"Wilson lower bound {wlb:.3f} on ordering reliability is too low: "
        f"{counted_successes}/{counted_trials} ordered correctly post-threshold."
    )


@pytest.mark.asyncio
async def test_per_class_isolation(store: HebbianStore) -> None:
    """Per-(pair, query_class) means observations in class X don't leak to Y.

    The point of the per-class shard is specialisation — code-style
    alignment shouldn't bleed into chat-style. We assert that by observing
    rounds *only* under class "code" and verifying class "chat" remains
    untouched (neutral boost).
    """
    responses = [
        _FakeResponse("alpha", "matching response text"),
        _FakeResponse("beta",  "matching response text"),
    ]
    for _ in range(SAMPLE_THRESHOLD * 2):
        await store.observe("code", responses)

    code_boost = await store.boost("alpha", "beta", "code")
    chat_boost = await store.boost("alpha", "beta", "chat")
    assert code_boost > chat_boost, "code class should be boosted, chat untouched"
    assert math.isclose(chat_boost, STORE_MIN_BOOST, abs_tol=1e-9), (
        f"chat class leaked from code class: chat_boost={chat_boost}"
    )


@pytest.mark.asyncio
async def test_observe_skips_errored_responses(store: HebbianStore) -> None:
    """An errored response forms no valid pair; observe() must skip it.

    Otherwise an upstream provider crash would pull every other model's
    co-activation down toward zero — exactly the wrong gradient.
    """
    responses = [
        _FakeResponse("alpha", "valid output"),
        _FakeResponse("beta", "", error="timeout"),
    ]
    updated = await store.observe("general", responses)
    assert updated == 0, "no pair should be recorded when only one model is valid"


@pytest.mark.asyncio
async def test_observe_validates_query_class(store: HebbianStore) -> None:
    """Empty query_class is a programming error, not a silent no-op."""
    responses = [
        _FakeResponse("alpha", "x"),
        _FakeResponse("beta", "x"),
    ]
    with pytest.raises(ValueError):
        await store.observe("", responses)


@pytest.mark.asyncio
async def test_self_pair_boost_is_neutral(store: HebbianStore) -> None:
    """A model paired with itself must always return MIN — no self-promotion."""
    responses = [
        _FakeResponse("alpha", "x"),
        _FakeResponse("beta", "x"),
    ]
    for _ in range(SAMPLE_THRESHOLD * 2):
        await store.observe("general", responses)

    self_boost = await store.boost("alpha", "alpha", "general")
    assert self_boost == STORE_MIN_BOOST


# Use module-level pytest-asyncio config so we don't need a fixture per test.
pytestmark = pytest.mark.asyncio
